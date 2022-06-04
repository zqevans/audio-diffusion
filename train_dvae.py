#!/usr/bin/env python3

import argparse
from contextlib import contextmanager
from copy import deepcopy
import math
from pathlib import Path

import sys
import torch
from torch import optim, nn
from torch.nn import functional as F
from torch.utils import data
from tqdm import trange
import pytorch_lightning as pl
from pytorch_lightning.utilities.distributed import rank_zero_only
from einops import rearrange

import torchaudio
import wandb

from dataset.dataset import SampleDataset
from diffusion.model import SkipBlock, FourierFeatures, expand_to_planes, ema_update
from diffusion.pqmf import CachedPQMF as PQMF
from encoders.encoders import RAVEEncoder, ResConvBlock, SoundStreamXLEncoder

from nwt_pytorch import Memcodes
from dvae.residual_memcodes import ResidualMemcodes

class DiffusionDecoder(nn.Module):
    def __init__(self, latent_dim, io_channels, depth=16):
        super().__init__()
        max_depth = 16
        depth = min(depth, max_depth)
        c_mults = [256, 256, 512, 512] + [512] * 12
        c_mults = c_mults[:depth]

        self.io_channels = io_channels
        self.timestep_embed = FourierFeatures(1, 16)
        block = nn.Identity()
        for i in range(depth, 0, -1):
            c = c_mults[i - 1]
            if i > 1:
                c_prev = c_mults[i - 2]
                block = SkipBlock(
                    nn.AvgPool1d(2),
                    ResConvBlock(c_prev, c, c),
                    ResConvBlock(c, c, c),
                    ResConvBlock(c, c, c),
                    block,
                    ResConvBlock(c * 2 if i != depth else c, c, c),
                    ResConvBlock(c, c, c),
                    ResConvBlock(c, c, c_prev),
                    nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
                )
            else:
                block = nn.Sequential(
                    ResConvBlock(io_channels + 16 + latent_dim, c, c),
                    ResConvBlock(c, c, c),
                    ResConvBlock(c, c, c),
                    block,
                    ResConvBlock(c * 2, c, c),
                    ResConvBlock(c, c, c),
                    ResConvBlock(c, c, io_channels, is_last=True),
                )
        self.net = block

    def forward(self, input, t, quantized):
        timestep_embed = expand_to_planes(self.timestep_embed(t[:, None]), input.shape)
        quantized = F.interpolate(quantized, (input.shape[2], ), mode='linear', align_corners=False)
        return self.net(torch.cat([input, timestep_embed, quantized], dim=1))

# Define the noise schedule and sampling loop
def get_alphas_sigmas(t):
    """Returns the scaling factors for the clean image (alpha) and for the
    noise (sigma), given a timestep."""
    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)

def get_crash_schedule(t):
    sigma = torch.sin(t * math.pi / 2) ** 2
    alpha = (1 - sigma ** 2) ** 0.5
    return alpha_sigma_to_t(alpha, sigma)

def alpha_sigma_to_t(alpha, sigma):
    """Returns a timestep, given the scaling factors for the clean image and for
    the noise."""
    return torch.atan2(sigma, alpha) / math.pi * 2

@torch.no_grad()
def sample(model, x, steps, eta, logits):
    """Draws samples from a model given starting noise."""
    ts = x.new_ones([x.shape[0]])

    # Create the noise schedule
    t = torch.linspace(1, 0, steps + 1)[:-1]
    alphas, sigmas = get_alphas_sigmas(get_crash_schedule(t))

    # The sampling loop
    for i in trange(steps):

        # Get the model output (v, the predicted velocity)
        with torch.cuda.amp.autocast():
            v = model(x, ts * t[i], logits).float()

        # Predict the noise and the denoised image
        pred = x * alphas[i] - v * sigmas[i]
        eps = x * sigmas[i] + v * alphas[i]

        # If we are not on the last timestep, compute the noisy image for the
        # next timestep.
        if i < steps - 1:
            # If eta > 0, adjust the scaling factor for the predicted noise
            # downward according to the amount of additional noise to add
            ddim_sigma = eta * (sigmas[i + 1]**2 / sigmas[i]**2).sqrt() * \
                (1 - alphas[i]**2 / alphas[i + 1]**2).sqrt()
            adjusted_sigma = (sigmas[i + 1]**2 - ddim_sigma**2).sqrt()

            # Recombine the predicted noise and predicted denoised image in the
            # correct proportions for the next step
            x = pred * alphas[i + 1] + eps * adjusted_sigma

            # Add the correct amount of fresh noise
            if eta:
                x += torch.randn_like(x) * ddim_sigma

    # If we are on the last timestep, output the denoised image
    return pred


class ToMode:
    def __init__(self, mode):
        self.mode = mode

    def __call__(self, image):
        return image.convert(self.mode)


def ramp(x1, x2, y1, y2):
    def wrapped(x):
        if x <= x1:
            return y1
        if x >= x2:
            return y2
        fac = (x - x1) / (x2 - x1)
        return y1 * (1 - fac) + y2 * fac
    return wrapped


class DiffusionDVAE(pl.LightningModule):
    def __init__(self, global_args):
        super().__init__()

        #self.encoder = Encoder(global_args.codebook_size, 2)
        #self.encoder = SoundStreamXLEncoder(32, global_args.latent_dim, n_io_channels=2, strides=[2, 2, 4, 5, 8], c_mults=[2, 4, 4, 8, 16])
        
        self.pqmf_bands = global_args.pqmf_bands

        if self.pqmf_bands > 1:
            self.pqmf = PQMF(2, 70, global_args.pqmf_bands)

        self.encoder = RAVEEncoder(2 * global_args.pqmf_bands, 64, global_args.latent_dim, ratios=[2, 2, 2, 2, 4, 4])
        self.encoder_ema = deepcopy(self.encoder)
        self.diffusion = DiffusionDecoder(global_args.latent_dim, 2)
        self.diffusion_ema = deepcopy(self.diffusion)
        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)
        self.ema_decay = global_args.ema_decay
        
        self.num_quantizers = global_args.num_quantizers
        if self.num_quantizers > 0:
            quantizer_class = ResidualMemcodes if global_args.num_quantizers > 1 else Memcodes
            
            quantizer_kwargs = {}
            if global_args.num_quantizers > 1:
                quantizer_kwargs["num_quantizers"] = global_args.num_quantizers

            self.quantizer = quantizer_class(
                dim=global_args.latent_dim,
                heads=global_args.num_heads,
                num_codes=global_args.codebook_size,
                temperature=1.,
                **quantizer_kwargs
            )

            self.quantizer_ema = deepcopy(self.quantizer)

        

    def encode(self, *args, **kwargs):
        if self.training:
            return self.encoder(*args, **kwargs)
        return self.encoder_ema(*args, **kwargs)

    def decode(self, *args, **kwargs):
        if self.training:
            return self.diffusion(*args, **kwargs)
        return self.diffusion_ema(*args, **kwargs)

    def configure_optimizers(self):
        return optim.Adam([*self.encoder.parameters(), *self.diffusion.parameters()], lr=2e-5)

  
    def training_step(self, batch, batch_idx):
        reals = batch[0]

        encoder_input = reals

        if self.pqmf_bands > 1:
            encoder_input = self.pqmf(reals)
        
        # Draw uniformly distributed continuous timesteps
        t = self.rng.draw(reals.shape[0])[:, 0].to(self.device)

        # Calculate the noise schedule parameters for those timesteps
        alphas, sigmas = get_alphas_sigmas(get_crash_schedule(t))

        # Combine the ground truth images and the noise
        alphas = alphas[:, None, None]
        sigmas = sigmas[:, None, None]
        noise = torch.randn_like(reals)
        noised_reals = reals * alphas + noise * sigmas
        targets = noise * alphas - reals * sigmas

        # Compute the model output and the loss.
        with torch.cuda.amp.autocast():
            tokens = self.encoder(encoder_input).float()

        if self.num_quantizers > 0:
            #Rearrange for Memcodes
            tokens = rearrange(tokens, 'b d n -> b n d')

            #Quantize into memcodes
            tokens, _ = self.quantizer(tokens)

            tokens = rearrange(tokens, 'b n d -> b d n')

        # p = torch.rand([reals.shape[0], 1], device=reals.device)
        # tokens = torch.where(p > 0.2, tokens, torch.zeros_like(tokens))

        with torch.cuda.amp.autocast():
            v = self.diffusion(noised_reals, t, tokens)
            mse_loss = F.mse_loss(v, targets)
            loss = mse_loss

        log_dict = {
            'train/loss': loss.detach(),
            'train/mse_loss': mse_loss.detach(),
        }

        self.log_dict(log_dict, prog_bar=True, on_step=True)
        return loss

    def on_before_zero_grad(self, *args, **kwargs):
        decay = 0.95 if self.current_epoch < 25 else self.ema_decay
        ema_update(self.diffusion, self.diffusion_ema, decay)
        ema_update(self.encoder, self.encoder_ema, decay)

        if self.num_quantizers > 0:
            ema_update(self.quantizer, self.quantizer_ema, decay)

class ExceptionCallback(pl.Callback):
    def on_exception(self, trainer, module, err):
        print(f'{type(err).__name__}: {err}', file=sys.stderr)


class DemoCallback(pl.Callback):
    def __init__(self, demo_dl, global_args):
        super().__init__()
        self.demo_every = global_args.demo_every
        self.demo_samples = global_args.sample_size
        self.demo_steps = global_args.demo_steps
        self.demo_dl = iter(demo_dl)
        self.sample_rate = global_args.sample_rate
        self.pqmf_bands = global_args.pqmf_bands
        self.quantized = global_args.num_quantizers > 0

        if self.pqmf_bands > 1:
            self.pqmf = PQMF(2, 70, global_args.pqmf_bands)

    @rank_zero_only
    @torch.no_grad()
    def on_train_epoch_end(self, trainer, module):
        #last_demo_step = -1
        #if (trainer.global_step - 1) % self.demo_every != 0 or last_demo_step == trainer.global_step:
        if trainer.current_epoch % self.demo_every != 0:
            return
        
        #last_demo_step = trainer.global_step

        demo_reals, _ = next(self.demo_dl)

        encoder_input = demo_reals
        
        if self.pqmf_bands > 1:
            encoder_input = self.pqmf(demo_reals)
        
        encoder_input = encoder_input.to(module.device)

        demo_reals = demo_reals.to(module.device)

        noise = torch.randn([demo_reals.shape[0], 2, self.demo_samples]).to(module.device)

        tokens = module.encoder_ema(encoder_input)

        if self.quantized:

            #Rearrange for Memcodes
            tokens = rearrange(tokens, 'b d n -> b n d')

            tokens, _= module.quantizer_ema(tokens)
            tokens = rearrange(tokens, 'b n d -> b d n')

        fakes = sample(module.diffusion_ema, noise, self.demo_steps, 1, tokens)

        # Put the demos together
        fakes = rearrange(fakes, 'b d n -> d (b n)')
        demo_reals = rearrange(demo_reals, 'b d n -> d (b n)')

        #demo_audio = torch.cat([demo_reals, fakes], -1)

        try:
            log_dict = {}
            
            filename = f'recon_{trainer.global_step:08}.wav'
            fakes = fakes.clamp(-1, 1).mul(32767).to(torch.int16).cpu()
            torchaudio.save(filename, fakes, self.sample_rate)

            reals_filename = f'reals_{trainer.global_step:08}.wav'
            demo_reals = demo_reals.clamp(-1, 1).mul(32767).to(torch.int16).cpu()
            torchaudio.save(reals_filename, demo_reals, self.sample_rate)


            log_dict[f'recon'] = wandb.Audio(filename,
                                                sample_rate=self.sample_rate,
                                                caption=f'Reconstructed')
            log_dict[f'real'] = wandb.Audio(reals_filename,
                                                sample_rate=self.sample_rate,
                                                caption=f'Real')
            trainer.logger.experiment.log(log_dict, step=trainer.global_step)
        except Exception as e:
            print(f'{type(e).__name__}: {e}', file=sys.stderr)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--training-dir', type=Path, required=True,
                   help='the training data directory')
    p.add_argument('--name', type=str, required=True,
                   help='the name of the run')
    p.add_argument('--num-workers', type=int, default=2,
                   help='number of CPU workers for the DataLoader')
    p.add_argument('--num-gpus', type=int, default=1,
                   help='number of GPUs to use for training')
    
    p.add_argument('--sample-rate', type=int, default=48000,
                   help='The sample rate of the audio')
    p.add_argument('--sample-size', type=int, default=65536,
                   help='Number of samples to train on, must be a multiple of 16384')
    p.add_argument('--demo-every', type=int, default=10,
                   help='Number of epochs between demos')
    p.add_argument('--demo-steps', type=int, default=250,
                   help='Number of denoising steps for the demos')
    p.add_argument('--num-demos', type=int, default=16,
                   help='Number of demos to create')
    p.add_argument('--checkpoint-every', type=int, default=10000,
                   help='Number of steps between checkpoints')
    p.add_argument('--accum-batches', type=int, default=2,
                   help='Batches for gradient accumulation')
    p.add_argument('--batch-size', '-bs', type=int, default=8,
                   help='the batch size')

    p.add_argument('--ema-decay', type=float, default=0.995,
                   help='the EMA decay')
    p.add_argument('--seed', type=int, default=0,
                   help='the random seed')
    
    p.add_argument('--latent-dim', type=int, default=32,
                   help='the validation set')
    p.add_argument('--codebook-size', type=int, default=2048,
                   help='the validation set')
    p.add_argument('--num-heads', type=int, default=8,
                   help='number of heads for the memcodes')
    p.add_argument('--num-quantizers', type=int, default=1,
                   help='number of quantizers')
    p.add_argument('--cache-training-data', type=bool, default=False,
                   help='If true, training data is kept in RAM')

    # p.add_argument('--val-set', type=str, required=True,
    #                help='the validation set')
    p.add_argument('--pqmf-bands', type=int, default=1,
                   help='number of sub-bands for the PQMF filter')

    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    torch.manual_seed(args.seed)

    train_set = SampleDataset([args.training_dir], args)
    train_dl = data.DataLoader(train_set, args.batch_size, shuffle=True,
                               num_workers=args.num_workers, persistent_workers=True, pin_memory=True)
    wandb_logger = pl.loggers.WandbLogger(project=args.name)
    demo_dl = data.DataLoader(train_set, args.num_demos, shuffle=True)
    
    exc_callback = ExceptionCallback()
    ckpt_callback = pl.callbacks.ModelCheckpoint(every_n_train_steps=args.checkpoint_every, save_top_k=-1)
    demo_callback = DemoCallback(demo_dl, args)
    diffusion_model = DiffusionDVAE(args)
    wandb_logger.watch(diffusion_model)

    diffusion_trainer = pl.Trainer(
        gpus=args.num_gpus,
        accelerator="gpu",
        strategy='fsdp',
        precision=16,
        accumulate_grad_batches={
            0:1, 
            1: args.accum_batches #Start without accumulation
            # 5:2,
            # 10:3, 
            # 12:4, 
            # 14:5, 
            # 16:6, 
            # 18:7,  
            # 20:8
            }, 
        callbacks=[ckpt_callback, demo_callback, exc_callback],
        logger=wandb_logger,
        log_every_n_steps=1,
        max_epochs=10000000,
    )

    diffusion_trainer.fit(diffusion_model, train_dl)

if __name__ == '__main__':
    main()
