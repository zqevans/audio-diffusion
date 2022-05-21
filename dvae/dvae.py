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

import torchaudio
import wandb

from dataset.dataset import SampleDataset
from diffusion.model import SkipBlock, FourierFeatures, expand_to_planes, ema_update
from encoders.encoders import ResConvBlock
#from vector_quantize_pytorch import ResidualVQ

class Encoder(nn.Sequential):
    def __init__(self, codes, io_channels):
        c_in = io_channels
        c_mults = [64, 64, 128, 128] + [codes] * 10
        layers = []
        c_mult_prev = c_in
        for i, c_mult in enumerate(c_mults):
            is_last = i == len(c_mults) - 1
            layers.append(ResConvBlock(c_mult_prev, c_mult, c_mult))
            layers.append(ResConvBlock(
                c_mult, c_mult, c_mult, is_last=is_last))
            if not is_last:
                layers.append(nn.AvgPool1d(2))
            else:
                layers.append(nn.AdaptiveAvgPool1d(1))
                layers.append(nn.Flatten())
            c_mult_prev = c_mult
        super().__init__(*layers)

        self.downsample_ratio = 2**len(c_mults)
        print(f'Encoder downsample ratio: {self.downsample_ratio}')

class DiffusionDecoder(nn.Module):
    def __init__(self, codes, io_channels):
        super().__init__()
        c_mults = [128, 128, 256, 256] + [512] * 12
        depth = len(c_mults)

        self.io_channels = io_channels
        self.timestep_embed = FourierFeatures(1, 16)
        self.logits_embed = nn.Conv1d(codes, 45, 1, bias=False)

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
                    ResConvBlock(16 + 45, c, c),
                    ResConvBlock(c, c, c),
                    ResConvBlock(c, c, c),
                    block,
                    ResConvBlock(c * 2, c, c),
                    ResConvBlock(c, c, c),
                    ResConvBlock(c, c, io_channels, is_last=True),
                )
        self.net = block

    def forward(self, input, t, logits):
        timestep_embed = expand_to_planes(self.timestep_embed(t[:, None]), input.shape)
        logits_embed = F.interpolate(self.logits_embed(logits), input.shape[self.io_channels])
        return self.net(torch.cat([input, timestep_embed, logits_embed], dim=1))

# Define the noise schedule and sampling loop
def get_alphas_sigmas(t):
    """Returns the scaling factors for the clean image (alpha) and for the
    noise (sigma), given a timestep."""
    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)

@torch.no_grad()
def sample(model, x, steps, eta, logits):
    """Draws samples from a model given starting noise."""
    ts = x.new_ones([x.shape[0]])

    # Create the noise schedule
    t = torch.linspace(1, 0, steps + 1)[:-1]
    alphas, sigmas = get_alphas_sigmas(t)

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


class RQDVAE(pl.LightningModule):
    def __init__(self, global_args):
        super().__init__()

        self.tau_ramp = ramp(0, 200, math.log(1), math.log(1/16))

        self.encoder = Encoder(global_args.codebook_size, 2)
        self.encoder_ema = deepcopy(self.encoder)
        self.diffusion = DiffusionDecoder(global_args.codebook_size, 2)
        self.diffusion_ema = deepcopy(self.diffusion)
        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)

        # self.quantizer = ResidualVQ(
        #     num_quantizers=global_args.num_quantizers,
        #     dim=128,
        #     codebook_size=2,
        #     kmeans_init=True,
        #     kmeans_iters=100,
        #     threshold_ema_dead_code=2,
        #     sample_codebook_temp = 0.1,
        #     channel_last=False,
        #     sync_codebook=True
        # )
        #self.pqmf = PQMF(2, 70, global_args.pqmf_bands)

    def encode(self, *args, **kwargs):
        if self.training:
            return self.encoder(*args, **kwargs)
        return self.encoder_ema(*args, **kwargs)

    def decode(self, *args, **kwargs):
        if self.training:
            return self.diffusion(*args, **kwargs)
        return self.diffusion_ema(*args, **kwargs)

    def configure_optimizers(self):
        return optim.Adam([*self.encoder.parameters(), *self.diffusion.parameters()], lr=2e-4)

    def eval_loss(self, reals, tau):
        # Draw uniformly distributed continuous timesteps
        t = self.rng.draw(reals.shape[0])[:, 0].to(self.device)

        # Calculate the noise schedule parameters for those timesteps
        alphas, sigmas = get_alphas_sigmas(t)

        # Combine the ground truth images and the noise
        alphas = alphas[:, None, None, None]
        sigmas = sigmas[:, None, None, None]
        noise = torch.randn_like(reals)
        noised_reals = reals * alphas + noise * sigmas
        targets = noise * alphas - reals * sigmas

        # Compute the model output and the loss.
        with torch.cuda.amp.autocast():
            logits = self.encoder(reals).float()

        print(f'logits shape {logits.shape}')
        
        one_hot = F.gumbel_softmax(logits, tau=tau, dim=1)

        print(f'one_hot shape {one_hot.shape}')
        with torch.cuda.amp.autocast():
            v = self.diffusion(noised_reals, t, one_hot)
            return F.mse_loss(v, targets)

    def training_step(self, batch, batch_idx):

        tau = math.exp(self.tau_ramp(self.current_epoch + batch_idx))

        loss = self.eval_loss(batch[0], tau)
        log_dict = {'train/loss': loss.detach()}
        self.log_dict(log_dict, prog_bar=True, on_step=True)
        return loss

    def on_before_zero_grad(self, *args, **kwargs):
        decay = 0.95 if self.current_epoch < 25 else args.ema_decay
        ema_update(self.diffusion, self.diffusion_ema, decay)
        ema_update(self.encoder, self.encoder_ema, decay)

class ExceptionCallback(pl.Callback):
    def on_exception(self, trainer, module, err):
        print(f'{type(err).__name__}: {err}', file=sys.stderr)


class DemoCallback(pl.Callback):
    def __init__(self, demo_reals, global_args):
        super().__init__()
        self.demo_every = global_args.demo_every
        self.demo_steps = global_args.demo_steps
        self.demo_reals = demo_reals

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx, unused=0):
        last_demo_step = -1
        if (trainer.global_step - 1) % self.demo_every != 0 or last_demo_step == trainer.global_step:
            return

        noise = torch.randn([32, 3, 128, 128], device=module.device)
        logits = module.encoder_ema(self.demo_reals)
        one_hot = F.one_hot(logits.argmax(1), 1024).movedim(3, 1).to(logits.dtype)
        fakes = sample(module.decoder_ema, noise, 1000, 1, one_hot)

        try:
            log_dict = {}
            for i, fake in enumerate(fakes):

                filename = f'demo_{trainer.global_step:08}_{i:02}.wav'
                fake = self.ms_encoder(fake).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
                torchaudio.save(filename, fake, 44100)
                log_dict[f'demo_{i}'] = wandb.Audio(filename,
                                                    sample_rate=44100,
                                                    caption=f'Demo {i}')
            trainer.logger.experiment.log(log_dict, step=trainer.global_step)
        except Exception as e:
            print(f'{type(e).__name__}: {e}', file=sys.stderr)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--training-dir', type=Path, required=True,
                   help='the training data directory')
    p.add_argument('--name', type=str, required=True,
                   help='the name of the run')
    p.add_argument('--demo-dir', type=Path, required=True,
                   help='path to a directory with audio files for demos')
    p.add_argument('--num-workers', type=int, default=2,
                   help='number of CPU workers for the DataLoader')
    p.add_argument('--num-gpus', type=int, default=1,
                   help='number of GPUs to use for training')
    
    p.add_argument('--sample-rate', type=int, default=48000,
                   help='The sample rate of the audio')
    p.add_argument('--sample-size', type=int, default=65536,
                   help='Number of samples to train on, must be a multiple of 16384')
    p.add_argument('--demo-every', type=int, default=1000,
                   help='Number of steps between demos')
    p.add_argument('--demo-steps', type=int, default=500,
                   help='Number of denoising steps for the demos')
    p.add_argument('--checkpoint-every', type=int, default=20000,
                   help='Number of steps between checkpoints')
    p.add_argument('--data-repeats', type=int, default=1,
                   help='Number of times to repeat the dataset. Useful to lengthen epochs on small datasets')
    p.add_argument('--accum-batches', type=int, default=8,
                   help='Batches for gradient accumulation')
    p.add_argument('--batch-size', '-bs', type=int, default=64,
                   help='the batch size')

    p.add_argument('--ema-decay', type=float, default=0.995,
                   help='the EMA decay')
    p.add_argument('--seed', type=int, default=0,
                   help='the random seed')
    
    p.add_argument('--codebook-size', type=int, required=True,
                   help='the validation set')
    p.add_argument('--num-quantizers', type=int, required=True,
                   help='number of quantizers')

    # p.add_argument('--val-set', type=str, required=True,
    #                help='the validation set')
    # p.add_argument('--pqmf-bands', type=int, default=8,
    #                help='number of sub-bands for the PQMF filter')

    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    torch.manual_seed(args.seed)

    train_set = SampleDataset([args.training_dir], args)
    train_dl = data.DataLoader(train_set, args.batch_size, shuffle=True,
                               num_workers=args.num_workers, persistent_workers=True, pin_memory=True)
    wandb_logger = pl.loggers.WandbLogger(project=args.name)

    demo_reals = next(iter(train_dl))[0].to(device)

    exc_callback = ExceptionCallback()
    ckpt_callback = pl.callbacks.ModelCheckpoint(every_n_train_steps=args.checkpoint_every, save_top_k=-1)
    demo_callback = DemoCallback(demo_reals, args)
    diffusion_model = RQDVAE(args)
    wandb_logger.watch(diffusion_model.diffusion)

    diffusion_trainer = pl.Trainer(
        gpus=args.num_gpus,
        strategy='ddp',
        precision=16,
        accumulate_grad_batches={0:1, 1:args.accum_batches}, #Start without accumulation
        callbacks=[ckpt_callback, demo_callback, exc_callback],
        logger=wandb_logger,
        log_every_n_steps=1,
        max_epochs=10000000,
    )

    diffusion_trainer.fit(diffusion_model, train_dl)

if __name__ == '__main__':
    main()