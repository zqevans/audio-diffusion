#!/usr/bin/env python3

from prefigure.prefigure import get_all_args, push_wandb_config
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
from diffusion.pqmf import CachedPQMF as PQMF
from encoders.encoders import AttnResEncoder1D

from nwt_pytorch import Memcodes
from dvae.residual_memcodes import ResidualMemcodes
from decoders.diffusion_decoder import DiffusionAttnUnet1D
from diffusion.model import ema_update
from viz.viz import embeddings_table, pca_point_cloud, audio_spectrogram_image, tokens_spectrogram_image


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
def sample(model, x, steps, eta, logits = None, use_crash = False):
    """Draws samples from a model given starting noise."""
    ts = x.new_ones([x.shape[0]])

    # Create the noise schedule
    t = torch.linspace(1, 0, steps + 1)[:-1]

    if use_crash:
        t = get_crash_schedule(t)

    alphas, sigmas = get_alphas_sigmas(t)

    # The sampling loop
    for i in trange(steps):

        # Get the model output (v, the predicted velocity)
        if logits is not None:
            with torch.cuda.amp.autocast():
                v = model(x, ts * t[i], logits).float()
        else:
            with torch.cuda.amp.autocast():
                v = model(x, ts * t[i]).float()

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

class DiffusionDVAE(pl.LightningModule):
    def __init__(self, global_args):
        super().__init__()

        self.pqmf_bands = global_args.pqmf_bands

        if self.pqmf_bands > 1:
            self.pqmf = PQMF(2, 70, global_args.pqmf_bands)

        self.encoder = AttnResEncoder1D(global_args, n_io_channels=2*global_args.pqmf_bands, depth=7, n_attn_layers=0, c_mults=[128, 256, 512, 512, 1024, 1024, 1024])
        self.encoder_ema = deepcopy(self.encoder)
        self.diffusion = DiffusionAttnUnet1D(global_args, cond_channels=global_args.latent_dim, n_attn_layers=0, c_mults=[256] * 3 + [512] * 6 + [1024]*5)
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


class LatentDiffusionDVAE(pl.LightningModule):
    def __init__(self, dvae, global_args):
        super().__init__()

        self.dvae = dvae
        self.encoder = dvae.encoder_ema
        self.decoder = dvae.diffusion_ema

        self.encoder.eval().requires_grad_(False)
        self.decoder.eval().requires_grad_(False)

        self.diffusion = DiffusionAttnUnet1D(global_args, io_channels=global_args.latent_dim, depth=8, n_attn_layers=3, c_mults=[1024]*8)
        self.diffusion_ema = deepcopy(self.diffusion)
        
        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)
        self.ema_decay = global_args.ema_decay
        
        
    def encode(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)

    def configure_optimizers(self):
        return optim.Adam([*self.diffusion.parameters()], lr=4e-5)

    def training_step(self, batch, batch_idx):
        audios = batch[0]

        encoder_input = audios

        if self.dvae.pqmf_bands > 1:
            encoder_input = self.dvae.pqmf(audios)
        
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                latents = self.encoder(encoder_input).float()

        # Draw uniformly distributed continuous timesteps
        t = self.rng.draw(latents.shape[0])[:, 0].to(self.device)

        # Calculate the noise schedule parameters for those timesteps
        alphas, sigmas = get_alphas_sigmas(t)

        # Combine the ground truth images and the noise
        alphas = alphas[:, None, None]
        sigmas = sigmas[:, None, None]
        noise = torch.randn_like(latents)
        noised_latents = latents * alphas + noise * sigmas
        targets = noise * alphas - latents * sigmas

        with torch.cuda.amp.autocast():
            v = self.diffusion(noised_latents, t)
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

class ExceptionCallback(pl.Callback):
    def on_exception(self, trainer, module, err):
        print(f'{type(err).__name__}: {err}', file=sys.stderr)


class DemoCallback(pl.Callback):
    def __init__(self, global_args):
        super().__init__()
        self.demo_every = global_args.demo_every
        self.demo_samples = global_args.sample_size
        self.num_demos = global_args.num_demos
        self.demo_steps = global_args.demo_steps
        self.sample_rate = global_args.sample_rate
        self.latent_dim = global_args.latent_dim
        self.latent_down_ratio = global_args.latent_down_ratio

    @rank_zero_only
    @torch.no_grad()
    def on_train_epoch_end(self, trainer, module):
        #last_demo_step = -1
        #if (trainer.global_step - 1) % self.demo_every != 0 or last_demo_step == trainer.global_step:
        if trainer.current_epoch % self.demo_every != 0:
            return

        demo_latent_size = self.demo_samples // self.latent_down_ratio

        latent_noise = torch.randn([self.num_demos, self.latent_dim, demo_latent_size]).to(module.device)

        with torch.no_grad():
            fake_latents = sample(module.diffusion_ema, latent_noise, self.demo_steps, 1)

            noise = torch.randn([self.num_demos, 2, self.demo_samples]).to(module.device)

            fakes = sample(module.decoder, noise, 250, 1, fake_latents, use_crash=True)

        # Put the demos together
        fakes = rearrange(fakes, 'b d n -> d (b n)')

        try:
            log_dict = {}
            
            filename = f'demo_{trainer.global_step:08}.wav'
            fakes = fakes.clamp(-1, 1).mul(32767).to(torch.int16).cpu()
            torchaudio.save(filename, fakes, self.sample_rate)


            log_dict[f'demo'] = wandb.Audio(filename,
                                                sample_rate=self.sample_rate,
                                                caption=f'Demo')

            log_dict[f'embeddings'] = embeddings_table(fake_latents)

            log_dict[f'embeddings_3dpca'] = pca_point_cloud(fake_latents)
            log_dict[f'embeddings_spec'] = wandb.Image(tokens_spectrogram_image(fake_latents))

            log_dict[f'demo_melspec_left'] = wandb.Image(audio_spectrogram_image(fakes))


            trainer.logger.experiment.log(log_dict, step=trainer.global_step)
        except Exception as e:
            print(f'{type(e).__name__}: {e}', file=sys.stderr)

def main():

    args = get_all_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    torch.manual_seed(args.seed)

    train_set = SampleDataset([args.training_dir], args)
    train_dl = data.DataLoader(train_set, args.batch_size, shuffle=True,
                               num_workers=args.num_workers)
    wandb_logger = pl.loggers.WandbLogger(project=args.name)
    
    exc_callback = ExceptionCallback()
    
    dvae = DiffusionDVAE(args)

    #Load DVAE config and weights
    dvae.load_state_dict(torch.load(args.dvae_ckpt_path, map_location="cpu")["state_dict"])

    #Free up non-EMA versions
    del dvae.encoder
    del dvae.diffusion

    latent_dvae = LatentDiffusionDVAE(dvae, args)

    args.latent_down_ratio = 2**(dvae.encoder_ema.depth-1)

    print(args.latent_down_ratio)

    ckpt_callback = pl.callbacks.ModelCheckpoint(every_n_train_steps=args.checkpoint_every, save_top_k=-1)
    demo_callback = DemoCallback(args)
    wandb_logger.watch(latent_dvae.diffusion)
    push_wandb_config(wandb_logger, args)

    diffusion_trainer = pl.Trainer(
        gpus=args.num_gpus,
        accelerator="gpu",
        #devices= args.num_gpus,
        #num_nodes = args.num_nodes,
        strategy='ddp',
        precision=16,
        accumulate_grad_batches=args.accum_batches, 
        callbacks=[ckpt_callback, demo_callback, exc_callback],
        logger=wandb_logger,
        log_every_n_steps=1,
        max_epochs=10000000,
    )

    diffusion_trainer.fit(latent_dvae, train_dl, ckpt_path=args.ckpt_path)

if __name__ == '__main__':
    main()

