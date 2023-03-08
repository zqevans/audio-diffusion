#!/usr/bin/env python3

from prefigure.prefigure import get_all_args, push_wandb_config
from contextlib import contextmanager
from copy import deepcopy
import math
import numpy as np
from pathlib import Path

import sys, random
import torch
from torch import optim, nn
from torch.nn import functional as F
from torch.utils import data
from tqdm import trange
import pytorch_lightning as pl
from pytorch_lightning.utilities.distributed import rank_zero_only
from einops import rearrange

import auraloss
import torchaudio

import wandb

from dataset.dataset import get_wds_loader
from autoencoders.soundstream import SoundStreamXLEncoder
from encodec.modules import SEANetEncoder, SEANetDecoder

from decoders.diffusion_decoder import DiffusionAttnUnet1D
from blocks.blocks import Upsample1d_2
from diffusion.model import ema_update
from aeiou.viz import pca_point_cloud, audio_spectrogram_image, tokens_spectrogram_image



# Define the noise schedule and sampling loop
def get_alphas_sigmas(t):
    """Returns the scaling factors for the clean audio (alpha) and for the
    noise (sigma), given a timestep."""
    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)

def alpha_sigma_to_t(alpha, sigma):
    """Returns a timestep, given the scaling factors for the clean audio and for
    the noise."""
    return torch.atan2(sigma, alpha) / math.pi * 2


@torch.no_grad()
def sample_v(model, x, steps, eta, **extra_args):
    """Draws samples from a model given starting noise."""
    ts = x.new_ones([x.shape[0]])

    # Create the noise schedule
    t = torch.linspace(1, 0, steps + 1)[:-1]

    alphas, sigmas = get_alphas_sigmas(t)

    # The sampling loop
    for i in trange(steps):

        # Get the model output (v, the predicted velocity)
        with torch.cuda.amp.autocast():
            v = model(x, ts * t[i], **extra_args).float()

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

@torch.no_grad()
def sample(model, x, steps, **extra_args):
    """Draws samples from a model given starting noise."""
    ts = x.new_ones([x.shape[0]])

    # Create the noise schedule
    t = torch.linspace(1, 0, steps + 1)[:-1]

    alphas, sigmas = get_alphas_sigmas(t)

    # The sampling loop
    for i in trange(steps):

        # Get the model output (pred, the predicted denoised audio)
        with torch.cuda.amp.autocast():
            pred = model(x, ts * t[i], **extra_args).float()

        # Predict the noise and the denoised audio
        #pred = x * alphas[i] - v * sigmas[i]
        eps = x - pred

        # If we are not on the last timestep, compute the noisy audio for the
        # next timestep.
        if i < steps - 1:
            
            # Recombine the predicted noise and predicted denoised audio in the
            # correct proportions for the next step
            x = pred * alphas[i + 1] + eps * sigmas[i + 1]

    # If we are on the last timestep, output the denoised audio
    return pred

@torch.no_grad()
def sample_eps(model, x, steps, **extra_args):
    """Draws samples from a model given starting noise."""
    ts = x.new_ones([x.shape[0]])

    # Create the noise schedule
    t = torch.linspace(1, 0, steps + 1)[:-1]

    alphas, sigmas = get_alphas_sigmas(t)

    # The sampling loop
    for i in trange(steps):

        # Get the model output (pred, the predicted denoised audio)
        with torch.cuda.amp.autocast():
            eps = model(x, ts * t[i], **extra_args).float()

        # Predict the noise and the denoised audio
        #pred = x * alphas[i] - v * sigmas[i]
        pred = x - eps

        # If we are not on the last timestep, compute the noisy audio for the
        # next timestep.
        if i < steps - 1:
            
            # Recombine the predicted noise and predicted denoised audio in the
            # correct proportions for the next step
            x = pred * alphas[i + 1] + eps * sigmas[i + 1]

    # If we are on the last timestep, output the denoised audio
    return pred

class DiffusionAutoencoder(nn.Module):
    def __init__(self):
        
        super().__init__()

        self.latent_dim = 32
        capacity = 32

        #c_mults = [2, 4, 8, 16, 32]
        
        strides = [2, 2, 2, 2, 2]

        self.downsample_ratio = np.prod(strides)

        # self.encoder = SoundStreamXLEncoder(
        #     in_channels=2, 
        #     capacity=capacity, 
        #     latent_dim=self.latent_dim,
        #     c_mults = c_mults,
        #     strides = strides
        # )

        self.encoder = SEANetEncoder(
            channels=2,
            dimension=self.latent_dim,
            n_filters=capacity,
            ratios = list(reversed(strides)),
            norm='time_group_norm',
        )

        # self.latent_upsampler = SEANetDecoder(
        #     channels=self.latent_dim,
        #     dimension=self.latent_dim,
        #     n_filters=capacity,
        #     ratios = strides,
        #     norm='time_group_norm',
        # )

        self.latent_upsampler = Upsample1d_2(self.latent_dim, self.latent_dim, self.downsample_ratio)


        self.diffusion = DiffusionAttnUnet1D(
            io_channels=2, 
            cond_dim=self.latent_dim,
            #cond_noise_aug=True,
            n_attn_layers=0, 
            depth=6,
            c_mults=[128, 256]+[512]*4,
            learned_resample=True,
            strides=[2, 2, 4, 4, 4, 4],
            kernel_size=5
        )

    def encode(self, audio):
        return torch.tanh(self.encoder(audio))

    def decode(self, latents, steps=100):
        upsampled_latents = self.latent_upsampler(latents)

        noise = torch.randn([latents.shape[0], 2, latents.shape[2] * self.downsample_ratio], device=latents.device)

        return sample_v(self.diffusion, noise, steps, 0, cond=upsampled_latents)

class DiffAETrainer(pl.LightningModule):
    def __init__(self, global_args):
        super().__init__()

        self.diffae = DiffusionAutoencoder()
        self.diffae_ema = deepcopy(self.diffae)
        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)
        self.ema_decay = global_args.ema_decay

        scales = [2048, 1024, 512, 256, 128]
        hop_sizes = []
        win_lengths = []
        overlap = 0.75
        for s in scales:
            hop_sizes.append(int(s * (1 - overlap)))
            win_lengths.append(s)

        self.sdstft = auraloss.freq.SumAndDifferenceSTFTLoss(
            fft_sizes=scales, 
            hop_sizes=hop_sizes, 
            win_lengths=win_lengths, 
            sample_rate=global_args.sample_rate, 
            perceptual_weighting=True, 
            # scale="mel", 
            # n_mels=64
        )

    def configure_optimizers(self):
        return optim.Adam([*self.diffae.parameters()], lr=1e-4)
  
    def training_step(self, batch, batch_idx):
        reals = batch[0][0]
        
        # Draw uniformly distributed continuous timesteps
        t = self.rng.draw(reals.shape[0])[:, 0].to(self.device)

        # Calculate the noise schedule parameters for those timesteps
        alphas, sigmas = get_alphas_sigmas(t)

        # Combine the ground truth images and the noise
        alphas = alphas[:, None, None]
        sigmas = sigmas[:, None, None]
        noise = torch.randn_like(reals)
        noised_reals = reals * alphas + noise * sigmas
        targets = noise * alphas - reals * sigmas
        #target_v = noise * alphas - reals * sigmas

        # Compute the model output and the loss.
        with torch.cuda.amp.autocast():
            latents = self.diffae.encode(reals).float()

        with torch.cuda.amp.autocast():
            latents_upsampled = self.diffae.latent_upsampler(latents)
            v = self.diffae.diffusion(noised_reals, t, latents_upsampled)

           # v = noise * alphas - denoised * sigmas

            mse_loss = F.mse_loss(v, targets)

            denoised = noise * alphas - v * sigmas

            stft_loss = self.sdstft(denoised, reals)
            loss = mse_loss + stft_loss

        log_dict = {
            'train/loss': loss.detach(),
            'train/stft_loss': stft_loss.detach(),
            'train/mse_loss': mse_loss.detach(),
        }

        self.log_dict(log_dict, prog_bar=True, on_step=True)
        return loss

    def on_before_zero_grad(self, *args, **kwargs):
        decay = 0.95 if self.current_epoch < 25 else self.ema_decay
        ema_update(self.diffae, self.diffae_ema, decay)

class ExceptionCallback(pl.Callback):
    def on_exception(self, trainer, module, err):
        print(f'{type(err).__name__}: {err}')


class DemoCallback(pl.Callback):
    def __init__(self, demo_dl, global_args):
        super().__init__()
        self.demo_every = global_args.demo_every
        self.demo_samples = global_args.sample_size
        self.demo_steps = global_args.demo_steps
        self.demo_dl = iter(demo_dl)
        self.sample_rate = global_args.sample_rate

    @rank_zero_only
    @torch.no_grad()
    #def on_train_epoch_end(self, trainer, module):
    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):   
        last_demo_step = -1
        if (trainer.global_step - 1) % self.demo_every != 0 or last_demo_step == trainer.global_step:
        #if trainer.current_epoch % self.demo_every != 0:
            return
        
        module.eval()

        last_demo_step = trainer.global_step

        demo_reals, _, _ = next(self.demo_dl)

        demo_reals = demo_reals[0]

        encoder_input = demo_reals.to(module.device)

        demo_reals = demo_reals.to(module.device)

        with torch.no_grad():

            latents = module.diffae_ema.encode(encoder_input)

            fakes = module.diffae_ema.decode(latents)

            

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

            log_dict[f'embeddings_3dpca'] = pca_point_cloud(latents, output_type='plotly')
            log_dict[f'embeddings_spec'] = wandb.Image(tokens_spectrogram_image(latents))

            log_dict[f'real_melspec_left'] = wandb.Image(audio_spectrogram_image(demo_reals))
            log_dict[f'recon_melspec_left'] = wandb.Image(audio_spectrogram_image(fakes))


            trainer.logger.experiment.log(log_dict, step=trainer.global_step)
        except Exception as e:
            print(f'{type(e).__name__}: {e}', file=sys.stderr)
        finally:
            module.train()

def main():

    args = get_all_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    torch.manual_seed(args.seed)

    names = [
    ]

    train_dl = get_wds_loader(
        batch_size=args.batch_size,
        s3_url_prefix=None,
        sample_size=args.sample_size,
        names=names,
        sample_rate=args.sample_rate,
        num_workers=args.num_workers,
        recursive=True,
        random_crop=True,
        #normalize_lufs=-14.0,
        epoch_steps=2000,
    )

    exc_callback = ExceptionCallback()
    ckpt_callback = pl.callbacks.ModelCheckpoint(every_n_train_steps=args.checkpoint_every, save_top_k=-1)
    demo_callback = DemoCallback(train_dl, args)
    diffusion_model = DiffAETrainer(args)

    wandb_logger = pl.loggers.WandbLogger(project=args.name)
    wandb_logger.watch(diffusion_model)
    push_wandb_config(wandb_logger, args)

    diffusion_trainer = pl.Trainer(
        gpus=args.num_gpus,
        accelerator="gpu",
        num_nodes = args.num_nodes,
        strategy='ddp',
        precision=16,
        accumulate_grad_batches=args.accum_batches, 
        callbacks=[ckpt_callback, demo_callback, exc_callback],
        logger=wandb_logger,
        log_every_n_steps=1,
        max_epochs=10000000,
    )

    diffusion_trainer.fit(diffusion_model, train_dl, ckpt_path=args.ckpt_path)

if __name__ == '__main__':
    main()

