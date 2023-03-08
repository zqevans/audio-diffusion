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
from autoencoders.models import AudioVAE, AudioAutoencoder
from encodec.modules import SEANetEncoder, SEANetDecoder

from typing import Literal
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
def sample_pred(model, x, steps, **extra_args):
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

class LatentDiffusionAutoencoder(nn.Module):
    def __init__(self, 
                 autoencoder,
                 first_stage_latent_dim=32,
                 encode_fn = lambda x,ae: ae.encode(x),
                 decode_fn = lambda z,ae: ae.decode(z),
                 objective: Literal["v", "eps", "pred", "pred_decode+v"] = 'v'
                 ):
        
        super().__init__()

        self.autoencoder = autoencoder.eval().requires_grad_(False)
        self.first_stage_latent_dim = first_stage_latent_dim
        self.encode_fn = encode_fn
        self.decode_fn = decode_fn
        self.objective = objective
        if self.objective == 'pred_decode+v':
            self.objective = 'pred'
        self.latent_dim = 32
        capacity = 128

        strides = [2, 2, 2, 2]

        self.downsample_ratio = np.prod(strides)

        self.encoder = SEANetEncoder(
            channels=self.first_stage_latent_dim,
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
            io_channels=self.first_stage_latent_dim, 
            cond_dim=self.latent_dim, 
            n_attn_layers=0, 
            depth=10,
            c_mults=[512]*10,
        )

    def encode(self, audio):
        first_stage_latents = self.encode_fn(audio, self.autoencoder)
        return self.encode_latents(first_stage_latents) 

    def encode_latents(self, first_stage_latents):
        return torch.tanh(self.encoder(first_stage_latents))

    def decode(self, latents, steps=100):
        upsampled_latents = self.latent_upsampler(latents)

        noise = torch.randn([latents.shape[0], self.first_stage_latent_dim, latents.shape[2] * self.downsample_ratio], device=latents.device)

        if self.objective == 'v':
            first_stage_sampled = sample_v(self.diffusion, noise, steps, 0, cond=upsampled_latents)
        elif self.objective == 'eps':
            first_stage_sampled = sample_eps(self.diffusion, noise, steps, cond=upsampled_latents)
        elif self.objective == 'pred':
            first_stage_sampled = sample_pred(self.diffusion, noise, steps, cond=upsampled_latents)

        return self.decode_fn(first_stage_sampled, self.autoencoder)

class LatentDiffAETrainer(pl.LightningModule):
    def __init__(self, global_args, 
                 autoencoder,
                 autoencoder_latent_dim,
                 encode_fn = lambda x,ae: ae.encode(x),
                 decode_fn = lambda x,ae: ae.decode(x),
                 objective: Literal["v", "eps", "pred"] = 'v'):
        super().__init__()

        self.encode_fn = encode_fn
        self.decode_fn = decode_fn
        self.objective = objective
        self.diffae = LatentDiffusionAutoencoder(autoencoder=autoencoder, encode_fn = encode_fn, first_stage_latent_dim=autoencoder_latent_dim, objective=objective)
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
        self.sdstft = auraloss.freq.SumAndDifferenceSTFTLoss(fft_sizes=scales, hop_sizes=hop_sizes, win_lengths=win_lengths, sample_rate=global_args.sample_rate, perceptual_weighting=True)

    def configure_optimizers(self):
        return optim.Adam([*self.diffae.parameters()], lr=1e-4)
  
    def training_step(self, batch, batch_idx):
        reals = batch[0][0]
        
        # Draw uniformly distributed continuous timesteps
        t = self.rng.draw(reals.shape[0])[:, 0].to(self.device)

        #t = get_crash_schedule(t)

        # Calculate the noise schedule parameters for those timesteps
        alphas, sigmas = get_alphas_sigmas(t)

        with torch.no_grad():
            first_stage_latents = self.diffae.encode_fn(reals, self.diffae.autoencoder)

        # Combine the ground truth images and the noise
        alphas = alphas[:, None, None]
        sigmas = sigmas[:, None, None]
        noise = torch.randn_like(first_stage_latents)
        noised_latents = first_stage_latents * alphas + noise * sigmas

        target_v = noise * alphas - first_stage_latents * sigmas

        if self.objective == 'v':
            targets = target_v
        elif self.objective == 'eps':
            targets = noise
        elif self.objective == 'pred':
            targets = first_stage_latents

        # Compute the model output and the loss.
        with torch.cuda.amp.autocast():
            latents = self.diffae.encode_latents(first_stage_latents)
    
        with torch.cuda.amp.autocast():
            latents_upsampled = self.diffae.latent_upsampler(latents)

            output = self.diffae.diffusion(noised_latents, t, latents_upsampled)

            if self.objective == 'v':
                mse_loss = F.mse_loss(output, targets)
                loss = mse_loss
            elif self.objective == 'eps':
                mse_loss = F.mse_loss(output, targets)
                loss = mse_loss
            elif self.objective == 'pred':
                first_stage_pred = output
                #first_stage_decoded = self.decode_fn(first_stage_pred, self.diffae.autoencoder)

                v = noise * alphas - first_stage_pred * sigmas

                #stft_loss = self.sdstft(first_stage_decoded, reals)
                
                v_mse_loss = F.mse_loss(v, target_v)
                pred_mse_loss = F.mse_loss(first_stage_pred, first_stage_latents)
                loss = v_mse_loss + pred_mse_loss


        log_dict = {
            'train/loss': loss.detach(),
            'train/mse_loss': mse_loss.detach(),
            #'train/stft_loss': stft_loss.detach() if self.objective == 'pred' else torch.tensor(0.0),
            #'train/v_mse_loss': v_mse_loss.detach(),
            #'train/pred_mse_loss': pred_mse_loss.detach(),
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
        epoch_steps=2000,
    )

    exc_callback = ExceptionCallback()
    ckpt_callback = pl.callbacks.ModelCheckpoint(every_n_train_steps=args.checkpoint_every, save_top_k=-1)
    demo_callback = DemoCallback(train_dl, args)

    # vae_config = {
    #     "capacity": 32
    # }

    # autoencoder = AudioVAE(**vae_config).requires_grad_(False).eval()
    # autoencoder.load_state_dict(torch.load(args.pretrained_ckpt_path)['state_dict'], strict=False)

    autoencoder = AudioAutoencoder().requires_grad_(False).eval()
    autoencoder.load_state_dict(torch.load(args.pretrained_ckpt_path)['state_dict'], strict=False)
    

    diffusion_model = LatentDiffAETrainer(args, autoencoder, autoencoder.latent_dim, encode_fn=lambda x,ae: ae.encode(x))

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

