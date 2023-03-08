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
import numpy as np
import torchaudio

import wandb
from blocks.utils import InverseLR

from diffusion.pqmf import CachedPQMF as PQMF
from autoencoders.soundstream import SoundStreamXLEncoder, SoundStreamXLDecoder
from audio_encoders_pytorch import Encoder1d

from quantizer_pytorch import Quantizer1d

from decoders.diffusion_decoder import DiffusionAttnUnet1D
from diffusion.model import ema_update
from aeiou.viz import embeddings_table, pca_point_cloud, audio_spectrogram_image, tokens_spectrogram_image
from aeiou.datasets import AudioDataset
from dataset.dataset import SampleDataset

# Define the noise schedule and sampling loop
def get_alphas_sigmas(t):
    """Returns the scaling factors for the clean image (alpha) and for the
    noise (sigma), given a timestep."""
    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)

def alpha_sigma_to_t(alpha, sigma):
    """Returns a timestep, given the scaling factors for the clean image and for
    the noise."""
    return torch.a2(sigma, alpha) / math.pi * 2

@torch.no_grad()
def sample(model, x, steps, eta, cond=None):
    """Draws samples from a model given starting noise."""
    ts = x.new_ones([x.shape[0]])

    # Create the noise schedule
    t = torch.linspace(1, 0, steps + 1)[:-1]

    alphas, sigmas = get_alphas_sigmas(t)

    # The sampling loop
    for i in trange(steps):

        # Get the model output (v, the predicted velocity)
        with torch.cuda.amp.autocast():
            v = model(x, ts * t[i], cond).float()

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

class AudioVAE(pl.LightningModule):
    def __init__(self, pqmf_bands=1, latent_dim=32):
        super().__init__()

        self.pqmf_bands = pqmf_bands

        if self.pqmf_bands > 1:
            self.pqmf = PQMF(2, PQMF_ATTN, self.pqmf_bands)

        capacity = 32

        c_mults = [2, 4, 8, 16, 32]
        
        strides = [2, 2, 2, 2, 2]

        self.latent_dim = latent_dim
        self.downsampling_ratio = np.prod(strides)

        self.encoder = SoundStreamXLEncoder(
            in_channels=2*self.pqmf_bands, 
            capacity=capacity, 
            latent_dim=2*self.latent_dim,
            c_mults = c_mults,
            strides = strides
        )

        self.decoder = SoundStreamXLDecoder(
            out_channels=2*self.pqmf_bands, 
            capacity=capacity, 
            latent_dim=self.latent_dim,
            c_mults = c_mults,
            strides = strides
        )
  
    def sample(self, mean, scale):
        stdev = nn.functional.softplus(scale) + 1e-4
        var = stdev * stdev
        logvar = torch.log(var)
        latents = torch.randn_like(mean) * stdev + mean

        kl = (mean * mean + var - logvar - 1).sum(1).mean()

        return latents, kl

    def encode(self, audio):
        if self.pqmf_bands > 1:
            audio = self.pqmf(audio)

        mean, scale = self.encoder(audio).chunk(2, dim=1)
        latents, _ = self.sample(mean, scale)

        return latents

    def decode(self, latents):
        decoded = self.decoder(latents)

        if self.pqmf_bands > 1:
            decoded = self.pqmf.inverse(decoded)

        return decoded

class LatentAudioDiffusionAutoencoder(pl.LightningModule):
    def __init__(self, global_args, autoencoder: AudioVAE):
        super().__init__()

        self.latent_dim = autoencoder.latent_dim

        self.second_stage_latent_dim = 32

        factors = [2, 2, 2, 2]

        self.latent_downsampling_ratio = np.prod(factors)
        
        self.downsampling_ratio = autoencoder.downsampling_ratio * self.latent_downsampling_ratio

        self.latent_encoder = Encoder1d(
            in_channels=self.latent_dim, 
            out_channels = 2 * self.second_stage_latent_dim,
            channels = 128,
            multipliers = [1, 2, 4, 8, 8],
            factors =  factors,
            num_blocks = [8, 8, 8, 8],
        )

        # Scale down the encoder parameters to avoid saturation
        with torch.no_grad():
            for param in self.latent_encoder.parameters():
                param *= 0.5

        self.latent_encoder_ema = deepcopy(self.latent_encoder)

        self.diffusion = DiffusionAttnUnet1D(
            io_channels=self.latent_dim, 
            cond_dim = self.second_stage_latent_dim,
            n_attn_layers=0, 
            c_mults=[512] * 8,
            depth=8
        )

        self.diffusion_ema = deepcopy(self.diffusion)

        self.diffusion_ema.requires_grad_(False)
        self.latent_encoder_ema.requires_grad_(False)

        self.autoencoder = autoencoder

        self.autoencoder.requires_grad_(False)
        
        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)
        self.ema_decay = global_args.ema_decay

    def encode(self, reals):
        first_stage_latents = self.autoencoder.encode(reals)

        if self.training:
            mean, scale = self.latent_encoder(first_stage_latents).chunk(2, dim=1)
            second_stage_latents, _ = self.sample(mean, scale)
        else:
            mean, scale = self.latent_encoder_ema(first_stage_latents).chunk(2, dim=1)
            second_stage_latents, _ = self.sample(mean, scale)

        return second_stage_latents

    def sample(self, mean, scale):
        stdev = nn.functional.softplus(scale) + 1e-4
        var = stdev * stdev
        logvar = torch.log(var)
        latents = torch.randn_like(mean) * stdev + mean

        kl = (mean * mean + var - logvar - 1).sum(1).mean()

        return latents, kl

    def decode(self, latents, steps=250):
        if self.training:
            first_stage_sampled = sample(self.diffusion, steps, 0, latents)
        else:
            first_stage_sampled = sample(self.diffusion_ema, steps, 0, latents)
        decoded = self.autoencoder.decode(first_stage_sampled)
        return decoded

    def configure_optimizers(self):
        optimizer = optim.Adam([*self.latent_encoder.parameters(), *self.diffusion.parameters()], lr=1e-4)

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        reals = batch

        with torch.cuda.amp.autocast():
            with torch.no_grad():
                first_stage_latents = self.autoencoder.encode(reals)
            
        # Draw uniformly distributed continuous timesteps
        t = self.rng.draw(reals.shape[0])[:, 0].to(self.device)

        # Calculate the noise schedule parameters for those timesteps
        alphas, sigmas = get_alphas_sigmas(t)

        # Combine the ground truth images and the noise
        alphas = alphas[:, None, None]
        sigmas = sigmas[:, None, None]
        noise = torch.randn_like(first_stage_latents)
        noised_latents = first_stage_latents * alphas + noise * sigmas
        targets = noise * alphas - first_stage_latents * sigmas

        with torch.cuda.amp.autocast():

            mean, scale = self.latent_encoder(first_stage_latents).chunk(2, dim=1)
            second_stage_latents, kl = self.sample(mean, scale)

            v = self.diffusion(noised_latents, t, second_stage_latents)

            kl_loss = 1e-6 * kl

            diffusion_loss = F.mse_loss(v, targets)

            loss = diffusion_loss + kl_loss

        log_dict = {
            'train/diffusion_loss': diffusion_loss.detach(),
            'train/kl_loss': kl_loss.detach(),
            'train/loss': loss.detach(),
            'train/lr': self.lr_schedulers().get_last_lr()[0],
        }

        self.log_dict(log_dict, prog_bar=True, on_step=True)
        return loss

    def on_before_zero_grad(self, *args, **kwargs):
        decay = 0.95 if self.current_epoch < 5 else self.ema_decay
        ema_update(self.diffusion, self.diffusion_ema, decay)
        ema_update(self.latent_encoder, self.latent_encoder_ema, decay)

class ExceptionCallback(pl.Callback):
    def on_exception(self, trainer, module, err):
        print(f'{type(err).__name__}: {err}', file=sys.stderr)


class DemoCallback(pl.Callback):
    def __init__(self, demo_dl, global_args):
        super().__init__()
        self.demo_every = global_args.demo_every
        self.demo_samples = global_args.sample_size
        self.demo_steps = global_args.demo_steps
        self.num_demos = global_args.num_demos
        self.sample_rate = global_args.sample_rate
        self.demo_dl = iter(demo_dl)

    @rank_zero_only
    @torch.no_grad()
    #def on_train_epoch_end(self, trainer, module):
    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):   
        last_demo_step = -1
        if (trainer.global_step - 1) % self.demo_every != 0 or last_demo_step == trainer.global_step:
        #if trainer.current_epoch % self.demo_every != 0:
            return
        
        last_demo_step = trainer.global_step
        
        print("Starting demo")
        try:

            demo_reals = next(self.demo_dl)

            demo_reals = demo_reals.to(module.device)

            with torch.no_grad():
                print("Encoding first stage")
                first_stage_latents = module.autoencoder.encode(demo_reals)
                print("Encoding second stage")
                mean, scale = module.latent_encoder(first_stage_latents).chunk(2, dim=1)
                second_stage_latents, _ = module.sample(mean, scale)

            print("Generating latent noise")
            latent_noise = torch.randn([self.num_demos, module.latent_dim, self.demo_samples//module.autoencoder.downsampling_ratio]).to(module.device)
            print("Reconstructing first stage")
            recon_latents = sample(module.diffusion_ema, latent_noise, self.demo_steps, 0, second_stage_latents)
            print("Reconstructing audio")
            reconstructed = module.autoencoder.decode(recon_latents)

            # Put the demos together
            reconstructed = rearrange(reconstructed, 'b d n -> d (b n)')
            demo_reals = rearrange(demo_reals, 'b d n -> d (b n)')

            log_dict = {}
            
            print("Saving files")
            filename = f'recon_demo_{trainer.global_step:08}.wav'
            reconstructed = reconstructed.clamp(-1, 1).mul(32767).to(torch.int16).cpu()
            torchaudio.save(filename, reconstructed, self.sample_rate)


            reals_filename = f'reals_{trainer.global_step:08}.wav'
            demo_reals = demo_reals.clamp(-1, 1).mul(32767).to(torch.int16).cpu()
            torchaudio.save(reals_filename, demo_reals, self.sample_rate)

            log_dict[f'recon'] = wandb.Audio(filename,
                                                sample_rate=self.sample_rate,
                                                caption=f'Reconstructed')
        
            log_dict[f'real'] = wandb.Audio(reals_filename,
                                                sample_rate=self.sample_rate,
                                                caption=f'Real')

            log_dict[f'embeddings_3dpca'] = pca_point_cloud(second_stage_latents, output_type="plotly", mode="lines+markers")
            log_dict[f'embeddings_spec'] = wandb.Image(tokens_spectrogram_image(second_stage_latents))

            log_dict[f'real_melspec_left'] = wandb.Image(audio_spectrogram_image(demo_reals))
            log_dict[f'recon_melspec_left'] = wandb.Image(audio_spectrogram_image(reconstructed))


            print("Done logging")
            trainer.logger.experiment.log(log_dict, step=trainer.global_step)

        except Exception as e:
            print(f'{type(e).__name__}: {e}')

def main():

    args = get_all_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    torch.manual_seed(args.seed)

    #args.random_crop = False

    train_set = AudioDataset(
        [args.training_dir],
        sample_rate=args.sample_rate,
        sample_size=args.sample_size,
        random_crop=args.random_crop,
        augs='Stereo(), PhaseFlipper()'
    )

    #args.random_crop = False

    #train_set = SampleDataset([args.training_dir], args)

    train_dl = data.DataLoader(train_set, args.batch_size, shuffle=True,
                               num_workers=args.num_workers, persistent_workers=True, pin_memory=True)
    wandb_logger = pl.loggers.WandbLogger(project=args.name)
    demo_dl = data.DataLoader(train_set, args.num_demos, num_workers=args.num_workers, shuffle=True)
    exc_callback = ExceptionCallback()
    ckpt_callback = pl.callbacks.ModelCheckpoint(every_n_train_steps=args.checkpoint_every, save_top_k=-1)
    demo_callback = DemoCallback(demo_dl, args)
    autoencoder = AudioVAE.load_from_checkpoint(args.pretrained_ckpt_path, global_args=args, strict=False).requires_grad_(False)
    if args.ckpt_path:
        latent_diffusion_model = LatentAudioDiffusionAutoencoder.load_from_checkpoint(args.ckpt_path, global_args=args, autoencoder=autoencoder, strict=False)
    else:
        latent_diffusion_model = LatentAudioDiffusionAutoencoder(args, autoencoder)
    wandb_logger.watch(latent_diffusion_model)
    push_wandb_config(wandb_logger, args)

    diffusion_trainer = pl.Trainer(
        devices=args.num_gpus,
        accelerator="gpu",
        num_nodes = args.num_nodes,
        strategy='ddp_find_unused_parameters_false',
        precision=16,
        accumulate_grad_batches=args.accum_batches, 
        callbacks=[ckpt_callback, demo_callback, exc_callback],
        logger=wandb_logger,
        log_every_n_steps=1,
        max_epochs=10000000,
        default_root_dir=args.save_dir
    )

    diffusion_trainer.fit(latent_diffusion_model, train_dl)

if __name__ == '__main__':
    main()

