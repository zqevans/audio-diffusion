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
from torch.nn.parameter import Parameter
from tqdm import trange
import pytorch_lightning as pl
from pytorch_lightning.utilities.distributed import rank_zero_only
from einops import rearrange
import numpy as np
import torchaudio

import wandb

from diffusion.pqmf import CachedPQMF as PQMF
from autoencoders.soundstream import SoundStreamXLEncoder, SoundStreamXLDecoder
from autoencoders.models import AudioAutoencoder
from audio_encoders_pytorch import Encoder1d
from ema_pytorch import EMA


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
    return torch.atan2(sigma, alpha) / math.pi * 2

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

class LatentAudioDiffusionAutoencoder(pl.LightningModule):
    def __init__(self, autoencoder: AudioAutoencoder):
        super().__init__()

        
        self.latent_dim = autoencoder.latent_dim
                
        self.second_stage_latent_dim = 32

        factors = [2, 2, 2, 2]

        self.latent_downsampling_ratio = np.prod(factors)
        
        self.downsampling_ratio = autoencoder.downsampling_ratio * self.latent_downsampling_ratio

        self.latent_encoder = Encoder1d(
            in_channels=self.latent_dim, 
            out_channels = self.second_stage_latent_dim,
            channels = 128,
            multipliers = [1, 2, 4, 8, 8],
            factors =  factors,
            num_blocks = [8, 8, 8, 8],
        )

        self.latent_encoder_ema = deepcopy(self.latent_encoder)

        self.diffusion = DiffusionAttnUnet1D(
            io_channels=self.latent_dim, 
            cond_dim = self.second_stage_latent_dim,
            n_attn_layers=0, 
            c_mults=[512] * 10,
            depth=10
        )

        self.diffusion_ema = deepcopy(self.diffusion)

        self.diffusion_ema.requires_grad_(False)
        self.latent_encoder_ema.requires_grad_(False)

        self.autoencoder = autoencoder

        self.autoencoder.requires_grad_(False)
        
        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)

    def encode(self, reals):
        first_stage_latents = self.autoencoder.encode(reals)

        second_stage_latents = self.latent_encoder(first_stage_latents)

        second_stage_latents = torch.tanh(second_stage_latents)

        return second_stage_latents

    def decode(self, latents, steps=250, device="cuda"):
        first_stage_latent_noise = torch.randn([latents.shape[0], self.latent_dim, latents.shape[2]*self.latent_downsampling_ratio]).to(device)

        first_stage_sampled = sample(self.diffusion, first_stage_latent_noise, steps, 0, latents)
        first_stage_sampled = first_stage_sampled.clamp(-1, 1)
        decoded = self.autoencoder.decode(first_stage_sampled)
        return decoded

class StackedAELatentDiffusion(pl.LightningModule):
    def __init__(self, latent_ae: LatentAudioDiffusionAutoencoder):
        super().__init__()

        self.latent_dim = latent_ae.second_stage_latent_dim
        self.downsampling_ratio = latent_ae.downsampling_ratio

        self.diffusion = DiffusionAttnUnet1D(
            io_channels=self.latent_dim, 
            n_attn_layers=6, 
            c_mults=[512] * 8,
            depth=8,
            kernel_size=9
        )

        self.diffusion_ema = EMA(
            self.diffusion,
            beta = 0.9999,
            power=3/4,
            update_every = 1,
            update_after_step = 1000
        )

        self.autoencoder = latent_ae

        self.autoencoder.requires_grad_(False)
        
        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)

    def encode(self, reals):
        return self.autoencoder.encode(reals)

    def decode(self, latents, steps=250):
        return self.autoencoder.decode(latents, steps, device=self.device)

    def configure_optimizers(self):
        return optim.Adam([*self.diffusion.parameters()], lr=4e-5)

    def training_step(self, batch, batch_idx):
        reals = batch

        with torch.cuda.amp.autocast():
            with torch.no_grad():
                latents = self.encode(reals)

        # Draw uniformly distributed continuous timesteps
        t = self.rng.draw(reals.shape[0])[:, 0].to(self.device)

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
        self.diffusion_ema.update()

class ExceptionCallback(pl.Callback):
    def on_exception(self, trainer, module, err):
        print(f'{type(err).__name__}: {err}', file=sys.stderr)


class DemoCallback(pl.Callback):
    def __init__(self, global_args):
        super().__init__()
        self.demo_every = global_args.demo_every
        self.demo_samples = global_args.sample_size
        self.demo_steps = global_args.demo_steps
        self.num_demos = global_args.num_demos
        self.sample_rate = global_args.sample_rate

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):   
        last_demo_step = -1
        if (trainer.global_step - 1) % self.demo_every != 0 or last_demo_step == trainer.global_step:
        #if trainer.current_epoch % self.demo_every != 0:
            return
        
        last_demo_step = trainer.global_step
        
        print("Starting demo")
        try:
            latent_noise = torch.randn([self.num_demos, module.latent_dim, self.demo_samples//module.downsampling_ratio]).to(module.device)
            fake_latents = sample(module.diffusion_ema, latent_noise, self.demo_steps, 0)
            print("Decoding fakes")
            fakes = module.decode(fake_latents)

            # Put the demos together
            fakes = rearrange(fakes, 'b d n -> d (b n)')

            log_dict = {}
            
            print("Saving files")
            filename = f'demo_{trainer.global_step:08}.wav'
            fakes = fakes.clamp(-1, 1).mul(32767).to(torch.int16).cpu()
            torchaudio.save(filename, fakes, self.sample_rate)


            log_dict[f'demo'] = wandb.Audio(filename,
                                                sample_rate=self.sample_rate,
                                                caption=f'Reconstructed')
        

            log_dict[f'demo_melspec_left'] = wandb.Image(audio_spectrogram_image(fakes))

            log_dict[f'embeddings_3dpca'] = pca_point_cloud(fake_latents)
            log_dict[f'embeddings_spec'] = wandb.Image(tokens_spectrogram_image(fake_latents))


            print("Done logging")
            trainer.logger.experiment.log(log_dict, step=trainer.global_step)

        except Exception as e:
            print(f'{type(e).__name__}: {e}')

def main():

    args = get_all_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    torch.manual_seed(args.seed)

    args.random_crop = False

    train_set = AudioDataset(
        [args.training_dir],
        sample_rate=args.sample_rate,
        sample_size=args.sample_size,
        random_crop=args.random_crop,
        augs='Stereo(), PhaseFlipper()'
    )

    #args.random_crop = False

    #train_set = SampleDataset([args.training_dir], args, keywords=["kick", "snare", "clap", "snap", "hat", "cymbal", "crash", "ride"])

    train_dl = data.DataLoader(train_set, args.batch_size, shuffle=True,
                               num_workers=args.num_workers, persistent_workers=True, pin_memory=True, drop_last=True)
    wandb_logger = pl.loggers.WandbLogger(project=args.name)
    exc_callback = ExceptionCallback()
    ckpt_callback = pl.callbacks.ModelCheckpoint(every_n_train_steps=args.checkpoint_every, save_top_k=-1)
    demo_callback = DemoCallback(args)

    first_stage_config = {"capacity": 64, "c_mults": [2, 4, 8, 16, 32], "strides": [2, 2, 2, 2, 2], "latent_dim": 32}

    first_stage_autoencoder = AudioAutoencoder( 
        **first_stage_config
    ).eval()

    #latent_diffae = LatentAudioDiffusionAutoencoder(first_stage_autoencoder)

    latent_diffae = LatentAudioDiffusionAutoencoder.load_from_checkpoint(args.pretrained_ckpt_path, autoencoder=first_stage_autoencoder, strict=False)

    latent_diffae.diffusion = latent_diffae.diffusion_ema
    del latent_diffae.diffusion_ema

    latent_diffae.latent_encoder = latent_diffae.latent_encoder_ema
    del latent_diffae.latent_encoder_ema

    #latent_diffae_state_dict = torch.load(args.pretrained_ckpt_path)["state_dict"]

    #latent_diffae.load_weights_from_ema(latent_diffae_load.state_dict())

    #del latent_diffae_load

    latent_diffusion_model = StackedAELatentDiffusion(latent_diffae)

    # if args.ckpt_path:
    #     latent_diffusion_model = LatentAudioDiffusionAutoencoder.load_from_checkpoint(args.ckpt_path, global_args=args, autoencoder=autoencoder, strict=False)
    # else:
    #     latent_diffusion_model = LatentAudioDiffusionAutoencoder(args, autoencoder)
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

    diffusion_trainer.fit(latent_diffusion_model, train_dl, ckpt_path=args.ckpt_path)

if __name__ == '__main__':
    main()

