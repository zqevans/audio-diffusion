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

from diffusion.pqmf import CachedPQMF as PQMF
from autoencoders.soundstream import SoundStreamXLEncoder, SoundStreamXLDecoder

from quantizer_pytorch import Quantizer1d

from audio_diffusion_pytorch import UNet1d
from diffusion.model import ema_update
from aeiou.viz import embeddings_table, pca_point_cloud, audio_spectrogram_image, tokens_spectrogram_image
from aeiou.datasets import AudioDataset
from dataset.dataset import SampleDataset

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
def sample(model, x, steps, eta):
    """Draws samples from a model given starting noise."""
    ts = x.new_ones([x.shape[0]])

    # Create the noise schedule
    t = torch.linspace(1, 0, steps + 1)[:-1]

    alphas, sigmas = get_alphas_sigmas(t)

    # The sampling loop
    for i in trange(steps):

        # Get the model output (v, the predicted velocity)
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

class AudioAutoencoder(pl.LightningModule):
    def __init__(self, global_args):
        super().__init__()

        self.latent_dim = global_args.latent_dim
        
        self.pqmf_bands = global_args.pqmf_bands

        if self.pqmf_bands > 1:
            self.pqmf = PQMF(2, 70, global_args.pqmf_bands)

        capacity = 64

        c_mults = [2, 4, 8, 16, 32]
        
        strides = [2, 2, 2, 2, 2]

        self.downsampling_ratio = np.prod(strides)

        self.encoder = SoundStreamXLEncoder(
            in_channels=2*global_args.pqmf_bands, 
            capacity=capacity, 
            latent_dim=global_args.latent_dim,
            c_mults = c_mults,
            strides = strides
        )

        self.decoder = SoundStreamXLDecoder(
            out_channels=2*global_args.pqmf_bands, 
            capacity=capacity, 
            latent_dim=global_args.latent_dim,
            c_mults = c_mults,
            strides = strides
            )

        self.quantizer = None

        self.num_residuals = global_args.num_residuals
        if self.num_residuals > 0:
            self.quantizer = Quantizer1d(
                channels = global_args.latent_dim,
                num_groups = 1,
                codebook_size = global_args.codebook_size,
                num_residuals = self.num_residuals,
                shared_codebook = False,
                expire_threshold=0.5
            )

    def encode(self, audio, with_info = False):
        return torch.tanh(self.encoder(audio))

    def decode(self, latents):
        if self.quantizer:
            latents, _ = self.quantizer(latents)
        return self.decoder(latents)

class LatentAudioDiffusion(pl.LightningModule):
    def __init__(self, global_args, autoencoder: AudioAutoencoder):
        super().__init__()

        
        self.latent_dim = autoencoder.latent_dim
        self.downsampling_ratio = autoencoder.downsampling_ratio

        self.diffusion = UNet1d(
            in_channels = self.latent_dim, 
            channels = 256,
            patch_blocks = 1,
            patch_factor = 8,
            resnet_groups = 8,
            kernel_multiplier_downsample = 2,
            multipliers = [1, 2, 4, 4, 4, 4, 4, 4, 4],
            factors = [2, 2, 2, 2, 2, 2, 4, 4],
            num_blocks = [4, 2, 2, 2, 2, 2, 4, 4],
            attentions = [0, 0, 0, 1, 1, 1, 2, 2, 2],
            attention_heads = 8,
            attention_features = 64,
            attention_multiplier = 2,
            use_nearest_upsample = False,
            use_skip_scale = True,
            use_context_time = True,
            use_magnitude_channels = False
        )

        self.diffusion_ema = deepcopy(self.diffusion)

        self.diffusion_ema.eval()

        self.autoencoder = autoencoder

        self.encoder = autoencoder.encoder
        self.decoder = autoencoder.decoder

        self.autoencoder.eval()
        
        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)
        self.ema_decay = global_args.ema_decay

    def encode(self, reals):
        return self.autoencoder.encode(reals)

    def decode(self, latents):
        return self.autoencoder.decode(latents)

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
        self.demo_steps = global_args.demo_steps
        self.num_demos = global_args.num_demos
        self.sample_rate = global_args.sample_rate

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
            latent_noise = torch.randn([self.num_demos, module.latent_dim, self.demo_samples//module.downsampling_ratio]).to(module.device)
            fake_latents = sample(module.diffusion_ema, latent_noise, self.demo_steps, 0)
            noise = torch.randn([self.num_demos, 2, self.demo_samples]).to(module.device)
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

            log_dict[f'embeddings'] = embeddings_table(fake_latents)
            log_dict[f'embeddings_3dpca'] = pca_point_cloud(fake_latents)
            log_dict[f'embeddings_spec'] = wandb.Image(tokens_spectrogram_image(fake_latents))


            print("Done logging")
            trainer.logger.experiment.log(log_dict, step=trainer.global_step)

        except Exception as e:
            print(f'{type(e).__name__}: {e}', file=sys.stderr)

def main():

    args = get_all_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    torch.manual_seed(args.seed)

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
                               num_workers=args.num_workers, persistent_workers=True, pin_memory=True)
    wandb_logger = pl.loggers.WandbLogger(project=args.name)
    
    exc_callback = ExceptionCallback()
    ckpt_callback = pl.callbacks.ModelCheckpoint(every_n_train_steps=args.checkpoint_every, save_top_k=-1)
    demo_callback = DemoCallback(args)
    autoencoder = AudioAutoencoder.load_from_checkpoint(args.pretrained_ckpt_path, global_args=args).eval()
    if args.ckpt_path:
        latent_diffusion_model = LatentAudioDiffusion.load_from_checkpoint(args.ckpt_path, global_args=args, autoencoder=autoencoder, strict=False)
    else:
        latent_diffusion_model = LatentAudioDiffusion(args, autoencoder)
    wandb_logger.watch(latent_diffusion_model)
    push_wandb_config(wandb_logger, args)

    diffusion_trainer = pl.Trainer(
        devices=args.num_gpus,
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

    diffusion_trainer.fit(latent_diffusion_model, train_dl)

if __name__ == '__main__':
    main()

