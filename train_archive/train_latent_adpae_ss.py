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
from audio_encoders_pytorch import AutoEncoder1d, TanhBottleneck, VariationalBottleneck, Bottleneck
from ema_pytorch import EMA

import auraloss


from quantizer_pytorch import Quantizer1d

from decoders.diffusion_decoder import DiffusionAttnUnet1D
from diffusion.model import ema_update
from aeiou.viz import embeddings_table, pca_point_cloud, audio_spectrogram_image, tokens_spectrogram_image
from aeiou.datasets import AudioDataset
from dataset.dataset import SampleDataset

class VAEBottleneck(Bottleneck):
    # copied/modified from RAVE code
    def __init__(self, channels, tanh_means = False):
        super().__init__()
        self.to_mean_scale = nn.Conv1d(
            in_channels=channels,
            out_channels=channels * 2,
            kernel_size=1,
        )

        self.tanh_means = tanh_means

    def sample(self, mean, scale):
        stdev = nn.functional.softplus(scale) + 1e-4
        var = stdev * stdev
        logvar = torch.log(var)
        latent = torch.randn_like(mean) * stdev + mean

        kl = (mean * mean + var - logvar - 1).sum(1).mean()

        return latent, dict(loss=kl, mean=mean, logvar=logvar)

    def forward(self, x, with_info = False):
        #Map input channels to 2x and split them out
        mean, scale = self.to_mean_scale(x).chunk(2, dim=1)

        if self.tanh_means:
            mean = torch.tanh(mean)

        latent, info = self.sample(mean, scale)

        return latent, info if with_info else latent

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

class AudioAutoencoder(pl.LightningModule):
    def __init__(self, global_args):
        super().__init__()

        
        self.pqmf_bands = global_args.pqmf_bands

        if self.pqmf_bands > 1:
            self.pqmf = PQMF(2, 70, global_args.pqmf_bands)

        capacity = 64

        c_mults = [2, 4, 8, 16, 32]
        
        strides = [2, 2, 2, 2, 2]

        global_args.latent_dim = 32

        self.downsampling_ratio = np.prod(strides)

        self.latent_dim = global_args.latent_dim

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

class LatentAudioAutoencoder(pl.LightningModule):
    def __init__(self, global_args, autoencoder: AudioAutoencoder):
        super().__init__()

        
        self.latent_dim = autoencoder.latent_dim
        self.downsampling_ratio = autoencoder.downsampling_ratio

        second_stage_latent_dim = 32

        self.latent_autoencoder = AutoEncoder1d(
            in_channels=self.latent_dim, 
            channels = 64,
            multipliers = [1, 2, 4, 8, 16],
            factors =  [2, 2, 2, 2],
            num_blocks = [8, 8, 8, 8],
            bottleneck_channels = second_stage_latent_dim,
            bottleneck = TanhBottleneck()
        )

        # Scale down the encoder parameters to avoid saturation
        with torch.no_grad():
            for param in self.latent_autoencoder.parameters():
                param *= 0.25

        self.latent_autoencoder_ema = EMA(
            self.latent_autoencoder,
            beta = 0.9999,
            update_every = 10,
            update_after_step = 1000
        )

        #self.latent_encoder_ema.requires_grad_(False)

        self.autoencoder = autoencoder

        self.encoder = autoencoder.encoder
        self.decoder = autoencoder.decoder

        self.autoencoder.requires_grad_(False)

        scales = [2048, 1024, 512, 256, 128]
        hop_sizes = []
        win_lengths = []
        overlap = 0.75
        for s in scales:
            hop_sizes.append(int(s * (1 - overlap)))
            win_lengths.append(s)

        self.sdstft = auraloss.freq.SumAndDifferenceSTFTLoss(fft_sizes=scales, hop_sizes=hop_sizes, win_lengths=win_lengths)

    def encode(self, reals):
        first_stage_latents = self.autoencoder.encode(reals)

        if self.training:
            return self.latent_autoencoder.encode(first_stage_latents)
        else:
            return self.latent_autoencoder_ema.encode(first_stage_latents)

    def decode(self, latents):
        

        if self.training:
            first_stage_decoded = self.latent_autoencoder.decode(latents)
        else:
            first_stage_decoded = self.latent_autoencoder_ema.decode(latents)
        decoded = self.autoencoder.decode(first_stage_decoded)
        return decoded

    def configure_optimizers(self):
        return optim.Adam([*self.latent_autoencoder.parameters()], lr=4e-5)

    def training_step(self, batch, batch_idx):
        reals = batch

        with torch.cuda.amp.autocast():
            first_stage_latents = self.autoencoder.encode(reals)
     
            second_stage_latents = self.latent_autoencoder.encode(first_stage_latents)

            decoded_first_stage = self.latent_autoencoder.decode(second_stage_latents)

            decoded = self.autoencoder.decode(decoded_first_stage)

        mrstft_loss = self.sdstft(reals, decoded)

        # kl_loss = info["bottleneck_loss"]

        # kl_loss = 1e-6 * kl_loss

        loss = mrstft_loss # + kl_loss

        log_dict = {
            'train/loss': loss.detach(),
          #  'train/kl_loss': kl_loss.detach(),
            'train/mrstft_loss': mrstft_loss.detach(),            
        }

        self.log_dict(log_dict, prog_bar=True, on_step=True)
        return loss

    def on_before_zero_grad(self, *args, **kwargs):
        self.latent_autoencoder_ema.update()

class ExceptionCallback(pl.Callback):
    def on_exception(self, trainer, module, err):
        print(f'{type(err).__name__}: {err}')


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
                first_stage_latents = module.autoencoder.encode(demo_reals)
                second_stage_latents = module.latent_autoencoder_ema.ema_model.encode(first_stage_latents)
                first_stage_decoded = module.latent_autoencoder_ema.ema_model.decode(second_stage_latents)

            print("Reconstructing")
            reconstructed = module.autoencoder.decode(first_stage_decoded)

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

            log_dict[f'embeddings_3dpca'] = pca_point_cloud(second_stage_latents)
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

    train_dl = data.DataLoader(train_set, args.batch_size, shuffle=True,
                               num_workers=args.num_workers, persistent_workers=True, pin_memory=True)
    wandb_logger = pl.loggers.WandbLogger(project=args.name)
    demo_dl = data.DataLoader(train_set, args.num_demos, num_workers=args.num_workers, shuffle=True)
    exc_callback = ExceptionCallback()
    ckpt_callback = pl.callbacks.ModelCheckpoint(every_n_train_steps=args.checkpoint_every, save_top_k=-1)
    demo_callback = DemoCallback(demo_dl, args)

    autoencoder = AudioAutoencoder.load_from_checkpoint(args.pretrained_ckpt_path, global_args=args).requires_grad_(False)
    
    latent_autoencoder = LatentAudioAutoencoder(args, autoencoder)
        
    wandb_logger.watch(latent_autoencoder)
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
    )

    diffusion_trainer.fit(latent_autoencoder, train_dl)

if __name__ == '__main__':
    main()

