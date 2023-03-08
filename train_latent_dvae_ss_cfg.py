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
#from autoencoders.models import AudioAutoencoder
from audio_encoders_pytorch import Encoder1d, Decoder1d
from ema_pytorch import EMA

from torch.nn.parameter import Parameter

from quantizer_pytorch import Quantizer1d

from decoders.diffusion_decoder import DiffusionAttnUnet1D
from diffusion.model import ema_update
from aeiou.viz import embeddings_table, pca_point_cloud, audio_spectrogram_image, tokens_spectrogram_image
from aeiou.datasets import AudioDataset
from dataset.dataset import SampleDataset, get_wds_loader

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
def sample(model, x, steps, eta, cond=None, cfg_scale=1.):
    """Draws samples from a model given starting noise."""
    ts = x.new_ones([x.shape[0]])

    # Create the noise schedule
    t = torch.linspace(1, 0, steps + 1)[:-1]

    alphas, sigmas = get_alphas_sigmas(t)

    # The sampling loop
    for i in trange(steps):

        # Get the model output (v, the predicted velocity)
        with torch.cuda.amp.autocast():
            x_in = torch.cat([x, x])
            ts_in = torch.cat([ts, ts])
            cond_input = torch.cat([cond, torch.zeros_like(cond)])
            v_cond, v_uncond = model(x_in, ts_in * t[i], cond_input).float().chunk(2)
        v = v_uncond + cfg_scale * (v_cond - v_uncond)

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

        self.autoencoder.eval().requires_grad_(False)
        
        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)

    def encode(self, reals):
        first_stage_latents = self.autoencoder.encode(reals)

        if self.training:
            second_stage_latents = self.latent_encoder(first_stage_latents)
        else:
            second_stage_latents = self.latent_encoder_ema(first_stage_latents)

        second_stage_latents = torch.tanh(second_stage_latents)

        return second_stage_latents

    def decode(self, latents, steps=250):
        if self.training:
            first_stage_sampled = sample(self.diffusion, steps, 0, latents)
        else:
            first_stage_sampled = sample(self.diffusion_ema, steps, 0, latents)

        first_stage_sampled = first_stage_sampled.clamp(-1, 1)
        decoded = self.autoencoder.decode(first_stage_sampled)
        return decoded

    def configure_optimizers(self):
        params = [*self.latent_encoder.parameters(), *self.diffusion.parameters()]

        optimizer = optim.Adam(params, lr=4e-5)

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500, eta_min=1e-6)

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        reals, jsons, timestamps = batch
        reals = reals[0]

        # if any([load_time > 10 for load_time in load_times]):
        #     print(f"Long load time in: {load_times}")
        #     print(f"File paths: {infos['path']}")

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
            # Finetune just the decoder
            # torch.no_grad():
            second_stage_latents = self.latent_encoder(first_stage_latents).float()

            second_stage_latents = torch.tanh(second_stage_latents)

            p = torch.rand([second_stage_latents.shape[0], 1, 1], device=second_stage_latents.device)
            second_stage_latents = torch.where(p > 0.2, second_stage_latents, torch.zeros_like(second_stage_latents))

            v = self.diffusion(noised_latents, t, second_stage_latents)
            loss = F.mse_loss(v, targets)

        log_dict = {
            'train/loss': loss.detach(),
            'train/lr': self.lr_schedulers().get_last_lr()[0]
        }

        self.log_dict(log_dict, prog_bar=True, on_step=True)
        return loss

    def on_before_zero_grad(self, *args, **kwargs):
        decay = 0.995
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

            demo_reals, _, _ = next(self.demo_dl)

            demo_reals = demo_reals[0]

            demo_reals = demo_reals.to(module.device)

            with torch.no_grad():
                first_stage_latents = module.autoencoder.encode(demo_reals)
                second_stage_latents = module.latent_encoder_ema(first_stage_latents)
                second_stage_latents = torch.tanh(second_stage_latents)
                
            log_dict = {}
            demo_reals = rearrange(demo_reals, 'b d n -> d (b n)')
            reals_filename = f'reals_{trainer.global_step:08}.wav'
            demo_reals = demo_reals.clamp(-1, 1).mul(32767).to(torch.int16).cpu()
            torchaudio.save(reals_filename, demo_reals, self.sample_rate)
            log_dict[f'real'] = wandb.Audio(reals_filename,
                                                    sample_rate=self.sample_rate,
                                                    caption=f'Real')
            log_dict[f'real_melspec_left'] = wandb.Image(audio_spectrogram_image(demo_reals))

            log_dict[f'embeddings_3dpca'] = pca_point_cloud(second_stage_latents, output_type="plotly", mode="lines+markers")
            log_dict[f'embeddings_spec'] = wandb.Image(tokens_spectrogram_image(second_stage_latents))


            trainer.logger.experiment.log(log_dict, step=trainer.global_step)

            demo_cfg_scales = [1, 3, 5]

            latent_noise = torch.randn_like(first_stage_latents).to(module.device)

            for cfg_scale in demo_cfg_scales:

                recon_latents = sample(module.diffusion_ema, latent_noise, self.demo_steps, 0, second_stage_latents, cfg_scale)
                print("Reconstructing")
                reconstructed = module.autoencoder.decode(recon_latents)

                # Put the demos together
                reconstructed = rearrange(reconstructed, 'b d n -> d (b n)')
                
                log_dict = {}
                
                print("Saving files")
                filename = f'recon_demo_{trainer.global_step:08}.wav'
                reconstructed = reconstructed.clamp(-1, 1).mul(32767).to(torch.int16).cpu()
                torchaudio.save(filename, reconstructed, self.sample_rate)


                log_dict[f'recon_cfg_{cfg_scale}'] = wandb.Audio(filename,
                                                    sample_rate=self.sample_rate,
                                                    caption=f'Reconstructed CFG {cfg_scale}')
            
                log_dict[f'recon_melspec_left_cfg_{cfg_scale}'] = wandb.Image(audio_spectrogram_image(reconstructed))

                trainer.logger.experiment.log(log_dict, step=trainer.global_step)

        except Exception as e:
            print(f'{type(e).__name__}: {e}')

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
        epoch_steps=10000
    )

    wandb_logger = pl.loggers.WandbLogger(project=args.name)
    #demo_dl = data.DataLoader(train_dl, args.num_demos, num_workers=args.num_workers, shuffle=True)
    exc_callback = ExceptionCallback()
    ckpt_callback = pl.callbacks.ModelCheckpoint(every_n_train_steps=args.checkpoint_every, save_top_k=-1)
    demo_callback = DemoCallback(train_dl, args)
    autoencoder = AudioAutoencoder.load_from_checkpoint(args.pretrained_ckpt_path, global_args=args).requires_grad_(False)

    if args.ckpt_path:
        latent_diffusion_model = LatentAudioDiffusionAutoencoder.load_from_checkpoint(args.ckpt_path, global_args=args, autoencoder=autoencoder, strict=False)
        # latent_diffusion_model = LatentAudioDiffusionAutoencoder(autoencoder)
        # latent_diffusion_model.load_encoder_weights(torch.load(args.ckpt_path)["state_dict"])
        #latent_diffusion_model.latent_encoder.requires_grad_(False)
    else:
        latent_diffusion_model = LatentAudioDiffusionAutoencoder(autoencoder)

    # # Finetune the decoder only
    # latent_diffusion_model.latent_encoder.requires_grad_(False)
    # # Use the EMA encoder so demos and training inputs are the same
    # latent_diffusion_model.latent_encoder = latent_diffusion_model.latent_encoder_ema

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