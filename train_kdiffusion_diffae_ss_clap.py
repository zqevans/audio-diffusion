#!/usr/bin/env python3

from prefigure.prefigure import get_all_args, push_wandb_config
from contextlib import contextmanager
from copy import deepcopy
import math
from pathlib import Path
import numpy as np

import random
import json

import sys
import torch
from torch import optim, nn
from torch.nn import functional as F
from torch.utils import data
from tqdm import trange
import pytorch_lightning as pl
from pytorch_lightning.utilities.distributed import rank_zero_only
from einops import rearrange, repeat

import laion_clap

import torchaudio

import wandb

from dataset.dataset import get_wds_loader
from audio_encoders_pytorch import Encoder1d
from autoencoders.models import AudioAutoencoder

from decoders.diffusion_decoder import AudioDenoiserModel
from diffusion.model import ema_update
from viz.viz import embeddings_table, pca_point_cloud, audio_spectrogram_image, tokens_spectrogram_image

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
def sample(model, x, steps, eta, **extra_args):
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

def unwrap_text(str_or_tuple):
    if type(str_or_tuple) is tuple:
        return random.choice(str_or_tuple)
    elif type(str_or_tuple) is str:
        return str_or_tuple

class ConditionedLatentDiffusionAutoencoder(pl.LightningModule):
    def __init__(self, autoencoder: AudioAutoencoder):
        super().__init__()

        self.autoencoder = autoencoder.eval().requires_grad_(False)

        self.clap_model = laion_clap.CLAP_Module(enable_fusion=False).requires_grad_(False).eval()

        self.clap_model.load_ckpt(model_id=1)

        self.clap_features_dim = 512

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

        self.latent_encoder_ema = deepcopy(self.latent_encoder).requires_grad_(False)
  
        self.diffusion = AudioDenoiserModel(
            c_in = self.latent_dim,
            feats_in = 1024,
            depths = [3] * 10,
            channels = [512] * 10,
            self_attn_depths = [False] * 10,
            strides = [2] * 9,
            dropout_rate = 0.0,
            unet_cond_dim = self.second_stage_latent_dim,
            mapping_cond_dim = self.clap_features_dim,
        )
        self.diffusion_ema = deepcopy(self.diffusion).requires_grad_(False)

        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)
        self.ema_decay = 0.995
        
    def encode(self, reals):
        first_stage_latents = self.autoencoder.encode(reals)

        if self.training:
            second_stage_latents = self.latent_encoder(first_stage_latents)
        else:
            second_stage_latents = self.latent_encoder_ema(first_stage_latents)

        second_stage_latents = torch.tanh(second_stage_latents)

        return second_stage_latents

    def decode(self, latents, prompt, steps=250):
        text_embed = self.clap_model.get_text_embedding([prompt]*latents.shape[0])
        text_embed = torch.from_numpy(text_embeddings).to(self.device)
        if self.training:
            first_stage_sampled = sample(self.diffusion, steps, 0, mapping_cond=text_embed, unet_cond=latents, log_sigma=False)
        else:
            first_stage_sampled = sample(self.diffusion_ema, steps, 0, mapping_cond=text_embed, unet_cond=latents, log_sigma=False)

        first_stage_sampled = first_stage_sampled.clamp(-1, 1)
        decoded = self.autoencoder.decode(first_stage_sampled)
        return decoded

    def configure_optimizers(self):
        return optim.Adam([*self.latent_encoder.parameters(), *self.diffusion.parameters()], lr=4e-5)
  
    def training_step(self, batch, batch_idx):
        reals, jsons, _ = batch
        reals = reals[0]

        condition_strings = [unwrap_text(json["text"][0]) for json in jsons]

        with torch.cuda.amp.autocast():
            with torch.no_grad():
                first_stage_latents = self.autoencoder.encode(reals)
                text_embeddings = self.clap_model.get_text_embedding(condition_strings)
                text_embeddings = torch.from_numpy(text_embeddings).to(self.device)
        
        # Draw uniformly distributed continuous timesteps
        t = self.rng.draw(reals.shape[0])[:, 0].to(self.device)

        # Calculate the noise schedule parameters for those timesteps
        alphas, sigmas = get_alphas_sigmas(t)

        # Combine the ground truth latents and the noise
        alphas = alphas[:, None, None]
        sigmas = sigmas[:, None, None]
        noise = torch.randn_like(first_stage_latents)
        noised_latents = first_stage_latents * alphas + noise * sigmas
        targets = noise * alphas - first_stage_latents * sigmas

        # Compute the model output and the loss.
        with torch.cuda.amp.autocast():
            second_stage_latents = self.latent_encoder(first_stage_latents).float()
            second_stage_latents = torch.tanh(second_stage_latents)

        with torch.cuda.amp.autocast():
            v = self.diffusion(noised_latents, t, mapping_cond=text_embeddings, unet_cond=second_stage_latents, log_sigma=False)
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
        self.demo_dl = iter(demo_dl)
        self.sample_rate = global_args.sample_rate

    @rank_zero_only
    @torch.no_grad()
    #def on_train_epoch_end(self, trainer, module):
    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):   
        if (trainer.global_step - 1) % self.demo_every != 0:
            return
    
        demo_reals, jsons, _ = next(self.demo_dl)

        demo_reals = demo_reals[0].to(module.device)
        
        condition_strings = [unwrap_text(json["text"][0]) for json in jsons]

        with torch.no_grad():
            first_stage_latents = module.autoencoder.encode(demo_reals)
            text_embeddings = module.clap_model.get_text_embedding(condition_strings)
            text_embeddings = torch.from_numpy(text_embeddings).to(module.device)
            latent_noise = torch.randn_like(first_stage_latents).to(module.device)
          
            second_stage_latents = module.latent_encoder_ema(first_stage_latents)
            second_stage_latents = torch.tanh(second_stage_latents)

            first_stage_sampled = sample(module.diffusion_ema, latent_noise, self.demo_steps, 0, mapping_cond=text_embeddings, unet_cond=second_stage_latents)
            fakes = module.autoencoder.decode(first_stage_sampled)

        # Put the demos together
        fakes = rearrange(fakes, 'b d n -> d (b n)')
        demo_reals = rearrange(demo_reals, 'b d n -> d (b n)')

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

            log_dict[f'embeddings'] = embeddings_table(second_stage_latents)

            log_dict[f'embeddings_3dpca'] = pca_point_cloud(second_stage_latents)
            log_dict[f'embeddings_spec'] = wandb.Image(tokens_spectrogram_image(second_stage_latents))

            log_dict[f'real_melspec_left'] = wandb.Image(audio_spectrogram_image(demo_reals))
            log_dict[f'recon_melspec_left'] = wandb.Image(audio_spectrogram_image(fakes))


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
        s3_url_prefix="s3://s-laion-audio/webdataset_tar/",
        sample_size=args.sample_size,
        names=names,
        sample_rate=args.sample_rate,
        num_workers=args.num_workers,
        recursive=True,
        random_crop=True,
        epoch_steps=10000,
    )
    
    exc_callback = ExceptionCallback()
    ckpt_callback = pl.callbacks.ModelCheckpoint(every_n_train_steps=args.checkpoint_every, save_top_k=-1)

    first_stage_config = {"capacity": 64, "c_mults": [2, 4, 8, 16, 32], "strides": [2, 2, 2, 2, 2], "latent_dim": 32}

    first_stage_autoencoder = AudioAutoencoder( 
        **first_stage_config
    ).requires_grad_(False).eval()

    first_stage_autoencoder.load_state_dict(torch.load(args.pretrained_ckpt_path)["state_dict"], strict=False)

    demo_callback = DemoCallback(train_dl, args)
    diffusion_model = ConditionedLatentDiffusionAutoencoder(first_stage_autoencoder)

    wandb_logger = pl.loggers.WandbLogger(project=args.name)
    wandb_logger.watch(diffusion_model)
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

    diffusion_trainer.fit(diffusion_model, train_dl, ckpt_path=args.ckpt_path)

if __name__ == '__main__':
    main()

