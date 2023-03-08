#!/usr/bin/env python3

from prefigure.prefigure import get_all_args, push_wandb_config
from contextlib import contextmanager
from copy import deepcopy
import math
from pathlib import Path

import sys, os
import random
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

import k_diffusion as K

from autoencoders.models import AudioAutoencoder
from audio_encoders_pytorch import Encoder1d
from ema_pytorch import EMA
from audio_diffusion_pytorch import T5Embedder, NumberEmbedder
from audio_diffusion_pytorch.modules import UNetCFG1d

from decoders.diffusion_decoder import DiffusionAttnUnet1D

# Define the noise schedule and sampling loop
def get_alphas_sigmas(t):
    """Returns the scaling factors for the clean image (alpha) and for the
    noise (sigma), given a timestep."""
    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)

def alpha_sigma_to_t(alpha, sigma):
    """Returns a timestep, given the scaling factors for the clean image and for
    the noise."""
    return torch.atan2(sigma, alpha) / math.pi * 2

def sample_v_ddim(model, x, steps, eta, **extra_args):
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


def sample(model_fn, noise, steps=100, sampler_type="v-iplms", device="cuda", **extra_args):
  #Check for k-diffusion
  if sampler_type.startswith('k-'):
    denoiser = K.external.VDenoiser(model_fn)
    sigmas = K.sampling.get_sigmas_vp(steps, device=device)

  elif sampler_type.startswith("v-"):
    t = torch.linspace(1, 0, steps + 1, device=device)[:-1]
    step_list = get_crash_schedule(t)

#   if sampler_type == "v-ddim":
#     return sample_v_ddim(model_fn, noise, step_list, eta, {})
#   elif sampler_type == "v-iplms":
#     return sampling.iplms_sample(model_fn, noise, step_list, {})

  elif sampler_type == "k-heun":
    return K.sampling.sample_heun(denoiser, noise, sigmas, disable=False, extra_args=extra_args)
  elif sampler_type == "k-lms":
    return K.sampling.sample_lms(denoiser, noise, sigmas, disable=False, extra_args=extra_args)
  elif sampler_type == "k-dpmpp_2s_ancestral":
    return K.sampling.sample_dpmpp_2s_ancestral(denoiser, noise, sigmas, disable=False, extra_args=extra_args)
  elif sampler_type == "k-dpm-2":
    return K.sampling.sample_dpm_2(denoiser, noise, sigmas, disable=False, extra_args=extra_args)
  elif sampler_type == "k-dpm-fast":
    return K.sampling.sample_dpm_fast(denoiser, noise, sigma_min, sigma_max, steps, disable=False, extra_args=extra_args)
  elif sampler_type == "k-dpm-adaptive":
    return K.sampling.sample_dpm_adaptive(denoiser, noise, sigma_min, sigma_max, rtol=rtol, atol=atol, disable=False)

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

        self.diffusion = DiffusionAttnUnet1D(
            io_channels=self.latent_dim, 
            cond_dim = self.second_stage_latent_dim,
            n_attn_layers=0, 
            c_mults=[512] * 10,
            depth=10
        )

        self.autoencoder = autoencoder

        self.autoencoder.requires_grad_(False)
        
        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)

    def encode(self, reals):
        first_stage_latents = self.autoencoder.encode(reals)

        second_stage_latents = self.latent_encoder(first_stage_latents)

        second_stage_latents = torch.tanh(second_stage_latents)

        return second_stage_latents

    def decode(self, latents, steps=100, device="cuda"):
        first_stage_latent_noise = torch.randn([latents.shape[0], self.latent_dim, latents.shape[2]*self.latent_downsampling_ratio]).to(device)

        first_stage_sampled = sample_v_ddim(self.diffusion, first_stage_latent_noise, steps, 0, cond=latents)

        # denoiser = K.external.VDenoiser(self.diffusion)

        # sigmas = K.sampling.get_sigmas_vp(steps, device=device)
        # first_stage_latent_noise = first_stage_latent_noise * sigmas[0]
        # #sigmas = K.sampling.get_sigmas_polyexponential(steps, sigma_min=0.1, sigma_max=1.0, rho=7.0, device=device)

        # first_stage_sampled = K.sampling.sample_dpmpp_2s_ancestral(denoiser, first_stage_latent_noise, sigmas, eta=0, extra_args=dict(cond=latents))

        first_stage_sampled = first_stage_sampled.clamp(-1, 1)
        decoded = self.autoencoder.decode(first_stage_sampled)
        return decoded

class StackedAELatentDiffusionCond(pl.LightningModule):
    def __init__(self, latent_ae: LatentAudioDiffusionAutoencoder):
        super().__init__()

        self.latent_dim = latent_ae.second_stage_latent_dim
        self.downsampling_ratio = latent_ae.downsampling_ratio

        embedding_max_len = 128

        self.embedder = T5Embedder(model='t5-base', max_length=embedding_max_len).requires_grad_(False)

        self.embedding_features = 768

        self.timestamp_embedder = NumberEmbedder(features=self.embedding_features)

        self.diffusion = UNetCFG1d(
            in_channels = self.latent_dim, 
            context_embedding_features = self.embedding_features,
            context_embedding_max_length = embedding_max_len + 2, #2 for timestep embeds
            channels = 256,
            resnet_groups = 8,
            kernel_multiplier_downsample = 2,
            multipliers = [2, 3, 4, 4, 4],
            factors = [1, 2, 2, 4],
            num_blocks = [3, 3, 3, 3],
            attentions = [0, 0, 3, 3, 3],
            attention_heads = 16,
            attention_features = 64,
            attention_multiplier = 4,
            attention_use_rel_pos=True,
            attention_rel_pos_max_distance=2048,
            attention_rel_pos_num_buckets=64,
            use_nearest_upsample = False,
            use_skip_scale = True,
            use_context_time = True,
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

    def encode(self, reals):
        return self.autoencoder.encode(reals)

    def decode(self, latents, steps=100):
        return self.autoencoder.decode(latents, steps, device=self.device)


class DemoCallback(pl.Callback):
    def __init__(self, global_args):
        super().__init__()
        self.demo_every = global_args.demo_every
        self.demo_samples = global_args.sample_size
        self.demo_steps = global_args.demo_steps
        self.num_demos = global_args.num_demos
        self.sample_rate = global_args.sample_rate

def main():

    args = get_all_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    seed = args.seed

    if seed == -1:
        seed = random.randint(0, sys.maxsize)
    
    torch.manual_seed(seed)

    first_stage_config = {"capacity": 64, "c_mults": [2, 4, 8, 16, 32], "strides": [2, 2, 2, 2, 2], "latent_dim": 32}

    first_stage_autoencoder = AudioAutoencoder( 
        **first_stage_config
    ).requires_grad_(False).eval()

    latent_diffae = LatentAudioDiffusionAutoencoder(first_stage_autoencoder).requires_grad_(False).eval()

    print("Loading model...")

    latent_diffusion_model = StackedAELatentDiffusionCond.load_from_checkpoint(args.ckpt_path, latent_ae=latent_diffae).to(device).requires_grad_(False).eval()

    print("Model loaded")

    prompt = ''

    timestamp_embeddings = latent_diffusion_model.timestamp_embedder(
        [
            #[0, 0.2],
            [0.2, 0.5],
            #[0.4, 0.70],
            #[0.7, 1.0]
        ]
    )

    text_embeddings = latent_diffusion_model.embedder([prompt] * len(timestamp_embeddings))

    latent_noise = torch.randn([len(text_embeddings), latent_diffusion_model.latent_dim, args.sample_size//latent_diffusion_model.downsampling_ratio]).to(device)

    embeddings = torch.cat([text_embeddings, timestamp_embeddings], dim=1)

    cfg_scale = args.cfg_scale

    print(f"Generating latents, CFG scale {cfg_scale}, seed: {seed}")
    fake_latents = sample_v_ddim(latent_diffusion_model.diffusion_ema, latent_noise, args.gen_steps, args.eta, embedding=embeddings, embedding_scale=cfg_scale)

    # model_fn = latent_diffusion_model.diffusion

    # denoiser = K.external.VDenoiser(model_fn)

    # sigmas = K.sampling.get_sigmas_vp(args.gen_steps, device=device)

    # latent_noise = latent_noise * sigmas[0]

    # fake_latents = K.sampling.sample_dpmpp_2s_ancestral(denoiser, latent_noise, sigmas, eta=args.eta, extra_args=dict(embedding=embeddings, embedding_scale=cfg_scale))
    
    fake_latents = fake_latents.clamp(-1, 1)

    print(f"Decoding latents, shape: {fake_latents.shape}")
    fakes = latent_diffusion_model.decode(fake_latents, steps=args.decoder_steps)

    print("Rearranging demos")
    # Put the demos together
    fakes = rearrange(fakes, 'b d n -> d (b n)')

    # Turn down the outputs
    fakes = fakes * 0.66

    log_dict = {}
    
    print("Saving files")

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    filename = os.path.join(args.save_dir, f"{seed}_cfg_{cfg_scale}.wav")

    # check if file already exists, add an incrementing number to the end to avoid duplicates
    if os.path.isfile(filename):
        i = 1
        while True:
            new_filename = os.path.join(args.save_dir, f"{seed}_cfg_{cfg_scale}_{i}.wav")
            if os.path.isfile(new_filename):
                i += 1
            else:
                filename = new_filename
                break

    fakes = fakes.clamp(-1, 1).mul(32767).to(torch.int16).cpu()
    torchaudio.save(filename, fakes, args.sample_rate)


if __name__ == '__main__':
    main()