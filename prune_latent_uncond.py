#@title Imports and definitions
import argparse 
from contextlib import contextmanager
from copy import deepcopy
import math
from pathlib import Path

import sys
import gc

from autoencoders.soundstream import SoundStreamXLEncoder, SoundStreamXLDecoder
from autoencoders.models import AudioAutoencoder
from audio_encoders_pytorch import Encoder1d
from ema_pytorch import EMA
from audio_diffusion_pytorch.modules import UNetConditional1d

from audio_diffusion_pytorch import T5Embedder, NumberEmbedder

import torch
from torch import optim, nn
from torch.nn import functional as F
from torch.utils import data
from tqdm import trange
from einops import rearrange

import torchaudio
from decoders.diffusion_decoder import DiffusionAttnUnet1D
import numpy as np

import random
from diffusion.utils import Stereo, PadCrop
from glob import glob

from torch.nn.parameter import Parameter

class LatentAudioDiffusion(nn.Module):
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

    def decode(self, latents, steps=250, device="cuda"):
        first_stage_latent_noise = torch.randn([latents.shape[0], self.latent_dim, latents.shape[2]*self.latent_downsampling_ratio]).to(device)

        t = torch.linspace(1, 0, steps + 1, device=device)[:-1]

        step_list = get_spliced_ddpm_cosine_schedule(t)

        first_stage_sampled = sampling.iplms_sample(self.diffusion, first_stage_latent_noise, step_list, {"cond":latents})
        #first_stage_sampled = sample(self.diffusion, first_stage_latent_noise, steps, 0, cond=latents)
        decoded = self.autoencoder.decode(first_stage_sampled)
        return decoded

def prune_ckpt_weights(stacked_state_dict):
  new_state_dict = {}
  for name, param in stacked_state_dict.items():
      if name.startswith("diffusion_ema.ema_model."):
          new_name = name.replace("diffusion_ema.ema_model.", "diffusion.")
          if isinstance(param, Parameter):
              # backwards compatibility for serialized parameters
              param = param.data
          new_state_dict[new_name] = param
      elif name.startswith("autoencoder") or name.startswith("timestamp_embedder"):
          new_state_dict[name] = param
          
  return new_state_dict


if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--ckpt_path', help='Path to the checkpoint to be pruned')
    args = parser.parse_args()

    print("Creating the model...")

    first_stage_config = {"capacity": 64, "c_mults": [2, 4, 8, 16, 32], "strides": [2, 2, 2, 2, 2], "latent_dim": 32}

    first_stage_autoencoder = AudioAutoencoder( 
        **first_stage_config
    ).eval()

    latent_diffusion_config = 

    latent_diffae = LatentAudioDiffusionAutoencoder(autoencoder=first_stage_autoencoder)

    model = StackedAELatentDiffusionCond(latent_diffae)

    ckpt_state_dict = torch.load(args.ckpt_path)["state_dict"]
    print(ckpt_state_dict.keys())

    new_ckpt = {}

    new_ckpt["state_dict"] = prune_ckpt_weights(ckpt_state_dict)

    model.load_state_dict(new_ckpt["state_dict"])

    torch.save(new_ckpt, f'./pruned.ckpt')