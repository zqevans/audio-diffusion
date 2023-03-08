#@title Imports and definitions
import argparse 
from contextlib import contextmanager
from copy import deepcopy
import math
from pathlib import Path

import sys
import gc
from truss import create

from autoencoders.soundstream import SoundStreamXLEncoder, SoundStreamXLDecoder
from autoencoders.models import AudioAutoencoder
from audio_encoders_pytorch import Encoder1d
from ema_pytorch import EMA
from audio_diffusion_pytorch.modules import UNetCFG1d

from audio_diffusion_pytorch import T5Embedder, NumberEmbedder

import torch
from torch import optim, nn
from torch.nn import functional as F
from torch.utils import data
from tqdm import trange
from einops import rearrange

# import torchaudio
from decoders.diffusion_decoder import DiffusionAttnUnet1D
import numpy as np
import laion_clap
import k_diffusion as K

from torch.nn.parameter import Parameter

class LatentAudioDiffusionAutoencoder(nn.Module):
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

    def decode(self, latents, steps=13, device="cuda"):
        first_stage_latent_noise = torch.randn([latents.shape[0], self.latent_dim, latents.shape[2]*self.latent_downsampling_ratio]).to(device)

        denoiser = K.external.VDenoiser(self.diffusion)

        sigma_max=50
        sigma_min=0.11

        sigmas = K.sampling.get_sigmas_polyexponential(steps, sigma_min, sigma_max, device=device)
        first_stage_latent_noise = first_stage_latent_noise * sigmas[0]

        first_stage_sampled = K.sampling.sample_dpmpp_2m(denoiser, first_stage_latent_noise, sigmas, extra_args=dict(cond=latents))

        first_stage_sampled = first_stage_sampled.clamp(-1, 1)
        decoded = self.autoencoder.decode(first_stage_sampled)
        return decoded

class StackedAELatentDiffusionCond(nn.Module):
    def __init__(self, latent_ae: LatentAudioDiffusionAutoencoder, clap_module: laion_clap.CLAP_Module, diffusion_config):
        super().__init__()

        self.latent_dim = latent_ae.second_stage_latent_dim
        self.downsampling_ratio = latent_ae.downsampling_ratio

        self.embedding_features = 512

        self.embedder = clap_module

        self.diffusion = UNetCFG1d(**diffusion_config)

        self.autoencoder = latent_ae

        self.autoencoder.requires_grad_(False)

    def encode(self, reals):
        return self.autoencoder.encode(reals)

    def decode(self, latents, steps=100, device="cuda"):
        return self.autoencoder.decode(latents, steps, device=device)


def prune_ckpt_weights(stacked_state_dict):
  new_state_dict = {}
  for name, param in stacked_state_dict.items():
      if name.startswith("diffusion_ema.ema_model."):
          new_name = name.replace("diffusion_ema.ema_model.", "diffusion.")
          if isinstance(param, Parameter):
              # backwards compatibility for serialized parameters
              param = param.data
          new_state_dict[new_name] = param
      elif name.startswith("autoencoder") or name.startswith("embedder"):
          new_state_dict[name] = param
          
  return new_state_dict


if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--ckpt_path', help='Path to the checkpoint to be pruned')
    parser.add_argument('--clap_ckpt_path', help='Path to the CLAP checkpoint')
    parser.add_argument('--clap_fusion', action='store_true', help='Enable CLAP fusion', default=False)
    parser.add_argument('--clap_amodel', help='CLAP amodel', default="HTSAT-tiny")
    #parser.add_argument('--output_path', help='Path to the checkpoint to be pruned')
    args = parser.parse_args()

    print("Creating the model...")

    first_stage_config = {"capacity": 64, "c_mults": [2, 4, 8, 16, 32], "strides": [2, 2, 2, 2, 2], "latent_dim": 32}

    first_stage_autoencoder = AudioAutoencoder( 
        **first_stage_config
    ).eval()

    diffusion_config = dict(
        in_channels = 32, 
        context_embedding_features = 512,
        context_embedding_max_length = 1, 
        channels = 256,
        resnet_groups = 8,
        kernel_multiplier_downsample = 2,
        multipliers = [2, 3, 4, 4, 4, 4],
        factors = [1, 2, 2, 4, 4],
        num_blocks = [3, 3, 3, 3, 3],
        attentions = [0, 0, 2, 2, 2, 2],
        attention_heads = 16,
        attention_features = 64,
        attention_multiplier = 4,
        attention_use_rel_pos=True,
        attention_rel_pos_max_distance=2048,
        attention_rel_pos_num_buckets=256,
        use_nearest_upsample = False,
        use_skip_scale = True,
        use_context_time = True,
    )

    clap_config = dict(
        clap_fusion = args.clap_fusion,
        clap_amodel = args.clap_amodel
    )

    latent_diffae = LatentAudioDiffusionAutoencoder(autoencoder=first_stage_autoencoder)

    clap_model = laion_clap.CLAP_Module(enable_fusion=args.clap_fusion, amodel= args.clap_amodel).requires_grad_(False).eval()

    if args.clap_ckpt_path:
        clap_model.load_ckpt(ckpt=args.clap_ckpt_path)
    else:
        clap_model.load_ckpt(model_id=1)

    model = StackedAELatentDiffusionCond(latent_diffae, clap_module=clap_model, diffusion_config=diffusion_config)

    ckpt_state_dict = torch.load(args.ckpt_path)["state_dict"]
    #print(ckpt_state_dict.keys())

    new_ckpt = {}

    new_ckpt["state_dict"] = prune_ckpt_weights(ckpt_state_dict)

    # new_ckpt["model_config"] = dict(
    #         version = (0, 0, 1),
    #         model_info = dict(
    #             name = 'Clap Conditioned Dance Diffusion Model',
    #             description = 'v1.0',
    #             type = 'CCDD',
    #             native_chunk_size = 2097152,
    #             sample_rate = 48000,
    #         ),
    #         autoencoder_config = first_stage_config,
    #         diffusion_config = diffusion_config,
    #         clap_config = clap_config
    #     )

    model.load_state_dict(new_ckpt["state_dict"], strict=False)

    # torch.save(new_ckpt, f'./pruned.ckpt')

    create(
        model,
        target_directory='./clap_truss/',
    )