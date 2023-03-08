## Modified from https://github.com/wesbz/SoundStream/blob/main/net.py
import numpy as np
import torch
import torch.nn as nn

from blocks.blocks import Downsample1d, SelfAttention1d, Upsample1d

from autoencoders.soundstream import SoundStreamXLEncoder, SoundStreamXLDecoder
from audio_encoders_pytorch import Encoder1d
from copy import deepcopy
from decoders.diffusion_decoder import DiffusionAttnUnet1D
from diffusion.pqmf import CachedPQMF as PQMF
from diffusion.sampling import sample

PQMF_ATTN = 100

# class AttnResEncoder1D(nn.Module):
#     def __init__(
#         self,  
#         n_io_channels=2, 
#         latent_dim=256,
#         depth=8, 
#         n_attn_layers = 5, 
#         c_mults = [256, 512, 1024, 1024, 1024, 1024, 1024, 1024]
#     ):
#         super().__init__()

#         max_depth = 12
#         depth = min(depth, max_depth)
                
#         self.act = torch.tanh
        
#         c_mults = c_mults[:depth]

#         conv_block = DilatedConvBlock

#         attn_start_layer = depth - n_attn_layers - 1

#         c = c_mults[0]
#         layers = [nn.Sequential(
#                     conv_block(n_io_channels, c, c),
#                     conv_block(c, c, c),
#                     conv_block(c, c, c),
#                     conv_block(c, c, c),
#                 )]

#         for i in range(1, depth):
#             c = c_mults[i]
#             c_prev = c_mults[i - 1]
#             add_attn = i >= attn_start_layer and n_attn_layers > 0
#             layers.append(nn.Sequential(
#                 Downsample1d(kernel="cubic"),
#                 conv_block(c_prev, c, c),
#                 SelfAttention1d(
#                     c, c // 32) if add_attn else nn.Identity(),
#                 conv_block(c, c, c),
#                 SelfAttention1d(
#                     c, c // 32) if add_attn else nn.Identity(),
#                 conv_block(c, c, c),
#                 SelfAttention1d(
#                     c, c // 32) if add_attn else nn.Identity(),
#                 conv_block(c, c, c),
#                 SelfAttention1d(
#                     c, c // 32) if add_attn else nn.Identity(),
#             ))
        

#         layers.append(nn.Sequential(
#                     conv_block(c, c, c),
#                     conv_block(c, c, c),
#                     conv_block(c, c, c),
#                     conv_block(c, c, latent_dim, is_last=True)
#                     )
#                 )

#         self.net = nn.Sequential(*layers)

#         with torch.no_grad():
#             for param in self.net.parameters():
#                 param *= 0.5

#     def forward(self, input):
#         return self.act(self.net(input))

# class AttnResDecoder1D(nn.Module):
#     def __init__(
#         self, 
#         n_io_channels=2, 
#         latent_dim=256,
#         depth=8, 
#         n_attn_layers = 5, 
#         c_mults = [256, 512, 1024, 1024, 1024, 1024, 1024, 1024]
#     ):
#         super().__init__()

#         max_depth = 12
#         depth = min(depth, max_depth)
                
#         self.act = torch.tanh
        
#         c_mults = c_mults[:depth]

#         c_mults = c_mults[::-1]

#         conv_block = DilatedConvBlock

#         c = c_mults[0]
#         layers = [nn.Sequential(
#                     conv_block(latent_dim, c, c),
#                     conv_block(c, c, c),
#                     conv_block(c, c, c),
#                     conv_block(c, c, c),
#                 )]

#         for i in range(1, depth):
#             c = c_mults[i]
#             c_prev = c_mults[i - 1]
#             add_attn = i < n_attn_layers and n_attn_layers > 0
#             layers.append(nn.Sequential(
#                 conv_block(c_prev, c, c),
#                 SelfAttention1d(
#                     c, c // 32) if add_attn else nn.Identity(),
#                 conv_block(c, c, c),
#                 SelfAttention1d(
#                     c, c // 32) if add_attn else nn.Identity(),
#                 conv_block(c, c, c),
#                 SelfAttention1d(
#                     c, c // 32) if add_attn else nn.Identity(),
#                 conv_block(c, c, c),
#                 SelfAttention1d(
#                     c, c // 32) if add_attn else nn.Identity(),
#                 Upsample1d(kernel="cubic"),
#             ))
        

#         layers.append(nn.Sequential(
#                     conv_block(c, c, c),
#                     conv_block(c, c, c),
#                     conv_block(c, c, c),
#                     conv_block(c, c, n_io_channels, is_last=True)
#                     )
#                 )

#         self.net = nn.Sequential(*layers)

#         with torch.no_grad():
#             for param in self.net.parameters():
#                 param *= 0.5

#     def forward(self, input):
#         return self.act(self.net(input))


class AudioAutoencoder(nn.Module):
    def __init__(
        self,
        capacity = 64,
        c_mults = [2, 4, 8, 16, 32],        
        strides = [2, 2, 2, 2, 2],
        latent_dim = 32,
        in_channels = 2,
        out_channels = 2
    ):
        super().__init__()

        self.downsampling_ratio = np.prod(strides)

        self.latent_dim = latent_dim

        self.encoder = SoundStreamXLEncoder(
            in_channels = in_channels, 
            capacity = capacity, 
            latent_dim = self.latent_dim,
            c_mults = c_mults,
            strides = strides
        )

        self.decoder = SoundStreamXLDecoder(
            out_channels=out_channels, 
            capacity = capacity, 
            latent_dim = self.latent_dim,
            c_mults = c_mults,
            strides = strides
        )

    def encode(self, audio):
        return torch.tanh(self.encoder(audio))

    def decode(self, latents):
        return self.decoder(latents)

class AudioVAE(nn.Module):
    def __init__(
        self,
        capacity = 64,
        c_mults = [2, 4, 8, 16, 32],        
        strides = [2, 2, 2, 2, 2],
        latent_dim = 32,
        in_channels = 2,
        out_channels = 2,
        pqmf_bands = 1
    ):
        super().__init__()

        self.pqmf_bands = pqmf_bands

        if self.pqmf_bands > 1:
            self.pqmf = PQMF(2, PQMF_ATTN, pqmf_bands)

        self.downsampling_ratio = np.prod(strides)

        self.latent_dim = latent_dim

        self.encoder = SoundStreamXLEncoder(
            in_channels = in_channels, 
            capacity = capacity, 
            latent_dim = 2 * self.latent_dim,
            c_mults = c_mults,
            strides = strides
        )

        self.decoder = SoundStreamXLDecoder(
            out_channels=out_channels, 
            capacity = capacity, 
            latent_dim = self.latent_dim,
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
        latents, kl = self.sample(mean, scale)

        return latents, kl

    def decode(self, latents):
        decoded = self.decoder(latents)

        if self.pqmf_bands > 1:
            decoded = self.pqmf.inverse(decoded)

        return decoded

class LatentAudioDiffusionAutoencoder(nn.Module):
    def __init__(
        self, 
        autoencoder: AudioAutoencoder,
        second_stage_latent_dim = 32,
        downsample_factors = [2, 2, 2, 2],
        encoder_base_channels = 128,
        encoder_channel_mults = [1, 2, 4, 8, 8],
        encoder_num_blocks = [8, 8, 8, 8],
        diffusion_channel_dims = [512] * 10
    ):
        super().__init__()

        self.latent_dim = autoencoder.latent_dim
                
        self.second_stage_latent_dim = second_stage_latent_dim

        self.latent_downsampling_ratio = np.prod(downsample_factors)
        
        self.downsampling_ratio = autoencoder.downsampling_ratio * self.latent_downsampling_ratio

        self.latent_encoder = Encoder1d(
            in_channels=self.latent_dim, 
            out_channels = self.second_stage_latent_dim,
            channels = encoder_base_channels,
            multipliers = encoder_channel_mults,
            factors =  downsample_factors,
            num_blocks = encoder_num_blocks,
        )

        self.latent_encoder_ema = deepcopy(self.latent_encoder)

        self.diffusion = DiffusionAttnUnet1D(
            io_channels=self.latent_dim, 
            cond_dim = self.second_stage_latent_dim,
            n_attn_layers=0, 
            c_mults=diffusion_channel_dims,
            depth=len(diffusion_channel_dims)
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

        first_stage_sampled = sample(self.diffusion, first_stage_latent_noise, steps, 0, cond=latents)
        first_stage_sampled = first_stage_sampled.clamp(-1, 1)
        decoded = self.autoencoder.decode(first_stage_sampled)
        return decoded

