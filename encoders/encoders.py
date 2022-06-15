## Modified from https://github.com/wesbz/SoundStream/blob/main/net.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.model import ResidualBlock, SelfAttention1d

from perceiver_pytorch import Perceiver

class ResConvBlock(ResidualBlock):
    def __init__(self, c_in, c_mid, c_out, is_last=False):
        skip = None if c_in == c_out else nn.Conv1d(c_in, c_out, 1, bias=False)
        super().__init__([
            nn.Conv1d(c_in, c_mid, 5, padding=2),
            nn.GroupNorm(1, c_mid),
            nn.SiLU(inplace=True),
            nn.Conv1d(c_mid, c_out, 5, padding=2),
            nn.GroupNorm(1, c_out) if not is_last else nn.Identity(),
            nn.SiLU(inplace=True) if not is_last else nn.Identity(),
        ], skip)

class AttnResEncoder1D(nn.Module):
    def __init__(
        self, 
        global_args, 
        n_io_channels=2, 
        depth=12, 
        n_attn_layers = 5, 
        downsamples = [0, 2, 2, 2] + [2] * 8,
        c_mults = [128, 128, 256, 256] + [512] * 8
    ):
        super().__init__()

        max_depth = 12
        depth = min(depth, max_depth)
                
        c_mults = c_mults[:depth]
        downsamples = downsamples[:depth]

        conv_block = ResConvBlock

        attn_start_layer = depth - n_attn_layers - 1

        c = c_mults[0]
        layers = [nn.Sequential(
                    conv_block(n_io_channels, c, c),
                    conv_block(c, c, c),
                    conv_block(c, c, c),
                    conv_block(c, c, c),
                )]

        for i in range(1, depth):
            c = c_mults[i]
            downsample = downsamples[i] 
            c_prev = c_mults[i - 1]
            layers.append(nn.Sequential(
                nn.AvgPool1d(downsample),
                conv_block(c_prev, c, c),
                SelfAttention1d(
                    c, c // 32) if i >= attn_start_layer else nn.Identity(),
                conv_block(c, c, c),
                SelfAttention1d(
                    c, c // 32) if i >= attn_start_layer else nn.Identity(),
                conv_block(c, c, c),
                SelfAttention1d(
                    c, c // 32) if i >= attn_start_layer else nn.Identity(), 
                conv_block(c, c, c),
                SelfAttention1d(
                    c, c // 32) if i >= attn_start_layer else nn.Identity(),                  
            ))
        

        layers.append(nn.Sequential(
                    conv_block(c, c, c),
                    conv_block(c, c, c),
                    conv_block(c, c, c),
                    conv_block(c, c, global_args.latent_dim, is_last=True)
                    )
                )

        self.net = nn.Sequential(*layers)

        print(f"Encoder downsampling ratio: {np.prod(downsamples[1:])}")

        with torch.no_grad():
            for param in self.net.parameters():
                param *= 0.5

    def forward(self, input):
        return self.net(input)

class GlobalEncoder(nn.Sequential):
    def __init__(self, latent_size, io_channels):
        c_in = io_channels
        c_mults = [64, 64, 128, 128] + [latent_size] * 10
        layers = []
        c_mult_prev = c_in
        for i, c_mult in enumerate(c_mults):
            is_last = i == len(c_mults) - 1
            layers.append(ResConvBlock(c_mult_prev, c_mult, c_mult))
            layers.append(ResConvBlock(
                c_mult, c_mult, c_mult, is_last=is_last))
            if not is_last:
                layers.append(nn.AvgPool1d(2))
            else:
                layers.append(nn.AdaptiveAvgPool1d(1))
                layers.append(nn.Flatten())
            c_mult_prev = c_mult
        super().__init__(*layers)

class AudioPerceiverEncoder(nn.Module):
    def __init__(self, n_io_channels, global_args):
        super().__init__()
        self.net = Perceiver(
            input_channels=n_io_channels,          # number of channels for each token of the input
            input_axis=1,# number of axis for input data (1 for audio, 2 for images, 3 for video)            
            num_freq_bands=128,# number of freq bands, with original value (2 * K + 1)
            max_freq=1000.,  # maximum frequency, hyperparameter depending on how fine the data is
            depth=10,# depth of net. The shape of the final attention mechanism will be:
                     # depth * (cross attention -> self_per_cross_attn * self attention)
            num_latents=256,  # number of latents, or induced set points, or centroids. different papers giving it different names
            latent_dim=512,            # latent dimension
            cross_heads=1,             # number of heads for cross attention. paper said 1
            latent_heads=8,            # number of heads for latent self attention, 8
            cross_dim_head=64,         # number of dimensions per cross attention head
            latent_dim_head=64,        # number of dimensions per latent self attention head
            num_classes=global_args.style_latent_size,          # output number of classes
            attn_dropout=0.,
            ff_dropout=0.,
            weight_tie_layers=True,# whether to weight tie layers (optional, as indicated in the diagram)
            fourier_encode_data=True,  # whether to auto-fourier encode the data, using the input_axis given. defaults to True, but can be turned off if you are fourier encoding the data yourself
            self_per_cross_attn=2      # number of self attention blocks per cross attention
        )

    def forward(self, input):
        return self.net(input)
