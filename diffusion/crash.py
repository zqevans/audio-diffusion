import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.utils import PixelShuffle1D, PixelUnshuffle1D
from blocks.blocks import SelfAttention1d, Downsample1d, Upsample1d

# class Conv1d(nn.Conv1d):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#        # self.reset_parameters()

#     def reset_parameters(self):
#         pass
#         # nn.init.orthogonal_(self.weight)
#         # nn.init.zeros_(self.bias)

Conv1d = nn.Conv1d

class RFF_MLP_Block(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.RFF_freq = nn.Parameter(
            16 * torch.randn([1, 32]), requires_grad=False).to(device)
        self.MLP = nn.ModuleList([
            nn.Linear(64, 128),
            nn.Linear(128, 256),
            nn.Linear(256, 512),
        ])

    def forward(self, sigma):
        """
        Arguments:
          sigma:
              (shape: [B, 1], dtype: float32)
        Returns:
          x: embedding of sigma
              (shape: [B, 512], dtype: float32)
        """
        x = self._build_RFF_embedding(sigma)
        for layer in self.MLP:
            x = F.relu(layer(x))
        return x

    def _build_RFF_embedding(self, sigma):
        """
        Arguments:
          sigma:
              (shape: [B, 1], dtype: float32)
        Returns:
          table:
              (shape: [B, 64], dtype: float32)
        """
        freqs = self.RFF_freq
        #freqs = freqs.to(device=torch.device("cuda"))
        table = 2 * np.pi * sigma * freqs
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table


class Film(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.output_layer = nn.Linear(512, 2 * output_dim)

    def forward(self, sigma_encoding):
        sigma_encoding = self.output_layer(sigma_encoding)
        sigma_encoding = sigma_encoding.unsqueeze(-1)
        gamma, beta = torch.chunk(sigma_encoding, 2, dim=1)
        return gamma, beta

class ResidualBlock(nn.Module):
    def __init__(self, main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, input):
        return self.main(input) + self.skip(input)

class DilatedResConvBlock(nn.Module):
    def __init__(self, c_in, c_mid, c_out, is_last=False):
        skip = None if c_in == c_out else nn.Conv1d(c_in, c_out, 1, bias=False)
        super().__init__([
            nn.Conv1d(c_in, c_mid, 5, padding=2),
            nn.GroupNorm(1, c_mid),
            nn.GELU(),
            nn.Conv1d(c_mid, c_out, 5, padding=2),
            nn.GroupNorm(1, c_out) if not is_last else nn.Identity(),
            nn.GELU() if not is_last else nn.Identity(),
        ], skip)


class UBlock(nn.Module):
    def __init__(self, input_size, hidden_size, factor, dilation, self_attn=False):
        super().__init__()
        assert isinstance(dilation, (list, tuple))
        assert len(dilation) == 4

        self.factor = factor
        self.residual_dense = Conv1d(input_size, hidden_size, 1)

        self.layers1 = nn.ModuleList([
            Conv1d(2 * input_size, hidden_size, 3,
                   dilation=dilation[0], padding=dilation[0]),
            Conv1d(hidden_size, hidden_size, 3,
                   dilation=dilation[1], padding=dilation[1]),
            Conv1d(hidden_size, hidden_size, 3,
                   dilation=dilation[2], padding=dilation[2]),
            Conv1d(hidden_size, hidden_size, 3,
                   dilation=dilation[3], padding=dilation[3]),
        ])

        self.layers2 = nn.ModuleList([
            Conv1d(hidden_size, hidden_size, 3,
                   dilation=dilation[0], padding=dilation[0]),
            Conv1d(hidden_size, hidden_size, 3,
                   dilation=dilation[1], padding=dilation[1]),
            Conv1d(hidden_size, hidden_size, 3,
                   dilation=dilation[2], padding=dilation[2]),
            Conv1d(hidden_size, hidden_size, 3,
                   dilation=dilation[3], padding=dilation[3]),
        ])

        self.self_attn = SelfAttention1d(hidden_size, hidden_size//8) if self_attn else nn.Identity()

        self.up = Upsample1d(kernel="cubic")

    def forward(self, x, x_dblock):
        size = x.shape[-1] * self.factor

        residual = F.interpolate(x, size=size)
        residual = self.residual_dense(residual)

        x = torch.cat([x, x_dblock], dim=1)
        x = F.leaky_relu(x, 0.2)

        if self.factor == 2:
            x = self.up(x)
        else:
            x = F.interpolate(x, size=size)
        
        for layer in self.layers:
            x = F.leaky_relu(x, 0.2)
            x = layer(x)

        x = x + residual



        for layer in self.layers:
            x = F.leaky_relu(x, 0.2)
            x = layer(x)
            
        x = x + residual
        
        return self.self_attn(x)


class DBlock(nn.Module):
    def __init__(self, input_size, hidden_size, factor, self_attn=False):
        super().__init__()
        self.factor = factor
        self.residual_dense = Conv1d(input_size, hidden_size, 1)
        self.layer_1 = Conv1d(input_size, hidden_size,
                              3, dilation=1, padding=1)
        self.convs1 = nn.ModuleList([
            Conv1d(hidden_size, hidden_size, 3, dilation=2, padding=2),
            Conv1d(hidden_size, hidden_size, 3, dilation=4, padding=4),
            Conv1d(hidden_size, hidden_size, 3, dilation=8, padding=8),
        ])

        self.convs2 = nn.ModuleList([
            Conv1d(hidden_size, hidden_size, 3, dilation=1, padding=1),
            Conv1d(hidden_size, hidden_size, 3, dilation=2, padding=2),
            Conv1d(hidden_size, hidden_size, 3, dilation=4, padding=4),
            Conv1d(hidden_size, hidden_size, 3, dilation=8, padding=8),
        ])

        self.self_attn = SelfAttention1d(hidden_size, hidden_size//8) if self_attn else nn.Identity()

        self.down = Downsample1d(kernel="cubic")

    def forward(self, x, gamma=1, beta=0):
        size = x.shape[-1] // self.factor

        residual = self.residual_dense(x)
        residual = F.interpolate(residual, size=size)

        if self.factor == 2:
            x = self.down(x)
        else:
            x = F.interpolate(x, size=size)

        x = F.leaky_relu(x, 0.2)
        x = self.layer_1(x)
        x = gamma * x + beta

        for layer in self.convs1:
            x = F.leaky_relu(x, 0.2)
            x = layer(x)
        x = x + residual
        
        residual = x

        for layer in self.convs2:
            x = F.leaky_relu(x, 0.2)
            x = layer(x)
            
        x = x + residual
        
        return self.self_attn(x)


class CrashEncoder(nn.Module):
    def __init__(self, in_channels = 2, latent_dim=256):
        super().__init__()
        
        self.downsample = nn.Sequential(
            Conv1d(in_channels, 128, 5, padding=2),
            DBlock(128, 128, 2), # 48 kHz -> 24 kHz
            DBlock(128, 128, 2), # 24 kHz -> 12 kHz
            DBlock(128, 256, 3), # 12 kHz -> 4 kHz
            DBlock(256, 512, 5), # 4 kHz -> 800 Hz
            DBlock(512, 512, 5), # 800 Hz -> 160 Hz
            Conv1d(512, latent_dim, 3, padding=1)
        )

    def forward(self, audio):
        return torch.tanh(self.downsample(audio))

class CrashUNet(nn.Module):
    def __init__(self, n_io_channels = 2, ps_ratio = 1, unet_cond_dim=0, device="cuda"):
        super().__init__()
        self.conv_1 = Conv1d(n_io_channels * ps_ratio + unet_cond_dim, 256, 5, padding=2)
        self.embedding = RFF_MLP_Block(device)

        self.downsample = nn.ModuleList([
            DBlock(256, 256, 2), # 48 kHz -> 24 kHz
            DBlock(256, 256, 2), # 24 kHz -> 12 kHz
            DBlock(256, 512, 3), # 12 kHz -> 4 kHz
            DBlock(512, 1024, 5), # 4 kHz -> 800 Hz
            DBlock(1024, 1024, 5, self_attn=True), # 800 Hz -> 160 Hz
        ])
        self.gamma_beta = nn.ModuleList([
            Film(256),
            Film(256),
            Film(512),
            Film(1024),
            Film(1024),
        ])
        self.upsample = nn.ModuleList([
            UBlock(1024, 1024, 5, [1, 2, 4, 8], self_attn=True),
            UBlock(1024, 512, 5, [1, 2, 4, 8]),
            UBlock(512, 256, 3, [1, 2, 4, 8]),
            UBlock(256, 256, 2, [1, 2, 4, 8]),
            UBlock(256, 256, 2, [1, 2, 4, 8]),
        ])

        self.last_conv = Conv1d(256, n_io_channels * ps_ratio, 3, padding=1)

        self.ps_ratio = ps_ratio

        if ps_ratio > 1:
            self.ps_down = PixelUnshuffle1D(ps_ratio)
            self.ps_up = PixelShuffle1D(ps_ratio)

    def forward(self, audio, sigma, unet_cond=None):
        if self.ps_ratio > 1:
            audio = self.ps_down(audio)

        inputs = [audio]

        if unet_cond is not None:
            unet_cond = F.interpolate(unet_cond, (audio.shape[2], ), mode='linear', align_corners=False)
            inputs.append(unet_cond)

        inputs = torch.cat(inputs, dim=1)

        x = self.conv_1(inputs)
        downsampled = []
        sigma_encoding = self.embedding(sigma.unsqueeze(1))

        for film, layer in zip(self.gamma_beta, self.downsample):
            gamma, beta = film(sigma_encoding)
            x = layer(x, gamma, beta)
            downsampled.append(x)

        for layer, x_dblock in zip(self.upsample, reversed(downsampled)):
            x = layer(x, x_dblock)

        x = self.last_conv(x)

        if self.ps_ratio > 1:
            x = self.ps_up(x)

        return x