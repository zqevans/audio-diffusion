## Modified from https://github.com/wesbz/SoundStream/blob/main/net.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

from diffusion.model import ResidualBlock

from vector_quantize_pytorch import ResidualVQ

# Generator
class CausalConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.causal_padding = self.dilation[0] * (self.kernel_size[0] - 1)

    def forward(self, x):
        return self._conv_forward(F.pad(x, [self.causal_padding, 0]), self.weight, self.bias)


class CausalConvTranspose1d(nn.ConvTranspose1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.causal_padding = self.dilation[0] * (self.kernel_size[0] - 1) + self.output_padding[0] + 1 - self.stride[0]
    
    def forward(self, x, output_size=None):
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose1d')

        assert isinstance(self.padding, tuple)
        output_padding = self._output_padding(
            x, output_size, self.stride, self.padding, self.kernel_size, self.dilation)
        return F.conv_transpose1d(
            x, self.weight, self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)[...,:-self.causal_padding]


class ResidualUnit(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        
        self.dilation = dilation

        self.layers = nn.Sequential(
            CausalConv1d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=7, dilation=dilation),
            nn.ELU(),
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=1)
        )

    def forward(self, x):
        return x + self.layers(x)


class EncoderBlock(nn.Module):
    def __init__(self, out_channels, stride):
        super().__init__()

        self.layers = nn.Sequential(
            ResidualUnit(in_channels=out_channels//2,
                         out_channels=out_channels//2, dilation=1),
            nn.ELU(),
            ResidualUnit(in_channels=out_channels//2,
                         out_channels=out_channels//2, dilation=3),
            nn.ELU(),
            ResidualUnit(in_channels=out_channels//2,
                         out_channels=out_channels//2, dilation=9),
            nn.ELU(),
            CausalConv1d(in_channels=out_channels//2, out_channels=out_channels,
                      kernel_size=2*stride, stride=stride)
        )

    def forward(self, x):
        return self.layers(x)


class DecoderBlock(nn.Module):
    def __init__(self, out_channels, stride):
        super().__init__()

        self.layers = nn.Sequential(
            CausalConvTranspose1d(in_channels=2*out_channels,
                               out_channels=out_channels,
                               kernel_size=2*stride, stride=stride),
            nn.ELU(),
            ResidualUnit(in_channels=out_channels, out_channels=out_channels,
                         dilation=1),
            nn.ELU(),
            ResidualUnit(in_channels=out_channels, out_channels=out_channels,
                         dilation=3),
            nn.ELU(),
            ResidualUnit(in_channels=out_channels, out_channels=out_channels,
                         dilation=9),

        )

    def forward(self, x):
        return self.layers(x)

class ResConvBlock(ResidualBlock):
    def __init__(self, c_in, c_mid, c_out, is_last=False):
        skip = None if c_in == c_out else nn.Conv1d(c_in, c_out, 1, bias=False)
        super().__init__([
            nn.Conv1d(c_in, c_mid, 5, padding=2),
            nn.GroupNorm(1, c_mid),
            nn.ReLU(inplace=True),
            nn.Conv1d(c_mid, c_out, 5, padding=2),
            nn.GroupNorm(1, c_out) if not is_last else nn.Identity(),
            nn.ReLU(inplace=True) if not is_last else nn.Identity(),
        ], skip)

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


class SoundStreamXLEncoder(nn.Module):
    def __init__(self, n_channels, latent_dim, n_io_channels=1):
        super().__init__()
        
        c_mults = [4, 4, 8, 8] + [16] * 8
  
        self.depth = len(c_mults)

        layers = [
            CausalConv1d(in_channels=n_io_channels, out_channels=n_channels, kernel_size=7),
            nn.ELU()
        ]
        
        for i in range(self.depth):
            layers.append(EncoderBlock(out_channels=c_mults[i]*n_channels, stride=2))
            layers.append(nn.ELU())

        layers.append(CausalConv1d(in_channels=16*n_channels, out_channels=latent_dim, kernel_size=3))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class SoundStreamXLDecoder(nn.Module):
    def __init__(self, n_channels, latent_dim, num_io_channels=1):
        super().__init__()

        c_mults = [4, 4, 8, 8] + [16] * 8
    
        self.depth = len(c_mults)

        layers = [
            CausalConv1d(in_channels=latent_dim, out_channels=16*n_channels, kernel_size=7),
            nn.ELU()
        ]
        
        for i in range(self.depth, 0, -1):
            layers.append(DecoderBlock(out_channels=c_mults[i-1]*n_channels, stride=2))
            layers.append(nn.ELU())

        layers.append( CausalConv1d(in_channels=c_mults[0] * n_channels, out_channels=num_io_channels, kernel_size=7))

        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)


class ResidualVQVAE(nn.Module):
    def __init__(self, encoder, decoder, latent_dim, n_quantizers=8, codebook_size=1024, sync_codebook=False):
        super().__init__()

        self.encoder = encoder

        self.quantizer = ResidualVQ(
            num_quantizers=n_quantizers, 
            dim=latent_dim, 
            codebook_size=codebook_size,
            kmeans_init=True, 
            kmeans_iters=100, 
            threshold_ema_dead_code=2, 
            channel_last=False,
            use_cosine_sim=True,
            orthogonal_reg_weight=10,
            shared_codebook=True,
            sync_codebook=sync_codebook
        )

        self.decoder = decoder
    
    def forward(self, x):
        encoded = self.encoder(x)
        quantized, indices, losses = self.quantizer(encoded)
        decoded = self.decoder(quantized)
        return decoded, indices, losses

class SoundStreamXL(ResidualVQVAE):
    def __init__(self, n_io_channels, n_feature_channels, latent_dim, *args, **kwargs):
        self.encoder = SoundStreamXLEncoder(n_io_channels=n_io_channels, n_channels=n_feature_channels, latent_dim=latent_dim)  
        self.decoder = SoundStreamXLDecoder(n_io_channels=n_io_channels, n_channels=n_feature_channels, latent_dim=latent_dim)

        super().__init__(self.encoder, self.decoder, latent_dim, *args, **kwargs)

# Wave-based Discriminator
def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


class WaveDiscriminatorBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.ReflectionPad1d(7),
                WNConv1d(in_channels=1, out_channels=16, kernel_size=15),
                nn.LeakyReLU(negative_slope=0.2)
            ),
            nn.Sequential(
                WNConv1d(in_channels=16, out_channels=64, kernel_size=41,
                         stride=4, padding=20, groups=4),
                nn.LeakyReLU(negative_slope=0.2)
            ),
            nn.Sequential(
                WNConv1d(in_channels=64, out_channels=256, kernel_size=41,
                         stride=4, padding=20, groups=16),
                nn.LeakyReLU(negative_slope=0.2)
            ),
            nn.Sequential(
                WNConv1d(in_channels=256, out_channels=1024, kernel_size=41,
                         stride=4, padding=20, groups=64),
                nn.LeakyReLU(negative_slope=0.2)
            ),
            nn.Sequential(
                WNConv1d(in_channels=1024, out_channels=1024, kernel_size=41,
                         stride=4, padding=20, groups=256),
                nn.LeakyReLU(negative_slope=0.2)
            ),
            nn.Sequential(
                WNConv1d(in_channels=1024, out_channels=1024, kernel_size=5,
                         stride=1, padding=2),
                nn.LeakyReLU(negative_slope=0.2)
            ),
            WNConv1d(in_channels=1024, out_channels=1, kernel_size=3, stride=1,
                     padding=1)
        ])
    
    def features_lengths(self, lengths):
        return [
            lengths,
            torch.div(lengths+3, 4, rounding_mode="floor"),
            torch.div(lengths+15, 16, rounding_mode="floor"),
            torch.div(lengths+63, 64, rounding_mode="floor"),
            torch.div(lengths+255, 256, rounding_mode="floor"),
            torch.div(lengths+255, 256, rounding_mode="floor"),
            torch.div(lengths+255, 256, rounding_mode="floor")
        ]

    def forward(self, x):
        feature_map = []
        for layer in self.layers:
            x = layer(x)
            feature_map.append(x)
        return feature_map


class WaveDiscriminator(nn.Module):
    def __init__(self, num_D, downsampling_factor):
        super().__init__()
        
        self.num_D = num_D
        self.downsampling_factor = downsampling_factor
        
        self.model = nn.ModuleDict({
            f"disc_{downsampling_factor**i}": WaveDiscriminatorBlock()
            for i in range(num_D)
        })
        self.downsampler = nn.AvgPool1d(kernel_size=4, stride=2, padding=1,
                                        count_include_pad=False)
    
    def features_lengths(self, lengths):
        return {
            f"disc_{self.downsampling_factor**i}": self.model[f"disc_{self.downsampling_factor**i}"].features_lengths(torch.div(lengths, 2**i, rounding_mode="floor")) for i in range(self.num_D)
        }
    
    def forward(self, x):
        results = {}
        for i in range(self.num_D):
            disc = self.model[f"disc_{self.downsampling_factor**i}"]
            results[f"disc_{self.downsampling_factor**i}"] = disc(x)
            x = self.downsampler(x)
        return results


# STFT-based Discriminator

class ResidualUnit2d(nn.Module):
    def __init__(self, in_channels, N, m, s_t, s_f):
        super().__init__()
        
        self.s_t = s_t
        self.s_f = s_f

        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=N,
                kernel_size=(3, 3),
                padding="same"
            ),
            nn.ELU(),
            nn.Conv2d(
                in_channels=N,
                out_channels=m*N,
                kernel_size=(s_f+2, s_t+2),
                stride=(s_f, s_t)
            )
        )
        
        self.skip_connection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=m*N,
            kernel_size=(1, 1), stride=(s_f, s_t)
        )

    def forward(self, x):
        return self.layers(F.pad(x, [self.s_t+1, 0, self.s_f+1, 0])) + self.skip_connection(x)


class STFTDiscriminator(nn.Module):
    def __init__(self, C, F_bins):
        super().__init__()

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=2, out_channels=32, kernel_size=(7, 7)),
                nn.ELU()
            ),
            nn.Sequential(
                ResidualUnit2d(in_channels=32,  N=C,   m=2, s_t=1, s_f=2),
                nn.ELU()
            ),
            nn.Sequential(
                ResidualUnit2d(in_channels=2*C, N=2*C, m=2, s_t=2, s_f=2),
                nn.ELU()
            ),
            nn.Sequential(
                ResidualUnit2d(in_channels=4*C, N=4*C, m=1, s_t=1, s_f=2),
                nn.ELU()
            ),
            nn.Sequential(
                ResidualUnit2d(in_channels=4*C, N=4*C, m=2, s_t=2, s_f=2),
                nn.ELU()
            ),
            nn.Sequential(
                ResidualUnit2d(in_channels=8*C, N=8*C, m=1, s_t=1, s_f=2),
                nn.ELU()
            ),
            nn.Sequential(
                ResidualUnit2d(in_channels=8*C,  N=8*C, m=2, s_t=2, s_f=2),
                nn.ELU()
            ),
            nn.Conv2d(in_channels=16*C, out_channels=1,
                      kernel_size=(F_bins//2**6, 1))
        ])
    
    def features_lengths(self, lengths):
        return [
            lengths-6,
            lengths-6,
            torch.div(lengths-5, 2, rounding_mode="floor"),
            torch.div(lengths-5, 2, rounding_mode="floor"),
            torch.div(lengths-3, 4, rounding_mode="floor"),
            torch.div(lengths-3, 4, rounding_mode="floor"),
            torch.div(lengths+1, 8, rounding_mode="floor"),
            torch.div(lengths+1, 8, rounding_mode="floor")
        ]

    def forward(self, x):
        feature_map = []
        for layer in self.layers:
            x = layer(x)
            feature_map.append(x)
        return feature_map
