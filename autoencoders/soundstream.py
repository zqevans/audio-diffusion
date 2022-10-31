import torch
import torch.nn as nn
import torch.nn.functional as F

from vector_quantize_pytorch import ResidualVQ

def mod_sigmoid(x):
    return 2 * torch.sigmoid(x)**2.3 + 1e-7

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
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=1)
        )

    def forward(self, x):
        return x + self.layers(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.layers = nn.Sequential(
            ResidualUnit(in_channels=in_channels,
                         out_channels=in_channels, dilation=1),
            nn.ELU(),
            ResidualUnit(in_channels=in_channels,
                         out_channels=in_channels, dilation=3),
            nn.ELU(),
            ResidualUnit(in_channels=in_channels,
                         out_channels=in_channels, dilation=9),
            nn.ELU(),
            ResidualUnit(in_channels=in_channels,
                         out_channels=in_channels, dilation=1),
            nn.ELU(),
            ResidualUnit(in_channels=in_channels,
                         out_channels=in_channels, dilation=3),
            nn.ELU(),
            ResidualUnit(in_channels=in_channels,
                         out_channels=in_channels, dilation=9),
            nn.ELU(),
            CausalConv1d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=2*stride, stride=stride)
        )

    def forward(self, x):
        return self.layers(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.layers = nn.Sequential(
            CausalConvTranspose1d(in_channels=in_channels,
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


class SoundStreamXLEncoder(nn.Module):
    def __init__(self, in_channels=2, capacity=32, latent_dim=128, c_mults = [2, 4, 4, 4, 8, 16], strides = [2, 2, 2, 4, 5, 8]):
        super().__init__()
          
        c_mults = [1] + c_mults

        self.depth = len(c_mults)

        layers = [
            CausalConv1d(in_channels=in_channels, out_channels=c_mults[0] * capacity, kernel_size=7),
            nn.ELU()
        ]
        
        for i in range(self.depth-1):
            layers.append(EncoderBlock(in_channels=c_mults[i]*capacity, out_channels=c_mults[i+1]*capacity, stride=strides[i]))
            layers.append(nn.ELU())

        layers.append(CausalConv1d(in_channels=c_mults[-1]*capacity, out_channels=latent_dim, kernel_size=3))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class SoundStreamXLDecoder(nn.Module):
    def __init__(self, out_channels=2, capacity=32, latent_dim=128, c_mults = [2, 4, 4, 4, 8, 16], strides = [2, 2, 2, 4, 5, 8]):
        super().__init__()

        c_mults = [1] + c_mults
        
        self.depth = len(c_mults)

        layers = [
            CausalConv1d(in_channels=latent_dim, out_channels=c_mults[-1]*capacity, kernel_size=7),
            nn.ELU()
        ]
        
        for i in range(self.depth-1, 0, -1):
            layers.append(DecoderBlock(in_channels=c_mults[i]*capacity, out_channels=c_mults[i-1]*capacity, stride=strides[i-1]))
            layers.append(nn.ELU())

        layers.append(CausalConv1d(in_channels=c_mults[0] * capacity, out_channels=out_channels, kernel_size=7))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


# class SoundStreamXL(nn.Module):
#     def __init__(self, n_io_channels, n_feature_channels, latent_dim, n_quantizers=8, codebook_size=1024):
#         super().__init__()

#         self.encoder = SoundStreamXLEncoder(in_channels=n_io_channels, capacity=n_feature_channels, latent_dim=latent_dim)  
#         self.decoder = SoundStreamXLDecoder(out_channels=n_io_channels, capacity=n_feature_channels, latent_dim=latent_dim)

#         self.quantizer = ResidualVQ(
#             num_quantizers=n_quantizers, 
#             dim=latent_dim, 
#             codebook_size=codebook_size,
#             kmeans_init=True, 
#             kmeans_iters=100, 
#             threshold_ema_dead_code=2, 
#             #use_cosine_sim=True,
#         )

#     def forward(self, x):
#         encoded = self.encoder(x)
#         quantized, indices, losses = self.quantizer(encoded)
#         decoded = self.decoder(quantized)
#         return decoded, indices, losses

        