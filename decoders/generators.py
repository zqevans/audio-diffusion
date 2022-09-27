#Copied and modified from https://github.com/acids-ircam/RAVE/blob/master/rave/model.py

import torch
import torch.nn as nn
import torch.nn.utils.weight_norm as wn

import cached_conv as cc
from functools import partial
from torch.nn import functional as F
from blocks.blocks import SkipBlock, SelfAttention1d, ResConvBlock, Downsample1d, Upsample1d, Downsample1d_2, Upsample1d_2

def mod_sigmoid(x):
    return 2 * torch.sigmoid(x)**2.3 + 1e-7

class Residual(nn.Module):

    def __init__(self, module, cumulative_delay=0):
        super().__init__()
        additional_delay = module.cumulative_delay
        self.aligned = cc.AlignBranches(
            module,
            nn.Identity(),
            delays=[additional_delay, 0],
        )
        self.cumulative_delay = additional_delay + cumulative_delay

    def forward(self, x):
        x_net, x_res = self.aligned(x)
        return x_net + x_res

class UpsampleLayer(nn.Module):

    def __init__(self,
                 in_dim,
                 out_dim,
                 ratio,
                 padding_mode,
                 cumulative_delay=0,
                 bias=False):
        super().__init__()
        net = [nn.LeakyReLU(.2)]
        if ratio > 1:
            net.append(
                wn(
                    cc.ConvTranspose1d(
                        in_dim,
                        out_dim,
                        2 * ratio,
                        stride=ratio,
                        padding=ratio // 2,
                        bias=bias,
                    )))
        else:
            net.append(
                wn(
                    cc.Conv1d(
                        in_dim,
                        out_dim,
                        3,
                        padding=cc.get_padding(3, mode=padding_mode),
                        bias=bias,
                    )))

        self.net = cc.CachedSequential(*net)
        self.cumulative_delay = self.net.cumulative_delay + cumulative_delay * ratio

    def forward(self, x):
        return self.net(x)


class RaveEncoder(nn.Module):

    def __init__(self,
                 data_size,
                 capacity,
                 latent_size,
                 ratios,
                 padding_mode="centered",
                 bias=True):
        super().__init__()
        net = [
            cc.Conv1d(data_size,
                      capacity,
                      7,
                      padding=cc.get_padding(7, mode=padding_mode),
                      bias=bias)
        ]

        for i, r in enumerate(ratios):
            in_dim = 2**i * capacity
            out_dim = 2**(i + 1) * capacity

            net.append(nn.BatchNorm1d(in_dim))
            net.append(nn.LeakyReLU(.2))
            net.append(
                cc.Conv1d(
                    in_dim,
                    out_dim,
                    2 * r + 1,
                    padding=cc.get_padding(2 * r + 1, r, mode=padding_mode),
                    stride=r,
                    bias=bias,
                    cumulative_delay=net[-3].cumulative_delay,
                ))

        net.append(nn.LeakyReLU(.2))
        net.append(
            cc.Conv1d(
                out_dim,
                latent_size,
                5,
                padding=cc.get_padding(5, mode=padding_mode),
                groups=2,
                bias=bias,
                cumulative_delay=net[-2].cumulative_delay,
            ))

        self.net = cc.CachedSequential(*net)
        self.cumulative_delay = self.net.cumulative_delay

    def forward(self, x):
        return self.net(x)
        

class ResidualStack(nn.Module):

    def __init__(self,
                 dim,
                 kernel_size,
                 padding_mode,
                 cumulative_delay=0,
                 bias=False):
        super().__init__()
        net = []

        res_cum_delay = 0
        # SEQUENTIAL RESIDUALS
        for i in range(3):
            # RESIDUAL BLOCK
            seq = [nn.LeakyReLU(.2)]
            seq.append(
                wn(
                    cc.Conv1d(
                        dim,
                        dim,
                        kernel_size,
                        padding=cc.get_padding(
                            kernel_size,
                            dilation=3**i,
                            mode=padding_mode,
                        ),
                        dilation=3**i,
                        bias=bias,
                    )))

            seq.append(nn.LeakyReLU(.2))
            seq.append(
                wn(
                    cc.Conv1d(
                        dim,
                        dim,
                        kernel_size,
                        padding=cc.get_padding(kernel_size, mode=padding_mode),
                        bias=bias,
                        cumulative_delay=seq[-2].cumulative_delay,
                    )))

            res_net = cc.CachedSequential(*seq)

            net.append(Residual(res_net, cumulative_delay=res_cum_delay))
            res_cum_delay = net[-1].cumulative_delay

        self.net = cc.CachedSequential(*net)
        self.cumulative_delay = self.net.cumulative_delay + cumulative_delay

    def forward(self, x):
        return self.net(x)

class RaveGenerator(nn.Module):

    def __init__(self,
                 latent_size,
                 capacity=64,
                 data_size=16,
                 ratios=[4, 4, 4, 2],
                 loud_stride=1,
                 padding_mode="centered",
                 bias=True):
        super().__init__()
        net = [
            wn(
                cc.Conv1d(
                    latent_size,
                    2**len(ratios) * capacity,
                    7,
                    padding=cc.get_padding(7, mode=padding_mode),
                    bias=bias,
                ))
        ]

        for i, r in enumerate(ratios):
            in_dim = 2**(len(ratios) - i) * capacity
            out_dim = 2**(len(ratios) - i - 1) * capacity

            net.append(
                UpsampleLayer(
                    in_dim,
                    out_dim,
                    r,
                    padding_mode,
                    cumulative_delay=net[-1].cumulative_delay,
                ))
            net.append(
                ResidualStack(
                    out_dim,
                    3,
                    padding_mode,
                    cumulative_delay=net[-1].cumulative_delay,
                ))

        self.net = cc.CachedSequential(*net)

        wave_gen = wn(
            cc.Conv1d(
                out_dim,
                data_size,
                7,
                padding=cc.get_padding(7, mode=padding_mode),
                bias=bias,
            ))

        loud_gen = wn(
            cc.Conv1d(
                out_dim,
                1,
                2 * loud_stride + 1,
                stride=loud_stride,
                padding=cc.get_padding(2 * loud_stride + 1,
                                       loud_stride,
                                       mode=padding_mode),
                bias=bias,
            ))

        branches = [wave_gen, loud_gen]

        self.synth = cc.AlignBranches(
            *branches,
            cumulative_delay=self.net.cumulative_delay,
        )
        
        self.loud_stride = loud_stride
        self.cumulative_delay = self.synth.cumulative_delay

        with torch.no_grad():
            for param in self.net.parameters():
                param *= 0.5

    def forward(self, x):
        x = self.net(x)
        
        waveform, loudness = self.synth(x)

        loudness = loudness.repeat_interleave(self.loud_stride)
        loudness = loudness.reshape(x.shape[0], 1, -1)

        waveform = torch.tanh(waveform) * mod_sigmoid(loudness)

        
        return waveform


class AudioResnet(nn.Module):

    def __init__(self,
                 in_channels=16,
                 out_channels=16,
                 width=64,
                 layers = 16,
                 loud_stride=1,
                 padding_mode="centered",
                 bias=True):
        super().__init__()
        net = [
            wn(
                cc.Conv1d(
                    in_channels,
                    width,
                    7,
                    padding=cc.get_padding(7, mode=padding_mode),
                    bias=bias,
                ))
        ]

        for i in range(layers):

            net.append(
                ResidualStack(
                    width,
                    3,
                    padding_mode,
                    cumulative_delay=net[-1].cumulative_delay,
                ))

        self.net = cc.CachedSequential(*net)

        wave_gen = wn(
            cc.Conv1d(
                width,
                out_channels,
                7,
                padding=cc.get_padding(7, mode=padding_mode),
                bias=bias,
            ))

        loud_gen = wn(
            cc.Conv1d(
                width,
                1,
                2 * loud_stride + 1,
                stride=loud_stride,
                padding=cc.get_padding(2 * loud_stride + 1,
                                       loud_stride,
                                       mode=padding_mode),
                bias=bias,
            ))

        branches = [wave_gen, loud_gen]

        self.synth = cc.AlignBranches(
            *branches,
            cumulative_delay=self.net.cumulative_delay,
        )
        
        self.loud_stride = loud_stride
        self.cumulative_delay = self.synth.cumulative_delay

        with torch.no_grad():
            for param in self.net.parameters():
                param *= 0.5

    def forward(self, x):
        x = self.net(x)
        
        waveform, loudness = self.synth(x)

        loudness = loudness.repeat_interleave(self.loud_stride)
        loudness = loudness.reshape(x.shape[0], 1, -1)

        waveform = torch.tanh(waveform) * mod_sigmoid(loudness)

        
        return waveform


class AttnUnet1D(nn.Module):
    def __init__(
        self, 
        io_channels = 2, 
        depth=14,
        n_attn_layers = 6,
        c_mults = [128, 128, 256, 256] + [512] * 10,
        cond_dim = 0,
        kernel_size = 5,
        learned_resample = False,
        strides = [2] * 14
    ):
        super().__init__()

        attn_layer = depth - n_attn_layers - 1

        block = nn.Identity()

        conv_block = partial(ResConvBlock, kernel_size=kernel_size)

        for i in range(depth, 0, -1):
            c = c_mults[i - 1]
            stride = strides[i-1]
            if i > 1:
                c_prev = c_mults[i - 2]
                add_attn = i >= attn_layer and n_attn_layers > 0
                block = SkipBlock(
                    Downsample1d_2(c_prev, c_prev, stride) if learned_resample else Downsample1d("cubic"),
                    conv_block(c_prev, c, c),
                    SelfAttention1d(
                        c, c // 32) if add_attn else nn.Identity(),
                    conv_block(c, c, c),
                    SelfAttention1d(
                        c, c // 32) if add_attn else nn.Identity(),
                    conv_block(c, c, c),
                    SelfAttention1d(
                        c, c // 32) if add_attn else nn.Identity(),
                    block,
                    conv_block(c * 2 if i != depth else c, c, c),
                    SelfAttention1d(
                        c, c // 32) if add_attn else nn.Identity(),
                    conv_block(c, c, c),
                    SelfAttention1d(
                        c, c // 32) if add_attn else nn.Identity(),
                    conv_block(c, c, c_prev),
                    SelfAttention1d(c_prev, c_prev //
                                    32) if add_attn else nn.Identity(),
                    Upsample1d_2(c_prev, c_prev, stride) if learned_resample else Upsample1d(kernel="cubic")
                )
            else:
                block = nn.Sequential(
                    conv_block(io_channels + cond_dim, c, c),
                    conv_block(c, c, c),
                    conv_block(c, c, c),
                    block,
                    conv_block(c * 2, c, c),
                    conv_block(c, c, c),
                    conv_block(c, c, io_channels, is_last=True),
                )
        self.net = block

        with torch.no_grad():
            for param in self.net.parameters():
                param *= 0.5

    def forward(self, x, cond=None):
        
        inputs = [x]

        if cond is not None:
            cond = F.interpolate(cond, (x.shape[2], ), mode='linear', align_corners=False)
            inputs.append(cond)

        outputs = self.net(torch.cat(inputs, dim=1))

        return outputs
