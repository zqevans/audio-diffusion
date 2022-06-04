## Modified from https://github.com/wesbz/SoundStream/blob/main/net.py
from xml.etree.ElementPath import prepare_predicate, prepare_star
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cached_conv as cc
#import torch.nn.utils.weight_norm as wn
import torch.nn.utils.weight_norm as weight_norm

import librosa as li
import torch.fft as fft
from einops import rearrange

# TODO:  Remove the RAVE code I'm not using anymore -SH

MAX_BATCH_SIZE = 64


def WNConv1d(*args, **kwargs):
    return weight_norm(cc.Conv1d(*args, **kwargs))

def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(cc.ConvTranspose1d(*args, **kwargs))

class CachedPadding1d(nn.Module):
    """
    Cached Padding implementation, replace zero padding with the end of
    the previous tensor.
    """
    def __init__(self, padding, crop=False):
        super().__init__()
        self.initialized = 0
        self.padding = padding
        self.crop = crop

    @torch.jit.unused
    @torch.no_grad()
    def init_cache(self, x):
        b, c, _ = x.shape
        self.register_buffer(
            "pad",
            torch.zeros(MAX_BATCH_SIZE, c, self.padding).to(x))
        self.initialized += 1

    def forward(self, x):
        if not self.initialized:
            self.init_cache(x)

        if self.padding:
            x = torch.cat([self.pad[:x.shape[0]], x.clone()], -1)
            self.pad[:x.shape[0]].copy_(x.clone()[..., -self.padding:])

            if self.crop:
                x = x.clone()[..., :-self.padding]

        return x


class AlignBranches(nn.Module):
    def __init__(self, *branches, delays=None, cumulative_delay=0, stride=1):
        super().__init__()
        self.branches = nn.ModuleList(branches)

        if delays is None:
            delays = list(map(lambda x: x.cumulative_delay, self.branches))

        max_delay = max(delays)

        self.paddings = nn.ModuleList([
            CachedPadding1d(p, crop=True)
            for p in map(lambda f: max_delay - f, delays)
        ])

        self.cumulative_delay = int(cumulative_delay * stride) + max_delay

    def forward(self, x):
        outs = []
        print("q  x.size() = ",x.size())
        for branch, pad in zip(self.branches, self.paddings):
            print("branch, pad = ",branch, pad)
            delayed_x = pad(x)
            bd = branch(delayed_x)
            print("delayed_x.size(), bd.size() = ",delayed_x.size(), bd.size())
            outs.append(bd)
        return outs



def mod_sigmoid(x):
    return 2 * torch.sigmoid(x)**2.3 + 1e-7


def amp_to_impulse_response(amp, target_size):
    """
    transforms frequecny amps to ir on the last dimension
    """
    amp = torch.stack([amp, torch.zeros_like(amp)], -1)
    amp = torch.view_as_complex(amp)
    amp = fft.irfft(amp.clone())

    filter_size = amp.shape[-1]

    amp = torch.roll(amp, filter_size // 2, -1)
    win = torch.hann_window(filter_size, dtype=amp.dtype, device=amp.device)

    amp = amp * win

    amp = nn.functional.pad(
        amp,
        (0, int(target_size) - int(filter_size)),
    )
    amp = torch.roll(amp, -filter_size // 2, -1)

    return amp



def fft_convolve(signal, kernel):
    """
    convolves signal by kernel on the last dimension
    """
    signal = nn.functional.pad(signal, (0, signal.shape[-1]))
    kernel = nn.functional.pad(kernel, (kernel.shape[-1], 0))

    output = fft.irfft(fft.rfft(signal.clone()) * fft.rfft(kernel.clone()))
    output = output[..., output.shape[-1] // 2:]

    return output


def multiscale_stft(signal, scales, overlap):
    """
    Compute a stft on several scales, with a constant overlap value.
    Parameters
    ----------
    signal: torch.Tensor
        input signal to process ( B X C X T )
    
    scales: list
        scales to use
    overlap: float
        overlap between windows ( 0 - 1 )
    """
    signal = rearrange(signal, "b c t -> (b c) t")
    stfts = []
    for s in scales:
        S = torch.stft(
            signal,
            s,
            int(s * (1 - overlap)),
            s,
            torch.hann_window(s).to(signal),
            True,
            normalized=True,
            return_complex=True,
        ).abs()
        stfts.append(S)
    return stfts


class Loudness(nn.Module):

    def __init__(self, sr, block_size, n_fft=2048):
        super().__init__()
        self.sr = sr
        self.block_size = block_size
        self.n_fft = n_fft

        f = np.linspace(0, sr / 2, n_fft // 2 + 1) + 1e-7
        a_weight = li.A_weighting(f).reshape(-1, 1)

        self.register_buffer("a_weight", torch.from_numpy(a_weight).float())
        self.register_buffer("window", torch.hann_window(self.n_fft))

    def forward(self, x):
        x = x.clone()[:,0,:] # mono loundness ? 
        print("x.squeeze(1).size() = ", x.squeeze(1).size())
        x = torch.stft(
            x.squeeze(1),
            self.n_fft,
            self.block_size,
            self.n_fft,
            center=True,
            window=self.window,
            return_complex=True,
        ).abs()
        x = torch.log(x + 1e-7) + self.a_weight
        return torch.mean(x, 1, keepdim=True)



class RAVEResidual(nn.Module):
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


class RAVEResidualStack(nn.Module):
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
                WNConv1d(
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
                ))

            seq.append(nn.LeakyReLU(.2))
            seq.append(
                WNConv1d(
                    dim,
                    dim,
                    kernel_size,
                    padding=cc.get_padding(kernel_size, mode=padding_mode),
                    bias=bias,
                    cumulative_delay=seq[-2].cumulative_delay,
                ))

            res_net = cc.CachedSequential(*seq)

            net.append(RAVEResidual(res_net, cumulative_delay=res_cum_delay))
            res_cum_delay = net[-1].cumulative_delay

        self.net = cc.CachedSequential(*net)
        self.cumulative_delay = self.net.cumulative_delay + cumulative_delay

    def forward(self, x):
        return self.net(x)

# RAVE bits grabbed from IRCAM-RAVE, https://github.com/acids-ircam/RAVE/blob/master/rave/model.py
class RAVEUpsampleLayer(nn.Module):
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
                WNConvTranspose1d(
                    in_dim,
                    out_dim,
                    2 * ratio,
                    stride=ratio,
                    padding=ratio // 2,
                    bias=bias,
                ))
        else:
            net.append(
                WNConv1d(
                    in_dim,
                    out_dim,
                    3,
                    padding=cc.get_padding(3, mode=padding_mode),
                    bias=bias,
                ))

        self.net = cc.CachedSequential(*net)
        self.cumulative_delay = self.net.cumulative_delay + cumulative_delay * ratio

    def forward(self, x):
        return self.net(x)


class RAVENoiseGenerator(nn.Module):
    def __init__(self, in_size, data_size, ratios, noise_bands, padding_mode):
        super().__init__()
        net = []
        channels = [in_size] * len(ratios) + [data_size * noise_bands]
        cum_delay = 0
        for i, r in enumerate(ratios):
            net.append(
                cc.Conv1d(
                    channels[i],
                    channels[i + 1],
                    3,
                    padding=cc.get_padding(3, r, mode=padding_mode),
                    stride=r,
                    cumulative_delay=cum_delay,
                ))
            cum_delay = net[-1].cumulative_delay
            if i != len(ratios) - 1:
                net.append(nn.LeakyReLU(.2))

        self.net = cc.CachedSequential(*net)
        self.data_size = data_size
        self.cumulative_delay = self.net.cumulative_delay * int(
            np.prod(ratios))

        self.register_buffer(
            "target_size",
            torch.tensor(np.prod(ratios)).long(),
        )

    def forward(self, x):
        amp = mod_sigmoid(self.net(x) - 5)
        amp = amp.permute(0, 2, 1)
        amp = amp.reshape(amp.shape[0], amp.shape[1], self.data_size, -1)

        ir = amp_to_impulse_response(amp, self.target_size)
        noise = torch.rand_like(ir) * 2 - 1

        noise = fft_convolve(noise.clone(), ir).permute(0, 2, 1, 3)
        noise = noise.reshape(noise.shape[0], noise.shape[1], -1)
        return noise



class RAVEGenerator(nn.Module):
    def __init__(self,
                 latent_size,
                 capacity,
                 data_size,
                 ratios,
                 loud_stride,
                 use_noise,
                 noise_ratios,
                 noise_bands,
                 padding_mode,
                 bias=False):
        super().__init__()
        net = [
            WNConv1d(
                latent_size,
                2**len(ratios) * capacity,
                7,
                padding=cc.get_padding(7, mode=padding_mode),
                bias=bias,
            )
        ]

        for i, r in enumerate(ratios):
            in_dim = 2**(len(ratios) - i) * capacity
            out_dim = 2**(len(ratios) - i - 1) * capacity
            print("i, r, in_dim, out_dim = ",i, r, in_dim, out_dim)

            net.append(
                RAVEUpsampleLayer(
                    in_dim,
                    out_dim,
                    r,
                    padding_mode,
                    cumulative_delay=net[-1].cumulative_delay,
                ))
            net.append(
                RAVEResidualStack(
                    out_dim,
                    3,
                    padding_mode,
                    cumulative_delay=net[-1].cumulative_delay,
                ))

        self.net = cc.CachedSequential(*net)

        wave_gen = WNConv1d(
                out_dim,
                data_size,
                7,
                padding=cc.get_padding(7, mode=padding_mode),
                bias=bias,
            )

        loud_gen = WNConv1d(
                out_dim,
                1,
                2 * loud_stride + 1,
                stride=loud_stride,
                padding=cc.get_padding(2 * loud_stride + 1,
                                       loud_stride,
                                       mode=padding_mode),
                bias=bias,
            )

        branches = [wave_gen, loud_gen]

        if use_noise:
            noise_gen = RAVENoiseGenerator(
                out_dim,
                data_size,
                noise_ratios,
                noise_bands,
                padding_mode=padding_mode,
            )
            branches.append(noise_gen)

        self.synth = AlignBranches(
            *branches,
            cumulative_delay=self.net.cumulative_delay,
        )

        self.use_noise = use_noise
        self.loud_stride = loud_stride
        self.cumulative_delay = self.synth.cumulative_delay

    def forward(self, x, add_noise: bool = True):
        print("\n RAVEGEnerator: in x.size() = ",x.size())
        x = self.net(x)
        print(" new x.size() = ",x.size())

        """
        if self.use_noise:
            waveform, loudness, noise = self.synth(x)
        else:
            waveform, loudness = self.synth(x)
            noise = torch.zeros_like(waveform)
        """
        waveform, loudness = x.clone()[:,0:1,:], x.clone()[:,2,:]

        print("1 waveform.size() =  ",waveform.size())
        print("1 loudness.size() =  ",loudness.size())

        loudness = loudness.repeat_interleave(self.loud_stride)
        loudness = loudness.reshape(x.shape[0], 1, -1)

        waveform = torch.tanh(waveform.clone()) * mod_sigmoid(loudness)
        print("2 waveform.size() =  ",waveform.size())

        if add_noise:
            waveform = waveform + noise

        waveform = waveform.clone()[:,:,0:32768] #truncate

        return waveform 


def GenBlock(input_channels, output_channels, kernel_size=4, stride=2, padding=1, final_block=False):
    if not final_block:
        return nn.Sequential(
            #nn.ConvTranspose1d(input_channels, output_channels, kernel_size, stride=stride, padding=padding),
            nn.Upsample(scale_factor=stride),
            nn.Conv1d(input_channels, output_channels, kernel_size, stride=1, padding=padding),
            nn.BatchNorm1d(output_channels),
            nn.LeakyReLU(0.2)
        )
    else: # Final block
        return nn.Sequential(
            #nn.ConvTranspose1d(input_channels, output_channels, kernel_size, stride=stride, padding=padding),
            nn.Upsample(scale_factor=stride),
            nn.Conv1d(input_channels, output_channels, kernel_size, stride=1, padding=padding),
            #nn.Tanh() # save tanh for end of loop, to rescale it
        )

class Upscale_new(nn.Module):
    def __init__(self, inc, outc, ksize, scale=2, final_block=False, add_noise=False):
        super().__init__()
        self.gb = GenBlock(inc, outc, final_block=final_block, stride=scale)
        self.conv_same_size1 = nn.Conv1d(outc,outc,ksize,stride=1,padding=1)
        self.conv_same_size2 = nn.Conv1d(outc,outc,ksize,stride=1,padding=1)

        self.add_noise = add_noise
        self.act = nn.Tanh()
        #self.bn = nn.BatchNorm1d(outc)

    def forward(self, x):
        x = self.gb(x)
        if self.add_noise: # some way of letting the network better match Zach's crazy Splice dataset
            noise = torch.rand_like(x) * 2 - 1
            # somehow we want the noise to be switched on or off based on what input signal is, but we only have latents x
            morph = self.act(self.conv_same_size1(x * noise))   # let x serve as the switch to allow more or less noise
            #morph = self.bn(morph) # output looked like it had a positive bias, so let's bn that
            x = self.conv_same_size2(x + morph)  
        return x 

"""    
class Upscale_old(nn.Module):
    def __init__(self, inc, outc, ksize, scale=2):
        super().__init__()
        #self.upsize = nn.ConvTranspose1d(inc, outc, ksize, stride=scale)  # can cause checkerboaring
        self.upsize = nn.Upsample(scale_factor=scale)
        self.conv1 = nn.Conv1d(inc,outc,ksize,stride=1,padding=1)
        self.act = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv1d(outc, outc, ksize, stride=1, padding=1)

    def forward(self, x):
        x = self.upsize(x)
        x = self.conv1(x) # filters to lower the number of channels        
        x = self.act(x)
        x = self.conv2(x)
        out = x
        return out
"""

class SimpleDecoder(nn.Module):
    """
    Scott trying Just making a basic expanding thingy
    """
    def __init__(self, latent_dim, io_channels, out_length=32768, depth=16, add_noise=False):
        super().__init__()
        channels = [latent_dim,32,16,8,4,4,2]
        scales = [2,2,4,4,2,2]
        ksize = 3
        assert len(scales) == (len(channels)-1)
        self.out_length = out_length

        self.up_layers = nn.ModuleList(
            [Upscale_new(channels[i],channels[i+1], ksize, scale=scales[i], 
                final_block=(i==len(scales)-1), add_noise=add_noise) for i in range(len(scales))]
        )
        #self.final_conv = nn.Conv1d(channels[-1],  channels[-1], ksize, stride=1, padding=1)
        self.final_act = nn.Tanh()  # so output waveform is on [-1,1]

    def forward(self, x):
        # initially, x = z = latents. then we upscale it
        for i in range(len(self.up_layers)):
            #print(f"{i} 1 x.size() = ",x.size())
            x = self.up_layers[i](x)
        x = 1.1*self.final_act(x)
        return x[:,:,0:self.out_length] # crop to desired length, throw away the rest

