import torch
from torch import nn 
import random 
import math

# Define the diffusion noise schedule
def get_alphas_sigmas(t):
    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)

class PadCrop(nn.Module):
    def __init__(self, n_samples, randomize=True):
        super().__init__()
        self.n_samples = n_samples
        self.randomize = randomize

    def __call__(self, signal):
        n, s = signal.shape
        start = 0 if (not self.randomize) else torch.randint(0, max(0, s - self.n_samples) + 1, []).item()
        end = start + self.n_samples
        output = signal.new_zeros([n, self.n_samples])
        output[:, :min(s, self.n_samples)] = signal[:, start:end]
        return output

class PhaseFlipper(nn.Module):
    "she was PHAAAAAAA-AAAASE FLIPPER, a random invert yeah"
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
    def __call__(self, signal):
        return -signal if (random.random() < self.p) else signal


class FillTheNoise(nn.Module):
    "randomly adds a bit of noise, just to spice things up"
    def __init__(self, p=0.33):
        super().__init__()
        self.p = p
    def __call__(self, signal):
        return signal + 0.25*random.random()*(2*torch.rand_like(signal)-1) if (random.random() < self.p) else signal


class OneMinus(nn.Module):
    "aka Destructo: subtracts the signal from +/- 1, just to spice things up"
    def __init__(self, p=0.2):
        super().__init__()
        self.p = p
    def __call__(self, signal):
        return 0.9*torch.sign(signal) - signal if (random.random() < self.p) else signal

class RandPool(nn.Module):
    def __init__(self, p=0.2):
        self.p, self.maxkern = p, 100
    def __call__(self, signal):
        if (random.random() < self.p):
            ksize = int(random.random()*self.maxkern)
            avger = nn.AvgPool1d(kernel_size=ksize, stride=1, padding=1)
            return avger(signal)
        else:
            return signal
        

class NormInputs(nn.Module):
    "useful for quiet inputs. intended to be part of augmentation chain; not activated by default"
    def __init__(self, do_norm=False):
        super().__init__()
        self.do_norm = do_norm
        self.eps = 1e-2
    def __call__(self, signal):
        return signal if (not self.do_norm) else signal/(torch.amax(signal,-1)[0] + self.eps)

class Mono(nn.Module):
  def __call__(self, signal):
    return torch.mean(signal, dim=0) if len(signal.shape) > 1 else signal

class Stereo(nn.Module):
  def __call__(self, signal):
    signal_shape = signal.shape
    # Check if it's mono
    if len(signal_shape) == 1: # s -> 2, s
        signal = signal.unsqueeze(0).repeat(2, 1)
    elif len(signal_shape) == 2:
        if signal_shape[0] == 1: #1, s -> 2, s
            signal = signal.repeat(2, 1)
        elif signal_shape[0] > 2: #?, s -> 2,s
            signal = signal[:2, :]    

    return signal

class RandomGain(nn.Module):
  def __init__(self, min_gain, max_gain):
      super().__init__()
      self.min_gain = min_gain
      self.max_gain = max_gain

  def __call__(self, signal):
    gain = random.uniform(self.min_gain, self.max_gain)
    signal = signal * gain

    return signal

class MidSideEncoding(nn.Module):
    def __call__(self, signal):
        #signal_shape should be 2, s
        left = signal[0]
        right = signal[1]
        mid = (left + right) / 2
        side = (left - right) / 2
        signal[0] = mid
        signal[1] = side

        return signal



# Taken from https://github.com/serkansulun/pytorch-pixelshuffle1d/blob/master/pixelshuffle1d.py

class PixelShuffle1D(torch.nn.Module):
    """
    1D pixel shuffler. https://arxiv.org/pdf/1609.05158.pdf
    Upscales sample length, downscales channel length
    "short" is input, "long" is output
    """
    def __init__(self, upscale_factor):
        super(PixelShuffle1D, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        batch_size = x.shape[0]
        short_channel_len = x.shape[1]
        short_width = x.shape[2]

        long_channel_len = short_channel_len // self.upscale_factor
        long_width = self.upscale_factor * short_width

        x = x.contiguous().view([batch_size, self.upscale_factor, long_channel_len, short_width])
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(batch_size, long_channel_len, long_width)

        return x

class PixelUnshuffle1D(torch.nn.Module):
    """
    Inverse of 1D pixel shuffler
    Upscales channel length, downscales sample length
    "long" is input, "short" is output
    """
    def __init__(self, downscale_factor):
        super(PixelUnshuffle1D, self).__init__()
        self.downscale_factor = downscale_factor

    def forward(self, x):
        batch_size = x.shape[0]
        long_channel_len = x.shape[1]
        long_width = x.shape[2]

        short_channel_len = long_channel_len * self.downscale_factor
        short_width = long_width // self.downscale_factor

        x = x.contiguous().view([batch_size, long_channel_len, short_width, self.downscale_factor])
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view([batch_size, short_channel_len, short_width])
        return x