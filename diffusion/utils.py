import torch
from torch import nn 
import random 
import math

# Define the diffusion noise schedule
def get_alphas_sigmas(t):
    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)

class PadCrop(nn.Module):
    def __init__(self, n_samples):
        super().__init__()
        self.n_samples = n_samples

    def __call__(self, signal):
        n, s = signal.shape
        start = 0 #torch.randint(0, max(0, s - self.n_samples) + 1, []).item()
        end = start + self.n_samples
        output = signal.new_zeros([n, self.n_samples])
        output[:, :min(s, self.n_samples)] = signal[:, start:end]
        return output

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