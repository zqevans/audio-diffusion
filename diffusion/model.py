from copy import deepcopy
import math
import pytorch_lightning as pl
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from functools import partial

from .utils import get_alphas_sigmas

from .pqmf import CachedPQMF as PQMF

from blocks.blocks import FourierFeatures
from autoencoders.models import AudioAutoencoder
from decoders.diffusion_decoder import DiffusionAttnUnet1D

@torch.no_grad()
def ema_update(model, averaged_model, decay):
    """Incorporates updated model parameters into an exponential moving averaged
    version of a model. It should be called after each optimizer step."""
    model_params = dict(model.named_parameters())
    averaged_params = dict(averaged_model.named_parameters())
    assert model_params.keys() == averaged_params.keys()

    for name, param in model_params.items():
        averaged_params[name].mul_(decay).add_(param, alpha=1 - decay)

    model_buffers = dict(model.named_buffers())
    averaged_buffers = dict(averaged_model.named_buffers())
    assert model_buffers.keys() == averaged_buffers.keys()

    for name, buf in model_buffers.items():
        averaged_buffers[name].copy_(buf)


class LatentAudioDiffusion(pl.LightningModule):
    def __init__(
        self, 
        autoencoder: AudioAutoencoder,
        io_channels = 32,
        n_attn_layers=4,
        channels = [512] * 6 + [1024] * 4,
        depth = 10
    ):
        super().__init__()

        self.latent_dim = autoencoder.latent_dim
        self.downsampling_ratio = autoencoder.downsampling_ratio

        self.diffusion = DiffusionAttnUnet1D(
            io_channels=self.latent_dim, 
            n_attn_layers=n_attn_layers, 
            c_mults=channels,
            depth=depth
        )

        self.autoencoder = autoencoder

    def encode(self, reals):
        return self.autoencoder.encode(reals)

    def decode(self, latents):
        return self.autoencoder.decode(latents)