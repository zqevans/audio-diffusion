from copy import deepcopy
import math
import pytorch_lightning as pl
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import numpy as np

from .utils import get_alphas_sigmas

from .pqmf import CachedPQMF as PQMF

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

class ResidualBlock(nn.Module):
    def __init__(self, main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, input):
        return self.main(input) + self.skip(input)

class ResConvBlock(ResidualBlock):
    def __init__(self, c_in, c_mid, c_out, is_last=False):
        skip = None if c_in == c_out else nn.Conv1d(c_in, c_out, 1, bias=False)
        super().__init__([
            nn.Conv1d(c_in, c_mid, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(c_mid, c_out, 5, padding=2),
            nn.ReLU(inplace=True) if not is_last else nn.Identity(),
        ], skip)

class SelfAttention1d(nn.Module):
    def __init__(self, c_in, n_head=1):
        super().__init__()
        assert c_in % n_head == 0
        self.norm = nn.GroupNorm(1, c_in)
        self.n_head = n_head
        self.qkv_proj = nn.Conv1d(c_in, c_in * 3, 1)
        self.out_proj = nn.Conv1d(c_in, c_in, 1)

    def forward(self, input):
        n, c, s = input.shape
        qkv = self.qkv_proj(self.norm(input))
        qkv = qkv.view([n, self.n_head * 3, c // self.n_head, s]).transpose(2, 3)
        q, k, v = qkv.chunk(3, dim=1)
        scale = k.shape[3]**-0.25
        att = ((q * scale) @ (k.transpose(2, 3) * scale)).softmax(3)
        y = (att @ v).transpose(2, 3).contiguous().view([n, c, s])
        return input + self.out_proj(y)

class SkipBlock(nn.Module):
    def __init__(self, *main):
        super().__init__()
        self.main = nn.Sequential(*main)

    def forward(self, input):
        return torch.cat([self.main(input), input], dim=1)

class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.):
        super().__init__()
        assert out_features % 2 == 0
        self.weight = nn.Parameter(torch.randn([out_features // 2, in_features]) * std)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)

def expand_to_planes(input, shape):
    return input[..., None].repeat([1, 1, shape[2]])

class AudioDiffusion(nn.Module):
    def __init__(self, global_args):
        super().__init__()

        c_mults = [128, 128, 256, 256] + [512] * 10
       
        depth = len(c_mults)

        #Number of input/output audio channels for the model
        n_io_channels = 2 * global_args.pqmf_bands #if global_args.mono else 2 * global_args.pqmf_bands

        self.timestep_embed = FourierFeatures(1, 16)

        attn_layer = depth - 4

        block = nn.Identity()
        for i in range(depth, 0, -1):
            c = c_mults[i - 1]
            if i > 1:
                c_prev = c_mults[i - 2]
                block = SkipBlock(
                    nn.AvgPool1d(2),
                    ResConvBlock(c_prev, c, c),
                    SelfAttention1d(c, c // 32) if i >= attn_layer else nn.Identity(),
                    ResConvBlock(c, c, c),
                    SelfAttention1d(c, c // 32) if i >= attn_layer else nn.Identity(),
                    ResConvBlock(c, c, c),
                    SelfAttention1d(c, c // 32) if i >= attn_layer else nn.Identity(),
                    block,
                    ResConvBlock(c * 2 if i != depth else c, c, c),
                    SelfAttention1d(c, c // 32) if i >= attn_layer else nn.Identity(),
                    ResConvBlock(c, c, c),
                    SelfAttention1d(c, c // 32) if i >= attn_layer else nn.Identity(),
                    ResConvBlock(c, c, c_prev),
                    SelfAttention1d(c_prev, c_prev // 32) if i >= attn_layer else nn.Identity(),
                    nn.Upsample(scale_factor=2 , mode='linear', align_corners=False),
                )
            else:
                block = nn.Sequential(
                    ResConvBlock(n_io_channels + 16, c, c),
                    ResConvBlock(c, c, c),
                    ResConvBlock(c, c, c),
                    block,
                    ResConvBlock(c * 2, c, c),
                    ResConvBlock(c, c, c),
                    ResConvBlock(c, c, n_io_channels, is_last=True),
                )
        self.net = block

    def forward(self, input, t):
        timestep_embed = expand_to_planes(self.timestep_embed(t[:, None]), input.shape)
        return self.net(torch.cat([input, timestep_embed], dim=1))

class LightningDiffusion(pl.LightningModule):
    def __init__(self, global_args):
        super().__init__()
        self.model = AudioDiffusion(global_args)
        self.model_ema = deepcopy(self.model)
        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)
        self.pqmf = PQMF(2, 100, global_args.pqmf_bands)

    def forward(self, *args, **kwargs):
        if self.training:
            return self.model(*args, **kwargs)
        return self.model_ema(*args, **kwargs)

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=1e-4)

    def eval_batch(self, batch):
        reals = batch[0]
        
        reals = self.pqmf(reals)

        # Sample timesteps
        t = self.rng.draw(reals.shape[0])[:, 0].to(reals)

        # Calculate the noise schedule parameters for those timesteps
        alphas, sigmas = get_alphas_sigmas(t)
        # Combine the ground truth images and the noise
        alphas = alphas[:, None, None]
        sigmas = sigmas[:, None, None]
        noise = torch.randn_like(reals)
        noised_reals = reals * alphas + noise * sigmas
        targets = noise * alphas - reals * sigmas

        # Compute the model output and the loss.
        v = self(noised_reals, t)
        return F.mse_loss(v, targets)

    def training_step(self, batch, batch_idx):
        loss = self.eval_batch(batch)
        log_dict = {'train/loss': loss.detach()}
        self.log_dict(log_dict, prog_bar=True, on_step=True)
        return loss

    def on_before_zero_grad(self, *args, **kwargs):
        decay = 0.98 if self.trainer.global_step < 10000 else 0.999
        ema_update(self.model, self.model_ema, decay)
