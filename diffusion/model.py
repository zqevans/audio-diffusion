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

class AudioDiffusion(nn.Module):
    def __init__(self, global_args):
        super().__init__()

        c_mults = [128, 128, 256, 256] + [512] * 10

        self.depth = len(c_mults)

        # Number of input/output audio channels for the model
        # if global_args.mono else 2 * global_args.pqmf_bands
        n_io_channels = 2 * global_args.pqmf_bands

        self.timestep_embed = FourierFeatures(1, 16)

        self.state = {}

        attn_layer = self.depth - 5

        block = nn.Identity()

        conv_block = partial(ResModConvBlock, self.state,
                             global_args.style_latent_size)

        for i in range(self.depth, 0, -1):
            c = c_mults[i - 1]
            if i > 1:
                c_prev = c_mults[i - 2]
                block = SkipBlock(
                    nn.AvgPool1d(2),
                    conv_block(c_prev, c, c),
                    SelfAttention1d(
                        c, c // 32) if i >= attn_layer else nn.Identity(),
                    conv_block(c, c, c),
                    SelfAttention1d(
                        c, c // 32) if i >= attn_layer else nn.Identity(),
                    conv_block(c, c, c),
                    SelfAttention1d(
                        c, c // 32) if i >= attn_layer else nn.Identity(),
                    block,
                    conv_block(c * 2 if i != self.depth else c, c, c),
                    SelfAttention1d(
                        c, c // 32) if i >= attn_layer else nn.Identity(),
                    conv_block(c, c, c),
                    SelfAttention1d(
                        c, c // 32) if i >= attn_layer else nn.Identity(),
                    conv_block(c, c, c_prev),
                    SelfAttention1d(c_prev, c_prev //
                                    32) if i >= attn_layer else nn.Identity(),
                    nn.Upsample(scale_factor=2, mode='linear',
                                align_corners=False),
                )
            else:
                block = nn.Sequential(
                    conv_block(n_io_channels + 16, c, c),
                    conv_block(c, c, c),
                    conv_block(c, c, c),
                    block,
                    conv_block(c * 2, c, c),
                    conv_block(c, c, c),
                    conv_block(c, c, n_io_channels, is_last=True),
                )
        self.net = block

        with torch.no_grad():
            for param in self.net.parameters():
                param *= 0.5

    def forward(self, input, t, cond_embed):
        self.state['cond'] = cond_embed
        timestep_embed = expand_to_planes(
            self.timestep_embed(t[:, None]), input.shape)
        out = self.net(torch.cat([input, timestep_embed], dim=1))
        self.state.clear()
        return out

class LightningDiffusion(pl.LightningModule):
    def __init__(self, encoder, global_args):
        super().__init__()

        self.encoder = encoder
        self.encoder_ema = deepcopy(self.encoder)
        self.diffusion = AudioDiffusion(global_args)
        self.diffusion_ema = deepcopy(self.diffusion)
        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)
        self.pqmf = PQMF(2, 70, global_args.pqmf_bands)

    def encode(self, *args, **kwargs):
        if self.training:
            return self.encoder(*args, **kwargs)
        return self.encoder_ema(*args, **kwargs)

    def decode(self, *args, **kwargs):
        if self.training:
            return self.diffusion(*args, **kwargs)
        return self.diffusion_ema(*args, **kwargs)

    def configure_optimizers(self):
        return optim.Adam(self.diffusion.parameters(), lr=1e-4)

    def eval_batch(self, batch):
        # Get the audio files
        reals = batch[0]

        reals = self.pqmf(reals)

        style_latents = self.encode(reals)

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
        v = self.decode(noised_reals, t, style_latents)
        return F.mse_loss(v, targets)

    def training_step(self, batch, batch_idx):
        loss = self.eval_batch(batch)
        log_dict = {'train/loss': loss.detach()}
        self.log_dict(log_dict, prog_bar=True, on_step=True)
        return loss

    def on_before_zero_grad(self, *args, **kwargs):
        decay = 0.98 if self.trainer.global_step < 10000 else 0.999
        ema_update(self.diffusion, self.diffusion_ema, decay)
        ema_update(self.encoder, self.encoder_ema, decay)
