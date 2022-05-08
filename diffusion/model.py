from copy import deepcopy
import math
from perceiver_pytorch import Perceiver
import pytorch_lightning as pl
from byol.byol_pytorch import BYOL
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from functools import partial

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


class Modulation1d(nn.Module):
    def __init__(self, state, feats_in, c_out):
        super().__init__()
        self.state = state
        self.layer = nn.Linear(feats_in, c_out * 2, bias=False)

    def forward(self, input):
        scales, shifts = self.layer(self.state['cond']).chunk(2, dim=-1)
        return torch.addcmul(shifts[..., None], input, scales[..., None] + 1)


class ResModConvBlock(ResidualBlock):
    def __init__(self, state, feats_in, c_in, c_mid, c_out, is_last=False):
        skip = None if c_in == c_out else nn.Conv1d(c_in, c_out, 1, bias=False)
        super().__init__([
            nn.Conv1d(c_in, c_mid, 5, padding=2),
            nn.GroupNorm(1, c_mid, affine=False),
            Modulation1d(state, feats_in, c_mid),
            nn.ReLU(inplace=True),
            nn.Conv1d(c_mid, c_out, 5, padding=2),
            nn.GroupNorm(
                1, c_out, affine=False) if not is_last else nn.Identity(),
            Modulation1d(state, feats_in,
                         c_out) if not is_last else nn.Identity(),
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
        qkv = qkv.view(
            [n, self.n_head * 3, c // self.n_head, s]).transpose(2, 3)
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
        self.weight = nn.Parameter(torch.randn(
            [out_features // 2, in_features]) * std)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)


def expand_to_planes(input, shape):
    return input[..., None].repeat([1, 1, shape[2]])


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

class Transpose(nn.Sequential):
    def __init__(self, dim0, dim1):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, input):
        return torch.transpose(input, self.dim0, self.dim1)

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

class AudioPerceiverEncoder(nn.Module):
    def __init__(self, global_args):
        super().__init__()
        self.net = Perceiver(
            input_channels=2,          # number of channels for each token of the input
            input_axis=1,# number of axis for input data (1 for audio, 2 for images, 3 for video)            
            num_freq_bands=200,# number of freq bands, with original value (2 * K + 1)
            max_freq=1000.,  # maximum frequency, hyperparameter depending on how fine the data is
            depth=20,# depth of net. The shape of the final attention mechanism will be:
                     # depth * (cross attention -> self_per_cross_attn * self attention)
            num_latents=256,  # number of latents, or induced set points, or centroids. different papers giving it different names
            latent_dim=512,            # latent dimension
            cross_heads=1,             # number of heads for cross attention. paper said 1
            latent_heads=8,            # number of heads for latent self attention, 8
            cross_dim_head=64,         # number of dimensions per cross attention head
            latent_dim_head=64,        # number of dimensions per latent self attention head
            num_classes=global_args.style_latent_size,          # output number of classes
            attn_dropout=0.,
            ff_dropout=0.,
            weight_tie_layers=True,# whether to weight tie layers (optional, as indicated in the diagram)
            fourier_encode_data=True,  # whether to auto-fourier encode the data, using the input_axis given. defaults to True, but can be turned off if you are fourier encoding the data yourself
            self_per_cross_attn=2      # number of self attention blocks per cross attention
        )

    def forward(self, input):
        return self.net(input)


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


class SelfSupervisedLearner(pl.LightningModule):
    def __init__(self, net, init_tensor, input_tf=None, **kwargs):
        super().__init__()
        self.input_tf = input_tf

        #Encode the mock input tensor as well
        if self.input_tf is not None:
            init_tensor = self.input_tf(init_tensor)

        self.learner = BYOL(net, init_tensor, **kwargs)

    def forward(self, inputs):
        if self.input_tf is not None:
            inputs = self.input_tf(inputs)

        return self.learner(inputs)

    def training_step(self, inputs, _):
        loss = self.forward(inputs)
        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def on_before_zero_grad(self, _):
        if self.learner.use_momentum:
            self.learner.update_moving_average()

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
