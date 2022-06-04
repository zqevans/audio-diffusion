#!/usr/bin/env python3

import argparse
from contextlib import contextmanager
from copy import deepcopy
import math
from pathlib import Path
import numpy as np


import sys
import torch
from torch import optim, nn
from torch.nn import functional as F
from torch.utils import data
from tqdm import trange
import pytorch_lightning as pl
from pytorch_lightning.utilities.distributed import rank_zero_only
from einops import rearrange

import torchaudio
import wandb
from prefigure.prefigure import get_all_args, push_wandb_config
import auraloss

from dataset.dataset import SampleDataset
from diffusion.model import SkipBlock, FourierFeatures, expand_to_planes, ema_update
from diffusion.pqmf import CachedPQMF as PQMF
from encoders.encoders import RAVEEncoder, ResConvBlock, SoundStreamXLEncoder
from decoders.decoders import RAVEGenerator, SimpleDecoder

from decoders.decoders import multiscale_stft, Loudness, mod_sigmoid


from nwt_pytorch import Memcodes
from dvae.residual_memcodes import ResidualMemcodes

# from RAVE core: 
def get_beta_kl(step, warmup, min_beta, max_beta):
    if step > warmup: return max_beta
    t = step / warmup
    min_beta_log = np.log(min_beta)
    max_beta_log = np.log(max_beta)
    beta_log = t * (max_beta_log - min_beta_log) + min_beta_log
    return np.exp(beta_log)


def get_beta_kl_cyclic(step, cycle_size, min_beta, max_beta):
    return get_beta_kl(step % cycle_size, cycle_size // 2, min_beta, max_beta)


def get_beta_kl_cyclic_annealed(step, cycle_size, warmup, min_beta, max_beta):
    min_beta = get_beta_kl(step, warmup, min_beta, max_beta)
    return get_beta_kl_cyclic(step, cycle_size, min_beta, max_beta)



# Define the noise schedule and sampling loop
def get_alphas_sigmas(t):
    """Returns the scaling factors for the clean image (alpha) and for the
    noise (sigma), given a timestep."""
    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)

def get_crash_schedule(t):
    sigma = torch.sin(t * math.pi / 2) ** 2
    alpha = (1 - sigma ** 2) ** 0.5
    return alpha_sigma_to_t(alpha, sigma)

def alpha_sigma_to_t(alpha, sigma):
    """Returns a timestep, given the scaling factors for the clean image and for
    the noise."""
    return torch.atan2(sigma, alpha) / math.pi * 2


@torch.no_grad()
def sample(model, inputs):
    """just runs the model in inference mode"""
    with torch.cuda.amp.autocast():
        v = model(inputs).float()
    return v 


class ToMode:
    def __init__(self, mode):
        self.mode = mode

    def __call__(self, image):
        return image.convert(self.mode)


def ramp(x1, x2, y1, y2):
    def wrapped(x):
        if x <= x1:
            return y1
        if x >= x2:
            return y2
        fac = (x - x1) / (x2 - x1)
        return y1 * (1 - fac) + y2 * fac
    return wrapped


class ZQVAE(pl.LightningModule):
    def __init__(self, global_args, 
                min_kl=1e-4,
                max_kl=5e-1):
        super().__init__()

        #self.encoder = Encoder(global_args.codebook_size, 2)
        #self.encoder = SoundStreamXLEncoder(32, global_args.latent_dim, n_io_channels=2, strides=[2, 2, 4, 5, 8], c_mults=[2, 4, 4, 8, 16])
        #self.loudness = Loudness(global_args.sample_rate, 512)

        self.pqmf_bands = global_args.pqmf_bands

        if self.pqmf_bands > 1:
            self.pqmf = PQMF(2, 70, global_args.pqmf_bands)

        self.min_kl = min_kl
        self.max_kl = max_kl
        self.warmup = 1000000

        #Model: 
        #   Encoder part
        self.encoder = RAVEEncoder(2 * global_args.pqmf_bands, 64, global_args.latent_dim, ratios=[2, 2, 2, 2, 4, 4])
        self.encoder_ema = deepcopy(self.encoder)

        #   Latent middle rep: Memcodes (see below)

        #   Decoder part 
        #self.diffusion = DiffusionDecoder(global_args.latent_dim, 2)
        #self.decoder = RAVEDecoder(global_args.latent_dim, 2)
                # default RAVE settings pulled from https://github.com/acids-ircam/RAVE/blob/master/train_rave.py
        '''DATA_SIZE = 2
        CAPACITY = 64
        LATENT_SIZE = 128
        BIAS = True
        NO_LATENCY = False
        RATIOS = [4, 4, 2, 2, 2, 2] #[4, 4, 4, 2]

        MIN_KL = 1e-1
        MAX_KL = 1e-1
        CROPPED_LATENT_SIZE = 0
        FEATURE_MATCH = True

        LOUD_STRIDE = 1

        USE_NOISE = False
        NOISE_RATIOS = [4, 4, 4]
        NOISE_BANDS = 5

        D_CAPACITY = 16
        D_MULTIPLIER = 4
        D_N_LAYERS = 4

        MODE = "hinge"
        CKPT = None

        #no_latency=False
        #PADDING_MODE =  "causal" if no_latency else "centered"
        PADDING_MODE = "centered"
        self.decoder = RAVEGenerator(global_args.latent_dim,
            capacity=CAPACITY,
            data_size=DATA_SIZE,
            ratios=RATIOS,
            loud_stride=LOUD_STRIDE,
            use_noise=USE_NOISE,
            noise_ratios=NOISE_RATIOS,
            noise_bands=NOISE_BANDS,
            padding_mode=PADDING_MODE,
            bias=True)'''
        self.decoder = SimpleDecoder(global_args.latent_dim, io_channels=2)
        self.decoder_ema = deepcopy(self.decoder)


        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)
        self.ema_decay = global_args.ema_decay
        
        self.num_quantizers = global_args.num_quantizers
        self.quantized = (self.num_quantizers > 0)
        if self.quantized:
            if self.quantized: print(f"Making a quantizer.")
            quantizer_class = ResidualMemcodes if global_args.num_quantizers > 1 else Memcodes
            
            quantizer_kwargs = {}
            if global_args.num_quantizers > 1:
                quantizer_kwargs["num_quantizers"] = global_args.num_quantizers

            self.quantizer = quantizer_class(
                dim=global_args.latent_dim,
                heads=global_args.num_heads,
                num_codes=global_args.codebook_size,
                temperature=1.,
                **quantizer_kwargs
            )

            self.quantizer_ema = deepcopy(self.quantizer)
            self.mrstft = auraloss.freq.MultiResolutionSTFTLoss()

    def lin_distance(self, x, y):
        return torch.norm(x - y) / torch.norm(x)

    def log_distance(self, x, y):
        return abs(torch.log(x + 1e-7) - torch.log(y + 1e-7)).mean()

    def distance(self, x, y):
        scales = [2048, 1024, 512, 256, 128]
        x = multiscale_stft(x, scales, .75)
        y = multiscale_stft(y, scales, .75)

        lin = sum(list(map(self.lin_distance, x, y)))
        log = sum(list(map(self.log_distance, x, y)))

        return lin + log

    def reparametrize(self, mean, scale):
        std = nn.functional.softplus(scale) + 1e-4
        var = std * std
        logvar = torch.log(var)

        z = torch.randn_like(mean) * std + mean

        kl = (mean * mean + var - logvar - 1).sum(1).mean()

        if self.cropped_latent_size:
            noise = torch.randn(
                z.shape[0],
                self.latent_size - self.cropped_latent_size,
                z.shape[-1],
            ).to(z.device)
            z = torch.cat([z, noise], 1)
        return z, kl
        

    def encode(self, *args, **kwargs):
        if self.training:
            return self.encoder(*args, **kwargs)
        return self.encoder_ema(*args, **kwargs)

    def decode(self, *args, **kwargs):
        if self.training:
            return self.decoder(*args, **kwargs)
        return self.decoder_ema(*args, **kwargs)

    def configure_optimizers(self):
        return optim.Adam([*self.encoder.parameters(), *self.decoder.parameters()], lr=2e-5)

  
    def training_step(self, batch, batch_idx):
        reals = batch[0]
        encoder_input = reals

        #if self.pqmf_bands > 1:
        #    encoder_input = self.pqmf(reals)
        
        targets = deepcopy(reals)

        # Compute the model output and the loss.
        with torch.cuda.amp.autocast():
            tokens = self.encoder(encoder_input).float()

        if self.num_quantizers > 0:
            #Rearrange for Memcodes
            tokens = rearrange(tokens, 'b d n -> b n d')

            #Quantize into memcodes
            tokens, _ = self.quantizer(tokens)

            tokens = rearrange(tokens, 'b n d -> b d n')

        # p = torch.rand([reals.shape[0], 1], device=reals.device)
        # quantized = torch.where(p > 0.2, quantized, torch.zeros_like(quantized))
            z = tokens  # ? Zach?

        with torch.cuda.amp.autocast():
            out_wave = self.decoder(z)
            mse_loss   = 2 * F.mse_loss(out_wave, targets) # 2 is just based on experience, to balance the losses
            mstft_loss = 0.1 * self.mrstft(out_wave, targets) # 0.2 is just based on experience, to balance the losses.
            loss = mse_loss + mstft_loss


        log_dict = {
            'train/loss': loss.detach(),
            'train/mse_loss': mse_loss.detach(),
            'train/mstft_loss': mstft_loss.detach(),
        }

        self.log_dict(log_dict, prog_bar=True, on_step=True, sync_dist=True)
        return loss

    def on_before_zero_grad(self, *args, **kwargs):
        decay = 0.95 if self.current_epoch < 25 else self.ema_decay
        ema_update(self.decoder, self.decoder_ema, decay)
        ema_update(self.encoder, self.encoder_ema, decay)

        if self.num_quantizers > 0:
            ema_update(self.quantizer, self.quantizer_ema, decay)

class ExceptionCallback(pl.Callback):
    def on_exception(self, trainer, module, err):
        print(f'{type(err).__name__}: {err}', file=sys.stderr)






class DemoCallback(pl.Callback):
    def __init__(self, demo_dl, global_args):
        super().__init__()
        self.demo_every = global_args.demo_every
        self.demo_samples = global_args.sample_size
        self.demo_steps = global_args.demo_steps
        self.demo_dl = iter(demo_dl)
        self.sample_rate = global_args.sample_rate
        self.pqmf_bands = global_args.pqmf_bands
        self.quantized = global_args.num_quantizers > 0

        if self.pqmf_bands > 1:
            self.pqmf = PQMF(2, 70, global_args.pqmf_bands)

    @rank_zero_only
    @torch.no_grad()
    def on_train_epoch_end(self, trainer, module):
        #last_demo_step = -1
        #if (trainer.global_step - 1) % self.demo_every != 0 or last_demo_step == trainer.global_step:
        if trainer.current_epoch % self.demo_every != 0:
            return
        
        #last_demo_step = trainer.global_step

        demo_reals, _ = next(self.demo_dl)

        encoder_input = demo_reals
        
        if self.pqmf_bands > 1:
            encoder_input = self.pqmf(demo_reals)
        
        encoder_input = encoder_input.to(module.device)

        demo_reals = demo_reals.to(module.device)

        #noise = torch.randn([demo_reals.shape[0], 2, self.demo_samples]).to(module.device)

        tokens = module.encoder_ema(encoder_input)

        if self.quantized:

            #Rearrange for Memcodes
            tokens = rearrange(tokens, 'b d n -> b n d')

            tokens, _= module.quantizer_ema(tokens)
            tokens = rearrange(tokens, 'b n d -> b d n')


        #fakes = sample(module.decoder_ema, encoder_input) # , noise, self.demo_steps, 1, tokens)
        fakes = module.decoder_ema(tokens)

        # Put the demos together
        fakes = rearrange(fakes, 'b d n -> d (b n)')
        demo_reals = rearrange(demo_reals, 'b d n -> d (b n)')

        #demo_audio = torch.cat([demo_reals, fakes], -1)

        try:
            log_dict = {}
            
            filename = f'recon_{trainer.global_step:08}.wav'
            fakes = fakes.clamp(-1, 1).mul(32767).to(torch.int16).cpu()
            torchaudio.save(filename, fakes, self.sample_rate)

            reals_filename = f'reals_{trainer.global_step:08}.wav'
            demo_reals = demo_reals.clamp(-1, 1).mul(32767).to(torch.int16).cpu()
            torchaudio.save(reals_filename, demo_reals, self.sample_rate)


            log_dict[f'recon'] = wandb.Audio(filename,
                                                sample_rate=self.sample_rate,
                                                caption=f'Reconstructed')
            log_dict[f'real'] = wandb.Audio(reals_filename,
                                                sample_rate=self.sample_rate,
                                                caption=f'Real')
            trainer.logger.experiment.log(log_dict, step=trainer.global_step)
        except Exception as e:
            print(f'{type(e).__name__}: {e}', file=sys.stderr)

def main():
    args = get_all_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    torch.manual_seed(args.seed)

    train_set = SampleDataset([args.training_dir], args)
    train_dl = data.DataLoader(train_set, args.batch_size, shuffle=True,
                               num_workers=args.num_workers, persistent_workers=True, pin_memory=True)
    wandb_logger = pl.loggers.WandbLogger(project=args.name)
    push_wandb_config(wandb_logger, args)
    demo_dl = data.DataLoader(train_set, args.num_demos, shuffle=True)
    
    exc_callback = ExceptionCallback()
    ckpt_callback = pl.callbacks.ModelCheckpoint(every_n_train_steps=args.checkpoint_every, save_top_k=-1)
    demo_callback = DemoCallback(demo_dl, args)
    model = ZQVAE(args)
    wandb_logger.watch(model)

    #torch.autograd.set_detect_anomaly(True)

    trainer = pl.Trainer(
        gpus=torch.cuda.device_count(),
        num_nodes=args.num_nodes,
        strategy='fsdp',
        precision=16,
        accumulate_grad_batches={
            0:1, 
            1: args.accum_batches #Start without accumulation
            # 5:2,
            # 10:3, 
            # 12:4, 
            # 14:5, 
            # 16:6, 
            # 18:7,  
            # 20:8
            }, 
        callbacks=[ckpt_callback, demo_callback, exc_callback],
        logger=wandb_logger,
        log_every_n_steps=1,
        max_epochs=10000000,
    )

    trainer.fit(model, train_dl)

if __name__ == '__main__':
    main()
