#!/usr/bin/env python3

from prefigure.prefigure import get_all_args, push_wandb_config
from contextlib import contextmanager
from copy import deepcopy
import math
from pathlib import Path

import sys
import torch
from torch import optim, nn
from torch.nn import functional as F
from torch.utils import data
from tqdm import trange
import pytorch_lightning as pl
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.plugins import DDPPlugin

from einops import rearrange

import torchaudio
from fairscale.nn import checkpoint_wrapper, auto_wrap, wrap

import wandb
import numpy as np
import pandas as pd

from dataset.dataset import SampleDataset
from encoders.encoders import SoundStreamXL, SoundStreamXLEncoder, SoundStreamXLDecoder
from vector_quantize_pytorch import ResidualVQ
from dvae.residual_memcodes import ResidualMemcodes
from diffusion.model import ema_update


# lonewater's auraloss fork:  pip install --no-cache-dir -U git+https://github.com/lonewater/auraloss.git@PWCmplxDif
from auraloss.freq import MultiResolutionSTFTLoss, PerceptuallyWeightedComplexLoss, MultiResolutionPrcptWghtdCmplxLoss

from viz.viz import embeddings_table, pca_point_cloud


class SoundStreamModule(pl.LightningModule):
    def __init__(self, global_args):
        super().__init__()

        n_io_channels = 2
        n_feature_channels = 8
        self.num_quantizers = global_args.num_quantizers
        self.unwrapped = False  # use single monolithic model or broken into parts?
        self.use_memcodes = True # only effects if unwrapped=True  Don't use ResidualVQ, which seems to break PyL stuff. Instead use memcodes for quantizer 
        self.ema_decay = global_args.ema_decay

        if not self.unwrapped:
            self.model = SoundStreamXL(n_io_channels, n_feature_channels, global_args.latent_dim, 
                n_quantizers=global_args.num_quantizers, codebook_size=global_args.codebook_size)
            self.model_ema = deepcopy(self.model)

        else:
            self.encoder = SoundStreamXLEncoder(n_io_channels=n_io_channels, n_channels=n_feature_channels, latent_dim=global_args.latent_dim)  
            self.decoder = SoundStreamXLDecoder(n_io_channels=n_io_channels, n_channels=n_feature_channels, latent_dim=global_args.latent_dim)

            self.quantizer = None
            if not self.use_memcodes:
                self.quantizer = ResidualVQ(
                    num_quantizers=self.num_quantizers, 
                    dim=global_args.latent_dim, 
                    codebook_size=global_args.codebook_size,
                    kmeans_init=True, 
                    kmeans_iters=100, 
                    threshold_ema_dead_code=2, 
                    channel_last=False,
                    sync_codebook=True
                )
            else: # memcodes
                self.num_quantizers = global_args.num_quantizers
                if self.num_quantizers > 0:
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

        self.mstft = MultiResolutionSTFTLoss()
        self.pwcl = PerceptuallyWeightedComplexLoss()
        self.mrpwcl = MultiResolutionPrcptWghtdCmplxLoss()

    def configure_optimizers(self):
        if not self.unwrapped:
            return optim.Adam([*self.model.parameters()], lr=2e-5)
        else:
            params = [*self.encoder.parameters(), *self.decoder.parameters()]
            if self.num_quantizers > 0:
                params += [*self.quantizer.parameters()]
            return optim.Adam(params, lr=2e-5)
  
    def training_step(self, batch, batch_idx):
        reals = batch[0]  # grab actual audio part of batch, not the filenames

        encoder_input = reals

        with torch.cuda.amp.autocast(): 
            if not self.unwrapped:
                preds, indices, cb_losses = self.model(encoder_input)  # cb_losses are codebook losses
            else:
                encoded = self.encoder(encoder_input)
                if not self.use_memcodes:
                    quantized, indices, cb_losses = self.quantizer(encoded)
                else:
                    encoded = rearrange(encoded, 'b d n -> b n d')
                    quantized, _= self.quantizer(encoded)
                    quantized = rearrange(quantized, 'b n d -> b d n')
                    indices, cb_losses = None, torch.zeros(2)  # so i don't have to write more ifs below

                preds = self.decoder(quantized)

            targets = reals  # autoencoder     
            preds = preds[:,:,0:targets.size()[-1]] # preds come out padded
            mse_loss   =  10 * F.mse_loss(preds, targets) # mult factor based on experience, to balance the losses
            mstft_loss =  0.1 * self.mstft(preds, targets) 
            pwc_loss =  0 # self.pwcl(preds, targets)
            mrpwc_loss = 0 # self.mrpwcl(preds,targets)
            cb_loss = 1e4 * cb_losses.sum()
            loss = mse_loss + mstft_loss + cb_loss + pwc_loss + mrpwc_loss

        log_dict = {
            'train/loss': loss.detach(),
            'train/mse_loss': mse_loss.detach(),
            'train/mstft_loss': mstft_loss.detach(),
            'train/cb_loss': cb_loss.detach(),
            #'train/pwc_loss': pwc_loss.detach(),
            #'train/mrpwc_loss': mrpwc_loss.detach(),
        }

        self.log_dict(log_dict, prog_bar=True, on_step=True)
        return loss

    def on_before_zero_grad(self, *args, **kwargs):
        decay = 0.95 if self.current_epoch < 25 else self.ema_decay
        ema_update(self.model, self.model_ema, decay)
        #ema_update(self.encoder, self.encoder_ema, decay)

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

    @rank_zero_only
    @torch.no_grad()
    def on_train_epoch_end(self, trainer, module):
 
        if trainer.current_epoch % self.demo_every != 0:
            return

        #print("\nStarting demo generation...")
        
        demo_reals, _ = next(self.demo_dl)
        encoder_input = demo_reals 
        encoder_input = encoder_input.to(module.device)
        demo_reals = demo_reals.to(module.device)

        module.model.eval()
        if not module.unwrapped:
            #encoder_input = encoder_input.half()  # do this or it breaks multi gpu
            module.model.to(encoder_input.dtype)
            fakes, indices, cb_losses = module.model(encoder_input)
        else:
            encoder_input = encoder_input.half() 
            encoded = module.encoder(encoder_input)
            if not module.use_memcodes:
                quantized, indices, cb_losses = self.quantizer(encoded)
            else:
                encoded = rearrange(encoded, 'b d n -> b n d')
                quantized, _= module.quantizer(encoded)
                quantized = rearrange(quantized, 'b n d -> b d n')
                indices, cb_losses = None, torch.zeros(2)  # so i don't have to write more ifs below
            fakes = module.decoder(quantized)

        # Put the demos together
        fakes = rearrange(fakes, 'b d n -> d (b n)')
        demo_reals = rearrange(demo_reals, 'b d n -> d (b n)')

        try: # loggins
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

            #log_dict[f'embeddings'] = embeddings_table(tokens)
            #log_dict[f'embeddings_3dpca'] = pca_point_cloud(tokens)

            trainer.logger.experiment.log(log_dict, step=trainer.global_step)

        except Exception as e:
            print(f'{type(e).__name__}: {e}', file=sys.stderr)

        module.model.train()
        return


def main():

    args = get_all_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    torch.manual_seed(args.seed)

    train_set = SampleDataset([args.training_dir], args)
    train_dl = data.DataLoader(train_set, args.batch_size, shuffle=True,
                               num_workers=args.num_workers, persistent_workers=True, pin_memory=True)
    wandb_logger = pl.loggers.WandbLogger(project=args.name)
    demo_dl = data.DataLoader(train_set, args.num_demos, num_workers=args.num_workers, shuffle=True)
    
    exc_callback = ExceptionCallback()
    ckpt_callback = pl.callbacks.ModelCheckpoint(every_n_train_steps=args.checkpoint_every, save_top_k=-1)
    demo_callback = DemoCallback(demo_dl, args)
    module = SoundStreamModule(args)
    wandb_logger.watch(module)
    push_wandb_config(wandb_logger, args)

    trainer = pl.Trainer(
        gpus=args.num_gpus,
        accelerator="gpu",
        strategy='ddp', # this worked for DVAE, but not soundstream. ....?
        #strategy = 'ddp_find_unused_parameters_false', #without this I get lots of warnings and it goes slow
        precision=32,
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

    trainer.fit(module, train_dl)

if __name__ == '__main__':
    main()
