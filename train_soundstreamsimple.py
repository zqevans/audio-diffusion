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

from auraloss.freq import MultiResolutionSTFTLoss
from viz.viz import embeddings_table, pca_point_cloud


class SoundStreamModule(pl.LightningModule):
    def __init__(self, global_args):
        super().__init__()

        n_io_channels = 2
        n_feature_channels = 8
        self.num_quantizers = global_args.num_quantizers

        self.unwrapped = False  # use single monolithic model or broken into parts?

        if not self.unwrapped:
            self.model = SoundStreamXL(n_io_channels, n_feature_channels, global_args.latent_dim, 
                n_quantizers=global_args.num_quantizers, codebook_size=global_args.codebook_size)
        else:
            self.encoder = SoundStreamXLEncoder(n_io_channels=n_io_channels, n_channels=n_feature_channels, latent_dim=global_args.latent_dim)  
            self.decoder = SoundStreamXLDecoder(n_io_channels=n_io_channels, n_channels=n_feature_channels, latent_dim=global_args.latent_dim)

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

        self.mrstft = MultiResolutionSTFTLoss()

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

        if True: # with torch.cuda.amp.autocast(): # can't get autocast to work so...
            if not self.unwrapped:
                preds, indices, cb_losses = self.model(encoder_input)  # cb_losses are codebook losses
            else:
                assert False, "TODO: You forgot to add the code here"

            targets = reals  # autoencoder     
            preds = preds[:,:,0:targets.size()[-1]] # preds come out padded
            mse_loss   = 10 * F.mse_loss(preds, targets) # mult factor based on experience, to balance the losses
            mstft_loss = 0.1 * self.mrstft(preds, targets) # mult factor based on experience, to balance the losses.
            cb_loss = 1e4 * cb_losses.sum()
            loss = mse_loss + mstft_loss + cb_loss

        log_dict = {
            'train/loss': loss.detach(),
            'train/mse_loss': mse_loss.detach(),
            'train/mstft_loss': mstft_loss.detach(),
            'train/cb_loss': cb_loss.detach(),
        }

        self.log_dict(log_dict, prog_bar=True, on_step=True)
        return loss

    def on_before_zero_grad(self, *args, **kwargs):
        decay = 0.95 if self.current_epoch < 25 else self.ema_decay
        #ema_update(self.diffusion, self.diffusion_ema, decay)
        #ema_update(self.encoder, self.encoder_ema, decay)

        #if self.num_quantizers > 0:
        #    ema_update(self.quantizer, self.quantizer_ema, decay)

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
        
        demo_reals, _ = next(self.demo_dl)
        encoder_input = demo_reals 
        encoder_input = encoder_input.to(module.device)
        demo_reals = demo_reals.to(module.device)

        if not module.unwrapped:
            fakes, indices, cb_losses = module.model(encoder_input)
        else:
            assert False, "TODO: You forgot to add the code here"

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
        #strategy='ddp', # this worked for DVAE, but not soundstream. ....?
        strategy = DDPStrategy(find_unused_parameters=False), #without this I get lots of warnings and it goes slow
        #precision=16,
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
