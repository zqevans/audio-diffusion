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
from einops import rearrange

import torchaudio
from fairscale.nn import checkpoint_wrapper, auto_wrap, wrap

import wandb
import numpy as np
import pandas as pd

from dataset.dataset import SampleDataset
from encoders.encoders import SoundStreamXL
from auraloss.freq import MultiResolutionSTFTLoss
from viz.viz import embeddings_table, pca_point_cloud


class SoundStreamModule(pl.LightningModule):
    def __init__(self, global_args):
        super().__init__()

        n_io_channels = 2
        n_feature_channels = 8
        self.num_quantizers = global_args.num_quantizers
        self.model = SoundStreamXL(n_io_channels, n_feature_channels, global_args.latent_dim, 
            n_quantizers=global_args.num_quantizers, codebook_size=global_args.codebook_size)

        self.mrstft = MultiResolutionSTFTLoss()

    def configure_optimizers(self):
        return optim.Adam([*self.model.parameters()], lr=2e-5)
  
    def training_step(self, batch, batch_idx):
        reals = batch[0]  # grab actual audio part of batch, not the filenames

        encoder_input = reals

        with torch.cuda.amp.autocast():
            preds, indices, cb_loss = self.model(encoder_input)  # cb_loss is codebook loss
            mse_loss   = 2 * F.mse_loss(preds, targets) # 2 is just based on experience, to balance the losses
            mstft_loss = 0.1 * self.mrstft(preds, targets) # 0.2 is just based on experience, to balance the losses.
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
        if self.pqmf_bands > 1:
            encoder_input = self.pqmf(demo_reals)

        fakes, indices, ss_losses = self.model(encoder_input)

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
    #diffusion_model = DiffusionDVAE(args)
    model = SoundStreamModule(args)
    wandb_logger.watch(model)
    push_wandb_config(wandb_logger, args)

    trainer = pl.Trainer(
        gpus=args.num_gpus,
        accelerator="gpu",
        #strategy='fsdp',
        strategy='ddp',
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
