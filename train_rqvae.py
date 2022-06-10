#!/usr/bin/env python3
import argparse
from contextlib import contextmanager
from pathlib import Path
from random import randint
import sys
from glob import glob

# from einops import rearrange
import pytorch_lightning as pl
from pytorch_lightning.utilities.distributed import rank_zero_only
import torch
from torch.utils import data
import torchaudio
from torchaudio import transforms as T
import torch.nn as nn
import wandb

from prefigure.prefigure import get_all_args, push_wandb_config

from dataset.dataset import SampleDataset
from diffusion.pqmf import CachedPQMF as PQMF
from diffusion.utils import PadCrop
from einops import rearrange
from encoders.learner import SoundStreamXLLearner

# Define utility functions


@contextmanager
def train_mode(model, mode=True):
    """A context manager that places a model into training mode and restores
    the previous mode on exit."""
    modes = [module.training for module in model.modules()]
    try:
        yield model.train(mode)
    finally:
        for i, module in enumerate(model.modules()):
            module.training = modes[i]


def eval_mode(model):
    """A context manager that places a model into evaluation mode and restores
    the previous mode on exit."""
    return train_mode(model, False)


class DemoCallback(pl.Callback):
    def __init__(self, demo_dl, global_args):
        super().__init__()
        self.demo_every = global_args.demo_every
        self.sample_rate = global_args.sample_rate
        self.demo_dl = iter(demo_dl)

    @rank_zero_only
    @torch.no_grad()
    def on_train_epoch_end(self, trainer, module, outputs, batch, batch_idx, unused=0):
        if trainer.current_epoch % self.demo_every != 0:
            return
        
        demo_reals, _ = next(self.demo_dl)

        demo_reals = demo_reals.to(module.device)

        fakes = module.soundstream(demo_reals)

        # Put the demos together
        fakes = rearrange(fakes, 'b d n -> d (b n)')
        demo_reals = rearrange(demo_reals, 'b d n -> d (b n)')

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


class ExceptionCallback(pl.Callback):
    def on_exception(self, trainer, module, err):
        print(f'{type(err).__name__}: {err}', file=sys.stderr)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--training-dir', type=Path, required=True,
                   help='the training data directory')
    p.add_argument('--name', type=str, required=True,
                   help='the name of the run')
    p.add_argument('--num-workers', type=int, default=4,
                   help='number of CPU workers for the DataLoader')
    p.add_argument('--batch-size', type=int, default=8,
                   help='number of audio samples per batch')
    p.add_argument('--num-gpus', type=int, default=1,
                   help='number of GPUs to use for training')
    p.add_argument('--sample-rate', type=int, default=48000,
                   help='The sample rate of the audio')
    p.add_argument('--sample-size', type=int, default=64000,
                   help='Number of samples to train on, must be a multiple of 640')
    p.add_argument('--demo-every', type=int, default=1000,
                   help='Number of steps between demos')                
    p.add_argument('--checkpoint-every', type=int, default=20000,
                   help='Number of steps between checkpoints')
    p.add_argument('--accum-batches', type=int, default=8,
                   help='Batches for gradient accumulation')  

    #Model hyperparameters
    p.add_argument('--style-latent-size', type=int, default=128,
                   help='Size of the style latents')     
    p.add_argument('--num-quantizers', type=int, default=8,
                   help='Number of residual vector quantizers')      
    p.add_argument('--codebook-size', type=int, default=2048,
                   help='Size of the style latents')                             
    args = p.parse_args()

    train_set = SampleDataset([args.training_dir], args)
    train_dl = data.DataLoader(train_set, batch_size=args.batch_size,
                               num_workers=args.num_workers, persistent_workers=True, pin_memory=True)
    demo_dl = data.DataLoader(train_set, args.num_demos, shuffle=True)
    wandb_logger = pl.loggers.WandbLogger(project=args.name)

    last_checkpoint = pl.callbacks.ModelCheckpoint(every_n_train_steps=10000, filename="last")
    
    exc_callback = ExceptionCallback()

    demo_callback = DemoCallback(demo_dl, args)

    soundstream = SoundStreamXLLearner(args)

    wandb_logger.watch(soundstream)

    latent_trainer = pl.Trainer(
        gpus=args.num_gpus,
        strategy="ddp_find_unused_parameters_false",
        #precision=16,
        accumulate_grad_batches=args.accum_batches,
        callbacks=[last_checkpoint, exc_callback, demo_callback],
        logger=wandb_logger,
        log_every_n_steps=1,
        max_epochs=1000000,
    )

    latent_trainer.fit(soundstream, train_dl)


if __name__ == '__main__':
    main()
