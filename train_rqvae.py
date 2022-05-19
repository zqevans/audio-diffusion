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

from dataset.dataset import SampleDataset
from diffusion.pqmf import CachedPQMF as PQMF
from diffusion.utils import PadCrop

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
    def __init__(self, global_args):
        super().__init__()
        self.demo_every = global_args.demo_every

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx, unused=0):
        last_demo_step = -1
        print(batch.shape)
        if (trainer.global_step - 1) % self.demo_every != 0 or last_demo_step == trainer.global_step:
            return

        last_demo_step = trainer.global_step

        try:
            log_dict = {}
            for i, reconstruction in enumerate(batch):

                filename = f'demo_{trainer.global_step:08}_{i:02}.wav'
                reconstruction = reconstruction.clamp(-1, 1).mul(32767).to(torch.int16).cpu()
                torchaudio.save(filename, reconstruction, 44100)
                log_dict[f'demo_{i}'] = wandb.Audio(filename,
                                                    sample_rate=44100,
                                                    caption=f'Demo {i}')
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

    def collate_fn(batch):
        lengths = torch.tensor([elem[0].shape[-1] for elem in batch])
        return nn.utils.rnn.pad_sequence([item[0] for item in batch], batch_first=True), lengths

    train_set = SampleDataset([args.training_dir], args)
    train_dl = data.DataLoader(train_set, batch_size=args.batch_size, collate_fn=collate_fn,
                               num_workers=args.num_workers, persistent_workers=True, pin_memory=True)
    wandb_logger = pl.loggers.WandbLogger(project=args.name)

    last_checkpoint = pl.callbacks.ModelCheckpoint(every_n_train_steps=10000, filename="last")
    
    exc_callback = ExceptionCallback()

    demo_callback = DemoCallback(args)

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
        max_epochs=100000,
    )

    latent_trainer.fit(soundstream, train_dl)


if __name__ == '__main__':
    main()
