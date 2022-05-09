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
import wandb

from byol.byol_pytorch import RandomApply

from diffusion.inference import sample
from diffusion.model import LightningDiffusion, AudioPerceiverEncoder, SelfSupervisedLearner, Transpose
from diffusion.dataset import SampleDataset
from diffusion.pqmf import CachedPQMF as PQMF
from diffusion.utils import MidSideEncoding, PadCrop, RandomGain

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
        self.pqmf = PQMF(2, 70, global_args.pqmf_bands)
        self.demo_dir = global_args.demo_dir
        self.demo_samples = global_args.sample_size
        self.demo_every = global_args.demo_every
        self.demo_steps = global_args.demo_steps
        self.ms_encoder = MidSideEncoding()
        self.pad_crop = PadCrop(global_args.sample_size)

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx, unused=0):
        last_demo_step = -1
        if (trainer.global_step - 1) % self.demo_every != 0 or last_demo_step == trainer.global_step:
            return

        last_demo_step = trainer.global_step

        demo_files = glob(f'{self.demo_dir}/**/*.wav', recursive=True)

        audio_batch = torch.zeros(len(demo_files), 2, self.demo_samples)

        for i, demo_file in enumerate(demo_files):
            audio, sr = torchaudio.load(demo_file)
            audio = audio.clamp(-1, 1)
            audio = self.pad_crop(audio)
            audio = self.ms_encoder(audio)
            audio_batch[i] = audio

        audio_batch = self.pqmf(audio_batch)

        audio_batch = audio_batch.to(module.device)

        with eval_mode(module):
            fakes = sample(module, audio_batch, self.demo_steps, 1)

        # undo the PQMF encoding
        fakes = self.pqmf.inverse(fakes.cpu())
        try:
            log_dict = {}
            for i, fake in enumerate(fakes):

                filename = f'demo_{trainer.global_step:08}_{i:02}.wav'
                fake = self.ms_encoder(fake).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
                torchaudio.save(filename, fake, 44100)
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
    p.add_argument('--demo-dir', type=Path, required=True,
                   help='path to a directory with audio files for demos')
    p.add_argument('--num-workers', type=int, default=2,
                   help='number of CPU workers for the DataLoader')
    p.add_argument('--batch-size', type=int, default=8,
                   help='number of audio samples per batch')
    p.add_argument('--num-gpus', type=int, default=1,
                   help='number of GPUs to use for training')
    p.add_argument('--pqmf-bands', type=int, default=8,
                   help='number of sub-bands for the PQMF filter')
    p.add_argument('--sample-size', type=int, default=16384,
                   help='Number of samples to train on, must be a multiple of 16384')
    p.add_argument('--demo-every', type=int, default=1000,
                   help='Number of steps between demos')
    p.add_argument('--demo-steps', type=int, default=500,
                   help='Number of denoising steps for the demos')                   
    p.add_argument('--checkpoint-every', type=int, default=20000,
                   help='Number of steps between checkpoints')
    p.add_argument('--data-repeats', type=int, default=1,
                   help='Number of times to repeat the dataset. Useful to lengthen epochs on small datasets')
    p.add_argument('--style-latent-size', type=int, default=512,
                   help='Size of the style latents')
    p.add_argument('--accum-batches', type=int, default=8,
                   help='Batches for gradient accumulation')        
    p.add_argument('--encoder-epochs', type=int, default=20,
                   help='Number of to train the encoder')                           
    p.add_argument('--skip-diffusion', type=bool, default=False, help='If true, diffusion model will not be trained')           
    args = p.parse_args()

    samples_set = SampleDataset([args.training_dir], args)
    train_set_size = int(len(samples_set) * 0.99)
    val_set_size = len(samples_set) - train_set_size
    train_set, val_set = torch.utils.data.random_split(samples_set, [train_set_size, val_set_size])
    train_dl = data.DataLoader(train_set, args.batch_size, shuffle=True,
                               num_workers=args.num_workers, persistent_workers=True, pin_memory=True)

    val_dl = data.DataLoader(val_set, args.batch_size,
                               num_workers=args.num_workers, persistent_workers=True, pin_memory=True)
    wandb_logger = pl.loggers.WandbLogger(project=args.name)

    validation_checkpoint = pl.callbacks.ModelCheckpoint(
        monitor="val/loss",
        filename="best",
    )

    last_checkpoint = pl.callbacks.ModelCheckpoint(filename="last")
    
    exc_callback = ExceptionCallback()
    
    encoder = AudioPerceiverEncoder(args)

    # Transform to go from data loader to encoder
    encoder_tf = torch.nn.Sequential(
        MidSideEncoding(),
        PQMF(2, 70, args.pqmf_bands),
        Transpose(-2, -1),
    )

    sr = 44100

    #Training augmentations for the PYOL
    pyol_augs = torch.nn.Sequential(
        RandomApply(
            RandomGain(0.8, 1),
            0.8
        )
    )

    latent_learner = SelfSupervisedLearner(
        encoder, 
        torch.randn(2, 2, args.sample_size),
        input_tf = encoder_tf,
        augment_fn=pyol_augs,
        hidden_layer=-1
    )
    wandb_logger.watch(latent_learner.learner)

    latent_trainer = pl.Trainer(
        gpus=args.num_gpus,
        strategy='ddp',
        precision=16,
        accumulate_grad_batches=args.accum_batches,
        callbacks=[validation_checkpoint, last_checkpoint, exc_callback],
        logger=wandb_logger,
        log_every_n_steps=1,
        max_epochs=args.encoder_epochs,
    )

    latent_trainer.fit(latent_learner, train_dl, val_dl)

    if not args.skip_diffusion:

        ckpt_callback = pl.callbacks.ModelCheckpoint(every_n_train_steps=args.checkpoint_every, save_top_k=-1)
        demo_callback = DemoCallback(args)
        diffusion_model = LightningDiffusion(encoder, args)
        wandb_logger.watch(diffusion_model.diffusion)
        
        diffusion_trainer = pl.Trainer(
            gpus=args.num_gpus,
            strategy='ddp',
            precision=16,
            accumulate_grad_batches={0:1, 1:args.accum_batches}, #Start without accumulation
            callbacks=[ckpt_callback, demo_callback, exc_callback],
            logger=wandb_logger,
            log_every_n_steps=1,
            max_epochs=10000000,
        )

        diffusion_trainer.fit(diffusion_model, train_dl)


if __name__ == '__main__':
    main()
