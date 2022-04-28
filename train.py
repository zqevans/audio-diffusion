#!/usr/bin/env python3
import argparse
from contextlib import contextmanager
from pathlib import Path
import sys
from glob import glob

# from einops import rearrange
import pytorch_lightning as pl
from pytorch_lightning.utilities.distributed import rank_zero_only
import torch
from torch.utils import data
import torchaudio
import wandb

from diffusion.inference import sample
from diffusion.model import LightningDiffusion
from diffusion.dataset import SampleDataset
#from diffusion.pqmf import CachedPQMF as PQMF
from diffusion.utils import MidSideEncoding, PadCrop

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
        #self.pqmf = PQMF(2, 70, global_args.pqmf_bands)
        self.demo_dir = global_args.demo_dir
        self.demo_samples = global_args.sample_size
        self.demo_every = global_args.demo_every
        self.demo_steps = global_args.demo_steps
        self.ms_encoder = MidSideEncoding()
        self.pad_crop = PadCrop(global_args.sample_size)

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx, unused=0):
        print(trainer.global_step, self.demo_every, batch_idx)
        if (trainer.global_step - 1) % self.demo_every != 0:
            return

        # noise = torch.zeros([4, 2, self.demo_samples])

        # #noise = self.pqmf(noise)

        # noise = torch.randn_like(noise)

        # noise = noise.to(module.device)

        # TODO: Load demo files, padcrop them, pass them in to the sampler

        demo_files = glob(f'{self.demo_dir}/**/*.wav', recursive=True)

        print(demo_files)

        audio_batch = torch.zeros(len(demo_files), 2, self.demo_samples)

        for i, demo_file in enumerate(demo_files):
            audio, sr = torchaudio.load(demo_file)
            audio = audio.clamp(-1, 1)
            audio = self.pad_crop(audio)
            audio = self.ms_encoder(audio)
            audio_batch[i] = audio

        audio_batch = audio_batch.to(module.device)

        with eval_mode(module):
            fakes = sample(module, audio_batch, self.demo_steps, 1)

        # undo the PQMF encoding
        #fakes = self.pqmf.inverse(fakes.cpu())
        try:
            print("Making log dict")
            log_dict = {}
            for i, fake in enumerate(fakes):

                filename = f'demo_{trainer.global_step:08}_{i:02}.wav'
                print(f"Decoding fake {i}")
                fake = self.ms_encoder(
                    fake).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
                print(f"Saving fake {i}")
                torchaudio.save(filename, fake, 44100)
                print(f"Adding log_dict item for fake {i}")
                log_dict[f'demo_{i}'] = wandb.Audio(filename,
                                                    sample_rate=44100,
                                                    caption=f'Demo {i}')
            print(f"Logging fakes")
            trainer.logger.experiment.log(log_dict, step=trainer.global_step)
            print(f"Logged fakes")
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
    # p.add_argument('--mono', type=int, default=True,
    #                help='whether or not the model runs in mono')
    p.add_argument('--pqmf-bands', type=int, default=1,
                   help='number of sub-bands for the PQMF filter')
    p.add_argument('--sample-size', type=int, default=65536,
                   help='Number of samples to train on, must be a multiple of 65536')
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
    args = p.parse_args()

    train_set = SampleDataset([args.training_dir], args)
    train_dl = data.DataLoader(train_set, args.batch_size, shuffle=True,
                               num_workers=args.num_workers, persistent_workers=True, pin_memory=True)

    model = LightningDiffusion(args)
    wandb_logger = pl.loggers.WandbLogger(project=args.name)
    wandb_logger.watch(model.diffusion)
    ckpt_callback = pl.callbacks.ModelCheckpoint(
        every_n_train_steps=args.checkpoint_every, save_top_k=-1)
    demo_callback = DemoCallback(args)
    exc_callback = ExceptionCallback()

    bottom_sample_size = args.sample_size / (2**model.diffusion.depth)
    print(f'bottom sample size: {bottom_sample_size}')

    trainer = pl.Trainer(
        gpus=args.num_gpus,
        strategy='ddp',
        precision=16,
        accumulate_grad_batches={0:1, 1:args.accum_batches}, #Start without accumulation
        callbacks=[ckpt_callback, demo_callback, exc_callback],
        logger=wandb_logger,
        log_every_n_steps=1,
        max_epochs=10000000,
    )

    trainer.fit(model, train_dl)


if __name__ == '__main__':
    main()
