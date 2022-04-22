#!/usr/bin/env python3
import argparse
from contextlib import contextmanager
from pathlib import Path
import sys

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
from diffusion.pqmf import CachedPQMF as PQMF
from diffusion.utils import MidSideDecoding

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
        self.pqmf = PQMF(2, 100, global_args.pqmf_bands)
        self.demo_samples = global_args.sample_size
        self.demo_every = global_args.demo_every
        #self.ms_decoder = MidSideDecoding()

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx, unused=0):
        if (trainer.global_step - 1) % self.demo_every != 0:
            return

        noise = torch.zeros([4, 2, self.demo_samples])

        noise = self.pqmf(noise)

        noise = torch.randn_like(noise)

        noise = noise.to(module.device)

        with eval_mode(module):
            fakes = sample(module, noise, 1000, 1)

        #undo the PQMF encoding
        fakes = self.pqmf.inverse(fakes.cpu())

        log_dict = {}
        for i, fake in enumerate(fakes):
            filename = f'demo_{trainer.global_step:08}_{i:02}.wav'
            
            #fake = self.ms_decoder(fake).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
            fake = fake.clamp(-1, 1).mul(32767).to(torch.int16).cpu()
            torchaudio.save(filename, fake, 44100)
            log_dict[f'demo_{i}'] = wandb.Audio(filename,
                                                sample_rate=44100,
                                                caption=f'Demo {i}')
        trainer.logger.experiment.log(log_dict, step=trainer.global_step)


class ExceptionCallback(pl.Callback):
    def on_exception(self, trainer, module, err):
        print(f'{type(err).__name__}: {err}', file=sys.stderr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--training-dir', type=Path, required=True,
                   help='the training data directory')      
    p.add_argument('--name', type=str, required=True,
                   help='the name of the run')                      
    p.add_argument('--num-workers', type=int, default=2,
                   help='number of CPU workers for the DataLoader')   
    p.add_argument('--batch-size', type=int, default=8,
                   help='number of audio samples per batch')   
    p.add_argument('--num-gpus', type=int, default=1,
                   help='number of GPUs to use for training')  
    # p.add_argument('--mono', type=int, default=True,
    #                help='whether or not the model runs in mono')  
    p.add_argument('--pqmf-bands', type=int, default=4,
                   help='number of sub-bands for the PQMF filter')  
    p.add_argument('--sample-size', type=int, default=131072,
                   help='Number of samples to train on, must be a multiple of 65536')  
    p.add_argument('--demo-every', type=int, default=1000,
                   help='Number of steps between demos')  
    p.add_argument('--data-repeats', type=int, default=1,
                   help='Number of times to repeat the dataset. Useful to lengthen epochs on small datasets')                
    args = p.parse_args()

    #Bottom level samples = ((sample_size / PQMF bands) / [2^model depth])

    bottom_sample_size = args.sample_size / args.pqmf_bands / (2**14)

    print(f'bottom sample size: {bottom_sample_size}')

    train_set = SampleDataset([args.training_dir], args)
    train_dl = data.DataLoader(train_set, args.batch_size, shuffle=True,
                               num_workers=args.num_workers, persistent_workers=True, pin_memory=True)

    model = LightningDiffusion(args)
    wandb_logger = pl.loggers.WandbLogger(project=args.name)
    wandb_logger.watch(model.model)
    ckpt_callback = pl.callbacks.ModelCheckpoint(every_n_train_steps=10000, save_top_k=-1)
    demo_callback = DemoCallback(args)
    exc_callback = ExceptionCallback()

    extra_trainer_args = {}

    # if (args.num_gpus > 1):
    #     extra_trainer_args["accelerator"] = 'ddp'

    trainer = pl.Trainer(
        gpus=args.num_gpus,
        strategy='ddp',
        precision=16,
        auto_scale_batch_size=True,
        callbacks=[ckpt_callback, demo_callback, exc_callback],
        logger=wandb_logger,
        log_every_n_steps=1,
        max_epochs=10000000,
       # **extra_trainer_args
    )

    trainer.fit(model, train_dl)


if __name__ == '__main__':
    main()