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

from RAVE.rave.pqmf import PQMF

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
    @rank_zero_only
    @torch.no_grad()
    def on_batch_end(self, trainer, module):
        if trainer.global_step % 1000 != 0:
            return

        noise = torch.randn([4, 2, 131072], device=module.device)
        with eval_mode(module):
            fakes = sample(module, noise, 500, 1)

        #ms_decoder = MidSideDecoding()

        log_dict = {}
        for i, fake in enumerate(fakes):
            filename = f'demo_{trainer.global_step:08}_{i:02}.wav'
            
            #fake = ms_decoder(fake).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
            fake = fake.clamp(-1, 1).mul(32767).to(torch.int16).cpu()
            torchaudio.save(filename, fake, 44100)
            log_dict[f'demo_{i}'] = wandb.Audio(filename,
                                                sample_rate=44100,
                                                caption=f'Demo {i}')
        trainer.logger.experiment.log(log_dict, step=trainer.global_step)


class ExceptionCallback(pl.Callback):
    def on_exception(self, module, err):
        print(f'{type(err).__name__}: {err}', file=sys.stderr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--training-dir', type=Path, required=True,
                   help='the training data directory')         
    p.add_argument('--num-workers', type=int, default=2,
                   help='number of CPU workers for the DataLoader')   
    p.add_argument('--batch-size', type=int, default=8,
                   help='number of audio samples per batch')   
    p.add_argument('--num-gpus', type=int, default=1,
                   help='number of GPUs to use for training')  
    p.add_argument('--mono', type=int, default=True,
                   help='whether or not the model runs in mono')  
    args = p.parse_args()

    # sample size needs to be a multiple of 2^16 for u-net compat
    args.training_sample_size = 131072 # (2 ^ 16) * 2, around 3 seconds at 44.1k

    
    train_set = SampleDataset([args.training_dir], *args)
    train_dl = data.DataLoader(train_set, args.batch_size, shuffle=True,
                               num_workers=args.num_workers, persistent_workers=True, pin_memory=True)

    model = LightningDiffusion()
    wandb_logger = pl.loggers.WandbLogger(project="break-diffusion")
    wandb_logger.watch(model.model)
    ckpt_callback = pl.callbacks.ModelCheckpoint(every_n_train_steps=10000, save_top_k=-1)
    demo_callback = DemoCallback()
    exc_callback = ExceptionCallback()

    trainer = pl.Trainer(
        gpus=args.num_gpus,
        accelerator='ddp',
        precision=16,
        callbacks=[ckpt_callback, demo_callback, exc_callback],
        logger=wandb_logger,
        log_every_n_steps=1,
        max_epochs=10000000,
    )

    trainer.fit(model, train_dl)


if __name__ == '__main__':
    main()