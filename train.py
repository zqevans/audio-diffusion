#!/usr/bin/env python3
import argparse
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import sys

# from einops import rearrange
import pytorch_lightning as pl
from pytorch_lightning.utilities.distributed import rank_zero_only
import torch
from torch import optim
from torch.nn import functional as F
from torch.utils import data
import torchaudio
import wandb

from diffusion.inference import sample
from diffusion.utils import Stereo, RandomGain, PadCrop, get_alphas_sigmas
from diffusion.model import AudioDiffusion
from diffusion.dataset import SampleDataset

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


@torch.no_grad()
def ema_update(model, averaged_model, decay):
    """Incorporates updated model parameters into an exponential moving averaged
    version of a model. It should be called after each optimizer step."""
    model_params = dict(model.named_parameters())
    averaged_params = dict(averaged_model.named_parameters())
    assert model_params.keys() == averaged_params.keys()

    for name, param in model_params.items():
        averaged_params[name].mul_(decay).add_(param, alpha=1 - decay)

    model_buffers = dict(model.named_buffers())
    averaged_buffers = dict(averaged_model.named_buffers())
    assert model_buffers.keys() == averaged_buffers.keys()

    for name, buf in model_buffers.items():
        averaged_buffers[name].copy_(buf)


class LightningDiffusion(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = AudioDiffusion()
        self.model_ema = deepcopy(self.model)
        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)

    def forward(self, *args, **kwargs):
        if self.training:
            return self.model(*args, **kwargs)
        return self.model_ema(*args, **kwargs)

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=1e-4)

    def eval_batch(self, batch):
        reals = batch[0]

        # Sample timesteps
        t = self.rng.draw(reals.shape[0])[:, 0].to(reals)

        # Calculate the noise schedule parameters for those timesteps
        alphas, sigmas = get_alphas_sigmas(t)

        # Combine the ground truth images and the noise
        alphas = alphas[:, None, None]
        sigmas = sigmas[:, None, None]
        noise = torch.randn_like(reals)
        noised_reals = reals * alphas + noise * sigmas
        targets = noise * alphas - reals * sigmas

        # Compute the model output and the loss.
        v = self(noised_reals, t)
        return F.mse_loss(v, targets)

    def training_step(self, batch, batch_idx):
        loss = self.eval_batch(batch)
        log_dict = {'train/loss': loss.detach()}
        self.log_dict(log_dict, prog_bar=True, on_step=True)
        return loss

    def on_before_zero_grad(self, *args, **kwargs):
        decay = 0.98 if self.trainer.global_step < 10000 else 0.999
        ema_update(self.model, self.model_ema, decay)


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
    args = p.parse_args()

    # sample size needs to be a multiple of 2^16 for u-net compat
    args.training_sample_size = 131072 # (2 ^ 16) * 2, around 3 seconds at 44.1k

    train_tf = torch.nn.Sequential(
        Stereo(),
        RandomGain(0.5, 1.0),
        PadCrop(args.training_sample_size)
    )
    train_set = SampleDataset([args.training_dir], train_tf)
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