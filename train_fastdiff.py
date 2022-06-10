#!/usr/bin/env python3

import argparse
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
from einops import rearrange

import torchaudio
import wandb

from dataset.dataset import MFCCDataset, SpecDataset

from diffusion.model import ema_update
from diffusion.FastDiff.FastDiff_model import FastDiff


# Define the noise schedule and sampling loop
def get_alphas_sigmas(t):
    """Returns the scaling factors for the clean image (alpha) and for the
    noise (sigma), given a timestep."""
    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)

def get_crash_schedule(t):
    sigma = torch.sin(t * math.pi / 2) ** 2
    alpha = (1 - sigma ** 2) ** 0.5
    return alpha_sigma_to_t(alpha, sigma)

def alpha_sigma_to_t(alpha, sigma):
    """Returns a timestep, given the scaling factors for the clean image and for
    the noise."""
    return torch.atan2(sigma, alpha) / math.pi * 2

@torch.no_grad()
def sample(model, reals, specs, steps, eta):
    """Draws samples from a model given starting noise."""
    ts = reals.new_ones([reals.shape[0]])

    # Create the noise schedule
    t = torch.linspace(1, 0, steps + 1)[:-1].to(reals.device)
    specs = specs.to(reals.device)
    alphas, sigmas = get_alphas_sigmas(get_crash_schedule(t))

    # The sampling loop
    for i in trange(steps):

        t_in = (ts * t[i]).unsqueeze(1).to(reals.device)

        # Get the model output (v, the predicted velocity)
        with torch.cuda.amp.autocast():
            v = model((reals, specs, t_in)).float()

        # Predict the noise and the denoised image
        pred = reals * alphas[i] - v * sigmas[i]
        eps = reals * sigmas[i] + v * alphas[i]

        # If we are not on the last timestep, compute the noisy image for the
        # next timestep.
        if i < steps - 1:
            # If eta > 0, adjust the scaling factor for the predicted noise
            # downward according to the amount of additional noise to add
            ddim_sigma = eta * (sigmas[i + 1]**2 / sigmas[i]**2).sqrt() * \
                (1 - alphas[i]**2 / alphas[i + 1]**2).sqrt()
            adjusted_sigma = (sigmas[i + 1]**2 - ddim_sigma**2).sqrt()

            # Recombine the predicted noise and predicted denoised image in the
            # correct proportions for the next step
            reals = pred * alphas[i + 1] + eps * adjusted_sigma

            # Add the correct amount of fresh noise
            if eta:
                reals += torch.randn_like(reals) * ddim_sigma

    # If we are on the last timestep, output the denoised image
    return pred


class FastDiffTrainer(pl.LightningModule):
    def __init__(self, global_args):
        super().__init__()

        self.diffusion = FastDiff(
            audio_channels=2,
            cond_channels=80,
            upsample_ratios=[8, 8, 4],
        )
        
        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)
        self.ema_decay = global_args.ema_decay

    def configure_optimizers(self):
        #optimizer =  optim.Adam([*self.diffusion.parameters()], lr=5e-4)
        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": {
        #         "scheduler": optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=1e-6, T_max=2000),
        #     },
        # }
        return optim.Adam([*self.diffusion.parameters()], lr=5e-4)

  
    def training_step(self, batch, batch_idx):
        specs, reals, _ = batch

        # Draw uniformly distributed continuous timesteps
        t = self.rng.draw(reals.shape[0])[:, 0].to(self.device)

        # Calculate the noise schedule parameters for those timesteps
        alphas, sigmas = get_alphas_sigmas(get_crash_schedule(t))

        # Combine the ground truth images and the noise
        alphas = alphas[:, None, None]
        sigmas = sigmas[:, None, None]
        noise = torch.randn_like(reals)
        noised_reals = reals * alphas + noise * sigmas
        targets = noise * alphas - reals * sigmas

        with torch.cuda.amp.autocast():
            v = self.diffusion((noised_reals, specs, t.unsqueeze(1)))
            loss = F.mse_loss(v, targets)

        log_dict = {
            'train/loss': loss.detach()
        }

        self.log_dict(log_dict, prog_bar=True, on_step=True)
        return loss

    def on_before_zero_grad(self, *args, **kwargs):
        decay = 0.95 if self.current_epoch < 25 else self.ema_decay
        #ema_update(self.diffusion, self.diffusion_ema, decay)


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
        
        demo_specs, demo_reals, _ = next(self.demo_dl)

        demo_reals = demo_reals.to(module.device)

        noise = torch.randn([demo_reals.shape[0], 2, self.demo_samples]).to(module.device)

        fakes = sample(module.diffusion, noise, demo_specs, self.demo_steps, 1)

        # Put the demos together
        fakes = rearrange(fakes, 'b d n -> d (b n)')
        demo_reals = rearrange(demo_reals, 'b d n -> d (b n)')

        #demo_audio = torch.stack([demo_reals, fakes], dim=0)

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

def main():
    args = get_all_args()
    """
    p = argparse.ArgumentParser()
    p.add_argument('--training-dir', type=Path, required=True,
                   help='the training data directory')
    p.add_argument('--name', type=str, required=True,
                   help='the name of the run')
    p.add_argument('--num-workers', type=int, default=2,
                   help='number of CPU workers for the DataLoader')
    p.add_argument('--num-gpus', type=int, default=1,
                   help='number of GPUs to use for training')
    
    p.add_argument('--sample-rate', type=int, default=48000,
                   help='The sample rate of the audio')
    p.add_argument('--sample-size', type=int, default=65536,
                   help='Number of samples to train on, must be a multiple of 16384')
    p.add_argument('--demo-every', type=int, default=10,
                   help='Number of epochs between demos')
    p.add_argument('--demo-steps', type=int, default=250,
                   help='Number of denoising steps for the demos')
    p.add_argument('--num-demos', type=int, default=16,
                   help='Number of demos to create')
    p.add_argument('--checkpoint-every', type=int, default=10000,
                   help='Number of steps between checkpoints')
    p.add_argument('--accum-batches', type=int, default=2,
                   help='Batches for gradient accumulation')
    p.add_argument('--batch-size', '-bs', type=int, default=8,
                   help='the batch size')

    p.add_argument('--ema-decay', type=float, default=0.995,
                   help='the EMA decay')
    p.add_argument('--seed', type=int, default=0,
                   help='the random seed')
    
    p.add_argument('--cache-training-data', type=bool, default=False,
                   help='If true, training data is kept in RAM')

    args = p.parse_args()
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    torch.manual_seed(args.seed)

    if args.use_mfcc:
        train_set = MFCCDataset([args.training_dir], args)
    else:
        train_set = SpecDataset([args.training_dir], args)

    train_dl = data.DataLoader(train_set, args.batch_size, shuffle=True,
                               num_workers=args.num_workers, persistent_workers=True, pin_memory=True)


    # def collate_fn(batch):
    #     print(len(batch))
    #     specs = []
    #     audios = []
    #     filenames = []
    #     for song in batch:
    #         specs = specs + song[0]
    #         audios = audios + song[1]
    #         filenames.append(song[2])
        
    #     return torch.FloatTensor(audios), torch.FloatTensor(specs), filenames

    # train_set = SongBatchDataset([args.training_dir], 8, args)
    # train_dl = data.DataLoader(train_set, args.batch_size, shuffle=True, collate_fn=collate_fn,
    #                            num_workers=args.num_workers, persistent_workers=True, pin_memory=True)
    
    wandb_logger = pl.loggers.WandbLogger(project=args.name)
    push_wandb_config(wandb_logger, args, omit=['training_dir'])
    demo_dl = data.DataLoader(train_set, args.num_demos, shuffle=True)
    
    exc_callback = ExceptionCallback()
    ckpt_callback = pl.callbacks.ModelCheckpoint(every_n_train_steps=args.checkpoint_every, save_top_k=-1)
    demo_callback = DemoCallback(demo_dl, args)
    
    diffusion_model = FastDiffTrainer(args)

    wandb_logger.watch(diffusion_model)

    diffusion_trainer = pl.Trainer(
        gpus=args.num_gpus,
        strategy="ddp_find_unused_parameters_false",
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

    diffusion_trainer.fit(diffusion_model, train_dl)

if __name__ == '__main__':
    main()