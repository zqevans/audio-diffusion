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
from einops import rearrange

from diffusion.pqmf import CachedPQMF as PQMF
import torchaudio

import auraloss

import wandb

from aeiou.datasets import AudioDataset
from dataset.dataset import SampleDataset

from audio_diffusion_pytorch import AudioDiffusionUpsampler
from audio_diffusion_pytorch.utils import downsample, upsample
from diffusion.model import ema_update
from aeiou.viz import embeddings_table, pca_point_cloud, audio_spectrogram_image, tokens_spectrogram_image

class DiffusionUncond(pl.LightningModule):
    def __init__(self, global_args):
        super().__init__()
      
        self.diffusion = AudioDiffusionUpsampler(
            factor = 3,
            in_channels = 2, 
            channels = 128,
            patch_blocks = 1,
            patch_factor = 8,
            resnet_groups = 8,
            kernel_multiplier_downsample = 2,
            multipliers = [1, 2, 4, 4, 4, 4, 4],
            factors = [2, 2, 2, 2, 2, 2],
            num_blocks = [2, 2, 2, 2, 2, 2],
            attentions = [0, 0, 0, 0, 1, 1, 1],
            attention_heads = 8,
            attention_features = 128,
            attention_multiplier = 4
        )

        self.diffusion_ema = deepcopy(self.diffusion)
        self.rng = torch.quasirandom.SobolEngine(1, scramble=True, seed=global_args.seed)
        self.ema_decay = global_args.ema_decay
        
    def configure_optimizers(self):
        return optim.Adam([*self.diffusion.parameters()], lr=1e-4)
  
    def training_step(self, batch, batch_idx):
        reals = batch
        
        loss = self.diffusion(reals)

        log_dict = {
            'train/loss': loss.detach()
        }

        self.log_dict(log_dict, prog_bar=True, on_step=True)
        return loss

    def on_before_zero_grad(self, *args, **kwargs):
        decay = 0.95 if self.current_epoch < 25 else self.ema_decay
        ema_update(self.diffusion, self.diffusion_ema, decay)

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
    #def on_train_epoch_end(self, trainer, module):
    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):        
        last_demo_step = -1
        if (trainer.global_step - 1) % self.demo_every != 0 or last_demo_step == trainer.global_step:
        #if trainer.current_epoch % self.demo_every != 0:
            return
        
        downsample_factor = 3

        last_demo_step = trainer.global_step
        
        demo_reals, _ = next(self.demo_dl)

        try:
            downsampled = downsample(demo_reals, downsample_factor)

            upsampled = module.diffusion_ema.sample(downsampled, downsample_factor)

            # Put the demos together
            upsampled = rearrange(upsampled, 'b d n -> d (b n)')

            log_dict = {}
            
            upsample_filename = f'upsampled_{trainer.global_step:08}.wav'
            upsampled = upsampled.clamp(-1, 1).mul(32767).to(torch.int16).cpu()
            torchaudio.save(upsample_filename, upsampled, self.sample_rate)


            downsample_filename = f'downsampled_{trainer.global_step:08}.wav'
            downsampled = downsampled.clamp(-1, 1).mul(32767).to(torch.int16).cpu()
            torchaudio.save(downsample_filename, downsampled, self.sample_rate // downsample_factor)

            reals_filename = f'reals_{trainer.global_step:08}.wav'
            demo_reals = demo_reals.clamp(-1, 1).mul(32767).to(torch.int16).cpu()
            torchaudio.save(reals_filename, demo_reals, self.sample_rate)


            log_dict[f'upsampled'] = wandb.Audio(upsample_filename,
                                                sample_rate=self.sample_rate,
                                                caption=f'Upsampled')
            log_dict[f'downsampled'] = wandb.Audio(downsample_filename,
                                                sample_rate=self.sample_rate // downsample_factor,
                                                caption=f'Downsampled')
            log_dict[f'real'] = wandb.Audio(reals_filename,
                                                sample_rate=self.sample_rate,
                                                caption=f'Real')

            log_dict[f'real_melspec_left'] = wandb.Image(audio_spectrogram_image(demo_reals))
            log_dict[f'downsample_melspec_left'] = wandb.Image(audio_spectrogram_image(downsampled))
            log_dict[f'upsample_melspec_left'] = wandb.Image(audio_spectrogram_image(upsampled))


            trainer.logger.experiment.log(log_dict, step=trainer.global_step)
        except Exception as e:
            print(f'{type(e).__name__}: {e}', file=sys.stderr)

def main():

    args = get_all_args()

    args.latent_dim = 0

    #args.random_crop = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    torch.manual_seed(args.seed)

    train_set = AudioDataset(
        [args.training_dir],
        sample_rate=args.sample_rate,
        sample_size=args.sample_size,
        random_crop=args.random_crop,
        augs='Stereo()'
    )

    #train_set = SampleDataset([args.training_dir], args, keywords=["kick", "snare", "clap", "snap", "hat", "cymbal", "crash", "ride"])

    train_dl = data.DataLoader(train_set, args.batch_size, shuffle=True,
                               num_workers=args.num_workers, persistent_workers=True, pin_memory=True)
    wandb_logger = pl.loggers.WandbLogger(project=args.name)
    demo_dl = data.DataLoader(train_set, args.num_demos, num_workers=args.num_workers, shuffle=True)
    exc_callback = ExceptionCallback()
    ckpt_callback = pl.callbacks.ModelCheckpoint(every_n_train_steps=args.checkpoint_every, save_top_k=-1)
    demo_callback = DemoCallback(demo_dl, args)
    diffusion_model = DiffusionUncond(args)
    wandb_logger.watch(diffusion_model)
    push_wandb_config(wandb_logger, args)

    diffusion_trainer = pl.Trainer(
        devices=args.num_gpus,
        accelerator="gpu",
        num_nodes = args.num_nodes,
        strategy='ddp',
        #precision=16,
        accumulate_grad_batches=args.accum_batches, 
        callbacks=[ckpt_callback, demo_callback, exc_callback],
        logger=wandb_logger,
        log_every_n_steps=1,
        max_epochs=10000000,
    )

    diffusion_trainer.fit(diffusion_model, train_dl, ckpt_path=args.ckpt_path)

if __name__ == '__main__':
    main()