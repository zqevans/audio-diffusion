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

from ema_pytorch import EMA

import auraloss

import wandb

from aeiou.datasets import AudioDataset

from audio_diffusion_pytorch import AudioDiffusionModel, LogNormalDistribution, Distribution
from diffusion.model import ema_update
from aeiou.viz import audio_spectrogram_image

class DiffusionUncond(pl.LightningModule):
    def __init__(self, global_args):
        super().__init__()

        self.diffusion = AudioDiffusionModel(
            in_channels=2,
            channels=256,
            patch_blocks=1,
            patch_factor=1,
            multipliers=[4, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3],
            factors=[1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2],
            num_blocks=[1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 4],
            attentions=[0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2],
            attention_heads=8,
            attention_features=64,
            attention_multiplier=2,
            attention_use_rel_pos=False,
            resnet_groups=8,
            kernel_multiplier_downsample=2,
            use_nearest_upsample=False,
            use_skip_scale=True,
            use_context_time=True,
            use_magnitude_channels=False,
            use_stft=True,
            stft_num_fft=1023,
            stft_hop_length=256,
            diffusion_type="v",
            diffusion_sigma_distribution=UniformDistribution(),
        )

        self.diffusion_ema = EMA(
            self.diffusion, 
            beta=0.9999,
            power=3/4,
            update_after_step=100,
            update_every=1
        )
        
    def configure_optimizers(self):
        return optim.Adam([*self.diffusion.parameters()], lr=4e-5)
  
    def training_step(self, batch, batch_idx):
        reals = batch
        
        batch_stdev = torch.std(batch.detach())

        with torch.cuda.amp.autocast():
            loss = self.diffusion(reals)

        log_dict = {
            'train/loss': loss.detach(),
            'train/data_stdev': batch_stdev
        }

        self.log_dict(log_dict, prog_bar=True, on_step=True)
        return loss

    def on_before_zero_grad(self, *args, **kwargs):
        self.diffusion_ema.update()

class ExceptionCallback(pl.Callback):
    def on_exception(self, trainer, module, err):
        print(f'{type(err).__name__}: {err}', file=sys.stderr)


class DemoCallback(pl.Callback):
    def __init__(self, global_args):
        super().__init__()
        self.demo_every = global_args.demo_every
        self.num_demos = global_args.num_demos
        self.demo_samples = global_args.sample_size
        self.demo_steps = global_args.demo_steps
        self.sample_rate = global_args.sample_rate

    @rank_zero_only
    @torch.no_grad()
    #def on_train_epoch_end(self, trainer, module):
    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):        
        last_demo_step = -1
        if (trainer.global_step - 1) % self.demo_every != 0 or last_demo_step == trainer.global_step:
        #if trainer.current_epoch % self.demo_every != 0:
            return
        
        last_demo_step = trainer.global_step
        
        try:
            print("Creating noise")
            noise = torch.randn([self.num_demos, 2, self.demo_samples]).to(module.device)

            print("Starting sampling")
            
            fakes = module.diffusion_ema.ema_model.sample(noise=noise, num_steps=self.demo_steps)

            print("Rearranging demos")
            # Put the demos together
            fakes = rearrange(fakes, 'b d n -> d (b n)')

            log_dict = {}
            
            filename = f'demo_{trainer.global_step:08}.wav'
            fakes = fakes.clamp(-1, 1).mul(32767).to(torch.int16).cpu()
            torchaudio.save(filename, fakes, self.sample_rate)


            log_dict[f'demo'] = wandb.Audio(filename,
                                                sample_rate=self.sample_rate,
                                                caption=f'Demos')
        

            log_dict[f'demo_melspec_left'] = wandb.Image(audio_spectrogram_image(fakes))


            trainer.logger.experiment.log(log_dict, step=trainer.global_step)
        except Exception as e:
            print(f'{type(e).__name__}: {e}')

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

    train_dl = data.DataLoader(train_set, args.batch_size, shuffle=True,
                               num_workers=args.num_workers, persistent_workers=True, pin_memory=True)
    wandb_logger = pl.loggers.WandbLogger(project=args.name)
    exc_callback = ExceptionCallback()
    ckpt_callback = pl.callbacks.ModelCheckpoint(every_n_train_steps=args.checkpoint_every, save_top_k=-1)
    demo_callback = DemoCallback(args)
    diffusion_model = DiffusionUncond(args)
    wandb_logger.watch(diffusion_model)
    push_wandb_config(wandb_logger, args)

    diffusion_trainer = pl.Trainer(
        devices=args.num_gpus,
        accelerator="gpu",
        num_nodes = args.num_nodes,
        strategy='ddp_find_unused_parameters_false',
        precision=16,
        accumulate_grad_batches=args.accum_batches, 
        callbacks=[ckpt_callback, demo_callback, exc_callback],
        logger=wandb_logger,
        log_every_n_steps=1,
        max_epochs=10000000,
    )

    diffusion_trainer.fit(diffusion_model, train_dl, ckpt_path=args.ckpt_path)

if __name__ == '__main__':
    main()