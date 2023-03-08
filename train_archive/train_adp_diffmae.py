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
import numpy as np
import torchaudio

import wandb

from ema_pytorch import EMA
from audio_encoders_pytorch import TanhBottleneck
from audio_diffusion_pytorch.model import AudioDiffusionMAE
from blocks.utils import InverseLR

from aeiou.viz import embeddings_table, pca_point_cloud, audio_spectrogram_image, tokens_spectrogram_image
from dataset.dataset import SampleDataset

class DiffMAE(pl.LightningModule):
    def __init__(self):
        super().__init__()


        self.diffMAE = AudioDiffusionMAE(
            in_channels = 2,
            encoder_channels = 32,
            encoder_factors = [1, 1, 1, 1, 1, 1],
            encoder_multipliers = [32, 16, 8, 8, 4, 2, 1],
            encoder_num_blocks = [4, 4, 4, 4, 4, 4],
            encoder_inject_depth = 1,
            encoder_patch_size = 1,
            #bottleneck = TanhBottleneck(),
            diffusion_type = "v",
            channels = 512,
            multipliers = [3, 2, 1, 1, 1, 1, 1, 1],
            factors = [1, 2, 2, 2, 2, 2, 2],
            num_blocks = [1, 1, 1, 2, 2, 2, 2],
            attentions = [0, 0, 0, 0, 0, 0, 0],
            stft_use_complex = True,
            stft_num_fft = 1023
        )

        self.diffMAE_ema = EMA(
            self.diffMAE,
            beta = 0.9999,
            power=3/4,
            update_every = 1,
            update_after_step = 1
        )
        
        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)      

    def configure_optimizers(self):
        optimizer = optim.Adam([*self.diffMAE.parameters()], lr=1e-4)

        scheduler = InverseLR(optimizer, inv_gamma=50000, power=1/2, warmup=0.9)

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        reals, _ = batch

        #with torch.cuda.amp.t():
        loss = self.diffMAE(reals)

        log_dict = {
            'train/loss': loss.detach(),
            'train/lr': self.lr_schedulers().get_last_lr()[0]
        }

        self.log_dict(log_dict, prog_bar=True, on_step=True)
        return loss

    def on_before_zero_grad(self, *args, **kwargs):
        self.diffMAE_ema.update()

class ExceptionCallback(pl.Callback):
    def on_exception(self, trainer, module, err):
        print(f'{type(err).__name__}: {err}', file=sys.stderr)


class DemoCallback(pl.Callback):
    def __init__(self, demo_dl, global_args):
        super().__init__()
        self.demo_every = global_args.demo_every
        self.demo_samples = global_args.sample_size
        self.demo_steps = global_args.demo_steps
        self.num_demos = global_args.num_demos
        self.sample_rate = global_args.sample_rate
        self.demo_dl = iter(demo_dl)

    @rank_zero_only
    @torch.no_grad()
    #def on_train_epoch_end(self, trainer, module):
    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):   
        last_demo_step = -1
        if (trainer.global_step - 1) % self.demo_every != 0 or last_demo_step == trainer.global_step:
        #if trainer.current_epoch % self.demo_every != 0:
            return
        
        last_demo_step = trainer.global_step
        
        print("Starting demo")
        try:

            demo_reals, _ = next(self.demo_dl)

            demo_reals = demo_reals.to(module.device)
            #demo_filenames = demo_filenames.to(module.device)

            with torch.no_grad():
                latents = module.diffMAE_ema.ema_model.encode(demo_reals)
                
                print("Reconstructing")
                reconstructed = module.diffMAE_ema.ema_model.decode(latents, num_steps=100)

            # Put the demos together
            reconstructed = rearrange(reconstructed, 'b d n -> d (b n)')
            
            log_dict = {}
            
            print("Saving files")
            filename = f'recon_demo_{trainer.global_step:08}.wav'
            reconstructed = reconstructed.clamp(-1, 1).mul(32767).to(torch.int16).cpu()
            torchaudio.save(filename, reconstructed, self.sample_rate)

            log_dict[f'recon_melspec_left'] = wandb.Image(audio_spectrogram_image(reconstructed))

            log_dict[f'recon'] = wandb.Audio(filename,
                                            sample_rate=self.sample_rate,
                                            caption=f'Reconstructed')


            demo_reals = rearrange(demo_reals, 'b d n -> d (b n)')

            reals_filename = f'reals_{trainer.global_step:08}.wav'
            demo_reals = demo_reals.clamp(-1, 1).mul(32767).to(torch.int16).cpu()
            torchaudio.save(reals_filename, demo_reals, self.sample_rate)

            
        
            log_dict[f'real'] = wandb.Audio(reals_filename,
                                                sample_rate=self.sample_rate,
                                                caption=f'Real')

            log_dict[f'embeddings_3dpca'] = pca_point_cloud(latents, output_type="plotly", mode="lines+markers")
            log_dict[f'embeddings_spec'] = wandb.Image(tokens_spectrogram_image(latents))

            log_dict[f'real_melspec_left'] = wandb.Image(audio_spectrogram_image(demo_reals))


            print("Done logging")
            trainer.logger.experiment.log(log_dict, step=trainer.global_step)

        except Exception as e:
            print(f'{type(e).__name__}: {e}')

def main():

    args = get_all_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    torch.manual_seed(args.seed)

    train_set = SampleDataset([args.training_dir], args, relpath=args.training_dir)

    train_dl = data.DataLoader(train_set, args.batch_size, shuffle=True,
                               num_workers=args.num_workers, persistent_workers=True, pin_memory=True)
    wandb_logger = pl.loggers.WandbLogger(project=args.name)
    demo_dl = data.DataLoader(train_set, args.num_demos, num_workers=args.num_workers, shuffle=True)
    exc_callback = ExceptionCallback()
    ckpt_callback = pl.callbacks.ModelCheckpoint(every_n_train_steps=args.checkpoint_every, save_top_k=-1)
    demo_callback = DemoCallback(demo_dl, args)

    model = DiffMAE()

    wandb_logger.watch(model)
    push_wandb_config(wandb_logger, args)

    diffusion_trainer = pl.Trainer(
        devices=args.num_gpus,
        accelerator="gpu",
        num_nodes = args.num_nodes,
        strategy='ddp',
        #precision=16,
        accumulate_grad_batches=args.accum_batches, 
        gradient_clip_val = 1.0,
        gradient_clip_algorithm="value",
        callbacks=[ckpt_callback, demo_callback, exc_callback],
        logger=wandb_logger,
        log_every_n_steps=1,
        max_epochs=10000000,
        default_root_dir=args.save_dir
    )

    diffusion_trainer.fit(model, train_dl, ckpt_path=args.ckpt_path)

if __name__ == '__main__':
    main()

