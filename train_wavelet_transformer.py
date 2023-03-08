#!/usr/bin/env python3

from prefigure.prefigure import get_all_args, push_wandb_config
from contextlib import contextmanager
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

from encoders.wavelets import WaveletEncode1d, WaveletDecode1d

from blocks.utils import InverseLR
from ema_pytorch import EMA
from aeiou.viz import embeddings_table, pca_point_cloud, audio_spectrogram_image, tokens_spectrogram_image
from aeiou.datasets import AudioDataset
from dataset.dataset import SampleDataset
from dataset.dataset import get_wds_loader

from x_transformers import ContinuousTransformerWrapper, ContinuousAutoregressiveWrapper, Decoder

class WaveletTransformer(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.levels = 8
        
        self.latent_dim = 2 ** (self.levels+1)
        self.downsampling_ratio = 2 ** self.levels

        self.transformer = ContinuousTransformerWrapper(
            dim_in=self.latent_dim,
            dim_out=self.latent_dim,
            max_seq_len=1024,
            attn_layers = Decoder(
                dim=768,
                depth=12,
                header=8
            )
        )

        self.transformer = ContinuousAutoregressiveWrapper(self.transformer)

        self.transformer_ema = EMA(
            self.transformer,
            beta = 0.9999,
            power=3/4,
            update_every = 1,
            update_after_step = 1
        )

        self.encoder = WaveletEncode1d(2, "bior4.4", levels = self.levels)
        self.decoder = WaveletDecode1d(2, "bior4.4", levels = self.levels)

    def encode(self, reals):
        return self.encoder(reals)

    def decode(self, wavelets):
        return self.decoder(wavelets)

    def configure_optimizers(self):
        optimizer = optim.Adam([*self.transformer.parameters()], lr=1e-4)

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500, eta_min=1e-6)

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        reals = batch
        #reals = reals[0]

        wavelets = self.encode(reals)

        mask = None

        wavelets = rearrange(wavelets, "b c n -> b n c")

        with torch.cuda.amp.autocast():
            loss = self.transformer(wavelets)

        log_dict = {
            'train/loss': loss.detach(),
            'train/lr': self.lr_schedulers().get_last_lr()[0]
        }

        self.log_dict(log_dict, prog_bar=True, on_step=True)
        return loss

    def on_before_zero_grad(self, *args, **kwargs):
        self.transformer_ema.update()

class ExceptionCallback(pl.Callback):
    def on_exception(self, trainer, module, err):
        print(f'{type(err).__name__}: {err}', file=sys.stderr)


class DemoCallback(pl.Callback):
    def __init__(self, global_args, demo_dl):
        super().__init__()
        self.demo_every = global_args.demo_every
        self.demo_samples = global_args.sample_size
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

        n_samples = self.demo_samples//module.downsampling_ratio

        demo_reals = next(self.demo_dl).to(module.device)

        try:
            real_wavelets = module.encode(demo_reals).to(module.device)

            real_wavelets = rearrange(real_wavelets, "b c n -> b n c")

            start_embeds = real_wavelets[:, :1, :]

            print(f"Start embeds: {start_embeds.shape}")

            fake_wavelets = module.transformer_ema.ema_model.generate(start_embeds, n_samples)
            
            print("Decoding")

            print(f"Fake wavelets: {fake_wavelets.shape}")

            fakes = module.decode(fake_wavelets)

            print()

            # Put the demos together
            fakes = rearrange(fakes, 'b d n -> d (b n)')

            log_dict = {}
            
            print("Saving files")
            filename = f'demo_{trainer.global_step:08}.wav'
            fakes = fakes.clamp(-1, 1).mul(32767).to(torch.int16).cpu()
            torchaudio.save(filename, fakes, self.sample_rate)


            log_dict[f'demo'] = wandb.Audio(filename,
                                                sample_rate=self.sample_rate,
                                                caption=f'Reconstructed')
        
            log_dict[f'demo_melspec_left'] = wandb.Image(audio_spectrogram_image(fakes))

            print("Done logging")
            trainer.logger.experiment.log(log_dict)

        except Exception as e:
            print(f'{type(e).__name__}: {e}')

def main():

    args = get_all_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    torch.manual_seed(args.seed)

    train_set = AudioDataset(
        [args.training_dir],
        sample_rate=args.sample_rate,
        sample_size=args.sample_size,
        random_crop=args.random_crop,
        augs='Stereo(), PhaseFlipper()'
    )

    train_dl = data.DataLoader(train_set, args.batch_size, shuffle=True,
                               num_workers=args.num_workers, persistent_workers=True, pin_memory=True, drop_last=True)

    demo_dl = data.DataLoader(train_set, args.num_demos, num_workers=args.num_workers, shuffle=True)
   

    wandb_logger = pl.loggers.WandbLogger(project=args.name)
    
    exc_callback = ExceptionCallback()
    ckpt_callback = pl.callbacks.ModelCheckpoint(every_n_train_steps=args.checkpoint_every, save_top_k=-1)
    demo_callback = DemoCallback(args, demo_dl)

    if args.ckpt_path:
        model = WaveletTransformer.load_from_checkpoint(args.ckpt_path, strict=False)
    else:
        model = WaveletTransformer()

    wandb_logger.watch(model)
    push_wandb_config(wandb_logger, args)

    trainer = pl.Trainer(
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
        default_root_dir=args.save_dir
    )

    trainer.fit(model, train_dl)

if __name__ == '__main__':
    main()

