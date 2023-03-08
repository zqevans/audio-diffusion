#!/usr/bin/env python3

from prefigure.prefigure import get_all_args, push_wandb_config
from contextlib import contextmanager
import math
from pathlib import Path
from copy import deepcopy

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
from autoencoders.models import AudioAutoencoder
from audio_encoders_pytorch import Encoder1d
from diffusion.sampling import sample
from ema_pytorch import EMA
from aeiou.viz import pca_point_cloud, audio_spectrogram_image, tokens_spectrogram_image
from aeiou.datasets import AudioDataset
from dataset.dataset import SampleDataset
from dataset.dataset import get_wds_loader

from decoders.diffusion_decoder import DiffusionAttnUnet1D

from x_transformers import ContinuousTransformerWrapper, ContinuousAutoregressiveWrapper, Decoder

class LatentAudioDiffusionAutoencoder(pl.LightningModule):
    def __init__(self, autoencoder: AudioAutoencoder):
        super().__init__()

        
        self.latent_dim = autoencoder.latent_dim
                
        self.second_stage_latent_dim = 32

        factors = [2, 2, 2, 2]

        self.latent_downsampling_ratio = np.prod(factors)
        
        self.downsampling_ratio = autoencoder.downsampling_ratio * self.latent_downsampling_ratio

        self.latent_encoder = Encoder1d(
            in_channels=self.latent_dim, 
            out_channels = self.second_stage_latent_dim,
            channels = 128,
            multipliers = [1, 2, 4, 8, 8],
            factors =  factors,
            num_blocks = [8, 8, 8, 8],
        )

        self.latent_encoder_ema = deepcopy(self.latent_encoder)

        self.diffusion = DiffusionAttnUnet1D(
            io_channels=self.latent_dim, 
            cond_dim = self.second_stage_latent_dim,
            n_attn_layers=0, 
            c_mults=[512] * 10,
            depth=10
        )

        self.diffusion_ema = deepcopy(self.diffusion)

        self.diffusion_ema.requires_grad_(False)
        self.latent_encoder_ema.requires_grad_(False)

        self.autoencoder = autoencoder

        self.autoencoder.requires_grad_(False)
        
        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)

    def encode(self, reals):
        first_stage_latents = self.autoencoder.encode(reals)

        second_stage_latents = self.latent_encoder(first_stage_latents)

        second_stage_latents = torch.tanh(second_stage_latents)

        return second_stage_latents

    def decode(self, latents, steps=100, device="cuda"):
        first_stage_latent_noise = torch.randn([latents.shape[0], self.latent_dim, latents.shape[2]*self.latent_downsampling_ratio]).to(device)

        first_stage_sampled = sample(self.diffusion, first_stage_latent_noise, steps, 0, cond=latents)
        first_stage_sampled = first_stage_sampled.clamp(-1, 1)
        decoded = self.autoencoder.decode(first_stage_sampled)
        return decoded

class LatentAudioTransformer(pl.LightningModule):
    def __init__(self, latent_ae: LatentAudioDiffusionAutoencoder):
        super().__init__()
        
        self.latent_dim = latent_ae.second_stage_latent_dim
        self.downsampling_ratio = latent_ae.downsampling_ratio

        self.transformer = ContinuousTransformerWrapper(
            dim_in=self.latent_dim,
            dim_out=self.latent_dim,
            max_seq_len=512,
            attn_layers = Decoder(
                dim=1024,
                depth=12,
                heads=8,
                rel_pos_bias = True
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

        self.autoencoder = latent_ae

        self.autoencoder.requires_grad_(False).eval()

    def encode(self, reals):
        return self.autoencoder.encode(reals)

    def decode(self, latents, steps=100):
        return self.autoencoder.decode(latents, steps, device=self.device)

    def configure_optimizers(self):
        optimizer = optim.Adam([*self.transformer.parameters()], lr=1e-4)

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500, eta_min=1e-6)

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        reals = batch
        #reals = reals[0]

        latents = self.encode(reals)

        mask = None

        latents = rearrange(latents, "b c n -> b n c")

        with torch.cuda.amp.autocast():
            loss = self.transformer(latents)

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
            real_latents = module.encode(demo_reals).to(module.device)

            real_latents = rearrange(real_latents, "b c n -> b n c")

            start_embeds = real_latents[:, :64, :]

            print(f"Start embeds: {start_embeds.shape}")

            fake_latents = module.transformer.generate(start_embeds, n_samples)

            fake_latents = rearrange(fake_latents, "b n c -> b c n")
            
            print("Decoding")

            print(f"Fake wavelets: {fake_latents.shape}")

            fakes = module.decode(fake_latents)

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
            log_dict[f'embeddings_spec'] = wandb.Image(tokens_spectrogram_image(fake_latents))

            print("Done logging")
            trainer.logger.experiment.log(log_dict)

        except Exception as e:
            print(f'{type(e).__name__}: {e}')

def main():

    args = get_all_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    torch.manual_seed(args.seed)

    args.random_crop = False

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

    first_stage_config = {"capacity": 64, "c_mults": [2, 4, 8, 16, 32], "strides": [2, 2, 2, 2, 2], "latent_dim": 32}

    first_stage_autoencoder = AudioAutoencoder( 
        **first_stage_config
    ).eval()

    latent_diffae = LatentAudioDiffusionAutoencoder.load_from_checkpoint(args.pretrained_ckpt_path, autoencoder=first_stage_autoencoder, strict=False)

    latent_diffae.diffusion = latent_diffae.diffusion_ema
    del latent_diffae.diffusion_ema

    latent_diffae.latent_encoder = latent_diffae.latent_encoder_ema
    del latent_diffae.latent_encoder_ema

    if args.ckpt_path:
        model = LatentAudioTransformer.load_from_checkpoint(args.ckpt_path, autoencoder = latent_diffae, strict=False)
    else:
        model = LatentAudioTransformer(latent_ae = latent_diffae)

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

