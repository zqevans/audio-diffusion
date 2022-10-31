#!/usr/bin/env python3

from prefigure.prefigure import get_all_args, push_wandb_config
from copy import deepcopy
import math
import random
import sys
import torch
from torch import optim, nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.utils import data
from tqdm import trange
import pytorch_lightning as pl
from pytorch_lightning.utilities.distributed import rank_zero_only
from einops import rearrange
import numpy as np
import torchaudio

import auraloss

import wandb

from aeiou.datasets import AudioDataset

from audio_diffusion_pytorch import AutoEncoder1d

from diffusion.utils import PadCrop, Stereo

from quantizer_pytorch import Quantizer1d

from aeiou.viz import embeddings_table, pca_point_cloud, audio_spectrogram_image, tokens_spectrogram_image


class AudioAutoencoder(pl.LightningModule):
    def __init__(self, global_args):
        super().__init__()

        self.quantizer = None

        self.num_residuals = global_args.num_residuals
        if self.num_residuals > 0:
            self.quantizer = Quantizer1d(
                channels = 32,
                num_groups = 1,
                codebook_size = global_args.codebook_size,
                num_residuals = self.num_residuals,
                shared_codebook = False,
                expire_threshold=0.5
            )

        self.autoencoder = AutoEncoder1d(
            in_channels = 2, 
            channels = 64,
            patch_factor = 4,
            patch_blocks = 1,
            resnet_groups = 8,
            multipliers = [1, 2, 4, 4, 4, 1],
            factors = [2, 2, 2, 2, 1],
            num_blocks = [8, 8, 8, 8, 8]
        )

        self.ema_decay = global_args.ema_decay
        
        scales = [2048, 1024, 512, 256, 128]
        hop_sizes = []
        win_lengths = []
        overlap = 0.75
        for s in scales:
            hop_sizes.append(int(s * (1 - overlap)))
            win_lengths.append(s)

        self.sdstft = auraloss.freq.SumAndDifferenceSTFTLoss(fft_sizes=scales, hop_sizes=hop_sizes, win_lengths=win_lengths)

    def configure_optimizers(self):
        return optim.Adam([*self.autoencoder.parameters()], lr=1e-4)
  
    def encode(self, audio, with_info = False):
        latents = torch.tanh(self.autoencoder.encode(audio))

        if self.quantizer:
            latents, info = self.quantizer(latents)

            if with_info:
                return (latents, info)

        return latents

    def decode(self, latents):
        return self.autoencoder.decode(latents)

    def training_step(self, batch):
        reals = batch
          
        latents = torch.tanh(self.autoencoder.encode(reals).float())            

        if self.quantizer:
            latents, quantizer_info = self.quantizer(latents, num_residuals = random.randint(1, self.num_residuals))
            quantizer_loss = quantizer_info["loss"]

        decoded = self.decode(latents)

        mrstft_loss = self.sdstft(reals, decoded)

        loss = mrstft_loss

        if self.quantizer:
            loss += quantizer_loss

        log_dict = {
            'train/loss': loss.detach(),
            'train/mrstft_loss': mrstft_loss.detach(),            
        }
  
        if self.quantizer:
            log_dict["train/quantizer_loss"] = quantizer_loss.detach()

            # Log perplexity of each codebook used
            for i, perplexity in enumerate(quantizer_info["perplexity"]):
                log_dict[f"train_perplexity_{i}"] = perplexity
            # Log replaced codes of each codebook used
            for i, replaced_codes in enumerate(quantizer_info["replaced_codes"]):
                log_dict[f"train_replaced_codes_{i}"] = replaced_codes
            # Log budget
            # for i, budget in enumerate(quantizer_info["budget"]):
            #     log_dict[f"budget_{i}"] = budget

        self.log_dict(log_dict, prog_bar=True, on_step=True)
        return loss

class ExceptionCallback(pl.Callback):
    def on_exception(self, trainer, module, err):
        print(f'{type(err).__name__}: {err}', file=sys.stderr)


class DemoCallback(pl.Callback):
    def __init__(self, demo_dl, global_args):
        super().__init__()
        self.demo_every = global_args.demo_every
        self.demo_samples = global_args.sample_size
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
        
        last_demo_step = trainer.global_step

        demo_reals = next(self.demo_dl)

        encoder_input = demo_reals
        
        encoder_input = encoder_input.to(module.device)

        demo_reals = demo_reals.to(module.device)

        with torch.no_grad():
            tokens = module.encode(encoder_input)
            fakes = module.decode(tokens)

        # Put the demos together
        fakes = rearrange(fakes, 'b d n -> d (b n)')
        demo_reals = rearrange(demo_reals, 'b d n -> d (b n)')

        #demo_audio = torch.cat([demo_reals, fakes], -1)

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

            log_dict[f'embeddings'] = embeddings_table(tokens)

            log_dict[f'embeddings_3dpca'] = pca_point_cloud(tokens)
            log_dict[f'embeddings_spec'] = wandb.Image(tokens_spectrogram_image(tokens))

            log_dict[f'real_melspec_left'] = wandb.Image(audio_spectrogram_image(demo_reals))
            log_dict[f'recon_melspec_left'] = wandb.Image(audio_spectrogram_image(fakes))


            trainer.logger.experiment.log(log_dict, step=trainer.global_step)
        except Exception as e:
            print(f'{type(e).__name__}: {e}', file=sys.stderr)

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
                               num_workers=args.num_workers, persistent_workers=True, pin_memory=True)
    wandb_logger = pl.loggers.WandbLogger(project=args.name)
    demo_dl = data.DataLoader(train_set, args.num_demos, num_workers=args.num_workers, shuffle=True)
    
    exc_callback = ExceptionCallback()
    ckpt_callback = pl.callbacks.ModelCheckpoint(every_n_train_steps=args.checkpoint_every, save_top_k=-1)
    demo_callback = DemoCallback(demo_dl, args)

    model = AudioAutoencoder(args)

    wandb_logger.watch(model)
    push_wandb_config(wandb_logger, args)

    trainer = pl.Trainer(
        devices=args.num_gpus,
        accelerator="gpu",
        num_nodes=args.num_nodes,
        strategy='ddp_find_unused_parameters_false',
        #precision=16,
        accumulate_grad_batches=args.accum_batches, 
        callbacks=[ckpt_callback, demo_callback, exc_callback],
        logger=wandb_logger,
        log_every_n_steps=1,
        max_epochs=10000000,
    )

    trainer.fit(model, train_dl, ckpt_path=args.ckpt_path)

if __name__ == '__main__':
    main()

