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

from losses.freq_losses import PerceptualSumAndDifferenceSTFTLoss

import wandb

from diffusion.pqmf import CachedPQMF as PQMF

from aeiou.datasets import AudioDataset

from autoencoders.soundstream import SoundStreamXLEncoder, SoundStreamXLDecoder

from diffusion.utils import PadCrop, Stereo

#from quantizer_pytorch import Quantizer1d

from aeiou.viz import embeddings_table, pca_point_cloud, audio_spectrogram_image, tokens_spectrogram_image


class AudioAutoencoder(pl.LightningModule):
    def __init__(self, global_args):
        super().__init__()

        self.pqmf_bands = global_args.pqmf_bands

        if self.pqmf_bands > 1:
            self.pqmf = PQMF(2, 70, global_args.pqmf_bands)

        capacity = 32

        c_mults = [2, 4, 8, 16, 32]
        
        strides = [2, 2, 2, 2, 2]

        self.encoder = SoundStreamXLEncoder(
            in_channels=2*global_args.pqmf_bands, 
            capacity=capacity, 
            latent_dim=global_args.latent_dim,
            c_mults = c_mults,
            strides = strides
        )

       # self.encoder_ema = deepcopy(self.encoder)
        self.decoder = SoundStreamXLDecoder(
            out_channels=2*global_args.pqmf_bands, 
            capacity=capacity, 
            latent_dim=global_args.latent_dim,
            c_mults = c_mults,
            strides = strides
        )

        self.quantizer = None

        self.num_residuals = global_args.num_residuals
        if self.num_residuals > 0:
            # self.quantizer = Quantizer1d(
            #     channels = global_args.latent_dim,
            #     num_groups = 1,
            #     codebook_size = global_args.codebook_size,
            #     num_residuals = self.num_residuals,
            #     shared_codebook = False,
            #     expire_threshold=0.5
            # )
            pass

      #  self.decoder_ema = deepcopy(self.diffusion)
        self.ema_decay = global_args.ema_decay
        
        scales = [2048, 1024, 512, 256, 128]
        hop_sizes = []
        win_lengths = []
        overlap = 0.75
        for s in scales:
            hop_sizes.append(int(s * (1 - overlap)))
            win_lengths.append(s)

        self.sdstft = auraloss.freq.SumAndDifferenceSTFTLoss(fft_sizes=scales, hop_sizes=hop_sizes, win_lengths=win_lengths)

    def encode(self, audio, with_info = False):
        latents = torch.tanh(self.encoder(audio))

        if self.quantizer:
            latents, _ = self.quantizer(latents)

        return latents

    def decode(self, latents):
        return self.decoder(latents)

    def configure_optimizers(self):
        return optim.Adam([*self.encoder.parameters(), *self.decoder.parameters()], lr=4e-5)
  
    def training_step(self, batch):
        reals = batch

        encoder_input = reals

        if self.pqmf_bands > 1:
            encoder_input = self.pqmf(reals)

        # Compute the model output and the loss.
        with torch.cuda.amp.autocast():
            # Freeze the encoder
            latents = torch.tanh(self.encoder(encoder_input).float())            

            if self.quantizer:
                latents, quantizer_info = self.quantizer(latents, num_residuals = random.randint(1, self.num_residuals))
                quantizer_loss = quantizer_info["loss"]

            decoded = self.decoder(latents)

            #Add pre-PQMF loss
            mb_distance = torch.tensor(0., device=self.device)

            if self.pqmf_bands > 1:
                mb_distance = F.mse_loss(encoder_input, decoded)
                decoded = self.pqmf.inverse(decoded)

            mrstft_loss = self.sdstft(reals, decoded)
            loss = mrstft_loss + mb_distance

            if self.quantizer:
                loss += quantizer_loss

        log_dict = {
            'train/loss': loss.detach(),
            'train/mb_distance': mb_distance.detach(),
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

    def load_encoder_weights_from_diffae(self, diffae_state_dict):
        own_state = self.state_dict()
        for name, param in diffae_state_dict.items():
            if name.startswith("encoder_ema."):
                new_name = name.replace("encoder_ema.", "encoder.")
                if isinstance(param, Parameter):
                    # backwards compatibility for serialized parameters
                    param = param.data
                own_state[new_name].copy_(param)

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
        self.pqmf_bands = global_args.pqmf_bands

        if self.pqmf_bands > 1:
            self.pqmf = PQMF(2, 70, global_args.pqmf_bands)

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
        
        if self.pqmf_bands > 1:
            encoder_input = self.pqmf(demo_reals)
        
        encoder_input = encoder_input.to(module.device)

        demo_reals = demo_reals.to(module.device)

        with torch.no_grad():
            tokens = module.encode(encoder_input)
            fakes = module.decode(tokens)

            if self.pqmf_bands > 1:
                fakes = self.pqmf.inverse(fakes.cpu())


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

    if args.ckpt_path:
        model = AudioAutoencoder.load_from_checkpoint(args.ckpt_path, global_args=args)
    else:
        model = AudioAutoencoder(args)

    if args.encoder_diffae_ckpt != '':
        diffae_state_dict = torch.load(args.encoder_diffae_ckpt, map_location='cpu')['state_dict']
        model.load_encoder_weights_from_diffae(diffae_state_dict)
        
    #model.encoder.requires_grad_(False)

    wandb_logger.watch(model)
    push_wandb_config(wandb_logger, args)

    trainer = pl.Trainer(
        devices=args.num_gpus,
        accelerator="gpu",
        num_nodes=args.num_nodes,
        strategy='ddp_find_unused_parameters_false',
        precision=16,
        accumulate_grad_batches=args.accum_batches, 
        callbacks=[ckpt_callback, demo_callback, exc_callback],
        logger=wandb_logger,
        log_every_n_steps=1,
        max_epochs=10000000,
    )

    trainer.fit(model, train_dl)

if __name__ == '__main__':
    main()

