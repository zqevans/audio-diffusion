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

from dataset.dataset import SampleDataset

from autoencoders.soundstream import SoundStreamXLEncoder, SoundStreamXLDecoder

from audio_encoders_pytorch import STFT

from diffusion.utils import PadCrop, Stereo

from aeiou.viz import  pca_point_cloud, audio_spectrogram_image, tokens_spectrogram_image

from blocks.utils import InverseLR


class AudioAutoencoder(pl.LightningModule):
    def __init__(self, global_args):
        super().__init__()

        self.n_fft = 1024
        self.stft = STFT(
            num_fft = self.n_fft,
            hop_length = 256,
            window_length = self.n_fft,
            use_complex = True
        )

        capacity = 64

        c_mults = [32, 16, 8, 4, 2]
        
        strides = [1, 1, 1, 1, 1]

        latent_dim = 32

        self.encoder = SoundStreamXLEncoder(
            in_channels=2052, 
            capacity=capacity, 
            latent_dim=latent_dim,
            c_mults = c_mults,
            strides = strides
        )

       # self.encoder_ema = deepcopy(self.encoder)
        self.decoder = SoundStreamXLDecoder(
            out_channels=2052, 
            capacity=capacity, 
            latent_dim=latent_dim,
            c_mults = c_mults,
            strides = strides
        )

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
        audio_stft_1d = self.stft.encode1d(audio)
        latents = torch.tanh(self.encoder(audio_stft_1d))

        return latents

    def decode(self, latents):
        decoded_stft_1d = self.decoder(latents)
        decoded_audio = self.stft.decode1d(decoded_stft_1d)

        return decoded_audio

    def configure_optimizers(self):
        optimizer = optim.Adam([*self.encoder.parameters(), *self.decoder.parameters()], lr=1e-4)

        scheduler = InverseLR(optimizer, inv_gamma=50000, power=1/2, warmup=0.9)

        return [optimizer], [scheduler]
  
    def training_step(self, batch):
        reals, _ = batch

        # Compute the model output and the loss.
        with torch.cuda.amp.autocast():
            # Freeze the encoder
            latents = self.encode(reals)
            decoded = self.decode(latents)


            mrstft_loss = self.sdstft(reals, decoded)
            loss = mrstft_loss


        log_dict = {
            'train/loss': loss.detach(),
            'train/mrstft_loss': mrstft_loss.detach(),
        }

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

    @rank_zero_only
    @torch.no_grad()
    #def on_train_epoch_end(self, trainer, module):
    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):        
        last_demo_step = -1
        if (trainer.global_step - 1) % self.demo_every != 0 or last_demo_step == trainer.global_step:
        #if trainer.current_epoch % self.demo_every != 0:
            return
        
        last_demo_step = trainer.global_step

        demo_reals, _ = next(self.demo_dl)

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

    train_set = SampleDataset([args.training_dir], args, relpath=args.training_dir)

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
        #precision=16,
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

