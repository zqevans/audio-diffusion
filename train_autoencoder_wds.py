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

from diffusion.pqmf import CachedPQMF as PQMF

from dataset.dataset import get_wds_loader

from autoencoders.soundstream import SoundStreamXLEncoder, SoundStreamXLDecoder

from diffusion.utils import PadCrop, Stereo

from quantizer_pytorch import Quantizer1d
from nwt_pytorch import Memcodes

from aeiou.viz import embeddings_table, pca_point_cloud, audio_spectrogram_image, tokens_spectrogram_image


class AudioAutoencoder(pl.LightningModule):
    def __init__(
        self,
        capacity=64,
        c_mults=[2, 4, 8, 16],
        strides=[2, 4, 4, 8],
        pqmf_bands=1,
        latent_dim=256,
        num_residuals=0,
        codebook_size=1024,
        sample_rate=48000
    ):
        super().__init__()

        self.pqmf_bands = pqmf_bands

        if self.pqmf_bands > 1:
            self.pqmf = PQMF(2, 70, self.pqmf_bands)

        self.encoder = SoundStreamXLEncoder(
            in_channels=2*self.pqmf_bands, 
            capacity=capacity, 
            latent_dim=latent_dim,
            c_mults = c_mults,
            strides = strides
        )
        
        self.decoder = SoundStreamXLDecoder(
            out_channels=2*self.pqmf_bands, 
            capacity=capacity, 
            latent_dim=latent_dim,
            c_mults = c_mults,
            strides = strides
        )

        self.quantizer = None

        self.num_residuals = num_residuals
        if self.num_residuals > 0:
            # self.quantizer = Quantizer1d(
            #     channels = global_args.latent_dim,
            #     num_groups = 1,
            #     codebook_size = global_args.codebook_size,
            #     num_residuals = self.num_residuals,
            #     shared_codebook = False,
            #     expire_threshold=0.6
            # )
            self.quantizer = Memcodes(
                dim=latent_dim,
                heads=num_residuals,
                num_codes=codebook_size,
                temperature=1.0
            )

        scales = [2048, 1024, 512, 256, 128]
        hop_sizes = []
        win_lengths = []
        overlap = 0.75
        for s in scales:
            hop_sizes.append(int(s * (1 - overlap)))
            win_lengths.append(s)

        #self.mrstft = auraloss.freq.MultiResolutionSTFTLoss(fft_sizes=scales, hop_sizes=hop_sizes, win_lengths=win_lengths, perceptual_weighting=True, scale="mel", sample_rate=sample_rate)

        self.sdstft = auraloss.freq.SumAndDifferenceSTFTLoss(fft_sizes=scales, hop_sizes=hop_sizes, win_lengths=win_lengths, perceptual_weighting=True, sample_rate=sample_rate, scale="mel", n_mels=64)

    def encode(self, audio, with_info = False):
        if self.pqmf_bands > 1:
            audio = self.pqmf(audio)

        latents = self.encoder(audio)

        if self.quantizer:
            #latents, _ = self.quantizer(latents)
            latents = rearrange(latents, "b d n -> b n d")
            latents, indices = self.quantizer(latents)
            latents = rearrange(latents, "b n d -> b d n")
        else:
            latents = torch.tanh(latents)

        if with_info:
            return latents, indices
        else:
            return latents

    def decode(self, latents):
        
        decoded = self.decoder(latents)

        if self.pqmf_bands > 1:
            decoded = self.pqmf.inverse(decoded)
        
        return decoded

    def configure_optimizers(self):
        parameters = [*self.encoder.parameters(), *self.decoder.parameters()]
        if self.quantizer:
            parameters += [*self.quantizer.parameters()]
        return optim.Adam(parameters, lr=1e-5)
  
    def training_step(self, batch):
        reals, _, _ = batch
        reals = reals[0]

        encoder_input = reals

        if self.pqmf_bands > 1:
            encoder_input = self.pqmf(reals)

        # Compute the model output and the loss.
        with torch.cuda.amp.autocast():
            latents = self.encoder(encoder_input).float()

            if self.quantizer:
                #latents, quantizer_info = self.quantizer(latents) #, num_residuals = random.randint(1, self.num_residuals))
                #quantizer_loss = quantizer_info["loss"]
                latents = rearrange(latents, "b d n -> b n d")
                latents, _ = self.quantizer(latents)
                latents = rearrange(latents, "b n d -> b d n")
            else:
                latents = torch.tanh(latents)

            decoded = self.decoder(latents)

            #Add pre-PQMF loss
            mb_distance = torch.tensor(0., device=self.device)

            if self.pqmf_bands > 1:
                #mb_distance = self.mrstft(encoder_input, decoded)
                decoded = self.pqmf.inverse(decoded)

            mrstft_loss = self.sdstft(reals, decoded)

            phase_loss = F.l1_loss(reals, decoded) * 0.1

            loss = mrstft_loss + mb_distance #+ phase_loss

            # if self.quantizer:
            #     loss += quantizer_loss

        log_dict = {
            'train/loss': loss.detach(),
            'train/mb_distance': mb_distance.detach(),
            'train/mrstft_loss': mrstft_loss.detach(),
            'train/phase_loss': phase_loss.detach()
        }

        # if self.quantizer:
        #     log_dict["train/quantizer_loss"] = quantizer_loss.detach()

        #     # Log perplexity of each codebook used
        #     for i, perplexity in enumerate(quantizer_info["perplexity"]):
        #         log_dict[f"train_perplexity_{i}"] = perplexity
        #     # Log replaced codes of each codebook used
        #     for i, replaced_codes in enumerate(quantizer_info["replaced_codes"]):
        #         log_dict[f"train_replaced_codes_{i}"] = replaced_codes
        #     # Log budget
        #     # for i, budget in enumerate(quantizer_info["budget"]):
        #     #     log_dict[f"budget_{i}"] = budget

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
        print(f'{type(err).__name__}: {err}')


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

        module.eval()

        demo_reals, _, _ = next(self.demo_dl)

        demo_reals = demo_reals[0].to(module.device)
        
        # encoder_input = encoder_input

        demo_reals = demo_reals.to(module.device)

        with torch.no_grad():
            tokens = module.encode(demo_reals)
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

            #log_dict[f'embeddings'] = embeddings_table(tokens)

            log_dict[f'embeddings_3dpca'] = pca_point_cloud(tokens)
            log_dict[f'embeddings_spec'] = wandb.Image(tokens_spectrogram_image(tokens))

            log_dict[f'real_melspec_left'] = wandb.Image(audio_spectrogram_image(demo_reals))
            log_dict[f'recon_melspec_left'] = wandb.Image(audio_spectrogram_image(fakes))


            trainer.logger.experiment.log(log_dict, step=trainer.global_step)
        except Exception as e:
            print(f'{type(e).__name__}: {e}')
        finally:
            module.train()

def main():

    args = get_all_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    torch.manual_seed(args.seed)

    names = [
   
    ]

    train_dl = get_wds_loader(
        batch_size=args.batch_size,
        s3_url_prefix=None,
        sample_size=args.sample_size,
        names=names,
        sample_rate=args.sample_rate,
        num_workers=args.num_workers,
        recursive=True,
        random_crop=True,
        #normalize_lufs=-14.0,
        epoch_steps=10000,
    )
   
    exc_callback = ExceptionCallback()
    ckpt_callback = pl.callbacks.ModelCheckpoint(every_n_train_steps=args.checkpoint_every, save_top_k=-1)
    demo_callback = DemoCallback(train_dl, args)

    ae_config = {
        'pqmf_bands': 1,
        'sample_rate': args.sample_rate,
        'c_mults': [2, 4, 8, 16, 32],
        'capacity': 64,
        'num_residuals': 0,
        'latent_dim': 32,
        'strides': [2, 2, 2, 2, 2]
    }

    if args.ckpt_path:
        model = AudioAutoencoder.load_from_checkpoint(args.ckpt_path, **ae_config, strict=False)
    else:
        model = AudioAutoencoder(**ae_config)

    wandb_logger = pl.loggers.WandbLogger(project=args.name)
    wandb_logger.watch(model)
    push_wandb_config(wandb_logger, args)

    trainer = pl.Trainer(
        devices=args.num_gpus,
        accelerator="gpu",
        num_nodes=args.num_nodes,
        strategy='ddp',
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

