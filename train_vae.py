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
from dataset.dataset import get_wds_loader
from autoencoders.soundstream import SoundStreamXLEncoder, SoundStreamXLDecoder

from diffusion.utils import PadCrop, Stereo
from diffusion.pqmf import CachedPQMF as PQMF

from quantizer_pytorch import Quantizer1d

from aeiou.viz import embeddings_table, pca_point_cloud, audio_spectrogram_image, tokens_spectrogram_image

from losses.adv_losses import StackDiscriminators
        
PQMF_ATTN = 100

class AudioVAE(pl.LightningModule):
    def __init__(self, global_args):
        super().__init__()

        # self.quantizer = None

        # self.num_residuals = global_args.num_residuals
        # if self.num_residuals > 0:
        #     self.quantizer = Quantizer1d(
        #         channels = 32,
        #         num_groups = 1,
        #         codebook_size = global_args.codebook_size,
        #         num_residuals = self.num_residuals,
        #         shared_codebook = False,
        #         expire_threshold=0.5
        #     )

        self.automatic_optimization = False

        self.pqmf_bands = global_args.pqmf_bands

        if self.pqmf_bands > 1:
            self.pqmf = PQMF(2, PQMF_ATTN, global_args.pqmf_bands)

        capacity = 32

        c_mults = [2, 4, 8, 16, 32]
        
        strides = [2, 2, 2, 2, 2]

        global_args.latent_dim = 32

        self.encoder = SoundStreamXLEncoder(
            in_channels=2*global_args.pqmf_bands, 
            capacity=capacity, 
            latent_dim=2*global_args.latent_dim,
            c_mults = c_mults,
            strides = strides
        )

        self.decoder = SoundStreamXLDecoder(
            out_channels=2*global_args.pqmf_bands, 
            capacity=capacity, 
            latent_dim=global_args.latent_dim,
            c_mults = c_mults,
            strides = strides
        )

        self.ema_decay = global_args.ema_decay

        self.warmed_up = False
        self.warmup_steps = global_args.warmup_steps
        
        scales = [2048, 1024, 512]
        hop_sizes = []
        win_lengths = []
        overlap = 0.75
        for s in scales:
            hop_sizes.append(int(s * (1 - overlap)))
            win_lengths.append(s)

        self.mrstft = auraloss.freq.MultiResolutionSTFTLoss(fft_sizes=scales, hop_sizes=hop_sizes, win_lengths=win_lengths)

        self.sdstft = auraloss.freq.SumAndDifferenceSTFTLoss(fft_sizes=scales, hop_sizes=hop_sizes, win_lengths=win_lengths, sample_rate=global_args.sample_rate, perceptual_weighting=True, scale="mel", n_bins=64)

        self.discriminator = StackDiscriminators(
            3,
            in_size=2, # Stereo
            capacity=16,
            multiplier=4,
            n_layers=4,
        )

    def configure_optimizers(self):
        opt_gen = optim.Adam([*self.encoder.parameters(), *self.decoder.parameters()], lr=4e-5)
        opt_disc = optim.Adam([*self.discriminator.parameters()], lr=1e-4, betas=(.5, .9))
        return [opt_gen, opt_disc]
  
    def sample(self, mean, scale):
        stdev = nn.functional.softplus(scale) + 1e-4
        var = stdev * stdev
        logvar = torch.log(var)
        latents = torch.randn_like(mean) * stdev + mean

        kl = (mean * mean + var - logvar - 1).sum(1).mean()

        return latents, kl

    def encode(self, audio):

        if self.pqmf_bands > 1:
            audio = self.pqmf(audio)

        mean, scale = self.encoder(audio).chunk(2, dim=1)
        latents, kl = self.sample(mean, scale)

        return latents, kl

    def decode(self, latents):
        decoded = self.decoder(latents)

        if self.pqmf_bands > 1:
            decoded = self.pqmf.inverse(decoded)

        return decoded

    def load_pretrained_ae(self, ae_state_dict):
        own_state = self.state_dict()
        for name, param in ae_state_dict.items():
            if name.startswith("encoder.") or name.startswith("decoder."):
                if isinstance(param, Parameter):
                    # backwards compatibility for serialized parameters
                    param = param.data
                own_state[name].copy_(param)

    def training_step(self, batch, batch_idx):
        reals, _, _ = batch

        reals = reals[0]
        
        if self.global_step >= self.warmup_steps:
            self.warmed_up = True

        opt_gen, opt_disc = self.optimizers()

        if self.pqmf_bands > 1:
            reals = self.pqmf(reals)

        # if self.warmed_up:
        #     with torch.no_grad():
        #         mean, scale = self.encoder(reals).chunk(2, dim=1)
        # else:
        mean, scale = self.encoder(reals).chunk(2, dim=1)

        latents, kl = self.sample(mean, scale)

        decoded = self.decoder(latents)

        if self.pqmf_bands > 1:
            mb_distance = self.mrstft(reals, decoded)
            decoded = self.pqmf.inverse(decoded)
            reals = self.pqmf.inverse(reals)
        else:
            mb_distance = torch.tensor(0.).to(reals)

        mrstft_loss = self.sdstft(reals, decoded)

        l1_time_loss = F.l1_loss(reals, decoded)

        if self.warmed_up:
            loss_dis, loss_adv, feature_matching_distance, _, _ = self.discriminator.loss(reals, decoded)
        else:
            loss_dis = torch.tensor(0.).to(reals)
            loss_adv = torch.tensor(0.).to(reals)
            feature_matching_distance = torch.tensor(0.).to(reals)

        # Train the discriminator
        if self.global_step % 2 and self.warmed_up:
            loss = loss_dis

            log_dict = {
                'train/discriminator_loss': loss_dis.detach()  
            }

            opt_disc.zero_grad()
            self.manual_backward(loss_dis)
            opt_disc.step()

        # Train the generator 
        else:

            kl_loss = 1e-4 * kl 

            loss_adv = 0.1 * loss_adv

            feature_matching_distance = 0.05 * feature_matching_distance

            # Combine spectral loss, KL loss, time-domain loss, and adversarial loss
            loss = mrstft_loss + mb_distance + loss_adv + feature_matching_distance + kl_loss #+ l1_time_loss 

            opt_gen.zero_grad()
            self.manual_backward(loss)
            opt_gen.step()

            log_dict = {
                'train/loss': loss.detach(),
                'train/mrstft_loss': mrstft_loss.detach(),   
                'train/mb_distance': mb_distance.detach(),
                'train/kl_loss': kl_loss.detach(),
                'train/l1_time_loss': l1_time_loss.detach(),
                'train/loss_adv': loss_adv.detach(),
                'train/feature_matching': feature_matching_distance.detach()
            }
            


        # if self.quantizer:
        #     loss += quantizer_loss
  
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
        self.last_demo_step = -1

    @rank_zero_only
    @torch.no_grad()
    #def on_train_epoch_end(self, trainer, module):
    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx): 
        if (trainer.global_step - 1) % self.demo_every != 0 or self.last_demo_step == trainer.global_step:
        #if trainer.current_epoch % self.demo_every != 0:
            return
        
        self.last_demo_step = trainer.global_step

        try:
            demo_reals, _, _ = next(self.demo_dl)

            demo_reals = demo_reals[0]

            encoder_input = demo_reals
            
            encoder_input = encoder_input.to(module.device)

            demo_reals = demo_reals.to(module.device)

            with torch.no_grad():
                tokens, _ = module.encode(encoder_input)

                fakes = module.decode(tokens)

            # Put the demos together
            fakes = rearrange(fakes, 'b d n -> d (b n)')
            demo_reals = rearrange(demo_reals, 'b d n -> d (b n)')

            #demo_audio = torch.cat([demo_reals, fakes], -1)

        
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
            print(f'{type(e).__name__}: {e}')

def main():

    args = get_all_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    torch.manual_seed(args.seed)

    names = []

    train_dl = get_wds_loader(
        batch_size=args.batch_size,
        s3_url_prefix=None,
        sample_size=args.sample_size,
        names=names,
        sample_rate=args.sample_rate,
        num_workers=args.num_workers,
        recursive=True,
        random_crop=True,
        epoch_steps=10000,
    )

    wandb_logger = pl.loggers.WandbLogger(project=args.name)
    
    exc_callback = ExceptionCallback()
    ckpt_callback = pl.callbacks.ModelCheckpoint(every_n_train_steps=args.checkpoint_every, save_top_k=-1)
    demo_callback = DemoCallback(train_dl, args)

    model = AudioVAE(args)

    if args.pretrained_ckpt_path:
        pretrained_state_dict = torch.load(args.pretrained_ckpt_path)["state_dict"]
        model.load_pretrained_ae(pretrained_state_dict)
        del pretrained_state_dict
       

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

    trainer.fit(model, train_dl, ckpt_path=args.ckpt_path)

if __name__ == '__main__':
    main()

