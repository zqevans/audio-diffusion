#!/usr/bin/env python3

from prefigure.prefigure import get_all_args, push_wandb_config
from copy import deepcopy
import math

from test.profiler import Profiler

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

import auraloss

import wandb

from dataset.dataset import DBDataset, SampleDataset

from decoders.generators import AttnUnet1D

from diffusion.model import ema_update
from diffusion.pqmf import CachedPQMF as PQMF
from autoencoders.soundstream import SoundStreamXLEncoder
from vector_quantize_pytorch import ResidualVQ

from viz.viz import embeddings_table, pca_point_cloud, audio_spectrogram_image, tokens_spectrogram_image

LAMBDA_QUANTIZER = 1

# PQMF stopband attenuation
PQMF_ATTN = 100

class AudioAutoencoder(pl.LightningModule):
    def __init__(self, global_args, depth=8, n_attn_layers = 0):
        super().__init__()

        self.pqmf_bands = global_args.pqmf_bands

        if self.pqmf_bands > 1:
            self.pqmf = PQMF(2, PQMF_ATTN, global_args.pqmf_bands)

        capacity = 32

        c_mults = [2, 4, 8, 16, 32]
        
        strides = [4, 4, 2, 2, 2]

        self.encoder = SoundStreamXLEncoder(
            in_channels=2*global_args.pqmf_bands, 
            capacity=capacity, 
            latent_dim=global_args.latent_dim,
            c_mults = c_mults,
            strides = strides
        )
        self.encoder_ema = deepcopy(self.encoder)

        self.decoder = AttnUnet1D(
            io_channels=2 * global_args.pqmf_bands, 
            cond_dim = global_args.latent_dim, 
            n_attn_layers=0, 
            depth=10,
            c_mults=[256, 256]+[512]*12
        )

        self.decoder_ema = deepcopy(self.decoder)

        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)
        self.ema_decay = global_args.ema_decay
        
        scales = [2048, 1024, 512, 256, 128]
        hop_sizes = []
        win_lengths = []
        overlap = 0.75
        for s in scales:
            hop_sizes.append(int(s * (1 - overlap)))
            win_lengths.append(s)

        self.mrstft = auraloss.freq.MultiResolutionSTFTLoss(fft_sizes=scales, hop_sizes=hop_sizes, win_lengths=win_lengths, w_log_mag=1.0, w_lin_mag=1.0)
        self.sdstft = auraloss.freq.SumAndDifferenceSTFTLoss(fft_sizes=scales, hop_sizes=hop_sizes, win_lengths=win_lengths)

        #self.aw_fir = auraloss.perceptual.FIRFilter(filter_type="aw", fs=global_args.sample_rate)

        self.num_quantizers = global_args.num_quantizers

        if self.num_quantizers > 0:

            self.quantizer = ResidualVQ(
                dim=global_args.latent_dim,
                codebook_size=global_args.codebook_size,
                num_quantizers = global_args.num_quantizers,
                kmeans_init = True,
                kmeans_iters = 100,
                threshold_ema_dead_code = 2,
                use_cosine_sim = True,
            )

    def configure_optimizers(self):
        params = [*self.encoder.parameters(), *self.decoder.parameters()]
        if self.num_quantizers > 0:
            params.append(*self.quantizer.parameters())
        return optim.Adam(params, lr=4e-5)
  
    def on_before_zero_grad(self, *args, **kwargs):
        decay = 0.95 if self.current_epoch < 25 else self.ema_decay
        ema_update(self.decoder, self.decoder_ema, decay)
        ema_update(self.encoder, self.encoder_ema, decay)

        if self.num_quantizers > 0:
            ema_update(self.quantizer, self.quantizer_ema, decay)

    def training_step(self, batch):

        p = Profiler()

        reals = batch[0]

        encoder_input = reals

        if self.pqmf_bands > 1:
            encoder_input = self.pqmf(reals)
            p.tick("pqmf")

        # Compute the model output and the loss.
        with torch.cuda.amp.autocast():
            latents = self.encoder(encoder_input).float()

            latents = torch.tanh(latents)

            p.tick("encoder")
            
            if self.num_quantizers > 0:

                latents = rearrange(latents, 'b d n -> b n d')

                latents, _, quantizer_loss = self.quantizer(latents)

                quantizer_loss = quantizer_loss.sum(dim=-1)
        
                latents = rearrange(latents, 'b n d -> b d n')

                p.tick("quantizer")

            noise = torch.randn_like(encoder_input)

            decoded = self.decoder(noise, latents)

            p.tick("decoder")

            #Add pre-PQMF loss

            if self.pqmf_bands > 1:

                # Multi-scale STFT loss on the PQMF for multiband harmonic content
                mb_distance = self.mrstft(encoder_input, decoded)
                p.tick("mb_distance")

                decoded = self.pqmf.inverse(decoded)
                p.tick("pqmf_inverse")

            # Multi-scale mid-side STFT loss for stereo/harmonic information
            mrstft_loss = self.sdstft(reals, decoded)
            p.tick("fb_distance")
            
            loss = mrstft_loss 
            
            if self.num_quantizers > 0:
                loss += quantizer_loss

            if self.pqmf_bands > 1:
                loss += mb_distance

        log_dict = {
            'train/loss': loss.detach(),
            'train/mrstft_loss': mrstft_loss.detach(),
            #'train/aw_mse_loss': aw_mse_loss.detach(),
        }

        if self.num_quantizers > 0:
            log_dict['quantizer_loss'] = quantizer_loss.detach()

        if self.pqmf_bands > 1:
            log_dict["mb_distance"] = mb_distance.detach()

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
        self.pqmf_bands = global_args.pqmf_bands

        if self.pqmf_bands > 1:
            self.pqmf = PQMF(2, PQMF_ATTN, global_args.pqmf_bands)

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        last_demo_step = -1
        if (trainer.global_step - 1) % self.demo_every != 0 or last_demo_step == trainer.global_step:
        #if trainer.global_step % self.demo_every != 0:
            return
        
        last_demo_step = trainer.global_step

        demo_reals, _ = next(self.demo_dl)

        encoder_input = demo_reals
        
        if self.pqmf_bands > 1:
            encoder_input = self.pqmf(demo_reals)
        
        encoder_input = encoder_input.to(module.device)

        demo_reals = demo_reals.to(module.device)

        with torch.no_grad():

            latents = module.encoder(encoder_input)

            latents = torch.tanh(latents)

            if module.num_quantizers > 0:

                latents = rearrange(latents, 'b d n -> b n d')

                latents, _, _ = module.quantizer(latents)

                latents = rearrange(latents, 'b n d -> b d n')

            noise = torch.randn_like(encoder_input)

            fakes = module.decoder(noise, latents)

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

            log_dict[f'embeddings'] = embeddings_table(latents)

            log_dict[f'embeddings_3dpca'] = pca_point_cloud(latents)
            log_dict[f'embeddings_spec'] = wandb.Image(tokens_spectrogram_image(latents))

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

    if args.preprocessed_dir != "":
        train_set = DBDataset(
            args.preprocessed_dir,
            [args.training_dir],
            args
        )
    else:
        train_set = SampleDataset([args.training_dir], args)

    train_dl = data.DataLoader(train_set, args.batch_size, shuffle=True,
                               num_workers=args.num_workers, persistent_workers=True, pin_memory=True)
    wandb_logger = pl.loggers.WandbLogger(project=args.name)
    demo_dl = data.DataLoader(train_set, args.num_demos, num_workers=args.num_workers, shuffle=True)
    
    exc_callback = ExceptionCallback()
    ckpt_callback = pl.callbacks.ModelCheckpoint(every_n_train_steps=args.checkpoint_every, save_top_k=-1)
    demo_callback = DemoCallback(demo_dl, args)

    model = AudioAutoencoder(args, depth=args.depth, n_attn_layers=args.n_attn_layers)

    wandb_logger.watch(model)
    push_wandb_config(wandb_logger, args)

    trainer = pl.Trainer(
        devices=args.num_gpus,
        accelerator="gpu",
        num_nodes = args.num_nodes,
        strategy='ddp',
        precision=16,
        accumulate_grad_batches=args.accum_batches, 
        callbacks=[ckpt_callback, demo_callback, exc_callback],
        logger=wandb_logger,
        log_every_n_steps=1,
        max_epochs=10000000,
    )

    trainer.fit(model, train_dl, ckpt_path=args.ckpt_path)

if __name__ == '__main__':
    try:
        main()
    except RuntimeError as err:
        import requests
        import datetime
        ts = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        resp = requests.get('http://169.254.169.254/latest/meta-data/instance-id')
        print(f'ERROR at {ts} on {resp.text}: {type(err).__name__}: {err}', flush=True)
        raise err


