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
import random

import wandb

from autoencoders.soundstream import SoundStreamXLEncoder, SoundStreamXLDecoder

from quantizer_pytorch import Quantizer1d

from ema_pytorch import EMA

from audio_diffusion_pytorch import T5Embedder
from audio_diffusion_pytorch.modules import UNetCFG1d

from aeiou.viz import embeddings_table, pca_point_cloud, audio_spectrogram_image, tokens_spectrogram_image
from dataset.dataset import get_wds_loader


# Define the noise schedule and sampling loop
def get_alphas_sigmas(t):
    """Returns the scaling factors for the clean image (alpha) and for the
    noise (sigma), given a timestep."""
    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)

def get_crash_schedule(t):
    sigma = torch.sin(t * math.pi / 2) ** 2
    alpha = (1 - sigma ** 2) ** 0.5
    return alpha_sigma_to_t(alpha, sigma)

def alpha_sigma_to_t(alpha, sigma):
    """Returns a timestep, given the scaling factors for the clean image and for
    the noise."""
    return torch.atan2(sigma, alpha) / math.pi * 2

@torch.no_grad()
def sample(model, x, steps, eta, **extra_args):
    """Draws samples from a model given starting noise."""
    ts = x.new_ones([x.shape[0]])

    # Create the noise schedule
    t = torch.linspace(1, 0, steps + 1)[:-1]

    alphas, sigmas = get_alphas_sigmas(t)

    # The sampling loop
    for i in trange(steps):

        # Get the model output (v, the predicted velocity)
        with torch.cuda.amp.autocast():
            v = model(x, ts * t[i], **extra_args).float()

        # Predict the noise and the denoised image
        pred = x * alphas[i] - v * sigmas[i]
        eps = x * sigmas[i] + v * alphas[i]

        # If we are not on the last timestep, compute the noisy image for the
        # next timestep.
        if i < steps - 1:
            # If eta > 0, adjust the scaling factor for the predicted noise
            # downward according to the amount of additional noise to add
            ddim_sigma = eta * (sigmas[i + 1]**2 / sigmas[i]**2).sqrt() * \
                (1 - alphas[i]**2 / alphas[i + 1]**2).sqrt()
            adjusted_sigma = (sigmas[i + 1]**2 - ddim_sigma**2).sqrt()

            # Recombine the predicted noise and predicted denoised image in the
            # correct proportions for the next step
            x = pred * alphas[i + 1] + eps * adjusted_sigma

            # Add the correct amount of fresh noise
            if eta:
                x += torch.randn_like(x) * ddim_sigma

    # If we are on the last timestep, output the denoised image
    return pred

class AudioVAE(pl.LightningModule):
    def __init__(self):
        super().__init__()

        capacity = 32

        c_mults = [2, 4, 8, 16, 32]
        
        strides = [2, 2, 2, 2, 2]

        self.latent_dim = 32
        self.downsampling_ratio = np.prod(strides)

        self.encoder = SoundStreamXLEncoder(
            in_channels=2,
            capacity=capacity, 
            latent_dim=2*self.latent_dim,
            c_mults = c_mults,
            strides = strides
        )

        self.decoder = SoundStreamXLDecoder(
            out_channels=2, 
            capacity=capacity, 
            latent_dim=self.latent_dim,
            c_mults = c_mults,
            strides = strides
        )
  
    def sample(self, mean, scale):
        stdev = nn.functional.softplus(scale) + 1e-4
        var = stdev * stdev
        logvar = torch.log(var)
        latents = torch.randn_like(mean) * stdev + mean

        kl = (mean * mean + var - logvar - 1).sum(1).mean()

        return latents, kl

    def encode(self, audio):
        mean, scale = self.encoder(audio).chunk(2, dim=1)
        latents, _ = self.sample(mean, scale)

        return latents

    def decode(self, latents):
        decoded = self.decoder(latents)

        return decoded

def unwrap_text(str_or_tuple):
    if type(str_or_tuple) is tuple:
        return random.choice(str_or_tuple)
    elif type(str_or_tuple) is str:
        return str_or_tuple

class LatentAudioDiffusion(pl.LightningModule):
    def __init__(self, autoencoder: AudioVAE):
        super().__init__()

        self.latent_dim = autoencoder.latent_dim
        self.downsampling_ratio = autoencoder.downsampling_ratio

        embedding_max_len = 128

        self.embedder = T5Embedder(model='t5-base', max_length=embedding_max_len).requires_grad_(False)

        self.embedding_features = 768

        self.diffusion = UNetCFG1d(
            in_channels = self.latent_dim, 
            context_embedding_features = self.embedding_features,
            context_embedding_max_length = embedding_max_len + 2, #2 for timestep embeds
            channels = 256,
            resnet_groups = 1,
            kernel_multiplier_downsample = 2,
            multipliers = [2, 3, 4, 4, 4],
            factors = [1, 2, 2, 4],
            num_blocks = [3, 3, 3, 3],
            attentions = [0, 0, 3, 3, 3],
            attention_heads = 16,
            attention_features = 64,
            attention_multiplier = 4,
            attention_use_rel_pos=True,
            attention_rel_pos_max_distance=2048,
            attention_rel_pos_num_buckets=64,
            use_nearest_upsample = False,
            use_skip_scale = True,
            use_context_time = True,
        )

        # with torch.no_grad():
        #     for param in self.diffusion.parameters():
        #         param *= 0.5

        self.diffusion_ema = EMA(
            self.diffusion,
            beta = 0.9999,
            power=3/4,
            update_every = 1,
            update_after_step = 1
        )

        self.autoencoder = autoencoder

        self.autoencoder.requires_grad_(False)
        
        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)

    def encode(self, reals):
        return self.autoencoder.encode(reals)

    def decode(self, latents):
        return self.autoencoder.decode(latents)

    def configure_optimizers(self):
        optimizer = optim.Adam([*self.diffusion.parameters()], lr=1e-4)

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500, eta_min=1e-6)

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        reals, jsons, timestamps = batch
        reals = reals[0]
        
        condition_string = [unwrap_text(json["text"][0]) for json in jsons]

        with torch.cuda.amp.autocast():
            with torch.no_grad():
                latents = self.encode(reals)
                text_embeddings = self.embedder(condition_string)
                

        embeddings = text_embeddings

        # Draw uniformly distributed continuous timesteps
        t = self.rng.draw(reals.shape[0])[:, 0].to(self.device)

        # Calculate the noise schedule parameters for those timesteps
        alphas, sigmas = get_alphas_sigmas(t)

        # Combine the ground truth images and the noise
        alphas = alphas[:, None, None]
        sigmas = sigmas[:, None, None]
        noise = torch.randn_like(latents)
        noised_latents = latents * alphas + noise * sigmas
        targets = noise * alphas - latents * sigmas

        with torch.cuda.amp.autocast():
            # 0.1 CFG dropout
            v = self.diffusion(noised_latents, t, embedding=embeddings, embedding_mask_proba = 0.1)
            mse_loss = F.mse_loss(v, targets)
            loss = mse_loss

        log_dict = {
            'train/loss': loss.detach(),
            'train/mse_loss': mse_loss.detach(),
            'train/lr': self.lr_schedulers().get_last_lr()[0]
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
        self.demo_samples = global_args.sample_size
        self.demo_steps = global_args.demo_steps
        self.num_demos = global_args.num_demos
        self.sample_rate = global_args.sample_rate

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):   
        last_demo_step = -1
        if (trainer.global_step - 1) % self.demo_every != 0 or last_demo_step == trainer.global_step:
        #if trainer.current_epoch % self.demo_every != 0:
            return
        
        last_demo_step = trainer.global_step
        
        print("Starting demo")
        try:
            latent_noise = torch.randn([8, module.latent_dim, self.demo_samples//module.downsampling_ratio]).to(module.device)

            text_embeddings = module.embedder([
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
            ])

            embeddings = text_embeddings # torch.cat([text_embeddings, timestamp_embeddings], dim=1)

            demo_cfg_scales = [3, 5, 7]

            for cfg_scale in demo_cfg_scales:
                print(f"Generating latents, CFG scale {cfg_scale}")
                fake_latents = sample(module.diffusion_ema, latent_noise, self.demo_steps, 0, embedding=embeddings, embedding_scale=cfg_scale)
                
                fake_latents = fake_latents.clamp(-1, 1)

                print(f"Decoding latents, shape: {fake_latents.shape}")
                fakes = module.decode(fake_latents)

                print("Rearranging demos")
                # Put the demos together
                fakes = rearrange(fakes, 'b d n -> d (b n)')

                log_dict = {}
                
                print("Saving files")
                filename = f'demo_{trainer.global_step:08}_cfg_{cfg_scale}.wav'
                fakes = fakes.clamp(-1, 1).mul(32767).to(torch.int16).cpu()
                torchaudio.save(filename, fakes, self.sample_rate)


                log_dict[f'demo_cfg_{cfg_scale}'] = wandb.Audio(filename,
                                                    sample_rate=self.sample_rate,
                                                    caption=f'Demo CFG {cfg_scale}')
            
                log_dict[f'demo_melspec_left_{cfg_scale}'] = wandb.Image(audio_spectrogram_image(fakes))

                log_dict[f'embeddings_3dpca_{cfg_scale}'] = pca_point_cloud(fake_latents)
                log_dict[f'embeddings_spec_{cfg_scale}'] = wandb.Image(tokens_spectrogram_image(fake_latents))

            
                trainer.logger.experiment.log(log_dict, step=trainer.global_step)

        except Exception as e:
            print(f'{type(e).__name__}: {e}')
def main():

    args = get_all_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    torch.manual_seed(args.seed)

    names = [
        #"songs_raw/songs_md/",
        "samples_raw/samples_all/",
        "samples_raw/samples_comm/",
        #"samples_raw/samples_zard/",
    ]

    train_dl = get_wds_loader(
        batch_size=args.batch_size,
        s3_url_prefix="s3://s-harmonai/datasets/",
        sample_size=args.sample_size,
        names=names,
        sample_rate=args.sample_rate,
        num_workers=args.num_workers,
        recursive=True,
        random_crop=True,
        #normalize_lufs=-14.0,
        epoch_steps=10000
    )

    wandb_logger = pl.loggers.WandbLogger(project=args.name)
    
    exc_callback = ExceptionCallback()
    ckpt_callback = pl.callbacks.ModelCheckpoint(every_n_train_steps=args.checkpoint_every, save_top_k=-1)
    demo_callback = DemoCallback(args)
    autoencoder = AudioVAE.load_from_checkpoint(args.pretrained_ckpt_path, strict=False).requires_grad_(False).eval()
    latent_diffusion_model = LatentAudioDiffusion(autoencoder)
    wandb_logger.watch(latent_diffusion_model)
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
        default_root_dir=args.save_dir
    )

    diffusion_trainer.fit(latent_diffusion_model, train_dl, ckpt_path=args.ckpt_path)

if __name__ == '__main__':
    main()

