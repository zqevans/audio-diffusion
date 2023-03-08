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
from torch.nn.parameter import Parameter
from tqdm import trange
import pytorch_lightning as pl
from pytorch_lightning.utilities.distributed import rank_zero_only
from einops import rearrange
import numpy as np
import torchaudio

import wandb

from diffusion.pqmf import CachedPQMF as PQMF
from archisound import ArchiSound
from ema_pytorch import EMA
from audio_diffusion_pytorch import UNetConditional1d, T5Embedder, NumberEmbedder

from diffusion.model import ema_update
from aeiou.viz import pca_point_cloud, audio_spectrogram_image, tokens_spectrogram_image
from aeiou.datasets import AudioDataset
from dataset.dataset import SampleDataset
from blocks.utils import InverseLR
from prompts.prompters import MetadataPrompter

# Define the noise schedule and sampling loop
def get_alphas_sigmas(t):
    """Returns the scaling factors for the clean image (alpha) and for the
    noise (sigma), given a timestep."""
    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)

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

class StackedAELatentDiffusionCond(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.autoencoder = ArchiSound.from_pretrained("dmae1d-ATC64-v1").requires_grad_(False)

        self.latent_dim = 32
        self.downsampling_ratio = 1024

        self.prompter = MetadataPrompter()

        embedding_max_len = 128

        self.embedder = T5Embedder(model='t5-base', max_length=embedding_max_len).requires_grad_(False)

        self.embedding_features = 768

        self.timestamp_embedder = NumberEmbedder(features=self.embedding_features)

        self.diffusion = UNetConditional1d(
            in_channels = self.latent_dim, 
            context_embedding_features = self.embedding_features,
            context_embedding_max_length = embedding_max_len + 2, #2 for timestep embeds
            channels = 256,
            patch_blocks = 1,
            patch_factor = 1,
            resnet_groups = 8,
            kernel_multiplier_downsample = 2,
            multipliers = [2, 2, 2, 2, 2, 2, 2, 2, 2,],#, 2, 2],
            factors = [2, 2, 2, 2, 2, 2, 2, 2],#, 2, 2],
            num_blocks = [3, 3, 3, 3, 3, 3, 4, 4],# 4, 4],
            attentions = [0, 3, 3, 3, 3, 3, 3, 3, 3],# 3, 3],
            attention_heads = 16,
            attention_features = 64,
            attention_multiplier = 4,
            attention_use_rel_pos=False,
            use_nearest_upsample = False,
            use_skip_scale = True,
            use_context_time = True,
            use_magnitude_channels = False
        )

        self.diffusion_ema = EMA(
            self.diffusion,
            beta = 0.9999,
            power=3/4,
            update_every = 1,
            update_after_step = 1000
        )
        
        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)

    def encode(self, reals):
        return self.autoencoder.encode(reals)

    def decode(self, latents, steps=50):
        return self.autoencoder.decode(latents, steps)

    def configure_optimizers(self):
        optimizer = optim.Adam([*self.diffusion.parameters(), *self.timestamp_embedder.parameters()], lr=1e-4)

        scheduler = InverseLR(optimizer, inv_gamma=50000, power=1/2, warmup=0.9)

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        reals, infos = batch

        prompts = [self.prompter.get_track_prompt_from_file_metadata(path) for path in infos["path"]]

       # print(prompts)

        timestamps = torch.stack(infos["timestamps"], dim=1).tolist()

      #  print(timestamps)

        timestamp_embeddings = self.timestamp_embedder(timestamps)

        with torch.cuda.amp.autocast():
            with torch.no_grad():
                latents = self.encode(reals)
                text_embeddings = self.embedder(prompts)

        embeddings = torch.cat([text_embeddings, timestamp_embeddings], dim=1)

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
                "",])

            timestamp_embeddings = module.timestamp_embedder(
                [
                    [0.45708426195742596, 0.7834296114608786], 
                    [0.5020567683088986, 0.6494035417573465], 
                    [0.49651725660710644, 0.7191410761600652], 
                    [0.08103689800013604, 0.303933299605469],
                    [0.4159263687966686, 0.8795373157655868], 
                    [0.29685109738873655, 0.7018188046652297], 
                    [0.13666634547505946, 0.8532489214454169], 
                    [0.30980256630030806, 0.5920307341289788]
                ]
            )

            embeddings = torch.cat([text_embeddings, timestamp_embeddings], dim=1)
            fake_latents = sample(module.diffusion_ema, latent_noise, self.demo_steps, 0, embedding=embeddings, embedding_scale=5.0)
            
            print(f"Decoding latents, shape: {fake_latents.shape}")
            fakes = module.decode(fake_latents)


            print("Rearranging demos")
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

            log_dict[f'embeddings_3dpca'] = pca_point_cloud(fake_latents)
            log_dict[f'embeddings_spec'] = wandb.Image(tokens_spectrogram_image(fake_latents))

            print("Done logging")
            trainer.logger.experiment.log(log_dict, step=trainer.global_step)

        except Exception as e:
            print(f'{type(e).__name__}: {e}')

def main():

    args = get_all_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    torch.manual_seed(args.seed)

    #args.random_crop = False

    # train_set = AudioDataset(
    #     [args.training_dir],
    #     sample_rate=args.sample_rate,
    #     sample_size=args.sample_size,
    #     random_crop=args.random_crop,
    #     augs='Stereo(), PhaseFlipper()'
    # )

    #args.random_crop = False

    train_set = SampleDataset([args.training_dir], args)


    train_dl = data.DataLoader(train_set, args.batch_size, shuffle=True,
                               num_workers=args.num_workers, persistent_workers=True, pin_memory=True)
    wandb_logger = pl.loggers.WandbLogger(project=args.name)
    exc_callback = ExceptionCallback()
    ckpt_callback = pl.callbacks.ModelCheckpoint(every_n_train_steps=args.checkpoint_every, save_top_k=-1)
    demo_callback = DemoCallback(args)

    latent_diffusion_model = StackedAELatentDiffusionCond()

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

    diffusion_trainer.fit(latent_diffusion_model, train_dl)

if __name__ == '__main__':
    main()

