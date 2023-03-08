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

import wandb

from diffusion.pqmf import CachedPQMF as PQMF
from autoencoders.soundstream import SoundStreamXLEncoder, SoundStreamXLDecoder
from autoencoders.models import AudioAutoencoder
from audio_encoders_pytorch import Encoder1d
from ema_pytorch import EMA
from audio_diffusion_pytorch import UNet1d, T5Embedder
from a_transformers_pytorch.transformers import Resampler

from aeiou.viz import embeddings_table, pca_point_cloud, audio_spectrogram_image, tokens_spectrogram_image
from dataset.dataset import SampleDataset
from blocks.utils import InverseLR

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

class LatentAudioDiffusionAutoencoder(pl.LightningModule):
    def __init__(self, global_args, autoencoder: AudioAutoencoder, embedder: T5Embedder = None):
        super().__init__()

        
        self.latent_dim = autoencoder.latent_dim
        self.downsampling_ratio = autoencoder.downsampling_ratio

        factors = [2, 2, 2, 2]

        self.latent_downsampling_ratio = np.prod(factors)
        second_stage_latent_dim = 32

        self.latent_encoder = Encoder1d(
            in_channels=self.latent_dim, 
            out_channels = second_stage_latent_dim,
            channels = 128,
            multipliers = [1, 2, 4, 8, 8],
            factors = factors,
            num_blocks = [8, 8, 8, 8],
        )

        # Scale down the encoder parameters to avoid saturation
        with torch.no_grad():
            for param in self.latent_encoder.parameters():
                param *= 0.5

        self.latent_encoder_ema = EMA(
            self.latent_encoder,
            beta = 0.9999,
            power=3/4,
            update_every = 1,
            update_after_step = 1
        )

        embedding_max_len = 128

        self.embedder = T5Embedder(model='t5-base', max_length=embedding_max_len).requires_grad_(False)

        self.embedding_features = 768

        self.resampler = Resampler(
            features=self.embedding_features, 
            in_tokens=embedding_max_len, 
            out_tokens=1, 
            num_layers=8,
            head_features=64,
            num_heads=12,
            multiplier=4,
        )

        self.diffusion = UNet1d(
            in_channels = self.latent_dim, 
            context_features = self.embedding_features,
            context_channels = [second_stage_latent_dim],
            channels = 256,
            patch_blocks = 1,
            patch_factor = 1,
            resnet_groups = 8,
            kernel_multiplier_downsample = 2,
            multipliers = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            factors = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            num_blocks = [3, 3, 3, 3, 3, 3, 4, 4, 4, 4],
            attentions = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
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
            update_after_step = 1
        )

        self.autoencoder = autoencoder

        self.autoencoder.requires_grad_(False)
        
        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)
        self.ema_decay = global_args.ema_decay       

    def encode(self, reals):
        first_stage_latents = self.autoencoder.encode(reals)

        if self.training:
            second_stage_latents = self.latent_encoder(first_stage_latents)
        else:
            second_stage_latents = self.latent_encoder_ema(first_stage_latents)

        second_stage_latents = torch.tanh(second_stage_latents)

        return second_stage_latents

    def decode(self, latents, steps=25, **extra_args):
        if self.training:
            first_stage_sampled = sample(self.diffusion, steps, 0, latents, **extra_args)
        else:
            first_stage_sampled = sample(self.diffusion_ema, steps, 0, latents, **extra_args)
        decoded = self.autoencoder.decode(first_stage_sampled)
        return decoded

    def configure_optimizers(self):
        optimizer = optim.Adam([*self.diffusion.parameters(), *self.latent_encoder.parameters(), *self.resampler.parameters()], lr=1e-4)

        scheduler = InverseLR(optimizer, inv_gamma=50000, power=1/2, warmup=0.9)

        return [optimizer], [scheduler]

    def get_context(self, latents):
        return [F.interpolate(latents, (int(latents.shape[2] * self.latent_downsampling_ratio), ), mode='linear', align_corners=False)]

    def training_step(self, batch, batch_idx):
        reals, infos = batch

        filenames = infos["path"]

        with torch.cuda.amp.autocast():
            with torch.no_grad():
                first_stage_latents = self.autoencoder.encode(reals)
                embeddings = self.embedder(filenames)
            
        # Draw uniformly distributed continuous timesteps
        t = self.rng.draw(reals.shape[0])[:, 0].to(self.device)

        # Calculate the noise schedule parameters for those timesteps
        alphas, sigmas = get_alphas_sigmas(t)

        # Combine the ground truth images and the noise
        alphas = alphas[:, None, None]
        sigmas = sigmas[:, None, None]
        noise = torch.randn_like(first_stage_latents)
        noised_latents = first_stage_latents * alphas + noise * sigmas
        targets = noise * alphas - first_stage_latents * sigmas

        with torch.cuda.amp.autocast():

            features = self.resampler(embeddings).squeeze()

            second_stage_latents = self.latent_encoder(first_stage_latents).float()

            second_stage_latents = torch.tanh(second_stage_latents)

            v = self.diffusion(noised_latents, t, features=features, channels_list=self.get_context(second_stage_latents))
            loss = F.mse_loss(v, targets)

        log_dict = {
            'train/loss': loss.detach(),
            'train/lr': self.lr_schedulers().get_last_lr()[0]
        }

        self.log_dict(log_dict, prog_bar=True, on_step=True)
        return loss

    def on_before_zero_grad(self, *args, **kwargs):
        self.latent_encoder_ema.update()
        self.diffusion_ema.update()

class ExceptionCallback(pl.Callback):
    def on_exception(self, trainer, module, err):
        print(f'{type(err).__name__}: {err}', file=sys.stderr)


class DemoCallback(pl.Callback):
    def __init__(self, demo_dl, global_args):
        super().__init__()
        self.demo_every = global_args.demo_every
        self.demo_samples = global_args.sample_size
        self.demo_steps = global_args.demo_steps
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
        try:

            demo_reals, infos = next(self.demo_dl)
            
            demo_filenames = infos["path"]
            print(demo_filenames)

            demo_reals = demo_reals.to(module.device)
            #demo_filenames = demo_filenames.to(module.device)

            with torch.no_grad():
                first_stage_latents = module.autoencoder.encode(demo_reals)
                second_stage_latents = module.latent_encoder(first_stage_latents)
                second_stage_latents = torch.tanh(second_stage_latents)
                embedding = module.embedder(list(demo_filenames))
                features = module.resampler(embedding).squeeze()

                latent_noise = torch.randn([self.num_demos, module.latent_dim, self.demo_samples//module.downsampling_ratio]).to(module.device)
                
                #demo_cfg_scales = [0, 1, 3, 5]

                #for cfg_scale in demo_cfg_scales:
                #print(f'Sampling with CFG scale {cfg_scale}')
                recon_latents = sample(module.diffusion, 
                                latent_noise, 
                                self.demo_steps, 
                                0, 
                                channels_list=module.get_context(second_stage_latents),
                                features=features)

                print("Reconstructing")
                reconstructed = module.autoencoder.decode(recon_latents)

            # Put the demos together
            reconstructed = rearrange(reconstructed, 'b d n -> d (b n)')
            
            log_dict = {}
            
            print("Saving files")
            filename = f'recon_demo_{trainer.global_step:08}.wav'
            reconstructed = reconstructed.clamp(-1, 1).mul(32767).to(torch.int16).cpu()
            torchaudio.save(filename, reconstructed, self.sample_rate)

            log_dict[f'recon_melspec_left'] = wandb.Image(audio_spectrogram_image(reconstructed))

            log_dict[f'recon'] = wandb.Audio(filename,
                                            sample_rate=self.sample_rate,
                                            caption=f'Reconstructed')


            demo_reals = rearrange(demo_reals, 'b d n -> d (b n)')

            reals_filename = f'reals_{trainer.global_step:08}.wav'
            demo_reals = demo_reals.clamp(-1, 1).mul(32767).to(torch.int16).cpu()
            torchaudio.save(reals_filename, demo_reals, self.sample_rate)

            
        
            log_dict[f'real'] = wandb.Audio(reals_filename,
                                                sample_rate=self.sample_rate,
                                                caption=f'Real')

            log_dict[f'embeddings_3dpca'] = pca_point_cloud(second_stage_latents, output_type="plotly", mode="lines+markers")
            log_dict[f'embeddings_spec'] = wandb.Image(tokens_spectrogram_image(second_stage_latents))

            log_dict[f'real_melspec_left'] = wandb.Image(audio_spectrogram_image(demo_reals))


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

    train_set = SampleDataset([args.training_dir], args, relpath=args.training_dir)

    train_dl = data.DataLoader(train_set, args.batch_size, shuffle=True,
                               num_workers=args.num_workers, persistent_workers=True, pin_memory=True)
    wandb_logger = pl.loggers.WandbLogger(project=args.name)
    demo_dl = data.DataLoader(train_set, args.num_demos, num_workers=args.num_workers, shuffle=True)
    exc_callback = ExceptionCallback()
    ckpt_callback = pl.callbacks.ModelCheckpoint(every_n_train_steps=args.checkpoint_every, save_top_k=-1)
    demo_callback = DemoCallback(demo_dl, args)

    first_stage_config = {"capacity": 64, "c_mults": [2, 4, 8, 16, 32], "strides": [2, 2, 2, 2, 2], "latent_dim": 32}

    first_stage_autoencoder = AudioAutoencoder( 
        **first_stage_config
    ).requires_grad_(False)

    first_stage_autoencoder.load_state_dict(torch.load(args.pretrained_ckpt_path)["state_dict"], strict=False)

    if args.ckpt_path:
        latent_diffusion_model = LatentAudioDiffusionAutoencoder.load_from_checkpoint(args.ckpt_path, global_args=args, autoencoder=first_stage_autoencoder, strict=False)
    else:
        latent_diffusion_model = LatentAudioDiffusionAutoencoder(args, first_stage_autoencoder)
    wandb_logger.watch(latent_diffusion_model)
    push_wandb_config(wandb_logger, args)

    diffusion_trainer = pl.Trainer(
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
        default_root_dir=args.save_dir
    )

    diffusion_trainer.fit(latent_diffusion_model, train_dl)

if __name__ == '__main__':
    main()

