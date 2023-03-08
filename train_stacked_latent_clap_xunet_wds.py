#!/usr/bin/env python3

from prefigure.prefigure import get_all_args, push_wandb_config
from contextlib import contextmanager
from copy import deepcopy
import math
from pathlib import Path

import sys, os
import random
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
import socket

import wandb

from autoencoders.models import AudioAutoencoder
from audio_encoders_pytorch import Encoder1d
from ema_pytorch import EMA
from audio_diffusion_pytorch import NumberEmbedder
from diffusion.xunet import UNetV0
import laion_clap

from decoders.diffusion_decoder import DiffusionAttnUnet1D
from diffusion.model import ema_update
from aeiou.viz import embeddings_table, pca_point_cloud, audio_spectrogram_image, tokens_spectrogram_image
from dataset.dataset import get_wds_loader
from blocks.utils import InverseLR
from diffusion.sampling import sample

from prompts.prompters import get_prompt_from_jmann_metadata, get_prompt_from_fma_metadata, get_prompt_from_audio_file_metadata

# Define the noise schedule and sampling loop
def get_alphas_sigmas(t):
    """Returns the scaling factors for the clean image (alpha) and for the
    noise (sigma), given a timestep."""
    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)

def alpha_sigma_to_t(alpha, sigma):
    """Returns a timestep, given the scaling factors for the clean image and for
    the noise."""
    return torch.atan2(sigma, alpha) / math.pi * 2

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

        self.autoencoder.requires_grad_(False).eval()

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

    def load_ema_weights(self, ema_state_dict):
        own_state = self.state_dict()
        for name, param in ema_state_dict.items():
            if name.startswith("latent_encoder_ema."):
                new_name = name.replace("latent_encoder_ema.", "latent_encoder.")
                if isinstance(param, Parameter):
                    # backwards compatibility for serialized parameters
                    param = param.data
                own_state[new_name].copy_(param)
            if name.startswith("diffusion_ema."):
                new_name = name.replace("diffusion_ema.", "diffusion.")
                if isinstance(param, Parameter):
                    # backwards compatibility for serialized parameters
                    param = param.data
                own_state[new_name].copy_(param)

def unwrap_text(str_or_tuple):
    if type(str_or_tuple) is tuple:
        return random.choice(str_or_tuple)
    elif type(str_or_tuple) is str:
        return str_or_tuple

class StackedAELatentDiffusionCond(pl.LightningModule):
    def __init__(self, latent_ae: LatentAudioDiffusionAutoencoder, clap_module: laion_clap.CLAP_Module):
        super().__init__()

        self.latent_dim = latent_ae.second_stage_latent_dim
        self.downsampling_ratio = latent_ae.downsampling_ratio

        embedding_max_len = 1

        self.embedder = clap_module

        self.embedding_features = 512

       # self.timestamp_embedder = NumberEmbedder(features=self.embedding_features)

        self.diffusion = UNetV0(
            dim = 1,
            in_channels = self.latent_dim,
            channels = [512, 768, 1024, 1024, 1024, 1024],
            factors = [1, 2, 2, 4, 4],
            items = [3, 3, 3, 3, 3],
            attentions = [0, 0, 2, 2, 2],
            cross_attentions = [0, 0, 2, 2, 2],
            attention_features = 64,
            attention_heads = 16,
            embedding_features = self.embedding_features,
            embedding_max_length = embedding_max_len,
            use_modulation = True,
            use_embedding_cfg = True,
            out_channels = self.latent_dim,
            use_text_conditioning = True,
        )

        self.diffusion_ema = EMA(
            self.diffusion,
            beta = 0.9999,
            power=3/4,
            update_every = 1,
            update_after_step = 1
        )

        self.autoencoder = latent_ae

        self.autoencoder.requires_grad_(False)

        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)

    def encode(self, reals):
        return self.autoencoder.encode(reals)

    def decode(self, latents, steps=100):
        return self.autoencoder.decode(latents, steps, device=self.device)

    def configure_optimizers(self):
        #optimizer = optim.Adam([*self.diffusion.parameters(), *self.timestamp_embedder.parameters()], lr=4e-5)
        optimizer = optim.Adam([*self.diffusion.parameters()], lr=4e-5)

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-6)

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        reals, jsons, timestamps = batch
        reals = reals[0]

        #timestamps = [[timestamp[0].item(), timestamp[1].item()] for timestamp in timestamps]

        #condition_strings = [unwrap_text(json["text"][0]) for json in jsons]

        condition_strings = [json["prompt"][0] for json in jsons]

        #print(condition_strings)

        with torch.cuda.amp.autocast():
            #timestamp_embeddings = self.timestamp_embedder(timestamps)

            with torch.no_grad():
                latents = self.encode(reals)
                text_embeddings = self.embedder.get_text_embedding(condition_strings)
                text_embeddings = torch.from_numpy(text_embeddings).unsqueeze(1).to(self.device)
                #print(text_embeddings.shape)

        embeddings = text_embeddings #torch.cat([text_embeddings, timestamp_embeddings], dim=1)

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
            'train/lr': self.lr_schedulers().get_last_lr()[0],
            'train/ema_decay': self.diffusion_ema.get_current_decay()
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

        torch.cuda.empty_cache()

        #module.autoencoder.autoencoder.decoder = module.autoencoder.autoencoder.decoder.to("cpu")

        try:
            latent_noise = torch.randn([4, module.latent_dim, self.demo_samples//module.downsampling_ratio]).to(module.device)


            text_embeddings = module.embedder.get_text_embedding([
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
            ])

            text_embeddings = torch.from_numpy(text_embeddings).unsqueeze(1).to(module.device)

            embeddings = text_embeddings

            demo_cfg_scales = [2, 4, 6]

            for cfg_scale in demo_cfg_scales:
                print(f"Generating latents, CFG scale {cfg_scale}")
                fake_latents = sample(module.diffusion_ema, latent_noise, self.demo_steps, 0, embedding=embeddings, embedding_scale=cfg_scale)

                fake_latents = fake_latents.clamp(-1, 1)

                torch.cuda.empty_cache()

                print(f"Decoding latents, shape: {fake_latents.shape}")
                fakes = module.decode(fake_latents, steps=100)

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

    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"CUDA is available on {socket.gethostname()} device.")
    else:
        # Print the hostname if CUDA is not available
        print(f"CUDA is not available on this device. Hostname: {socket.gethostname()}")

    names = [
    ]

    metadata_prompt_funcs = {}

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
        metadata_prompt_funcs=metadata_prompt_funcs,
    )

    exc_callback = ExceptionCallback()
    demo_callback = DemoCallback(args)

    first_stage_config = {"capacity": 64, "c_mults": [2, 4, 8, 16, 32], "strides": [2, 2, 2, 2, 2], "latent_dim": 32}

    first_stage_autoencoder = AudioAutoencoder(
        **first_stage_config
    ).eval()

    if args.ckpt_path:
        latent_diffae = LatentAudioDiffusionAutoencoder(first_stage_autoencoder).eval()
    else:
        latent_diffae = LatentAudioDiffusionAutoencoder.load_from_checkpoint(args.pretrained_ckpt_path, autoencoder=first_stage_autoencoder, strict=False).eval()

    latent_diffae.diffusion = latent_diffae.diffusion_ema
    del latent_diffae.diffusion_ema

    latent_diffae.latent_encoder = latent_diffae.latent_encoder_ema
    del latent_diffae.latent_encoder_ema

    ckpt_dir = f"{args.name}/{args.run_name}/checkpoints" if args.run_name else None

    ckpt_callback = pl.callbacks.ModelCheckpoint(dirpath=ckpt_dir, every_n_train_steps=args.checkpoint_every, save_top_k=-1, save_last=True)

    if args.run_name:
        ckpt_path = os.path.join(ckpt_dir, "last.ckpt")
        print(f"Looking for latest checkpoint in {ckpt_path}")
        if os.path.exists(ckpt_path):
            print("Found latest checkpoint")
            args.ckpt_path = ckpt_path

    clap_model = laion_clap.CLAP_Module(enable_fusion=args.clap_fusion, device=device, amodel= args.clap_amodel).requires_grad_(False).eval()

    if args.clap_ckpt_path:
        clap_model.load_ckpt(ckpt=args.clap_ckpt_path)
    else:
        clap_model.load_ckpt(model_id=1)

    # We don't need the audio encoder taking up VRAM
    clap_model.model.audio_branch.to("cpu")
    del clap_model.model.audio_branch

    if args.ckpt_path:
        latent_diffusion_model = StackedAELatentDiffusionCond.load_from_checkpoint(args.ckpt_path, latent_ae=latent_diffae, clap_module=clap_model, strict=False)
    else:
        latent_diffusion_model = StackedAELatentDiffusionCond(latent_ae=latent_diffae, clap_module=clap_model)

    wandb_logger = pl.loggers.WandbLogger(project=args.name)
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
        default_root_dir=args.save_dir,
        #gradient_clip_val=1.0,
        #track_grad_norm=2,
        #detect_anomaly = True
    )

    diffusion_trainer.fit(latent_diffusion_model, train_dl)

if __name__ == '__main__':
    main()