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

import wandb

from autoencoders.models import AudioAutoencoder
from audio_encoders_pytorch import Encoder1d
from ema_pytorch import EMA
from audio_diffusion_pytorch import NumberEmbedder
from audio_diffusion_pytorch.modules import UNetCFG1d
import laion_clap

from decoders.diffusion_decoder import DiffusionAttnUnet1D
from diffusion.model import ema_update
from aeiou.viz import embeddings_table, pca_point_cloud, audio_spectrogram_image, tokens_spectrogram_image
from dataset.dataset import get_wds_loader
from blocks.utils import InverseLR

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
        #with torch.cuda.amp.autocast():
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

        self.diffusion = UNetCFG1d(
            in_channels = self.latent_dim, 
            context_embedding_features = self.embedding_features,
            context_embedding_max_length = embedding_max_len, # + 2, #2 for timestep embeds
            channels = 256,
            resnet_groups = 8,
            kernel_multiplier_downsample = 2,
            multipliers = [2, 3, 4, 5],
            factors = [1, 2, 4],
            num_blocks = [3, 3, 3],
            attentions = [1, 1, 1, 1],
            attention_heads = 16,
            attention_features = 64,
            attention_multiplier = 4,
            attention_use_rel_pos=True,
            attention_rel_pos_max_distance=512,
            attention_rel_pos_num_buckets=128,
            use_nearest_upsample = False,
            use_skip_scale = True,
            use_context_time = True,
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

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500, eta_min=1e-6)

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        reals, jsons, timestamps = batch
        reals = reals[0]

        #timestamps = [[timestamp[0].item(), timestamp[1].item()] for timestamp in timestamps]
        mono_reals = reals.mean(dim=1)

        #with torch.cuda.amp.autocast():
        #timestamp_embeddings = self.timestamp_embedder(timestamps)

        with torch.no_grad():
            latents = self.encode(reals)
            audio_embeddings = self.embedder.get_audio_embedding_from_data(mono_reals, use_tensor=True)
            audio_embeddings = audio_embeddings.unsqueeze(1).to(self.device)
            #print(audio_embeddings)
            #print(audio_embeddings.shape)

        embeddings = audio_embeddings #torch.cat([text_embeddings, timestamp_embeddings], dim=1)

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

        #with torch.cuda.amp.autocast():
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
    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        last_demo_step = -1
        if (trainer.global_step - 1) % self.demo_every != 0 or last_demo_step == trainer.global_step:
        #if trainer.current_epoch % self.demo_every != 0:
            return

        last_demo_step = trainer.global_step

        print("Starting demo")

        training_device = module.device
        offload_device = "cpu"

        try:
            demo_reals, _, _ = next(self.demo_dl)

            demo_reals = demo_reals[0]

            demo_reals = demo_reals.to(module.device)

            latent_noise = torch.randn([demo_reals.shape[0], module.latent_dim, self.demo_samples//module.downsampling_ratio]).to(module.device)

            audio_embeddings = module.embedder.get_audio_embedding_from_data(demo_reals.mean(dim=1), use_tensor=True)
            audio_embeddings = audio_embeddings.unsqueeze(1).to(module.device)
            

            embeddings = audio_embeddings

            demo_cfg_scales = [2, 4, 6]

            log_dict = {}

            reals_filename = f'reals_{trainer.global_step:08}.wav'
            demo_reals = rearrange(demo_reals, 'b d n -> d (b n)')
            demo_reals = demo_reals.clamp(-1, 1).mul(32767).to(torch.int16).cpu()
            torchaudio.save(reals_filename, demo_reals, self.sample_rate)

            log_dict['real'] = wandb.Audio(reals_filename,
                                                sample_rate=self.sample_rate,
                                                caption=f'Real')

            log_dict[f'real_melspec_left'] = wandb.Image(audio_spectrogram_image(demo_reals))

            trainer.logger.experiment.log(log_dict, step=trainer.global_step)

            for cfg_scale in demo_cfg_scales:
                print(f"Generating latents, CFG scale {cfg_scale}")
                fake_latents = sample(module.diffusion_ema, latent_noise, self.demo_steps, 0, embedding=embeddings, embedding_scale=cfg_scale)

                fake_latents = fake_latents.clamp(-1, 1)

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

    names = []

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
        epoch_steps=10000,
    )

    exc_callback = ExceptionCallback()
    demo_callback = DemoCallback(train_dl, args)

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

    clap_module = laion_clap.CLAP_Module(enable_fusion=args.clap_fusion, device=device, amodel= args.clap_amodel).requires_grad_(False).eval()

    if args.clap_ckpt_path:
        clap_module.load_ckpt(ckpt=args.clap_ckpt_path)
    else:
        clap_module.load_ckpt(model_id=1)

    if args.ckpt_path:
        latent_diffusion_model = StackedAELatentDiffusionCond.load_from_checkpoint(args.ckpt_path, latent_ae=latent_diffae, clap_module=clap_module, strict=False)
    else:
        latent_diffusion_model = StackedAELatentDiffusionCond(latent_ae=latent_diffae, clap_module=clap_module)

    wandb_logger = pl.loggers.WandbLogger(project=args.name)
    wandb_logger.watch(latent_diffusion_model)
    push_wandb_config(wandb_logger, args)

    diffusion_trainer = pl.Trainer(
        devices=args.num_gpus,
        accelerator="gpu",
        num_nodes = args.num_nodes,
        strategy='ddp_find_unused_parameters_false',
        precision=32,
        accumulate_grad_batches=args.accum_batches,
        callbacks=[ckpt_callback, demo_callback, exc_callback],
        logger=wandb_logger,
        log_every_n_steps=1,
        max_epochs=10000000,
        default_root_dir=args.save_dir,
    )

    diffusion_trainer.fit(latent_diffusion_model, train_dl)

if __name__ == '__main__':
    main()