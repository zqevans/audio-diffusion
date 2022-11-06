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
from autoencoders.soundstream import SoundStreamXLEncoder, SoundStreamXLDecoder
from autoencoders.models import AudioAutoencoder
from audio_encoders_pytorch import Encoder1d
from ema_pytorch import EMA
from audio_diffusion_pytorch import UNetConditional1d
from audio_diffusion_pytorch import T5Embedder
from torchaudio import transforms as T

from decoders.diffusion_decoder import DiffusionAttnUnet1D
from diffusion.model import ema_update
from aeiou.viz import embeddings_table, pca_point_cloud, audio_spectrogram_image, tokens_spectrogram_image
from aeiou.datasets import HybridAudioDataset, get_all_s3_urls, PadCrop, Stereo, PhaseFlipper

import webdataset as wds

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

        self.autoencoder.requires_grad_(False)
        
        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)

    def encode(self, reals):
        first_stage_latents = self.autoencoder.encode(reals)

        second_stage_latents = self.latent_encoder(first_stage_latents)

        second_stage_latents = torch.tanh(second_stage_latents)

        return second_stage_latents

    def decode(self, latents, steps=100, device="cuda"):
        first_stage_latent_noise = torch.randn([latents.shape[0], self.latent_dim, latents.shape[2]*self.latent_downsampling_ratio]).to(device)

        first_stage_sampled = sample(self.diffusion, first_stage_latent_noise, steps, 0, cond=latents)
        decoded = self.autoencoder.decode(first_stage_sampled)
        return decoded

class StackedAELatentDiffusionCond(pl.LightningModule):
    def __init__(self, latent_ae: LatentAudioDiffusionAutoencoder):
        super().__init__()

        self.latent_dim = latent_ae.second_stage_latent_dim
        self.downsampling_ratio = latent_ae.downsampling_ratio

        embedding_max_len = 64

        self.embedder = T5Embedder(model='t5-base', max_length=embedding_max_len).requires_grad_(False)

        self.embedding_features = 768

        self.diffusion = UNetConditional1d(
            in_channels = self.latent_dim, 
            context_embedding_features = self.embedding_features,
            context_embedding_max_length= embedding_max_len,
            channels = 256,
            patch_blocks = 1,
            patch_factor = 1,
            resnet_groups = 8,
            kernel_multiplier_downsample = 2,
            multipliers = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            factors = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            num_blocks = [3, 3, 3, 3, 3, 3, 4, 4, 4, 4],
            attentions = [0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3],
            attention_heads = 16,
            attention_features = 64,
            attention_multiplier = 4,
            attention_use_rel_pos=False,
            use_nearest_upsample = False,
            use_skip_scale = True,
            use_context_time = True,
            use_magnitude_channels = False
        )

        # with torch.no_grad():
        #     for param in self.diffusion.parameters():
        #         param *= 0.5

        self.diffusion_ema = EMA(
            self.diffusion,
            beta = 0.9999,
            power=3/4,
            update_every = 1,
            update_after_step = 1000
        )

        self.autoencoder = latent_ae

        self.autoencoder.requires_grad_(False)
        
        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)

    def encode(self, reals):
        return self.autoencoder.encode(reals)

    def decode(self, latents, steps=100):
        return self.autoencoder.decode(latents, steps, device=self.device)

    def configure_optimizers(self):
        return optim.Adam([*self.diffusion.parameters()], lr=1e-4)

    def training_step(self, batch, batch_idx):
        reals, json, condition_string = batch
        reals = reals[0]
        condition_string = [cond[0] for cond in condition_string]

        #print(condition_string)

        with torch.cuda.amp.autocast():
            with torch.no_grad():
                latents = self.encode(reals)
                embeddings = self.embedder(condition_string)

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

            embedding = module.embedder([
                "06 - Luuli - Fluux Incapacitator - 2012 [Experimental,Psycore]", 
                "08 - Interconnekted - Oneness - 2012 [Full-On,Morning]", 
                "09 - Goch - Cooking - 2011 [Darkpsy,Forest]",
                "05 - Xenofish - Paradoxal Cycle - 2015 [Downtempo,Drum 'n Bass,Twilight]", 
                "04 - Brainstalker - Self Defeating - 2016 [Hi-tech]", 
                "04 - Flembaz - Orobas (Daäna Remix) - 2015 [Progressive,Techno]", 
                "02 - Conexion Animas - Espiritus Silenciosos - 2010 [Darkpsy,Full-On,Twilight]", 
                "08 - Sphingida - Alone In Aqua Endless - 2007 [Deep Trance,Downtempo]"])

            fake_latents = sample(module.diffusion_ema, latent_noise, self.demo_steps, 0, embedding=embedding, embedding_scale=5.0)
            
            fakes = module.decode(fake_latents)

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

def wds_preprocess(sample, sample_size=1048576, sample_rate=48000, verbose=False):
    "utility routine for QuickWebDataLoader, below"
    audio_keys = ("flac", "wav", "mp3", "aiff")
    found_key, rewrite_key = '', ''
    for k,v in sample.items():  # print the all entries in dict
        for akey in audio_keys:
            if k.endswith(akey): 
                found_key, rewrite_key = k, akey  # to rename long/weird key with its simpler counterpart
                break
        if '' != found_key: break 
    if '' == found_key:  # got no audio!   
        print("  Error: No audio in this sample:")
        for k,v in sample.items():  # print the all entries in dict
            print(f"    {k:20s} {repr(v)[:50]}")
        print("       Skipping it.")
        return None  # try returning None to tell WebDataset to skip this one ?   
    
    audio, in_sr = sample[found_key]
    if in_sr != sample_rate:
        if verbose: print(f"Resampling {filename} from {in_sr} Hz to {sample_rate} Hz",flush=True)
        resample_tf = T.Resample(in_sr, sample_rate)
        audio = resample_tf(audio)        
    myop = torch.nn.Sequential(PadCrop(sample_size), Stereo(), PhaseFlipper())
    audio = myop(audio)
    if found_key != rewrite_key:   # rename long/weird key with its simpler counterpart
        del sample[found_key]
    sample[rewrite_key] = audio   

    metadata = sample["json"]

    condition_string = f'{metadata["file"][:-5]} - {metadata["year"]} [{",".join(metadata["tags"])}]'

    sample["cond"] = condition_string
     
    return sample

def main():

    args = get_all_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    torch.manual_seed(args.seed)

    #args.random_crop = False

    # train_set = AudioDataset(
    #     [args.training_dir],
        # sample_rate=args.sample_rate,
        # sample_size=args.sample_size,
        # random_crop=args.random_crop,
    #     augs='Stereo(), PhaseFlipper()'
    # )

    #args.random_crop = False

#    train_set = SampleDataset([args.training_dir], args)

    # keywords=["kick", "snare", "clap", "snap", "hat", "cymbal", "crash", "ride"]

    urls = get_all_s3_urls(names=['ekto/1'], s3_url_prefix="s3://s-harmonai/datasets/", recursive=True) 

    shuffle_vals=[10, 10]

    # train_set = HybridAudioDataset(
    #     local_paths=[],
    #     webdataset_names=[],
    #     sample_rate=args.sample_rate,
    #     sample_size=args.sample_size,
    #     random_crop=args.random_crop,
    #     augs='Stereo(), PhaseFlipper()'
    # )  

    dataset = wds.DataPipeline(
        wds.ResampledShards(urls), # Yields a single .tar URL
        wds.tarfile_to_samples(), # Opens up a stream to the TAR file, yields files grouped by keys
        #wds.shuffle(bufsize=100, initial=10), # Pulls from iterator until initial value
        wds.decode(wds.torch_audio),
        wds.map(wds_preprocess),
        wds.shuffle(bufsize=100, initial=10), # Pulls from iterator until initial value
        wds.to_tuple("flac", "json", "cond"),
        wds.batched(args.batch_size)
    ).with_epoch(5)

    train_dl = wds.WebLoader(dataset, num_workers=args.num_workers)
    #train_dl = data.DataLoader(train_set, args.batch_size, num_workers=args.num_workers, persistent_workers=True, pin_memory=True)

                                 
    wandb_logger = pl.loggers.WandbLogger(project=args.name)
    exc_callback = ExceptionCallback()
    ckpt_callback = pl.callbacks.ModelCheckpoint(every_n_train_steps=args.checkpoint_every, save_top_k=-1)
    demo_callback = DemoCallback(args)

    first_stage_config = {"capacity": 64, "c_mults": [2, 4, 8, 16, 32], "strides": [2, 2, 2, 2, 2], "latent_dim": 32}

    first_stage_autoencoder = AudioAutoencoder( 
        **first_stage_config
    ).eval()

    latent_diffae = LatentAudioDiffusionAutoencoder.load_from_checkpoint(args.pretrained_ckpt_path, autoencoder=first_stage_autoencoder, strict=False)

    latent_diffae.diffusion = latent_diffae.diffusion_ema
    del latent_diffae.diffusion_ema

    latent_diffae.latent_encoder = latent_diffae.latent_encoder_ema
    del latent_diffae.latent_encoder_ema

    latent_diffusion_model = StackedAELatentDiffusionCond(latent_diffae)

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
    )

    diffusion_trainer.fit(latent_diffusion_model, train_dl, ckpt_path=args.ckpt_path)

if __name__ == '__main__':
    main()

