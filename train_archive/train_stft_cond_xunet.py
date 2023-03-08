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
import random

import wandb

from diffusion.pqmf import CachedPQMF as PQMF
from audio_encoders_pytorch import STFT
from ema_pytorch import EMA
from blocks.utils import InverseLR
from a_unet import TimeConditioningPlugin, ClassifierFreeGuidancePlugin, T5Embedder, NumberEmbedder
from a_unet.apex import (
    XUNet,
    XBlock,
    ResnetItem as ResItem,
    AttentionItem as AttnItem,
    CrossAttentionItem as CrossAttnItem,
    ModulationItem as ModItem,
    FeedForwardItem,
    SkipCat
)
from torchaudio import transforms as T

from diffusion.model import ema_update
from aeiou.viz import embeddings_table, pca_point_cloud, audio_spectrogram_image, tokens_spectrogram_image
#from aeiou.datasets import HybridAudioDataset, get_all_s3_urls, PadCrop, Stereo, PhaseFlipper
from dataset.dataset import get_laion_630k_loader, get_wds_loader, SampleDataset

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

def unwrap_text(str_or_tuple):
    if type(str_or_tuple) is tuple:
        return random.choice(str_or_tuple)
    elif type(str_or_tuple) is str:
        return str_or_tuple

class STFTDiffusionCond(pl.LightningModule):
    def __init__(self):
        super().__init__()


        embedding_max_len = 128

        self.embedder = T5Embedder(model='t5-base', max_length=embedding_max_len).requires_grad_(False)

        self.embedding_features = 768


        channels = [128, 128, 256, 512, 512]

        factors = [1, 2, 2, 4, 4]

        num_blocks = [3, 3, 3, 3, 4]

        acfs = [0, 0, 0, 0, 1]

        cfs = [0, 0, 1, 1, 0]
        
        # channels = [512]*11

        # factors = [2] * 10 + [1] 

        # num_blocks = [3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4]

        # acfs = [0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3]

        # cfs = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        block_defs = channels, factors, num_blocks, acfs, cfs

        # Add time conditioning and CFG
        UNet = ClassifierFreeGuidancePlugin(XUNet, embedding_max_length=embedding_max_len)
        UNet = TimeConditioningPlugin(UNet)

        self.diffusion = UNet(
            dim=2,
            in_channels = 4, 
            out_channels = 4,
            embedding_features = self.embedding_features,
            modulation_features=1024,
            resnet_groups = 8,
            blocks = [
                XBlock(
                    channels = num_channels,
                    factor = factor,
                    items=([ResItem, ModItem] * num_blocks + [AttnItem, CrossAttnItem, FeedForwardItem] * num_acfs + [CrossAttnItem, FeedForwardItem] * num_cross_attentions)
                ) for num_channels, factor, num_blocks, num_acfs, num_cross_attentions in zip(*block_defs)
            ],
            attention_heads = 16,
            attention_features = 64,
            attention_multiplier = 4,
            skip_t=SkipCat
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


        self.stft = STFT(num_fft=1023, hop_length=256, use_complex = True)
        
        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)

    def configure_optimizers(self):
        optimizer = optim.Adam([*self.diffusion.parameters()], lr=1e-4)

        scheduler = InverseLR(optimizer, inv_gamma=500, power=1/2, warmup=0.75)

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        # reals, jsons, timestamps = batch
        # reals = reals[0]

        reals, infos = batch
        
        condition_string = infos["path"] #[unwrap_text(json["text"][0]) for json in jsons]

        #timestamps = [[timestamp[0].item(), timestamp[1].item()] for timestamp in timestamps]

        #print(condition_string)
        #print(timestamps)

        #timestamp_embeddings = self.timestamp_embedder(timestamps)

        stft_real, stft_imag = self.stft.encode(reals)

        stft_encoded = torch.cat([stft_real, stft_imag], dim=1)

        with torch.cuda.amp.autocast():
            with torch.no_grad():
                text_embeddings = self.embedder(condition_string)
                

        embeddings = text_embeddings #torch.cat([text_embeddings, timestamp_embeddings], dim=1)

        # Draw uniformly distributed continuous timesteps
        t = self.rng.draw(reals.shape[0])[:, 0].to(self.device)

        
        # Calculate the noise schedule parameters for those timesteps
        alphas, sigmas = get_alphas_sigmas(t)

        # Combine the ground truth images and the noise
        alphas = alphas[:, None, None, None]
        sigmas = sigmas[:, None, None, None]
        noise = torch.randn_like(stft_encoded)
        noised_stft = stft_encoded * alphas + noise * sigmas
        targets = noise * alphas - stft_encoded * sigmas

        with torch.cuda.amp.autocast():
            # 0.1 CFG dropout
            v = self.diffusion(noised_stft, time=t, embedding=embeddings, embedding_mask_proba = 0.1)
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
        print(f'{type(err).__name__}: {err}')


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
            n_bins = (module.stft.num_fft + 1)//2
            stft_noise = torch.randn([8, 4, n_bins, self.demo_samples//n_bins]).to(module.device)

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


            embeddings = text_embeddings 

            demo_cfg_scales = [2, 5, 9]

            for cfg_scale in demo_cfg_scales:
                print(f"Generating stft, CFG scale {cfg_scale}")
                fake_stft = sample(module.diffusion_ema, stft_noise, self.demo_steps, 0, embedding=embeddings, embedding_scale=cfg_scale)
                
                print(f"Decoding stft, shape: {fake_stft.shape}")
                
                fake_stft_real, fake_stft_imag = fake_stft.chunk(2, dim=1)

                fakes = module.stft.decode(fake_stft_real, fake_stft_imag)

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

            
                trainer.logger.experiment.log(log_dict, step=trainer.global_step)

        except Exception as e:
            print(f'{type(e).__name__}: {e}')

def main():

    args = get_all_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    torch.manual_seed(args.seed)
  
    names = [
   
    ]

    train_dl = get_wds_loader(batch_size=args.batch_size, s3_url_prefix=None, sample_size=args.sample_size, names=names, sample_rate=args.sample_rate, num_workers=args.num_workers, recursive=True)
                                 
    wandb_logger = pl.loggers.WandbLogger(project=args.name)
    exc_callback = ExceptionCallback()
    ckpt_callback = pl.callbacks.ModelCheckpoint(every_n_train_steps=args.checkpoint_every, save_top_k=-1)
    demo_callback = DemoCallback(args)

    if args.ckpt_path:
        stft_diffusion_model = STFTDiffusionCond.load_from_checkpoint(args.ckpt_path, strict=False)
    else:
        stft_diffusion_model = STFTDiffusionCond()

    wandb_logger.watch(stft_diffusion_model)
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

    diffusion_trainer.fit(stft_diffusion_model, train_dl)

if __name__ == '__main__':
    main()

