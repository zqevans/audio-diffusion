#!/usr/bin/env python3

from prefigure.prefigure import get_all_args, push_wandb_config

import sys, os
import random
import torch
from torch import optim, nn
from torch.nn import functional as F
from torch.utils import data
from tqdm import trange
import pytorch_lightning as pl
from einops import rearrange
import numpy as np
import torchaudio

import wandb

from ema_pytorch import EMA
import laion_clap

from diffusion.model import ema_update
from dataset.dataset import get_wds_loader
from blocks.utils import InverseLR
from autoencoders.transformer_ae import TransformerEncoder1D

from diffusion.pqmf import CachedPQMF as PQMF


def unwrap_text(str_or_tuple):
    if type(str_or_tuple) is tuple:
        return random.choice(str_or_tuple)
    elif type(str_or_tuple) is str:
        return str_or_tuple

class ClapAudioEncoder(nn.Module):
    def __init__(
        self, 
        in_channels = 1,
        pqmf_bands = 32,
    ):
        super().__init__()

        self.embedding_features = 512

        self.pqmf_bands = pqmf_bands

        if self.pqmf_bands > 1:
            self.pqmf = PQMF(in_channels, 70, self.pqmf_bands)

        self.audio_encoder = TransformerEncoder1D(
            in_channels = in_channels * self.pqmf_bands,
            out_channels = self.embedding_features,
            embed_dims = [96, 192, 384, 768],
            heads = [4, 8, 16, 32],
            depths = [2, 2, 2, 12],
            ratios = [4, 4, 2, 2],
            local_attn_window_size = 64
        )

        self.pooling = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        if self.pqmf_bands > 1:
            x = self.pqmf(x)
        x = self.audio_encoder(x)
        x = self.pooling(x)
        x = x.squeeze(-1)
        x = F.normalize(x, dim=-1)
        return x

class ClapAudioEncoderTrainer(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.text_embedder = laion_clap.CLAP_Module(enable_fusion=False).requires_grad_(False).eval()

        self.text_embedder.load_ckpt(model_id=1)

        self.embedding_features = 512

        self.audio_encoder = ClapAudioEncoder()

        self.audio_encoder_ema = EMA(
            self.audio_encoder,
            beta = 0.9999,
            power=3/4,
            update_every = 1,
            update_after_step = 1
        )

    def configure_optimizers(self):
        optimizer = optim.Adam([*self.audio_encoder.parameters()], lr=4e-5)

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500, eta_min=1e-6)

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        reals, jsons, timestamps = batch
        reals = reals[0]

        # Mono input
        reals = reals.mean(1, keepdim=True)

        condition_strings = [unwrap_text(json["text"][0]) for json in jsons]

        #print(condition_strings)

        with torch.cuda.amp.autocast():

            with torch.no_grad():
                text_embeddings = self.text_embedder.get_text_embedding(condition_strings)
                text_embeddings = torch.from_numpy(text_embeddings).to(self.device)

            audio_embeddings = self.audio_encoder(reals)

            cosine_sim = F.cosine_similarity(audio_embeddings, text_embeddings, dim=1)
            loss = -cosine_sim.mean()

        log_dict = {
            'train/loss': loss.detach(),
            'train/cosine_sim': cosine_sim.mean().detach(),
            'train/lr': self.lr_schedulers().get_last_lr()[0],
            'train/ema_decay': self.audio_encoder_ema.get_current_decay()
        }

        self.log_dict(log_dict, prog_bar=True, on_step=True)
        return loss

    def on_before_zero_grad(self, *args, **kwargs):
        self.audio_encoder_ema.update()

class ExceptionCallback(pl.Callback):
    def on_exception(self, trainer, module, err):
        print(f'{type(err).__name__}: {err}')

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

    ckpt_callback = pl.callbacks.ModelCheckpoint(every_n_train_steps=args.checkpoint_every, save_top_k=-1, save_last=True)

    clap_audio_encoder = ClapAudioEncoderTrainer()

    wandb_logger = pl.loggers.WandbLogger(project=args.name)
    wandb_logger.watch(clap_audio_encoder)
    push_wandb_config(wandb_logger, args)

    pl_trainer = pl.Trainer(
        devices=args.num_gpus,
        accelerator="gpu",
        num_nodes = args.num_nodes,
        strategy='ddp',
        precision=16,
        accumulate_grad_batches=args.accum_batches,
        callbacks=[ckpt_callback, exc_callback],
        logger=wandb_logger,
        log_every_n_steps=1,
        max_epochs=10000000,
        default_root_dir=args.save_dir,
        #gradient_clip_val=1.0,
        #track_grad_norm=2,
        #detect_anomaly = True
    )

    pl_trainer.fit(clap_audio_encoder, train_dl)

if __name__ == '__main__':
    main()