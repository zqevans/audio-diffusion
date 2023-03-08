#!/usr/bin/env python3

from prefigure.prefigure import get_all_args, push_wandb_config
import math, random

import sys
import torch
from torch import optim, nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities.distributed import rank_zero_only
from einops import rearrange
import numpy as np
import torchaudio
import laion_clap

from dataset.dataset import get_wds_loader

def unwrap_text(str_or_tuple):
    if type(str_or_tuple) is tuple:
        return random.choice(str_or_tuple)
    elif type(str_or_tuple) is str:
        return str_or_tuple

class ClapDurationPredictor(pl.LightningModule):
    def __init__(self, clap_model: laion_clap.CLAP_Module):
        super().__init__()

        self.clap_model = clap_model

        # CLAP embeddings are 512-dim
        self.embedding_features = 512

        self.hidden_dim = 1024

        self.max_seconds = 512

        self.to_seconds_embed = nn.Sequential(
            nn.Linear(self.embedding_features, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.max_seconds + 1),
            nn.ReLU(),
            nn.Softmax(dim=-1)
        )

    def get_clap_features(self, prompts, layer_ix=-2):
        prompt_tokens = self.clap_model.tokenizer(prompts)
        prompt_features = self.clap_model.model.text_branch(
            input_ids=prompt_tokens["input_ids"].to(device=self.device, non_blocking=True),
            attention_mask=prompt_tokens["attention_mask"].to(
                device=self.device, non_blocking=True
            ),
            output_hidden_states=True
        )["hidden_states"][layer_ix]

        masks = prompt_tokens["attention_mask"].to(device=self.device, non_blocking=True)

        return prompt_features, masks

    def configure_optimizers(self):
        return optim.Adam([*self.to_seconds_embed.parameters()], lr=4e-5)

    def training_step(self, batch, batch_idx):
        _, jsons, _ = batch

        condition_strings = [unwrap_text(json["prompt"][0]) for json in jsons]

        seconds_totals = [json["seconds_total"][0] for json in jsons]

        seconds_totals = torch.tensor(seconds_totals).to(self.device)
        seconds_totals = seconds_totals.clamp(0, self.max_seconds)

        with torch.no_grad():
            # Get text embeds
            text_embeddings = self.clap_model.get_text_embedding(condition_strings, use_tensor=True)

        second_predictions = self.to_seconds_embed(text_embeddings)
        
        seconds_totals_one_hot = F.one_hot(seconds_totals, num_classes=self.max_seconds + 1).float()

        loss = F.binary_cross_entropy(seconds_totals_one_hot, second_predictions)

        log_dict = {
            'train/loss': loss.detach(),
        }

        self.log_dict(log_dict, prog_bar=True, on_step=True)
        return loss

class ExceptionCallback(pl.Callback):
    def on_exception(self, trainer, module, err):
        print(f'{type(err).__name__}: {err}')


def main():

    args = get_all_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    torch.manual_seed(args.seed)

    names = [
    ]

    train_dl = get_wds_loader(
        batch_size=args.batch_size, 
        s3_url_prefix=None, 
        sample_size=args.sample_size, 
        names=names, 
        sample_rate=args.sample_rate, 
        num_workers=args.num_workers, 
        recursive=True,
        random_crop=True,
        epoch_steps=10000
    )

    exc_callback = ExceptionCallback()
    ckpt_callback = pl.callbacks.ModelCheckpoint(every_n_train_steps=args.checkpoint_every, save_top_k=-1)
    clap_model = laion_clap.CLAP_Module(enable_fusion=args.clap_fusion, device=device, amodel= args.clap_amodel).requires_grad_(False).eval()

    if args.clap_ckpt_path:
        clap_model.load_ckpt(ckpt=args.clap_ckpt_path)
    else:
        clap_model.load_ckpt(model_id=1)

    duration_predictor = ClapDurationPredictor(clap_model)

    wandb_logger = pl.loggers.WandbLogger(project=args.name)
    wandb_logger.watch(duration_predictor)
    push_wandb_config(wandb_logger, args)

    diffusion_trainer = pl.Trainer(
        devices=args.num_gpus,
        accelerator="gpu",
        num_nodes = args.num_nodes,
        strategy='ddp_find_unused_parameters_false',
        #precision=16,
        accumulate_grad_batches=args.accum_batches, 
        callbacks=[ckpt_callback, exc_callback],
        logger=wandb_logger,
        log_every_n_steps=1,
        max_epochs=10000000,
        default_root_dir=args.save_dir
    )

    diffusion_trainer.fit(duration_predictor, train_dl, ckpt_path=args.ckpt_path)

if __name__ == '__main__':
    main()

