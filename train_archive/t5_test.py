#!/usr/bin/env python3
from prefigure.prefigure import get_all_args, push_wandb_config
import argparse

import torch
from torch.utils import data
from torch import nn
from tqdm import tqdm

from dataset.dataset import SampleDataset

from encoders.t5 import t5_encode_text, get_encoded_dim, DEFAULT_T5_NAME

from encoders.perceiver_resampler import PerceiverResampler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    args = get_all_args()

    train_set = SampleDataset([args.training_dir], args)
    train_dl = data.DataLoader(train_set, args.batch_size, shuffle=True,
                               num_workers=args.num_workers, persistent_workers=True, pin_memory=True, drop_last=True)



    text_embed_dim = get_encoded_dim(DEFAULT_T5_NAME)

    max_text_len = 256

    cond_dim = 512

    cond_mapping = nn.Linear(text_embed_dim, cond_dim, device=device)

    resampler = PerceiverResampler(
        dim=cond_dim, 
        depth=2, 
        dim_head=64,
        heads=8,
        num_latents = 512,
        num_latents_mean_pooled= 1
    ).to(device)

    try:
        
        for batch in tqdm(train_dl):
            
            audios, filenames = batch
            
            audios.to(device)
           
            print(filenames)
            t5_encodings, attn_masks = t5_encode_text(filenames)
            print(f'T5 output shape: {t5_encodings.shape}')

            t5_encodings = cond_mapping(t5_encodings)
            print(f'mapping output shape: {t5_encodings.shape}')

            t5_encodings = resampler(t5_encodings)
            print(f'Resampler output shape: {t5_encodings.shape}')
            
              
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()