## Modified from https://github.com/wesbz/SoundStream/blob/main/net.py
import numpy as np
import torch
import torch.nn as nn

from blocks.blocks import Downsample1d, SelfAttention1d, ResConvBlock, Upsample1d

class AttnResEncoder1D(nn.Module):
    def __init__(
        self,  
        n_io_channels=2, 
        latent_dim=256,
        depth=8, 
        n_attn_layers = 5, 
        c_mults = [256, 512, 1024, 1024, 1024, 1024, 1024, 1024]
    ):
        super().__init__()

        max_depth = 12
        depth = min(depth, max_depth)
                
        self.act = torch.tanh
        
        c_mults = c_mults[:depth]

        conv_block = ResConvBlock

        attn_start_layer = depth - n_attn_layers - 1

        c = c_mults[0]
        layers = [nn.Sequential(
                    conv_block(n_io_channels, c, c),
                    conv_block(c, c, c),
                    conv_block(c, c, c),
                    conv_block(c, c, c),
                )]

        for i in range(1, depth):
            c = c_mults[i]
            c_prev = c_mults[i - 1]
            add_attn = i >= attn_start_layer and n_attn_layers > 0
            layers.append(nn.Sequential(
                Downsample1d(kernel="cubic"),
                conv_block(c_prev, c, c),
                SelfAttention1d(
                    c, c // 32) if add_attn else nn.Identity(),
                conv_block(c, c, c),
                SelfAttention1d(
                    c, c // 32) if add_attn else nn.Identity(),
                conv_block(c, c, c),
                SelfAttention1d(
                    c, c // 32) if add_attn else nn.Identity(),
                conv_block(c, c, c),
                SelfAttention1d(
                    c, c // 32) if add_attn else nn.Identity(),
            ))
        

        layers.append(nn.Sequential(
                    conv_block(c, c, c),
                    conv_block(c, c, c),
                    conv_block(c, c, c),
                    conv_block(c, c, latent_dim, is_last=True)
                    )
                )

        self.net = nn.Sequential(*layers)

        with torch.no_grad():
            for param in self.net.parameters():
                param *= 0.5

    def forward(self, input):
        return self.act(self.net(input))

class AttnResDecoder1D(nn.Module):
    def __init__(
        self, 
        n_io_channels=2, 
        latent_dim=256,
        depth=8, 
        n_attn_layers = 5, 
        c_mults = [256, 512, 1024, 1024, 1024, 1024, 1024, 1024]
    ):
        super().__init__()

        max_depth = 12
        depth = min(depth, max_depth)
                
        self.act = torch.tanh
        
        c_mults = c_mults[:depth]

        c_mults = c_mults[::-1]

        conv_block = ResConvBlock

        c = c_mults[0]
        layers = [nn.Sequential(
                    conv_block(latent_dim, c, c),
                    conv_block(c, c, c),
                    conv_block(c, c, c),
                    conv_block(c, c, c),
                )]

        for i in range(1, depth):
            c = c_mults[i]
            c_prev = c_mults[i - 1]
            add_attn = i < n_attn_layers and n_attn_layers > 0
            layers.append(nn.Sequential(
                conv_block(c_prev, c, c),
                SelfAttention1d(
                    c, c // 32) if add_attn else nn.Identity(),
                conv_block(c, c, c),
                SelfAttention1d(
                    c, c // 32) if add_attn else nn.Identity(),
                conv_block(c, c, c),
                SelfAttention1d(
                    c, c // 32) if add_attn else nn.Identity(),
                conv_block(c, c, c),
                SelfAttention1d(
                    c, c // 32) if add_attn else nn.Identity(),
                Upsample1d(kernel="cubic"),
            ))
        

        layers.append(nn.Sequential(
                    conv_block(c, c, c),
                    conv_block(c, c, c),
                    conv_block(c, c, c),
                    conv_block(c, c, n_io_channels, is_last=True)
                    )
                )

        self.net = nn.Sequential(*layers)

        with torch.no_grad():
            for param in self.net.parameters():
                param *= 0.5

    def forward(self, input):
        return self.act(self.net(input))