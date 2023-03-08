from x_transformers import ContinuousTransformerWrapper, Encoder
from einops import rearrange
import torch
from torch import nn
from encoders.wavelets import WaveletEncode1d, WaveletDecode1d
from blocks.blocks import FourierFeatures

class DiffusionTransformer(nn.Module):
    def __init__(self, 
        io_channels=32, 
        input_length=512,
        cond_token_dim=0,
        embed_dim=768,
        depth=12,
        num_heads=8,
        wavelet_levels=0):

        super().__init__()
        
        self.cond_token_dim = cond_token_dim
        self.wavelet_levels = wavelet_levels

        data_channels = io_channels

        # Wavelet decomposition
        if self.wavelet_levels > 0:
            self.wavelet_encoder = WaveletEncode1d(io_channels, "bior4.4", levels = self.wavelet_levels)
            self.wavelet_decoder = WaveletDecode1d(io_channels, "bior4.4", levels=self.wavelet_levels)
            data_channels = data_channels * (2**self.wavelet_levels)
            input_length = input_length // (2**self.wavelet_levels)


        # Timestep embeddings
        timestep_features_dim = 256

        self.timestep_features = FourierFeatures(1, timestep_features_dim)

        self.to_timestep_embed = nn.Sequential(
            nn.Linear(timestep_features_dim, embed_dim, bias=True),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim, bias=True),
        )

        if cond_token_dim > 0:
            # Conditioning tokens
            self.to_cond_embed = nn.Sequential(
                nn.Linear(cond_token_dim, embed_dim, bias=False),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim, bias=False)
            )

        # Transformer

        self.transformer = ContinuousTransformerWrapper(
            dim_in=data_channels,
            dim_out=data_channels,
            max_seq_len=input_length + 1, #1 for time conditioning
            attn_layers = Encoder(
                dim=embed_dim,
                depth=depth,
                heads=num_heads,
                cross_attend = True,
                zero_init_branch_output=True,
                rotary_pos_emb =True,
                ff_swish = True, # set this to True
                ff_glu = True 
            )
        )

        self.preprocess_conv = nn.Conv1d(data_channels, data_channels, 3, padding=1, bias=False)
        nn.init.zeros_(self.preprocess_conv.weight)
        self.postprocess_conv = nn.Conv1d(data_channels, data_channels, 3, padding=1, bias=False)
        nn.init.zeros_(self.postprocess_conv.weight)

    def forward(
        self, 
        x, 
        t, 
        cond_tokens=None,
        cond_token_mask=None,
        cfg_scale=1.0,
        cfg_dropout_prob=0.0):

        # Get the batch of timestep embeddings
        timestep_embed = self.to_timestep_embed(self.timestep_features(t[:, None])) # (b, embed_dim)

        timestep_embed = timestep_embed.unsqueeze(1)

        if cond_tokens is not None:

            cond_tokens = self.to_cond_embed(cond_tokens)

            # CFG dropout
            if cfg_dropout_prob > 0.0:
                null_embed = torch.zeros_like(cond_tokens, device=cond_tokens.device)
                dropout_mask = torch.bernoulli(torch.full((cond_tokens.shape[0], 1, 1), cfg_dropout_prob, device=cond_tokens.device)).to(torch.bool)
                cond_tokens = torch.where(dropout_mask, null_embed, cond_tokens)
                    
        if self.wavelet_levels > 0:
            x = self.wavelet_encoder(x)

        x = self.preprocess_conv(x) + x

        x = rearrange(x, "b c t -> b t c")

        if cond_tokens is not None and cfg_scale != 1.0:
            # Classifier-free guidance
            # Concatenate conditioned and unconditioned inputs on the batch dimension            
            batch_inputs = torch.cat([x, x], dim=0)
            
            null_embed = torch.zeros_like(cond_tokens, device=cond_tokens.device)

            batch_timestep = torch.cat([timestep_embed, timestep_embed], dim=0)
            batch_cond = torch.cat([cond_tokens, null_embed], dim=0)
            if cond_token_mask is not None:
                batch_masks = torch.cat([cond_token_mask, cond_token_mask], dim=0)
            else:
                batch_masks = None
            
            output = self.transformer(batch_inputs, prepend_embeds=batch_timestep, context=batch_cond, context_mask=batch_masks)

            cond_output, uncond_output = torch.chunk(output, 2, dim=0)
            output = uncond_output + (cond_output - uncond_output) * cfg_scale
            
        else:
            output = self.transformer(x, prepend_embeds=timestep_embed, context=cond_tokens, context_mask=cond_token_mask)

        output = rearrange(output, "b t c -> b c t")[:,:,1:]

        if self.wavelet_levels > 0:
            output = self.wavelet_decoder(output)

        output = self.postprocess_conv(output) + output

        return output