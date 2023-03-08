import torch
from torch import nn
from torch.nn import functional as F
from functools import partial
from blocks.blocks import DBlock, MappingNet, SkipBlock, FourierFeatures, UBlock, UNet, expand_to_planes, SelfAttention1d, ResConvBlock,OutConvBlock, Downsample1d, Upsample1d, Downsample1d_2, Upsample1d_2
from blocks.utils import append_dims
from diffusion.pqmf import CachedPQMF as PQMF
from einops import rearrange

from autoencoders.transformer_ae import ContinuousLocalTransformer

class DiffusionResConvUnet(nn.Module):
    def __init__(self, latent_dim, io_channels, depth=16):
        super().__init__()
        max_depth = 16
        depth = min(depth, max_depth)
        c_mults = [256, 256, 512, 512] + [512] * 12
        c_mults = c_mults[:depth]

        self.io_channels = io_channels
        self.timestep_embed = FourierFeatures(1, 16)
        block = nn.Identity()
        for i in range(depth, 0, -1):
            c = c_mults[i - 1]
            if i > 1:
                c_prev = c_mults[i - 2]
                block = SkipBlock(
                    nn.AvgPool1d(2),
                    ResConvBlock(c_prev, c, c),
                    ResConvBlock(c, c, c),
                    ResConvBlock(c, c, c),
                    block,
                    ResConvBlock(c * 2 if i != depth else c, c, c),
                    ResConvBlock(c, c, c),
                    ResConvBlock(c, c, c_prev),
                    nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
                )
            else:
                block = nn.Sequential(
                    ResConvBlock(io_channels + 16 + latent_dim, c, c),
                    ResConvBlock(c, c, c),
                    ResConvBlock(c, c, c),
                    block,
                    ResConvBlock(c * 2, c, c),
                    ResConvBlock(c, c, c),
                    ResConvBlock(c, c, io_channels, is_last=True),
                )
        self.net = block

    def forward(self, input, t, quantized):
        timestep_embed = expand_to_planes(self.timestep_embed(t[:, None]), input.shape)
        quantized = F.interpolate(quantized, (input.shape[2], ), mode='linear', align_corners=False)
        return self.net(torch.cat([input, timestep_embed, quantized], dim=1))


class DiffusionAttnUnet1D(nn.Module):
    def __init__(
        self, 
        io_channels = 2, 
        depth=14,
        n_attn_layers = 6,
        c_mults = [128, 128, 256, 256] + [512] * 10,
        cond_dim = 0,
        cond_noise_aug = False,
        pqmf_bands = 1,
        kernel_size = 5,
        learned_resample = False,
        strides = [2] * 14,
        conv_bias = True
    ):
        super().__init__()

        self.pqmf_bands = pqmf_bands

        self.cond_noise_aug = cond_noise_aug

        if self.cond_noise_aug:
            self.rng = torch.quasirandom.SobolEngine(1, scramble=True)

        if self.pqmf_bands > 1:
            self.pqmf = PQMF(2, 70, self.pqmf_bands)

        self.timestep_embed = FourierFeatures(1, 16)

        attn_layer = depth - n_attn_layers - 1

        block = nn.Identity()

        conv_block = partial(ResConvBlock, kernel_size=kernel_size, conv_bias = conv_bias)

        for i in range(depth, 0, -1):
            c = c_mults[i - 1]
            stride = strides[i-1]
            if i > 1:
                c_prev = c_mults[i - 2]
                add_attn = i >= attn_layer and n_attn_layers > 0
                block = SkipBlock(
                    Downsample1d_2(c_prev, c_prev, stride) if learned_resample else Downsample1d("cubic"),
                    conv_block(c_prev, c, c),
                    SelfAttention1d(
                        c, c // 32) if add_attn else nn.Identity(),
                    conv_block(c, c, c),
                    SelfAttention1d(
                        c, c // 32) if add_attn else nn.Identity(),
                    conv_block(c, c, c),
                    SelfAttention1d(
                        c, c // 32) if add_attn else nn.Identity(),
                    block,
                    conv_block(c * 2 if i != depth else c, c, c),
                    SelfAttention1d(
                        c, c // 32) if add_attn else nn.Identity(),
                    conv_block(c, c, c),
                    SelfAttention1d(
                        c, c // 32) if add_attn else nn.Identity(),
                    conv_block(c, c, c_prev),
                    SelfAttention1d(c_prev, c_prev //
                                    32) if add_attn else nn.Identity(),
                    Upsample1d_2(c_prev, c_prev, stride) if learned_resample else Upsample1d(kernel="cubic")
                    # nn.Upsample(scale_factor=2, mode='linear',
                    #             align_corners=False),
                )
            else:
                cond_embed_dim = 16 if not self.cond_noise_aug else 32
                block = nn.Sequential(
                    conv_block((io_channels * self.pqmf_bands + cond_dim) + cond_embed_dim, c, c),
                    conv_block(c, c, c),
                    conv_block(c, c, c),
                    block,
                    conv_block(c * 2, c, c),
                    conv_block(c, c, c),
                    conv_block(c, c, io_channels * self.pqmf_bands, is_last=True),
                )
        self.net = block

        with torch.no_grad():
            for param in self.net.parameters():
                param *= 0.5

    def forward(self, x, t, cond=None, cond_aug_scale=None):

        if self.pqmf_bands > 1:
            x = self.pqmf(x)

        timestep_embed = expand_to_planes(self.timestep_embed(t[:, None]), x.shape)
        
        inputs = [x, timestep_embed]

        if cond is not None:
            if cond.shape[2] != x.shape[2]:
                cond = F.interpolate(cond, (x.shape[2], ), mode='linear', align_corners=False)
                
            if self.cond_noise_aug:
                # Get a random number between 0 and 1, uniformly sampled
                if cond_aug_scale is None:
                    aug_level = self.rng.draw(cond.shape[0])[:, 0].to(cond)  
                else:
                    aug_level = torch.tensor([cond_aug_scale]).repeat([cond.shape[0]]).to(cond)
                    print(f"Using cond_aug_scale {aug_level}")
             

                # Add noise to the conditioning signal
                cond = cond + torch.randn_like(cond) * aug_level[:, None, None]

                # Get embedding for noise cond level, reusing timestamp_embed
                aug_level_embed = expand_to_planes(self.timestep_embed(aug_level[:, None]), x.shape)

                inputs.append(aug_level_embed)

            inputs.append(cond)

        outputs = self.net(torch.cat(inputs, dim=1))

        if self.pqmf_bands > 1:
            outputs = self.pqmf.inverse(outputs)

        return outputs

class TransformerDiffusionDecoder(nn.Module):
    def __init__(self, 
        io_channels=32, 
        cond_dim=0,
        embed_dim=768,
        depth=12,
        num_heads=8, 
        local_attn_window_size=64):

        super().__init__()
        
        # Timestep embeddings
        timestep_features_dim = 16

        self.timestep_features = FourierFeatures(1, timestep_features_dim)

        # Transformer

        self.transformer = ContinuousLocalTransformer(
            dim_in = io_channels + cond_dim + timestep_features_dim,
            dim_out = io_channels,
            dim = embed_dim,
            depth = depth,
            heads = num_heads,     
            local_attn_window_size = local_attn_window_size       
        )

    def forward(
        self, 
        x, 
        t, 
        cond=None,
        #cfg_scale=1.0,
        #cfg_dropout_prob=0.0
        ):

        # if cond_tokens is not None:

        #     cond_tokens = self.to_cond_embed(cond_tokens)

        #     # CFG dropout
        #     if cfg_dropout_prob > 0.0:
        #         null_embed = torch.zeros_like(cond_tokens, device=cond_tokens.device)
        #         dropout_mask = torch.bernoulli(torch.full((cond_tokens.shape[0], 1, 1), cfg_dropout_prob, device=cond_tokens.device)).to(torch.bool)
        #         cond_tokens = torch.where(dropout_mask, null_embed, cond_tokens)
  
        
        timestep_embed = expand_to_planes(self.timestep_features(t[:, None]), x.shape)
        
        inputs = [x, timestep_embed]

        if cond is not None:
            cond = F.interpolate(cond, (x.shape[2], ), mode='linear', align_corners=False)
            inputs.append(cond)

        x = torch.cat(inputs, dim=1)

        x = rearrange(x, "b c t -> b t c")

        # if cond_tokens is not None and cfg_scale != 1.0:
        #     # Classifier-free guidance
        #     # Concatenate conditioned and unconditioned inputs on the batch dimension            
        #     batch_inputs = torch.cat([x, x], dim=0)
            
        #     null_embed = torch.zeros_like(cond_tokens, device=cond_tokens.device)

        #     batch_timestep = torch.cat([timestep_embed, timestep_embed], dim=0)
        #     batch_cond = torch.cat([cond_tokens, null_embed], dim=0)
        #     batch_masks = torch.cat([cond_token_mask, cond_token_mask], dim=0)
            
        #     output = self.transformer(batch_inputs, prepend_embeds=batch_timestep, context=batch_cond, context_mask=batch_masks)

        #     cond_output, uncond_output = torch.chunk(output, 2, dim=0)
        #     output = uncond_output + (cond_output - uncond_output) * cfg_scale
            
        # else:
            
        output = self.transformer(x)

        output = rearrange(output, "b t c -> b c t")

        return output

class AudioDenoiserModel(nn.Module):
    def __init__(
        self, 
        c_in,
        feats_in,
        depths, 
        channels,
        self_attn_depths,
        strides,
        mapping_cond_dim=0,
        unet_cond_dim=0,
        dropout_rate=0.,
    ):
        super().__init__()
        self.timestep_embed = FourierFeatures(1, feats_in)
        if mapping_cond_dim > 0:
            self.mapping_cond = nn.Linear(mapping_cond_dim, feats_in, bias=False)
        self.mapping = MappingNet(feats_in, feats_in)
        self.proj_in = nn.Conv1d(c_in + unet_cond_dim, channels[0], 1)
        self.proj_out = nn.Conv1d(channels[0], c_in, 1)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)
        d_blocks, u_blocks = [], []
        for i in range(len(depths)):
            my_c_in = channels[i] if i==0 else channels[i-1]
            my_stride = 1 if i==0 else strides[i-1]
            d_blocks.append(DBlock(depths[i], feats_in, my_c_in, channels[i], channels[i], downsample_ratio=my_stride, self_attn=self_attn_depths[i], dropout_rate=dropout_rate))

        for i in range(len(depths)):
            my_c_in = channels[i] * 2 if i < len(depths) - 1 else channels[i]
            my_c_out = channels[i] if i == 0 else channels[i - 1]
            my_stride = 1 if i==0 else strides[i-1]
            u_blocks.append(UBlock(depths[i], feats_in, my_c_in, channels[i], my_c_out, upsample_ratio=my_stride, self_attn=self_attn_depths[i], dropout_rate=dropout_rate))
        self.u_net = UNet(d_blocks, reversed(u_blocks))

    def forward(self, input, sigma, mapping_cond=None, unet_cond=None, log_sigma = True):
        if log_sigma:
            sigma = sigma.log() / 4
        c_noise = sigma
        timestep_embed = self.timestep_embed(append_dims(c_noise, 2))
        mapping_cond_embed = torch.zeros_like(timestep_embed) if mapping_cond is None else self.mapping_cond(mapping_cond)
        

        mapping_out = self.mapping(timestep_embed + mapping_cond_embed)
        cond = {'cond': mapping_out}
        if unet_cond is not None:
            unet_cond = F.interpolate(unet_cond, (input.shape[2], ), mode='linear', align_corners=False)
            input = torch.cat([input, unet_cond], dim=1)
        input = self.proj_in(input)
        input = self.u_net(input, cond)
        input = self.proj_out(input)
        return input
        
class Denoiser(nn.Module):
    """A Karras et al. preconditioner for denoising diffusion models."""

    def __init__(self, inner_model, sigma_data=1.):
        super().__init__()
        self.inner_model = inner_model
        self.sigma_data = sigma_data

    def get_scalings(self, sigma):
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2)**0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2)**0.5
        return c_skip, c_out, c_in

    def loss(self, x, noise, sigma, **kwargs):
        c_skip, c_out, c_in = [append_dims(x, x.ndim) for x in self.get_scalings(sigma)]
        noised_input = x + noise * append_dims(sigma, x.ndim)
        model_output = self.inner_model(noised_input * c_in, sigma, **kwargs)
        target = (x - c_skip * noised_input) / c_out
        return (model_output - target).pow(2).flatten(1).mean(1)

    def forward(self, x, sigma, **kwargs):
        c_skip, c_out, c_in = [append_dims(x, x.ndim) for x in self.get_scalings(sigma)]
        return self.inner_model(x * c_in, sigma, **kwargs) * c_out + x * c_skip
