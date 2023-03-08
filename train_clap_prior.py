#!/usr/bin/env python3

from prefigure.prefigure import get_all_args, push_wandb_config
from contextlib import contextmanager
from copy import deepcopy
import math, random
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
import laion_clap

import wandb

from ema_pytorch import EMA

from diffusion.transformers import DiffusionTransformer
from diffusion.model import ema_update
from aeiou.viz import embeddings_table, pca_point_cloud, audio_spectrogram_image, tokens_spectrogram_image
from dataset.dataset import get_wds_loader

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

class ClapDiffusionPrior(pl.LightningModule):
    def __init__(self, clap_module: laion_clap.CLAP_Module):
        super().__init__()

        embedding_max_len = 1

        self.embedder = clap_module

        self.embedding_features = 512

        self.diffusion = DiffusionTransformer(
            io_channels=self.latent_dim, 
            input_length = 1,
            embed_dim=1024,
            depth=16,
            num_heads=16,
            cond_token_dim=self.embedding_features,
            wavelet_levels=0
        )

        self.diffusion_ema = EMA(
            self.diffusion,
            beta = 0.9999,
            power=3/4,
            update_every = 1,
            update_after_step = 1
        )

        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)

    def get_clap_features(self, prompts, layer_ix=-2):
        prompt_tokens = self.embedder.tokenizer(prompts)
        prompt_features = self.embedder.model.text_branch(
            input_ids=prompt_tokens["input_ids"].to(device=self.device, non_blocking=True),
            attention_mask=prompt_tokens["attention_mask"].to(
                device=self.device, non_blocking=True
            ),
            output_hidden_states=True
        )["hidden_states"][layer_ix]

        masks = prompt_tokens["attention_mask"].to(device=self.device, non_blocking=True)

        return prompt_features, masks

    def configure_optimizers(self):
        return optim.Adam([*self.diffusion.parameters()], lr=4e-5)

    def training_step(self, batch, batch_idx):
        reals, jsons, _ = batch
        reals = reals[0]

        #condition_strings = [get_prompt_from_metadata(json) for json in jsons]

        condition_strings = [unwrap_text(json["text"][0]) for json in jsons]

        #print(condition_strings)

        with torch.cuda.amp.autocast():            
             with torch.no_grad():
                mono_reals = reals.mean(dim=1)
                audio_embeddings = self.embedder.get_audio_embedding_from_data(mono_reals, use_tensor=True)
                audio_embeddings = audio_embeddings.unsqueeze(1).to(self.device)

                # Get text embeds
                text_embeddings = self.embedder.get_text_embedding(condition_strings)
                text_embeddings = torch.from_numpy(text_embeddings).unsqueeze(1).to(self.device)

                # Get full text features
                text_features, masks = self.get_clap_features(condition_strings)

        embeddings = torch.cat([text_embeddings, text_features], dim=1)])

        # Create mask tensor, adding an unmasked token at the beginning for the text embedding
        masks = torch.cat([torch.ones_like(masks[:, :1]).to(torch.bool), masks], dim=1)

        # Draw uniformly distributed continuous timesteps
        t = self.rng.draw(reals.shape[0])[:, 0].to(self.device)

        # Calculate the noise schedule parameters for those timesteps
        alphas, sigmas = get_alphas_sigmas(t)

        # Combine the ground truth images and the noise
        alphas = alphas[:, None, None]
        sigmas = sigmas[:, None, None]
        noise = torch.randn_like(audio_embeddings)
        noised_audio_embed = audio_embeddings * alphas + noise * sigmas
        targets = noise * alphas - audio_embeddings * sigmas

        with torch.cuda.amp.autocast():
            # 0.1 CFG dropout
            v = self.diffusion(noised_audio_embed, t, cond_tokens=embeddings, cfg_dropout_prob = 0.1)
            mse_loss = F.mse_loss(v, targets)
            loss = mse_loss

        log_dict = {
            'train/loss': loss.detach(),
            'train/mse_loss': mse_loss.detach(),
            #'train/lr': self.lr_schedulers().get_last_lr()[0]
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

            with torch.cuda.amp.autocast():  
                text_embeddings = module.embedder.get_text_embedding([
                    "Drums/Drum Samples/Snares/Tight_Snare_1_C.wav", 
                    "Splice/Virtual Riot/Basses/Growls/wet_growl_f#.wav", 
                    "Sounds of KSHMR/Risers/Long Riser 2.wav", 
                    "Samples/Vocal Atmospheres/HDVA_Cm_1.wav", 
                    "Drum Samples/Breaks/amen_break_174.wav", 
                    "Drum Loops/Vengeance/VEC Old School Drum Loops/Loops/Old School Drum Loop 01 128.wav", 
                    "Synth Loops/Synth Lead Loops/Synth_Melody_Cm_128.wav",
                    "Samples/QL/Quannum Logic/Basses/Neuro Reese.wav"
                ])

                # text_embeddings = module.embedder([
                #     "A dog barking next to a waterfall", 
                #     "a gunshot, cartoon sound effect", 
                #     "loud running footsteps in a hallway",
                #     "a woman laughing at a restaurant, people talking nearby, recorded in New Orleans", 
                #     "the sounds of glass shattering, a window breaking", 
                #     "a car honks its horn on a busy street", 
                #     "amen break 174 BPM", 
                #     "A crowd clapping in a stadium"
                # ])
                text_embeddings = torch.from_numpy(text_embeddings).unsqueeze(1).to(module.device)

            embeddings = text_embeddings

            demo_cfg_scales = [2, 3, 4]

            for cfg_scale in demo_cfg_scales:
                print(f"Generating latents, CFG scale {cfg_scale}")
                with torch.cuda.amp.autocast():  
                    fake_latents = sample(module.diffusion_ema, latent_noise, self.demo_steps, 0, cond_tokens=embeddings, cfg_scale=cfg_scale)
                
                fake_latents = fake_latents.clamp(-1, 1)

                print(f"Decoding latents, shape: {fake_latents.shape}")
                with torch.cuda.amp.autocast():  
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

            
                trainer.logger.experiment.log(log_dict)

        except Exception as e:
            print(f'{type(e).__name__}: {e}')

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
    demo_callback = DemoCallback(args)

    first_stage_config = {"capacity": 64, "c_mults": [2, 4, 8, 16, 32], "strides": [2, 2, 2, 2, 2], "latent_dim": 32}

    first_stage_autoencoder = AudioAutoencoder( 
        **first_stage_config
    ).eval()

    #latent_diffae = LatentAudioDiffusionAutoencoder(first_stage_autoencoder)

    latent_diffae = LatentAudioDiffusionAutoencoder.load_from_checkpoint(args.pretrained_ckpt_path, autoencoder=first_stage_autoencoder, strict=False)

    latent_diffae.diffusion = latent_diffae.diffusion_ema
    del latent_diffae.diffusion_ema

    latent_diffae.latent_encoder = latent_diffae.latent_encoder_ema
    del latent_diffae.latent_encoder_ema

    clap_model = laion_clap.CLAP_Module(enable_fusion=args.clap_fusion, device=device, amodel= args.clap_amodel).requires_grad_(False).eval()

    if args.clap_ckpt_path:
        clap_model.load_ckpt(ckpt=args.clap_ckpt_path)
    else:
        clap_model.load_ckpt(model_id=1)

    latent_diffusion_model = StackedAELatentDiffusion(latent_diffae, clap_model, sample_size=args.sample_size)

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
        default_root_dir=args.save_dir
    )

    diffusion_trainer.fit(latent_diffusion_model, train_dl, ckpt_path=args.ckpt_path)

if __name__ == '__main__':
    main()

