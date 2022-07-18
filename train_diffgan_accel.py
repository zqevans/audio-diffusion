#!/usr/bin/env python3

from prefigure.prefigure import get_all_args, push_wandb_config
from copy import deepcopy
import math
import auraloss
import json

import accelerate
import sys
import torch
from torch import optim, nn
from torch import multiprocessing as mp
from torch.nn import functional as F
from torch.utils import data
from tqdm import tqdm, trange
from einops import rearrange, repeat

import torchaudio

import wandb

from blocks import utils
from dataset.dataset import SampleDataset
from diffusion.pqmf import CachedPQMF as PQMF
from autoencoders.soundstream import SoundStreamXLEncoder, SoundStreamXLDecoder

from nwt_pytorch import Memcodes
from dvae.residual_memcodes import ResidualMemcodes
from decoders.diffusion_decoder import DiffusionAttnUnet1D
from diffusion.model import ema_update
from viz.viz import embeddings_table, pca_point_cloud, audio_spectrogram_image, tokens_spectrogram_image


# Define the noise schedule and sampling loop
def get_alphas_sigmas(t):
    """Returns the scaling factors for the clean image (alpha) and for the
    noise (sigma), given a timestep."""
    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)

def get_crash_schedule(t):
    sigma = torch.sin(t * math.pi / 2) ** 2
    alpha = (1 - sigma ** 2) ** 0.5
    return alpha_sigma_to_t(alpha, sigma)

def alpha_sigma_to_t(alpha, sigma):
    """Returns a timestep, given the scaling factors for the clean image and for
    the noise."""
    return torch.atan2(sigma, alpha) / math.pi * 2

@torch.no_grad()
def sample(model, x, steps, eta, logits):
    """Draws samples from a model given starting noise."""
    ts = x.new_ones([x.shape[0]])

    # Create the noise schedule
    t = torch.linspace(1, 0, steps + 1)[:-1]
    alphas, sigmas = get_alphas_sigmas(get_crash_schedule(t))

    # The sampling loop
    for i in trange(steps):

        # Get the model output (v, the predicted velocity)
        with torch.cuda.amp.autocast():
            v = model(x, ts * t[i], logits).float()

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

LAMBDA_DIFFUSION = 10

class DiffGAN(nn.Module):
    def __init__(self, global_args, device):
        super().__init__()

        self.device = device

        self.pqmf_bands = global_args.pqmf_bands

        if self.pqmf_bands > 1:
            self.pqmf = PQMF(2, 100, global_args.pqmf_bands)

        capacity = 64

        c_mults = [2, 4, 8]
        
        strides = [2, 2, 2]

        #self.encoder = AttnResEncoder1D(global_args, n_io_channels=2*global_args.pqmf_bands, depth=8, n_attn_layers=0)
        self.encoder = SoundStreamXLEncoder(
            in_channels=2*global_args.pqmf_bands, 
            capacity=capacity, 
            latent_dim=global_args.latent_dim,
            c_mults = c_mults,
            strides = strides
        )

        self.decoder = SoundStreamXLDecoder(
            out_channels=2*global_args.pqmf_bands, 
            capacity=capacity, 
            latent_dim=global_args.latent_dim,
            c_mults = c_mults,
            strides = strides
        )

        # self.encoder_ema = deepcopy(self.encoder)
        # self.decoder_ema = deepcopy(self.decoder)

        self.diffusion = DiffusionAttnUnet1D(global_args, io_channels=2*global_args.pqmf_bands, n_attn_layers=0, c_mults=[256] + [512]*12, depth=6)
        #self.diffusion_ema = deepcopy(self.diffusion)

        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)
        self.ema_decay = global_args.ema_decay
        
        self.num_quantizers = global_args.num_quantizers
        if self.num_quantizers > 0:
            quantizer_class = ResidualMemcodes if global_args.num_quantizers > 1 else Memcodes
            
            quantizer_kwargs = {}
            if global_args.num_quantizers > 1:
                quantizer_kwargs["num_quantizers"] = global_args.num_quantizers

            self.quantizer = quantizer_class(
                dim=global_args.latent_dim,
                heads=global_args.num_heads,
                num_codes=global_args.codebook_size,
                temperature=1.,
                **quantizer_kwargs
            )

            self.quantizer_ema = deepcopy(self.quantizer)

        scales = [2048, 1024, 512, 256, 128]
        hop_sizes = []
        win_lengths = []
        overlap = 0.75
        for s in scales:
            hop_sizes.append(int(s * (1 - overlap)))
            win_lengths.append(s)

        self.mrstft = auraloss.freq.MultiResolutionSTFTLoss(fft_sizes=scales, hop_sizes=hop_sizes, win_lengths=win_lengths, w_log_mag=1.0, w_lin_mag=1.0, device=device)
  
    def loss(self, reals):

        if self.pqmf_bands > 1:    
            encoder_input = self.pqmf(reals)
        else:
            encoder_input = reals

        # Draw uniformly distributed continuous timesteps
        t = self.rng.draw(reals.shape[0])[:, 0].to(self.device)

        # Calculate the noise schedule parameters for those timesteps
        alphas, sigmas = get_alphas_sigmas(get_crash_schedule(t))

        # Combine the ground truth images and the noise
        alphas = alphas[:, None, None]
        sigmas = sigmas[:, None, None]
        noise = torch.randn_like(encoder_input)
        noised_reals = encoder_input * alphas + noise * sigmas
        targets = noise * alphas - encoder_input * sigmas

        #print(f"noised_reals_shape:{noised_reals.shape}")

        # Compute the model output and the loss.
        with torch.cuda.amp.autocast():
            latents = self.encoder(encoder_input).float()

        if self.num_quantizers > 0:
            #Rearrange for Memcodes
            latents = rearrange(latents, 'b d n -> b n d')

            #Quantize into memcodes
            latents, _ = self.quantizer(latents)

            latents = rearrange(latents, 'b n d -> b d n')

        # p = torch.rand([tokens.shape[0], 1], device=tokens.device)
        # tokens = torch.where(p > 0.2, tokens, torch.zeros_like(tokens))

        with torch.cuda.amp.autocast():
            v = self.diffusion(noised_reals, t, latents)
            #print(f"v shape:{v.shape}")
            mse_loss = LAMBDA_DIFFUSION * F.mse_loss(v, targets)

        #Decode audio
        decoded_audio = self.decoder(latents)

        if self.pqmf_bands > 1:
            mb_distance = self.mrstft(encoder_input, decoded_audio)
            decoded_audio = self.pqmf.inverse(decoded_audio)

        mrstft_loss = self.mrstft(reals, decoded_audio)
            

        loss = mse_loss + mrstft_loss

        if self.pqmf_bands > 1:
            loss += mb_distance
        
        log_dict = {
            'train/mse_loss': mse_loss.detach(),
            'train/mrstft_loss': mrstft_loss.detach(),
            'train/mb_distance': mb_distance.detach(),
        }

        return loss, log_dict

def main():

    args = get_all_args()
    
    args.random_crop = False

    torch.manual_seed(args.seed)

    try:
        mp.set_start_method(args.start_method)
    except RuntimeError:
        pass
     
    accelerator = accelerate.Accelerator()
    device = accelerator.device
    print('Using device:', device, flush=True)

    diffusion_model = DiffGAN(args, device)

    accelerator.print('Parameters:', utils.n_params(diffusion_model))

    # If logging to wandb, initialize the run
    use_wandb = accelerator.is_main_process and args.name
    if use_wandb:
        import wandb
        config = vars(args)
        config['params'] = utils.n_params(diffusion_model)
        wandb.init(project=args.name, config=config, save_code=True)

    opt = optim.Adam([*diffusion_model.encoder.parameters(), *diffusion_model.diffusion.parameters(), *diffusion_model.decoder.parameters()], lr=4e-5)

    sched = utils.InverseLR(opt, inv_gamma=50000, power=1/2, warmup=0.99)
    ema_sched = utils.EMAWarmup(power=2/3, max_value=0.9999)

    train_set = SampleDataset([args.training_dir], args)
    train_dl = data.DataLoader(train_set, args.batch_size, shuffle=True,
                               num_workers=args.num_workers, persistent_workers=True, pin_memory=True)

    diffusion_model, opt, train_dl = accelerator.prepare(diffusion_model, opt, train_dl)

    diffusion_model_ema = deepcopy(diffusion_model)

    demo_dl = data.DataLoader(train_set, args.num_demos, num_workers=args.num_workers, shuffle=True)
    
    demo_dl = iter(demo_dl)

    if use_wandb:
        wandb.watch(diffusion_model)
        
    if args.ckpt_path:
        ckpt = torch.load(args.ckpt_path, map_location='cpu')
        accelerator.unwrap_model(diffusion_model).load_state_dict(ckpt['model'])
        accelerator.unwrap_model(diffusion_model_ema).load_state_dict(ckpt['model_ema'])
        opt.load_state_dict(ckpt['opt'])
        sched.load_state_dict(ckpt['sched'])
        ema_sched.load_state_dict(ckpt['ema_sched'])
        epoch = ckpt['epoch'] + 1
        step = ckpt['step'] + 1
        del ckpt
    else:
        epoch = 0
        step = 0


    @torch.no_grad()
    @utils.eval_mode(diffusion_model_ema)
    def demo():
        demo_reals, _ = next(demo_dl)

        encoder_input = demo_reals
        
        model_ema = accelerator.unwrap_model(diffusion_model_ema)

        encoder_input = encoder_input.to(device)

        demo_reals = demo_reals.to(device)

        if model_ema.pqmf_bands > 1:
            encoder_input = model_ema.pqmf(demo_reals)
        
        noise = torch.randn([demo_reals.shape[0], 2*args.pqmf_bands, args.sample_size//args.pqmf_bands]).to(device)

        with torch.no_grad():

            latents = model_ema.encoder(encoder_input)

            if model_ema.num_quantizers > 0:

                #Rearrange for Memcodes
                latents = rearrange(latents, 'b d n -> b n d')

                latents, _= model_ema.quantizer(latents)
                latents = rearrange(latents, 'b n d -> b d n')

            fakes = sample(model_ema.diffusion, noise, args.demo_steps, 1, latents)

            one_shot_fakes = model_ema.decoder(latents)

            if model_ema.pqmf_bands > 1:
                fakes = model_ema.pqmf.inverse(fakes)
                one_shot_fakes = model_ema.pqmf.inverse(one_shot_fakes)

        # Put the demos together
        fakes = rearrange(fakes, 'b d n -> d (b n)')
        one_shot_fakes = rearrange(one_shot_fakes, 'b d n -> d (b n)')
        demo_reals = rearrange(demo_reals, 'b d n -> d (b n)')

        if use_wandb:
            try:
                log_dict = {}
                
                filename = f'recon_{step:08}.wav'
                fakes = fakes.clamp(-1, 1).mul(32767).to(torch.int16).cpu()
                torchaudio.save(filename, fakes, args.sample_rate)

                decoder_filename = f'decoded_{step:08}.wav'
                one_shot_fakes = one_shot_fakes.clamp(-1, 1).mul(32767).to(torch.int16).cpu()
                torchaudio.save(decoder_filename, one_shot_fakes, args.sample_rate)

                reals_filename = f'reals_{step:08}.wav'
                demo_reals = demo_reals.clamp(-1, 1).mul(32767).to(torch.int16).cpu()
                torchaudio.save(reals_filename, demo_reals, args.sample_rate)


                log_dict[f'recon'] = wandb.Audio(filename,
                                                    sample_rate=args.sample_rate,
                                                    caption=f'Diffusion decoder')
                log_dict[f'real'] = wandb.Audio(reals_filename,
                                                    sample_rate=args.sample_rate,
                                                    caption=f'Real')
                log_dict[f'decoded'] = wandb.Audio(decoder_filename,
                                                    sample_rate=args.sample_rate,
                                                    caption=f'One-shot decoder')                                                    

                log_dict[f'embeddings'] = embeddings_table(latents)

                log_dict[f'embeddings_3dpca'] = pca_point_cloud(latents)
                log_dict[f'embeddings_spec'] = wandb.Image(tokens_spectrogram_image(latents))

                log_dict[f'real_melspec_left'] = wandb.Image(audio_spectrogram_image(demo_reals))
                log_dict[f'recon_melspec_left'] = wandb.Image(audio_spectrogram_image(fakes))
                log_dict[f'oneshot_recon_melspec_left'] = wandb.Image(audio_spectrogram_image(one_shot_fakes))


                wandb.log(log_dict, step=step)
            except Exception as e:
                print(f'{type(e).__name__}: {e}', file=sys.stderr)

    def save():
        accelerator.wait_for_everyone()
        filename = f'{args.name}_{step:08}.pth'
        if accelerator.is_main_process:
            tqdm.write(f'Saving to {filename}...')
        obj = {
            'model': accelerator.unwrap_model(diffusion_model).state_dict(),
            'model_ema': accelerator.unwrap_model(diffusion_model_ema).state_dict(),
            'opt': opt.state_dict(),
            'sched': sched.state_dict(),
            'ema_sched': ema_sched.state_dict(),
            'epoch': epoch,
            'step': step
        }
        accelerator.save(obj, filename)
    try:
        while True:
            for batch in tqdm(train_dl, disable=not accelerator.is_main_process):
                opt.zero_grad()
                loss, log_dict = accelerator.unwrap_model(diffusion_model).loss(batch[0])
                accelerator.backward(loss)
                opt.step()
                sched.step()
                ema_decay = ema_sched.get_value()
                
                utils.ema_update(
                    accelerator.unwrap_model(diffusion_model), 
                    accelerator.unwrap_model(diffusion_model_ema),
                    ema_decay
                )

                ema_sched.step()

                if accelerator.is_main_process:
                    if step % 25 == 0:
                        tqdm.write(f'Epoch: {epoch}, step: {step}, loss: {loss.item():g}')

                    if use_wandb:
                        log_dict = {
                            **log_dict,
                            'epoch': epoch,
                            'loss': loss.item(),
                            'lr': sched.get_last_lr()[0],
                            'ema_decay': ema_decay,
                        }
                        wandb.log(log_dict, step=step)

                    if step % args.demo_every == 0:
                        demo()

                if step > 0 and step % args.checkpoint_every == 0:
                    save()

                step += 1
            epoch += 1
    except RuntimeError as err:
            import requests
            import datetime
            ts = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
            resp = requests.get('http://169.254.169.254/latest/meta-data/instance-id')
            print(f'ERROR at {ts} on {resp.text} {device}: {type(err).__name__}: {err}', flush=True)
            raise err
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()

