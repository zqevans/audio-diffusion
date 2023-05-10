#!/usr/bin/env python3

from prefigure.prefigure import get_all_args, push_wandb_config
from contextlib import contextmanager
from copy import deepcopy
import math
from pathlib import Path

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

from decoders.diffusion_decoder import AudioDenoiserModel
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
def sample(model, x, steps, eta, mapping_cond=None, unet_cond=None):
    """Draws samples from a model given starting noise."""
    ts = x.new_ones([x.shape[0]])

    # Create the noise schedule
    t = torch.linspace(1, 0, steps + 1)[:-1]
    alphas, sigmas = get_alphas_sigmas(get_crash_schedule(t))

    # The sampling loop
    for i in trange(steps):

        # Get the model output (v, the predicted velocity)
        with torch.cuda.amp.autocast():
            v = model(x, ts * t[i], mapping_cond=mapping_cond, unet_cond=unet_cond, log_sigma=False).float()

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



class AudioDiffusion(nn.Module):
    def __init__(self, global_args, device):
        super().__init__()

        self.device = device

        model_config = json.load(open(global_args.model_config))
  
        
        #size = model_config['input_size'] # Input size is determined by global_args.sample_size instead

        self.diffusion = AudioDenoiserModel(
            model_config['input_channels'],
            model_config['mapping_out'],
            model_config['depths'],
            model_config['channels'],
            model_config['self_attn_depths'],
            dropout_rate=model_config['dropout_rate']
        )

        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)
        self.ema_decay = global_args.ema_decay
  
    def loss(self, reals):
        
        # Draw uniformly distributed continuous timesteps
        t = self.rng.draw(reals.shape[0])[:, 0].to(self.device)

        # Calculate the noise schedule parameters for those timesteps
        alphas, sigmas = get_alphas_sigmas(get_crash_schedule(t))

        # Combine the ground truth images and the noise
        alphas = alphas[:, None, None]
        sigmas = sigmas[:, None, None]
        noise = torch.randn_like(reals)
        noised_reals = reals * alphas + noise * sigmas
        targets = noise * alphas - reals * sigmas

        with torch.cuda.amp.autocast():
            v = self.diffusion(noised_reals, t, log_sigma=False)
            mse_loss = F.mse_loss(v, targets)
            loss = mse_loss

        return loss

def main():

    args = get_all_args()
    
    try:
        mp.set_start_method(args.start_method)
    except RuntimeError:
        pass

    torch.manual_seed(args.seed)

    model_config = json.load(open(args.model_config))

    accelerator = accelerate.Accelerator()
    device = accelerator.device
    print('Using device:', device, flush=True)

    diffusion_model = AudioDiffusion(args, device)

    accelerator.print('Parameters:', utils.n_params(diffusion_model))
    
    # If logging to wandb, initialize the run
    use_wandb = accelerator.is_main_process and args.name
    if use_wandb:
        import wandb
        config = vars(args)
        config['model_config'] = model_config
        config['params'] = utils.n_params(diffusion_model)
        wandb.init(project=args.name, config=config, save_code=True)


    opt = optim.Adam([*diffusion_model.parameters()], lr=4e-5)

    sched = utils.InverseLR(opt, inv_gamma=50000, power=1/2, warmup=0.99)
    ema_sched = utils.EMAWarmup(power=2/3, max_value=0.9999)

    train_set = SampleDataset([args.training_dir], args)
    train_dl = data.DataLoader(train_set, args.batch_size, shuffle=True,
                               num_workers=args.num_workers, persistent_workers=True, pin_memory=True)

    diffusion_model, opt, train_dl = accelerator.prepare(diffusion_model, opt, train_dl)

    diffusion_model_ema = deepcopy(diffusion_model)

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
        
        model_ema = accelerator.unwrap_model(diffusion_model_ema)

        noise = torch.randn([args.num_demos, 2, args.sample_size]).to(device)

        fakes = sample(model_ema.diffusion, noise, args.demo_steps, 1)

        # Put the demos together
        fakes = rearrange(fakes, 'b d n -> d (b n)')

        try:
            if use_wandb:
                log_dict = {}
                
                filename = f'demo_{step:08}.wav'
                fakes = fakes.clamp(-1, 1).mul(32767).to(torch.int16).cpu()
                torchaudio.save(filename, fakes, args.sample_rate)

                log_dict[f'demo'] = wandb.Audio(filename,
                                                    sample_rate=args.sample_rate,
                                                    caption=f'Demo')
                
                log_dict[f'recon_melspec_left'] = wandb.Image(audio_spectrogram_image(fakes))

            
                wandb.log(log_dict, step=step)
        except Exception as e:
            print(f'{type(e).__name__}: {e}', file=sys.stderr)

    def save():
        print("Waiting for everyone:")
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
        if args.wandb_save_model and use_wandb:
            wandb.save(filename)

    try:
        while True:
            for batch in tqdm(train_dl, disable=not accelerator.is_main_process):
                opt.zero_grad()
                loss = accelerator.unwrap_model(diffusion_model).loss(batch[0])
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
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()

