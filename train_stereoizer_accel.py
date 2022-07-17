#!/usr/bin/env python3

from prefigure.prefigure import get_all_args, push_wandb_config
from copy import deepcopy
import numpy as np

from test.profiler import Profiler

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

import auraloss

import wandb

from blocks import utils
from dataset.dataset import SampleDataset
from diffusion.pqmf import CachedPQMF as PQMF

from decoders.generators import AudioResnet

from losses.freq_losses import PerceptualSumAndDifferenceSTFTLoss

from viz.viz import embeddings_table, pca_point_cloud, audio_spectrogram_image, tokens_spectrogram_image

# PQMF stopband attenuation
PQMF_ATTN = 100

class AudioStereoizer(nn.Module):
    def __init__(self, global_args, device):
        super().__init__()

        self.device = device

        self.pqmf_bands = global_args.pqmf_bands

        if self.pqmf_bands > 1:
            self.mono_pqmf = PQMF(2, PQMF_ATTN, global_args.pqmf_bands)
            self.stereo_pqmf = PQMF(2, PQMF_ATTN, global_args.pqmf_bands)

        width = global_args.latent_dim

        layers = 16

        self.resnet = AudioResnet(in_channels = 1 * self.pqmf_bands, out_channels=2 * self.pqmf_bands, width=width, layers=layers)

        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)
        self.ema_decay = global_args.ema_decay
        
        scales = [2048, 1024, 512, 256, 128]
        hop_sizes = []
        win_lengths = []
        overlap = 0.75
        for s in scales:
            hop_sizes.append(int(s * (1 - overlap)))
            win_lengths.append(s)

        self.mrstft = auraloss.freq.MultiResolutionSTFTLoss(fft_sizes=scales, hop_sizes=hop_sizes, win_lengths=win_lengths, w_log_mag=1.0, w_lin_mag=1.0, w_phs=0.5, device=device)
        self.sdstft = PerceptualSumAndDifferenceSTFTLoss(fft_sizes=scales, hop_sizes=hop_sizes, win_lengths=win_lengths, w_lin_mag = 1.0, w_log_mag = 1.0, device=device)

        self.num_quantizers = global_args.num_quantizers
        
  
    def loss(self, stereo_input):

        p = Profiler()
        
        mono_input = stereo_input.mean(dim=1, keepdim=True)

        if self.pqmf_bands > 1:
            mono_input = self.mono_pqmf(mono_input)
            p.tick("pqmf")

        # Compute the model output and the loss.
        with torch.cuda.amp.autocast():
            stereo_output = self.resnet(mono_input).float()

            p.tick("resnet")

            #Add pre-PQMF loss

            if self.pqmf_bands > 1:

                # Multi-scale STFT loss on the PQMF for multiband harmonic content
                mb_distance = self.mrstft(mono_input, stereo_output)
                p.tick("mb_distance")

            
                stereo_output = self.stereo_pqmf.inverse(stereo_output)
                p.tick("pqmf_inverse")

          
            # Multi-scale mid-side STFT loss for stereo/harmonic information
            mrstft_loss = self.sdstft(stereo_input, stereo_output)
            p.tick("fb_distance")
            
            loss = mrstft_loss

            if self.pqmf_bands > 1:
                loss += mb_distance


            #print(p)
        log_dict = {
            'train/loss': loss.detach(),
            'train/mrstft_loss': mrstft_loss.detach(),
            #'train/aw_mse_loss': aw_mse_loss.detach(),
        }

        if self.pqmf_bands > 1:
            log_dict["mb_distance"] = mb_distance.detach()

        return loss, log_dict

def main():

    args = get_all_args()
    
    torch.manual_seed(args.seed)

    try:
        mp.set_start_method(args.start_method)
    except RuntimeError:
        pass
     
    accelerator = accelerate.Accelerator()
    device = accelerator.device
    print('Using device:', device, flush=True)

    model = AudioStereoizer(args, device)

    accelerator.print('Parameters:', utils.n_params(model))

    # If logging to wandb, initialize the run
    use_wandb = accelerator.is_main_process and args.name
    if use_wandb:
        import wandb
        config = vars(args)
        config['params'] = utils.n_params(model)
        wandb.init(project=args.name, config=config, save_code=True)

    opt = optim.Adam([*model.resnet.parameters()], lr=4e-5)

    #sched = utils.InverseLR(opt, inv_gamma=50000, power=1/2, warmup=0.99)

    train_set = SampleDataset([args.training_dir], args)
    train_dl = data.DataLoader(train_set, args.batch_size, shuffle=True,
                               num_workers=args.num_workers, persistent_workers=True, pin_memory=True)

    model, opt, train_dl = accelerator.prepare(model, opt, train_dl)

    demo_dl = data.DataLoader(train_set, args.num_demos, num_workers=args.num_workers, shuffle=True)
    
    demo_dl = iter(demo_dl)

    if use_wandb:
        wandb.watch(model)
        
    if args.ckpt_path:
        ckpt = torch.load(args.ckpt_path, map_location='cpu')
        accelerator.unwrap_model(model).load_state_dict(ckpt['model'])
        opt.load_state_dict(ckpt['opt'])
        #sched.load_state_dict(ckpt['sched'])
        epoch = ckpt['epoch'] + 1
        step = ckpt['step'] + 1
        del ckpt
    else:
        epoch = 0
        step = 0


    @torch.no_grad()
    def demo():
        demo_reals, _ = next(demo_dl)

        stereo_input = demo_reals
        
        model_unwrap = accelerator.unwrap_model(model)
        
        stereo_input = stereo_input.to(device)

        mono_input = stereo_input.mean(dim=1, keepdim=True)

        if args.pqmf_bands > 1:
            mono_input = model_unwrap.mono_pqmf(mono_input)
        
        with torch.no_grad():

            stereo_output = model_unwrap.resnet(mono_input)

            if args.pqmf_bands > 1:
                stereo_output = model_unwrap.stereo_pqmf.inverse(stereo_output)


        # Put the demos together
        stereo_output = rearrange(stereo_output, 'b d n -> d (b n)')
        mono_input = rearrange(mono_input, 'b d n -> d (b n)')

        if use_wandb:
            try:
                log_dict = {}
                
                filename = f'recon_{step:08}.wav'
                stereo_output = stereo_output.clamp(-1, 1).mul(32767).to(torch.int16).cpu()
                torchaudio.save(filename, stereo_output, args.sample_rate)

                reals_filename = f'reals_{step:08}.wav'
                mono_input = mono_input.clamp(-1, 1).mul(32767).to(torch.int16).cpu()
                torchaudio.save(reals_filename, mono_input, args.sample_rate)


                log_dict[f'recon'] = wandb.Audio(filename,
                                                    sample_rate=args.sample_rate,
                                                    caption=f'Reconstructed Stereo')
                log_dict[f'real'] = wandb.Audio(reals_filename,
                                                    sample_rate=args.sample_rate,
                                                    caption=f'Mono')

                log_dict[f'real_melspec_mono'] = wandb.Image(audio_spectrogram_image(mono_input))
                log_dict[f'recon_melspec_left'] = wandb.Image(audio_spectrogram_image(stereo_output))


                wandb.log(log_dict, step=step)
            except Exception as e:
                print(f'{type(e).__name__}: {e}', file=sys.stderr)

    def save():
        accelerator.wait_for_everyone()
        filename = f'{args.name}_{step:08}.pth'
        if accelerator.is_main_process:
            tqdm.write(f'Saving to {filename}...')
        obj = {
            'model': accelerator.unwrap_model(model).state_dict(),
            'opt': opt.state_dict(),
            #'sched': sched.state_dict(),
            'epoch': epoch,
            'step': step
        }
        accelerator.save(obj, filename)
    try:
        while True:
            for batch in tqdm(train_dl, disable=not accelerator.is_main_process):
                opt.zero_grad()
                loss, log_dict = accelerator.unwrap_model(model).loss(batch[0])
                accelerator.backward(loss)
                opt.step()
                #sched.step()

                if accelerator.is_main_process:
                    if step % 25 == 0:
                        tqdm.write(f'Epoch: {epoch}, step: {step}, loss: {loss.item():g}')

                    if use_wandb:
                        log_dict['epoch'] = epoch
                        log_dict['loss'] = loss.item(),
                       # log_dict['lr'] = sched.get_last_lr()[0]
                        
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

