#!/usr/bin/env python3

from prefigure.prefigure import get_all_args, push_wandb_config
from copy import deepcopy
import numpy as np
import math
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
from autoencoders.models import AttnResEncoder1D
from losses.adv_losses import StackDiscriminators

from decoders.generators import RaveEncoder, RaveGenerator
from vector_quantize_pytorch import ResidualVQ

from viz.viz import embeddings_table, pca_point_cloud, audio_spectrogram_image, tokens_spectrogram_image

def l2norm(t):
    return F.normalize(t, p = 2, dim = -1)

LAMBDA_QUANTIZER = 1

# PQMF stopband attenuation
PQMF_ATTN = 100

class AudioAutoencoder(nn.Module):
    def __init__(self, global_args, device, depth=8, n_attn_layers = 0):
        super().__init__()

        self.device = device

        self.pqmf_bands = global_args.pqmf_bands

        if self.pqmf_bands > 1:
            self.pqmf = PQMF(2, PQMF_ATTN, global_args.pqmf_bands)

        #c_mults = [512] * depth

        #c_mults = [128, 128, 256] + [512] * (depth-3)

        #ratios = [2]*(depth-1)

        ratios = [4, 4, 4, 2]

        capacity = 64

        #assert np.prod(ratios) == 2**(depth-1)

        #self.encoder = AttnResEncoder1D(n_io_channels=2*global_args.pqmf_bands, latent_dim=global_args.latent_dim, depth=depth, n_attn_layers=n_attn_layers, c_mults = c_mults)
       
        self.encoder = RaveEncoder(2*global_args.pqmf_bands, capacity=capacity, latent_size=global_args.latent_dim, ratios=ratios)

        self.decoder = RaveGenerator(latent_size=global_args.latent_dim, capacity=capacity, data_size=2*global_args.pqmf_bands, ratios = ratios)

        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)
        self.ema_decay = global_args.ema_decay
        
        scales = [2048, 1024, 512, 256, 128]
        hop_sizes = []
        win_lengths = []
        overlap = 0.75
        for s in scales:
            hop_sizes.append(int(s * (1 - overlap)))
            win_lengths.append(s)

        self.mrstft = auraloss.freq.MultiResolutionSTFTLoss(fft_sizes=scales, hop_sizes=hop_sizes, win_lengths=win_lengths, w_log_mag=1.0, w_lin_mag=1.0, device=device)
        self.sdstft = auraloss.freq.SumAndDifferenceSTFTLoss(fft_sizes=scales, hop_sizes=hop_sizes, win_lengths=win_lengths)

        #self.aw_fir = auraloss.perceptual.FIRFilter(filter_type="aw", fs=global_args.sample_rate)

        self.num_quantizers = global_args.num_quantizers
        
        self.quantizer = ResidualVQ(
            dim=global_args.latent_dim,
            codebook_size=global_args.codebook_size,
            num_quantizers = global_args.num_quantizers,
            kmeans_init = True,
            kmeans_iters = 100,
            threshold_ema_dead_code = 2,
            use_cosine_sim = True,
            # commitment_weight=0,
            # orthogonal_reg_weight = 1,
            # orthogonal_reg_active_codes_only = True,
            # shared_codebook = True
        )
  
    def loss(self, reals):

        p = Profiler()

        encoder_input = reals

        if self.pqmf_bands > 1:
            encoder_input = self.pqmf(reals)
            p.tick("pqmf")

        # Compute the model output and the loss.
        with torch.cuda.amp.autocast():
            latents = self.encoder(encoder_input).float()

            p.tick("encoder")

            tokens = rearrange(latents, 'b d n -> b n d')

            tokens, _, quantizer_loss = self.quantizer(tokens)

            if self.num_quantizers > 0:
                quantizer_loss = quantizer_loss.sum(dim=-1)
                latents = latents / math.sqrt(self.num_quantizers)
        
            tokens = rearrange(tokens, 'b n d -> b d n')

            p.tick("quantizer")

            decoded = self.decoder(tokens)

            p.tick("decoder")

            #Add pre-PQMF loss

            if self.pqmf_bands > 1:

                # Multi-scale STFT loss on the PQMF for multiband harmonic content
                mb_distance = self.mrstft(encoder_input, decoded)
                p.tick("mb_distance")

            
                decoded = self.pqmf.inverse(decoded)
                p.tick("pqmf_inverse")

          
            # aw_mse_loss_l = torch.nn.functional.mse_loss(self.aw_fir(reals[:, 0, :], decoded[:, 0, :]))
            # aw_mse_loss_r = torch.nn.functional.mse_loss(self.aw_fir(reals[:, 1, :], decoded[:, 1, :]))
            # aw_mse_loss = aw_mse_loss_l + aw_mse_loss_r
            # p.tick("aw_mse_loss")

            # Multi-scale mid-side STFT loss for stereo/harmonic information
            mrstft_loss = self.sdstft(reals, decoded)
            p.tick("fb_distance")
            
            loss = mrstft_loss + quantizer_loss #+ aw_mse_loss

            if self.pqmf_bands > 1:
                loss += mb_distance


            #print(p)
        log_dict = {
            'train/loss': loss.detach(),
            'train/mrstft_loss': mrstft_loss.detach(),
            'quantizer_loss': quantizer_loss.detach(),
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

    model = AudioAutoencoder(args, device, depth=args.depth)

    accelerator.print('Parameters:', utils.n_params(model))

    # If logging to wandb, initialize the run
    use_wandb = accelerator.is_main_process and args.name
    if use_wandb:
        import wandb
        config = vars(args)
        config['params'] = utils.n_params(model)
        wandb.init(project=args.name, config=config, save_code=True)

    opt = optim.Adam([*model.encoder.parameters(), *model.quantizer.parameters(), *model.decoder.parameters()], lr=4e-4)

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

        
        model_unwrap = accelerator.unwrap_model(model)
        
        demo_reals = demo_reals.to(device)
        encoder_input = demo_reals

        if args.pqmf_bands > 1:
            encoder_input = model_unwrap.pqmf(demo_reals)
        
        with torch.no_grad():

            latents = model_unwrap.encoder(encoder_input)

            tokens = rearrange(latents, 'b d n -> b n d')

            tokens, _, _ = model_unwrap.quantizer(tokens)

            tokens = rearrange(tokens, 'b n d -> b d n')
            fakes = model_unwrap.decoder(tokens)

            if args.pqmf_bands > 1:
                fakes = model_unwrap.pqmf.inverse(fakes)


        # Put the demos together
        fakes = rearrange(fakes, 'b d n -> d (b n)')
        demo_reals = rearrange(demo_reals, 'b d n -> d (b n)')

        #demo_audio = torch.cat([demo_reals, fakes], -1)
        if use_wandb:
            try:
                log_dict = {}
                
                filename = f'recon_{step:08}.wav'
                fakes = fakes.clamp(-1, 1).mul(32767).to(torch.int16).cpu()
                torchaudio.save(filename, fakes, args.sample_rate)

                reals_filename = f'reals_{step:08}.wav'
                demo_reals = demo_reals.clamp(-1, 1).mul(32767).to(torch.int16).cpu()
                torchaudio.save(reals_filename, demo_reals, args.sample_rate)


                log_dict[f'recon'] = wandb.Audio(filename,
                                                    sample_rate=args.sample_rate,
                                                    caption=f'Reconstructed')
                log_dict[f'real'] = wandb.Audio(reals_filename,
                                                    sample_rate=args.sample_rate,
                                                    caption=f'Real')

                log_dict[f'embeddings'] = embeddings_table(tokens)

                log_dict[f'embeddings_3dpca'] = pca_point_cloud(tokens)
                log_dict[f'embeddings_spec'] = wandb.Image(tokens_spectrogram_image(tokens))

                log_dict[f'real_melspec_left'] = wandb.Image(audio_spectrogram_image(demo_reals))
                log_dict[f'recon_melspec_left'] = wandb.Image(audio_spectrogram_image(fakes))


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

if __name__ == '__main__':
    main()

