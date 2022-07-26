#!/usr/bin/env python3

from prefigure.prefigure import get_all_args, push_wandb_config
from copy import deepcopy
import numpy as np
import math
from test.profiler import Profiler
from losses.perceptual_losses import Loudness

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

from losses.adv_losses import StackDiscriminators

from autoencoders.soundstream import SoundStreamXLEncoder, SoundStreamXLDecoder

from vector_quantize_pytorch import ResidualVQ

from dvae.residual_memcodes import ResidualMemcodes

from nwt_pytorch import Memcodes

from viz.viz import embeddings_table, pca_point_cloud, audio_spectrogram_image, tokens_spectrogram_image

def l2norm(t):
    return F.normalize(t, p = 2, dim = -1)

@torch.no_grad()
def ema_update(model, averaged_model, decay):
    """Incorporates updated model parameters into an exponential moving averaged
    version of a model. It should be called after each optimizer step."""
    model_params = dict(model.named_parameters())
    averaged_params = dict(averaged_model.named_parameters())
    assert model_params.keys() == averaged_params.keys()

    for name, param in model_params.items():
        averaged_params[name].mul_(decay).add_(param, alpha=1 - decay)

    model_buffers = dict(model.named_buffers())
    averaged_buffers = dict(averaged_model.named_buffers())
    assert model_buffers.keys() == averaged_buffers.keys()

    for name, buf in model_buffers.items():
        averaged_buffers[name].copy_(buf)

LAMBDA_QUANTIZER = 1

# PQMF stopband attenuation
PQMF_ATTN = 100

class SoundStream(nn.Module):
    def __init__(self, global_args, device, n_attn_layers = 0):
        super().__init__()

        self.device = device

        self.pqmf_bands = global_args.pqmf_bands

        if self.pqmf_bands > 1:
            self.pqmf = PQMF(2, PQMF_ATTN, global_args.pqmf_bands)

        capacity = 32

        #c_mults = [2, 4, 8, 16]
        
        #strides = [2, 4, 5, 8]

        c_mults = [2, 4, 8, 16]
        
        strides = [2, 2, 2, 2]

        #self.loudness = Loudness(global_args.sample_rate, 512)

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

        self.discriminator = StackDiscriminators(
            3,
            in_size=2, # Stereo
            capacity=16,
            multiplier=4,
            n_layers=4,
        )

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

        self.num_quantizers = global_args.num_quantizers
        
        # self.quantizer = ResidualVQ(
        #     dim=global_args.latent_dim,
        #     codebook_size=global_args.codebook_size,
        #     num_quantizers = global_args.num_quantizers,
        #     kmeans_init = True,
        #     kmeans_iters = 100,
        #     threshold_ema_dead_code = 2,
        #     use_cosine_sim = True,
        #     sync_codebook = True
        #     #orthogonal_reg_weight = 1,
        #     # orthogonal_reg_active_codes_only = True,
        #     # shared_codebook = True
        # )

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

        self.warmed_up = False

    def loss(self, reals):

        p = Profiler()

        encoder_input = reals

        if self.pqmf_bands > 1:
            encoder_input = self.pqmf(reals)
            p.tick("pqmf")

        # Compute the model output and the loss.
        with torch.cuda.amp.autocast():

            if self.warmed_up:
                with torch.no_grad():
                    latents = self.encoder(encoder_input).float()
                    #p.tick("encoder")
                    tokens = rearrange(latents, 'b d n -> b n d')
                    tokens, _ = self.quantizer(tokens)
                    tokens = rearrange(tokens, 'b n d -> b d n')
            else:
                latents = self.encoder(encoder_input).float()
                #p.tick("encoder")
                tokens = rearrange(latents, 'b d n -> b n d')
                # tokens, _ = self.quantizer(tokens)
                tokens = l2norm(tokens)
                tokens = rearrange(tokens, 'b n d -> b d n')
                

            # if self.num_quantizers > 0:
            #     tokens /= self.num_quantizers
            

            #p.tick("quantizer")

            decoded = self.decoder(tokens)

            #p.tick("decoder")

            #Add pre-PQMF loss

            if self.pqmf_bands > 1:

                # Multi-scale STFT loss on the PQMF for multiband harmonic content
                mb_distance = self.mrstft(encoder_input, decoded)
                #p.tick("mb_distance")

            
                decoded = self.pqmf.inverse(decoded)
                #p.tick("pqmf_inverse")

            # Multi-scale mid-side STFT loss for stereo/harmonic information
            mrstft_loss = self.sdstft(reals, decoded)
            #p.tick("fb_distance")
            
            # loud_reals = self.loudness(reals)
            # loud_decoded = self.loudness(decoded)
            # loud_dist = (loud_reals - loud_decoded).pow(2).mean()
            # p.tick("loudness distance")

            if self.warmed_up:
                loss_disc, loss_adv, feature_dist, _, _ = self.discriminator.loss(reals, decoded)
                p.tick("discriminator")
            else:
                loss_disc = torch.tensor(0.).to(self.device)
                loss_adv = torch.tensor(0.).to(self.device)
                feature_dist = torch.tensor(0.).to(self.device)

            gen_loss = mrstft_loss + loss_adv + feature_dist #+ quantizer_loss  #+ loud_dist

            if self.pqmf_bands > 1:
                gen_loss += mb_distance

            #print(p)

        log_dict = {
            'train/loss': gen_loss.detach(),
            'train/mrstft_loss': mrstft_loss.detach(),
            #'quantizer_loss': quantizer_loss.detach(),
            'adv_loss': loss_adv.detach(),
            'train/disc_loss': loss_disc.detach(),
            'train/feature_match_distance': feature_dist.detach(),
            #'train/loudness_distance': loud_dist.detach(),
            #'train/aw_mse_loss': aw_mse_loss.detach(),
        }

        if self.pqmf_bands > 1:
            log_dict["mb_distance"] = mb_distance.detach()

        return gen_loss, loss_disc, log_dict

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

    model = SoundStream(args, device)

    accelerator.print('Parameters:', utils.n_params(model))

    # If logging to wandb, initialize the run
    use_wandb = accelerator.is_main_process and args.name
    if use_wandb:
        import wandb
        config = vars(args)
        config['params'] = utils.n_params(model)
        wandb.init(project=args.name, config=config, save_code=True)

    opt_gen = optim.Adam([*model.encoder.parameters(), *model.quantizer.parameters(), *model.decoder.parameters()], lr=4e-5, betas=(.5, .9))
    opt_disc = optim.Adam([*model.discriminator.parameters()], lr=4e-5, betas=(.5, .9))
    #ema_sched = utils.EMAWarmup(power=2/3, max_value=0.9999)
    #sched = utils.InverseLR(opt, inv_gamma=50000, power=1/2, warmup=0.99)

    train_set = SampleDataset([args.training_dir], args)
    train_dl = data.DataLoader(train_set, args.batch_size, shuffle=True,
                               num_workers=args.num_workers, persistent_workers=True, pin_memory=True)

    model, opt_gen, opt_disc, train_dl = accelerator.prepare(model, opt_gen, opt_disc, train_dl)

    #model_ema = deepcopy(model)

    demo_dl = data.DataLoader(train_set, args.num_demos, num_workers=args.num_workers, shuffle=True)
    
    demo_dl = iter(demo_dl)

    if use_wandb:
        wandb.watch(model)
        
    if args.ckpt_path:
        ckpt = torch.load(args.ckpt_path, map_location='cpu')
        accelerator.unwrap_model(model).load_state_dict(ckpt['model'])
        #accelerator.unwrap_model(model_ema).load_state_dict(ckpt['model_ema'])
        opt_gen.load_state_dict(ckpt['opt_gen'])
        opt_disc.load_state_dict(ckpt['opt_disc'])
        #sched.load_state_dict(ckpt['sched'])
       # ema_sched.load_state_dict(ckpt['ema_sched'])
        epoch = ckpt['epoch'] + 1
        step = ckpt['step'] + 1
        del ckpt
    else:
        epoch = 0
        step = 0


    @torch.no_grad()
    @utils.eval_mode(model)
    def demo():
        demo_reals, _ = next(demo_dl)

        #model_unwrap = accelerator.unwrap_model(model_ema)
        model_unwrap = accelerator.unwrap_model(model)
        
        demo_reals = demo_reals.to(device)
        encoder_input = demo_reals

        if args.pqmf_bands > 1:
            encoder_input = model_unwrap.pqmf(demo_reals)
        
        with torch.no_grad():
            latents = model_unwrap.encoder(encoder_input)

            tokens = rearrange(latents, 'b d n -> b n d')

            #tokens, _ = model_unwrap.quantizer(tokens)
            tokens = l2norm(tokens)

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
            #'model_ema': accelerator.unwrap_model(model_ema).state_dict(),
            'opt_gen': opt_gen.state_dict(),
            'opt_disc': opt_gen.state_dict(),
            #'sched': sched.state_dict(),
            #'ema_sched': ema_sched.state_dict(),
            'epoch': epoch,
            'step': step
        }
        accelerator.save(obj, filename)
    try:
        while True:
            for batch in tqdm(train_dl, disable=not accelerator.is_main_process):
                
                model_unwrap = accelerator.unwrap_model(model)

                gen_loss, disc_loss, log_dict = model_unwrap.loss(batch[0])

                if step >= args.warmup_steps:
                    model_unwrap.warmed_up = True
                
                if step%2 and model_unwrap.warmed_up:
                    opt_disc.zero_grad()
                    accelerator.backward(disc_loss)
                    opt_disc.step()
                else:
                    opt_gen.zero_grad()
                    accelerator.backward(gen_loss)
                    opt_gen.step()

                #sched.step()
                #ema_decay = ema_sched.get_value()

                # utils.ema_update(
                #     accelerator.unwrap_model(model), 
                #     accelerator.unwrap_model(model_ema),
                #     ema_decay
                # )

               # ema_sched.step()

                if accelerator.is_main_process:
                    if step % 25 == 0:
                        tqdm.write(f'Epoch: {epoch}, step: {step}, loss: {gen_loss.item():g}')

                    if use_wandb:
                        log_dict['epoch'] = epoch
                        log_dict['gen_loss'] = gen_loss.item(),
                        log_dict['disc_loss'] = disc_loss.item(),
                        #log_dict['ema_decay'] = ema_decay
                        #log_dict['lr'] = sched.get_last_lr()[0]
                        
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

