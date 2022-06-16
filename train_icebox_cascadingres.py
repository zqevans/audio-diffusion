#!/usr/bin/env python3

from prefigure.prefigure import get_all_args, push_wandb_config
from contextlib import contextmanager
from copy import deepcopy
import math
from pathlib import Path

import sys
import os
import torch
from torch import optim, nn
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from tqdm import trange
import pytorch_lightning as pl
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.plugins import DDPPlugin

from einops import rearrange
from pprint import pprint

import torchaudio
from torchaudio import transforms as T
import torch.nn.functional as F
from fairscale.nn import checkpoint_wrapper, auto_wrap, wrap

import wandb
import numpy as np
import pandas as pd

from dataset.dataset import SampleDataset
from encoders.encoders import SoundStreamXL, SoundStreamXLEncoder, SoundStreamXLDecoder
from vector_quantize_pytorch import ResidualVQ
from dvae.residual_memcodes import ResidualMemcodes
from decoders.diffusion_decoder import DiffusionDecoder
from diffusion.model import ema_update

from icebox.tagbox_utils import audio_for_jbx, load_audio_for_jbx
from jukebox.make_models import make_vqvae, make_prior, MODELS, make_model
from jukebox.hparams import Hyperparams, setup_hparams
#from jukebox.sample import sample_single_window, _sample, sample_partial_window, upsample
#from jukebox.utils.dist_utils import setup_dist_from_mpi
#from jukebox.utils.torch_utils import empty_cache



# lonewater's auraloss fork:  pip install --no-cache-dir -U git+https://github.com/lonewater/auraloss.git@PWCmplxDif
from auraloss.freq import MultiResolutionSTFTLoss, PerceptuallyWeightedComplexLoss, MultiResolutionPrcptWghtdCmplxLoss

from viz.viz import embeddings_table, pca_point_cloud


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


def prep_reals_for_diffusion(lower_res_reals=None, higher_res_reals=None, main_sr=44100, new_res=0,diff=True, device='cuda'): 
    """
    higher res = the new resolution, i.e. the "current" resolution in the loop that would be calling this
    if higher res reals are given it means to producude the different between upscaled & true
    """
    # reals are stereo, new_res=0,1,2 from high to low
    # only res=2 involves predicting raw audio otherwise we upsample (and provide the difference)
    if new_res==2: return higher_res_reals
    new_sr = main_sr // (4**new_res)
    prev_sr = new_sr // 4 # self.sample_rate // (4**(res+1))
    #upscale_op = T.Resample(prev_sr, new_sr).to(self.device)
    upscaled_reals = F.interpolate(lower_res_reals, scale_factor=4, mode='linear').to(device)
    if diff and (higher_res_reals is not None):   # then do just the difference/residual
        return higher_res_reals - upscaled_reals  # diff
    else:
        return upscaled_reals  # Sould be unused: or do the full thing, eh? 



class IceBoxModule(pl.LightningModule):
    def __init__(self, global_args):
        super().__init__()

        n_io_channels = 2
        n_feature_channels = 8
        self.num_quantizers = global_args.num_quantizers
        self.ema_decay = global_args.ema_decay
        self.sample_rate = global_args.sample_rate

        #print(os.environ)
        #rank, local_rank, device = setup_dist_from_mpi()
        #dist_url = "tcp://127.0.0.1:9500"
        dist.init_process_group(backend="nccl")
        rank, local_rank, device = int(os.getenv('RANK')), int(os.getenv('LOCAL_RANK')), self.device

        self.hps = Hyperparams()
        assert global_args.sample_rate == 44100, "Jukebox was pretrained at 44100 Hz."
        self.hps.sr = global_args.sample_rate #44100
        self.hps.levels = 3
        self.hps.hop_fraction = [.5,.5,.125]

        vqvae = "vqvae"
        self.vqvae = make_vqvae(setup_hparams(vqvae, dict(sample_length = 1048576)), self.device)
        for param in self.vqvae.parameters():  # FREEZE IT
            param.requires_grad = False

        self.encoder = self.vqvae.encode
        self.encoder_ema = deepcopy(self.encoder)

        latent_dim = 64 # global_args.latent_dim. Jukebox is 64
        io_channels = 2#1 # 2.  Jukebox is mono but we decode in stereo
        self.num_resolutions = 1#self.hps.levels # jukebox: 0=high, 1=mid, 2 = low

        ## tried to make a list but PyL wouldn't give them gradients so we're doing them explicitly
        self.diffusion0 = DiffusionDecoder(latent_dim, io_channels,depth=16) #hires
        self.diffusion1 = DiffusionDecoder(latent_dim, io_channels, depth=13) #mid
        self.diffusion2 = DiffusionDecoder(latent_dim, io_channels, depth=10) #low
        self.diffusion_ema0 = deepcopy(self.diffusion0)
        self.diffusion_ema1 = deepcopy(self.diffusion1)
        self.diffusion_ema2 = deepcopy(self.diffusion2)
        self.diffusion = [self.diffusion0, self.diffusion1, self.diffusion2]
        self.diffusion_ema = [self.diffusion_ema0, self.diffusion_ema1, self.diffusion_ema2]

        '''self.diffusion, self.diffusion_ema = [], []
        for r in range(self.num_resolutions):
            self.diffusion += [DiffusionDecoder(latent_dim, io_channels)]
            self.diffusion_ema += [deepcopy(self.diffusion[r])]

        for r in range(len(self.diffusion)):
            self.diffusion[r].to(self.device)
            self.diffusion_ema[r].to(self.device)'''

        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)
        self.ema_decay = global_args.ema_decay
 
        self.num_quantizers = 0 # turn off all quantizer stuff from train_dvae.py for now 
        self.quantized = False 

        #self.jukebox_layers = [global_args.jukebox_layer]


        # losses
        #self.mstft = MultiResolutionSTFTLoss()
        #self.pwcl = PerceptuallyWeightedComplexLoss()
        #self.mrpwcl = MultiResolutionPrcptWghtdCmplxLoss()

    def configure_optimizers(self): 
        return optim.Adam([*self.diffusion0.parameters(),*self.diffusion1.parameters(),*self.diffusion2.parameters()], lr=2e-5)
        '''param_list = []
        for i in range(len(self.diffusion)):
            param_list += [*self.diffusion[i].parameters()]
        return optim.Adam(param_list, lr=2e-5)'''

    def encode(self, *args, **kwargs):
        if self.training:
            return self.encoder(*args, **kwargs)
        return self.encoder_ema(*args, **kwargs)

    def decode(self, *args, **kwargs):
        if self.training:
            return self.diffusion(*args, **kwargs)
        return self.diffusion_ema(*args, **kwargs)

  
    def get_nr_t_targets(self, reals):
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
        return noised_reals.to(self.device), t, targets.to(self.device)



    def training_step(self, batch, batch_idx):
        reals = batch[0]  # grab actual audio part of batch, not the filenames

        # ENCODING: Jukebox
        reals_mono = reals[:,0,:] # Jukebox encodes mono only, sorry
        encoder_input = audio_for_jbx(reals_mono, device=reals.device)
        with torch.cuda.amp.autocast():  # Encode and get embeddings / "tokens"
            zs = self.encoder(encoder_input) # indices at 3 resolutions
            xs = self.vqvae.bottleneck.decode(zs) # vectors vectors vectors! [hires, mid, lowres]
            #tokens = self.package_3layer_tokens(xs).float() # combine resolutions
        tokens = [x.float().to(self.device) for x in xs]
        #print("reals.size() = ",reals.size())
        #print("tokens sizes = ",[x.size() for x in tokens])

        # DECODING: (Cascading) Diffusion
        prev_reals = None
        mse_losses = []
        for res in reversed(range(len(self.diffusion))): 
            diffusion = self.diffusion[res]
            diffusion.to(self.device)

            new_sr = self.sample_rate // (4**res) # TODO: there's redundancy in here vs in prep_reals_for_diffusion
            if new_sr != self.sample_rate:
                resample_tf = T.Resample(self.sample_rate, new_sr).to(self.device)
                res_reals = resample_tf(reals).to(self.device)
            else:
                res_reals = reals
            diffuson_reals_input = prep_reals_for_diffusion(lower_res_reals=prev_reals, higher_res_reals=res_reals, main_sr=self.sample_rate, new_res=res, device=self.device)

            #print(f"res = {res}: diffuson_reals_input.shape =",diffuson_reals_input.shape)
            noised_reals, t, targets = self.get_nr_t_targets(diffuson_reals_input)
            #print(f"res = {res}: noised_reals, t, targets .shape =",noised_reals.shape, t.shape, targets.shape)

            # Compute the model output and the loss.
            with torch.cuda.amp.autocast():
                v = diffusion(noised_reals, t, tokens[res]) # diffusion 
                mse_losses += [F.mse_loss(v, targets)]

            prev_reals = res_reals

        mse_loss = sum(mse_losses)
        loss = mse_loss

        log_dict = {
            'train/loss': loss.detach(),
            'train/mse_loss': mse_loss.detach(),
        }

        self.log_dict(log_dict, prog_bar=True, on_step=True)
        return loss

    def on_before_zero_grad(self, *args, **kwargs):
        decay = 0.95 if self.current_epoch < 25 else self.ema_decay
        for i in range(len(self.diffusion)):
            ema_update(self.diffusion[i].to(self.device), self.diffusion_ema[i].to(self.device), decay)
        #ema_update(self.encoder, self.encoder_ema, decay) # frozen

        if self.num_quantizers > 0:
            ema_update(self.quantizer, self.quantizer_ema, decay)

    def package_3layer_tokens(self, tokens_list):
        "jukebox vqvae returns a list of 3 1-dim tensor. Here we package them...somehow"
        print("WARNING: We ain't doing this no more")
        return tokens_list
        #ind = self.jukebox_layers[0]
        #return tokens_list[ind] # TODO: just grab one set of tokens for now

class ExceptionCallback(pl.Callback):
    def on_exception(self, trainer, module, err):
        print(f'{type(err).__name__}: {err}', file=sys.stderr)


class DemoCallback(pl.Callback):
    def __init__(self, demo_dl, global_args):
        super().__init__()
        self.demo_every = global_args.demo_every
        self.demo_samples = global_args.sample_size
        self.demo_steps = global_args.demo_steps
        self.demo_dl = iter(demo_dl)
        self.sample_rate = global_args.sample_rate
        self.pqmf_bands = global_args.pqmf_bands
        self.quantized = global_args.num_quantizers > 0

    @rank_zero_only
    @torch.no_grad()
    def on_train_epoch_end(self, trainer, module):
        if trainer.current_epoch % self.demo_every != 0:
            return
        
        print("\nStarting demo")

        demo_reals, _ = next(self.demo_dl) # these are full resolution
        demo_reals = demo_reals.to(module.device)
        resample_to_lowest = T.Resample(module.sample_rate, (module.sample_rate//4**2)).to(module.device)
        resample_to_mid = T.Resample(module.sample_rate, (module.sample_rate//4)).to(module.device)
        lowest_res_reals = resample_to_lowest(demo_reals)
        #mid_res_reals = resample_to_mid(demo_reals) #unused

        # ENCODING: encode all token levels using full res signal
        encoder_input = audio_for_jbx(demo_reals[:,0,:]).to(module.device) 
        zs = module.encoder(encoder_input) 
        xs = module.vqvae.bottleneck.decode(zs) 
        tokens = [x.float() for x in xs]


        # Could do a loop for the next bit easily, but writing everything out helps keep my head straight

        res = 2  #------------- Low res
        print(f"res = {res}")
        diffusion_reals_input = lowest_res_reals
        diffusion_ema = module.diffusion_ema[res].to(module.device)
        noise = torch.randn([diffusion_reals_input.shape[0], 2, self.demo_samples//(4**res)]).to(module.device)
        fakes_low = sample(diffusion_ema, noise, self.demo_steps, 1, tokens[res]) # at higher res these will be diffs rel to upscaled


        res = 1  #------------- mid res
        print(f"res = {res}")
        upscaled_fakes = F.interpolate(fakes_low, scale_factor=4, mode='linear').to(module.device)
        diffusion_reals_input = upscaled_fakes
        diffusion_ema = module.diffusion_ema[res].to(module.device)
        noise = torch.randn([diffusion_reals_input.shape[0], 2, self.demo_samples//(4**res)]).to(module.device)
        fakes_mid = sample(diffusion_ema, noise, self.demo_steps, 1, tokens[res]) # at higher res these will be diffs rel to upscaled
        fakes_mid += upscaled_fakes  # add in upscaled version to diffs for full thing

        res = 0  #------------- highest res
        print(f"res = {res}")
        upscaled_fakes = F.interpolate(fakes_mid, scale_factor=4, mode='linear').to(module.device)
        diffusion_reals_input = upscaled_fakes
        diffusion_ema = module.diffusion_ema[res].to(module.device)
        noise = torch.randn([diffusion_reals_input.shape[0], 2, self.demo_samples//(4**res)]).to(module.device)
        fakes_high = sample(diffusion_ema, noise, self.demo_steps, 1, tokens[res]) # at higher res these will be diffs rel to upscaled
        fakes_high += upscaled_fakes  # add in upscaled version to diffs for full thing

        # Put the demos together from batches
        fakes = rearrange(fakes_high, 'b d n -> d (b n)')
        demo_reals = rearrange(demo_reals, 'b d n -> d (b n)')

        try: # loggins
            log_dict = {}

            filename = f'recon_{trainer.global_step:08}.wav'
            fakes = fakes.clamp(-1, 1).mul(32767).to(torch.int16).cpu()
            torchaudio.save(filename, fakes, self.sample_rate)

            reals_filename = f'reals_{trainer.global_step:08}.wav'
            demo_reals = demo_reals.clamp(-1, 1).mul(32767).to(torch.int16).cpu()
            torchaudio.save(reals_filename, demo_reals, self.sample_rate)

            log_dict[f'recon'] = wandb.Audio(filename,
                                                sample_rate=self.sample_rate,
                                                caption=f'Reconstructed')
            log_dict[f'real'] = wandb.Audio(reals_filename,
                                                sample_rate=self.sample_rate,
                                                caption=f'Real')

            #log_dict[f'embeddings'] = embeddings_table(tokens)
            #log_dict[f'embeddings_3dpca'] = pca_point_cloud(tokens)

            trainer.logger.experiment.log(log_dict, step=trainer.global_step)

        except Exception as e:
            print(f'{type(e).__name__}: {e}', file=sys.stderr)

        return


def main():    
    args = get_all_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    torch.manual_seed(args.seed)
    #dist.init_process_group(backend="nccl")

    train_set = SampleDataset([args.training_dir], args)
    train_dl = data.DataLoader(train_set, args.batch_size, shuffle=True,
                               num_workers=args.num_workers, persistent_workers=True, pin_memory=True)
    wandb_logger = pl.loggers.WandbLogger(project=args.name)
    demo_dl = data.DataLoader(train_set, args.num_demos, num_workers=args.num_workers, shuffle=True)
    
    exc_callback = ExceptionCallback()
    ckpt_callback = pl.callbacks.ModelCheckpoint(every_n_train_steps=args.checkpoint_every, save_top_k=-1)
    demo_callback = DemoCallback(demo_dl, args)
    module = IceBoxModule(args)
    wandb_logger.watch(module)
    push_wandb_config(wandb_logger, args)

    #print(os.environ)
    #for env in ['MASTER_ADDR','MASTER_PORT','RANK','LOCAL_RANK','WORLD_SIZE','GLOBAL_RANK']:
    #    env_val = os.getenv(env)
    #    print(f"{env}={env_val}")

    trainer = pl.Trainer(
        gpus=args.num_gpus,
        accelerator="gpu",
        #strategy='fsdp', 
        #strategy = 'ddp_find_unused_parameters_false', #without this I get lots of warnings and it goes slow
        precision=32,
        accumulate_grad_batches={
            0:1, 
            1: args.accum_batches #Start without accumulation
            # 5:2,
            # 10:3, 
            # 12:4, 
            # 14:5, 
            # 16:6, 
            # 18:7,  
            # 20:8
            }, 
        callbacks=[ckpt_callback, demo_callback, exc_callback],
        logger=wandb_logger,
        log_every_n_steps=1,
        max_epochs=10000000,
    )

    trainer.fit(module, train_dl)

if __name__ == '__main__':
    main()

