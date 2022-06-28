#!/usr/bin/env python3

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path

from prefigure import get_all_args, push_wandb_config

import accelerate
import torch
from torch import optim
from torch import multiprocessing as mp
from torch.utils import data
from torchvision import utils
from tqdm import tqdm

from dataset.dataset import SampleDataset
from decoders.diffusion_decoder import AudioDenoiserModel, Denoiser

from blocks import utils

from diffusion import inference

def main():  
    
    args = get_all_args()

    mp.set_start_method(args.start_method)

    model_config = json.load(open(args.model_config))
  
    size = model_config['input_size']

    accelerator = accelerate.Accelerator()
    device = accelerator.device
    print('Using device:', device, flush=True)

    assert model_config['type'] == 'audio_v1'

    inner_model = AudioDenoiserModel(
        model_config['input_channels'],
        model_config['mapping_out'],
        model_config['depths'],
        model_config['channels'],
        model_config['self_attn_depths'],
        dropout_rate=model_config['dropout_rate'],
        mapping_cond_dim=9,
    )
    accelerator.print('Parameters:', utils.n_params(inner_model))

    # If logging to wandb, initialize the run
    use_wandb = accelerator.is_main_process and args.wandb_project
    if use_wandb:
        import wandb
        config = vars(args)
        config['model_config'] = model_config
        config['params'] = utils.n_params(inner_model)
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, group=args.wandb_group, config=config, save_code=True)

    opt = optim.AdamW(inner_model.parameters(), lr=args.lr, betas=(0.95, 0.999), eps=1e-6, weight_decay=1e-3)
    sched = utils.InverseLR(opt, inv_gamma=50000, power=1/2, warmup=0.99)
    ema_sched = utils.EMAWarmup(power=2/3, max_value=0.9999)

    train_set = SampleDataset([args.training_dir], args)
    train_dl = data.DataLoader(train_set, args.batch_size, shuffle=True,
                               num_workers=args.num_workers, persistent_workers=True, pin_memory=True, drop_last=True)

    inner_model, opt, train_dl = accelerator.prepare(inner_model, opt, train_dl)

    if use_wandb:
        wandb.watch(inner_model)
    
    model = Denoiser(inner_model, sigma_data=model_config['sigma_data'])
    model_ema = deepcopy(model)

    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu')
        accelerator.unwrap_model(model.inner_model).load_state_dict(ckpt['model'])
        accelerator.unwrap_model(model_ema.inner_model).load_state_dict(ckpt['model_ema'])
        opt.load_state_dict(ckpt['opt'])
        sched.load_state_dict(ckpt['sched'])
        ema_sched.load_state_dict(ckpt['ema_sched'])
        epoch = ckpt['epoch'] + 1
        step = ckpt['step'] + 1
        del ckpt
    else:
        epoch = 0
        step = 0

    
    sigma_min = model_config['sigma_min']
    sigma_max = model_config['sigma_max']
    assert model_config['sigma_sample_density']['type'] == 'lognormal'
    sigma_mean = model_config['sigma_sample_density']['mean']
    sigma_std = model_config['sigma_sample_density']['std']

    @torch.no_grad()
    @utils.eval_mode(model_ema)
    def demo():
        if accelerator.is_local_main_process:
            tqdm.write('Sampling...')
        filename = f'{args.name}_demo_{step:08}.png'
        n_per_proc = math.ceil(args.n_to_sample / accelerator.num_processes)
        x = torch.randn([n_per_proc, model_config['input_channels'], size[0]], device=device) * sigma_max
        sigmas = inference.get_sigmas_karras(50, sigma_min, sigma_max, rho=7., device=device)
        x_0 = inference.sample_lms(model_ema, x, sigmas, disable=not accelerator.is_local_main_process)
        x_0 = accelerator.gather(x_0)[:args.n_to_sample]

        if accelerator.is_main_process:
            # Demo sample logging. If you get here, great job!
            print(x_0.shape)

    def save():
        accelerator.wait_for_everyone()
        filename = f'{args.name}_{step:08}.pth'
        if accelerator.is_local_main_process:
            tqdm.write(f'Saving to {filename}...')
        obj = {
            'model': accelerator.unwrap_model(model.inner_model).state_dict(),
            'model_ema': accelerator.unwrap_model(model_ema.inner_model).state_dict(),
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
            for batch in tqdm(train_dl, disable=not accelerator.is_local_main_process):
                opt.zero_grad()
                reals = batch[0]
                noise = torch.randn_like(reals)
                sigma = torch.distributions.LogNormal(sigma_mean, sigma_std).sample([reals.shape[0]]).to(device)
                loss = model.loss(reals, noise, sigma).mean()
                accelerator.backward(loss)
                opt.step()
                sched.step()
                ema_decay = ema_sched.get_value()
                utils.ema_update(model, model_ema, ema_decay)
                ema_sched.step()

                if accelerator.is_local_main_process:
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

                if step > 0 and step % args.save_every == 0:
                    save()

                step += 1
            epoch += 1
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()