#!/usr/bin/env python3

## Just mixes 'reals' and 'recon' files with a delay and gain reduction on the latter
## Puts new files with "delay_00" in filename
## Reads & writes from current directory
## Author: Scott H. Hawley (scott.hawley at belmont.edu)

import torchaudio
from torchaudio import transforms as T
import torch.nn.functional as F 
import librosa 
from glob import glob
import numpy as np
import argparse 

MAIN_SR = 48000

def load_file(filename):
    audio, sr = torchaudio.load(filename)
    if sr != MAIN_SR:
        resample_tf = T.Resample(sr, MAIN_SR)
        audio = resample_tf(audio)
    return audio

def main():
    
    p = argparse.ArgumentParser()
    p.add_argument('--delay', type=float, default=1.0,
                   help='relative delay (in terms of sample size)')
    p.add_argument('--gain', type=float, default=0.75,
                   help='relative gain for echo')
    args = p.parse_args()

    path = '.'
    reals_fnames = []
    recon_fnames = []

    for ext in ['wav','flac','ogg','aiff','aif','mp3']:
        reals_fnames += sorted(glob(f'{path}/reals_*.{ext}'))
        recon_fnames += sorted(glob(f'{path}/recon_*.{ext}'))

    assert len(reals_fnames) == len(recon_fnames)

    for (reals_file, recon_file) in zip(reals_fnames, recon_fnames):
        mixed_file = reals_file.replace('reals','delay')
        print(f"{mixed_file}")
        real = load_file(reals_file)
        recon = load_file(recon_file)
        clip_len = real.size()[1]//16  # 16 demos per clip is so far consisten
        delay_samples = int(clip_len * args.delay)
        echo =  F.pad(recon,  (delay_samples, 0, 0, 0) ) * args.gain
        padreal = F.pad(real, (0, delay_samples, 0, 0) )
        mixed = padreal + echo  # TODO: more sophisticated mixing
        torchaudio.save(mixed_file, mixed, MAIN_SR)

if __name__ == '__main__':
    main()


