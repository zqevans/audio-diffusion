#! /usr/bin/env python3

"""
chunkadelic.py 
Author: Scott Hawley
Purpose: Preprocesses a dataset of disparate-sized audio files into entirely uniform chunks

Creates a copy of the filesystem referenced by input paths
"""

import argparse 
from glob import glob 
import os 
from multiprocessing import Pool, cpu_count, Barrier
from functools import partial
import tqdm
from tqdm.contrib.concurrent import process_map  
import torch
import torchaudio
from torchaudio import transforms as T
import math
from dataset.dataset import get_audio_filenames

def is_silence(
    audio,       # torch tensor of multichannel audio
    thresh=-70,  # threshold in dB below which we declare to be silence
    ):
    "checks if entire clip is 'silence' below some dB threshold"
    dBmax = 20*torch.log10(torch.flatten(audio.abs()).max()).cpu().numpy()
    return dBmax < thresh


def load_file(filename, sr=48000):
    audio, in_sr = torchaudio.load(filename)
    if in_sr != sr:
        print(f"Resampling {filename} from {in_sr} Hz to {sr} Hz")
        resample_tf = T.Resample(in_sr, sr)
        audio = resample_tf(audio)
    return audio


def makedir(path):
    if os.path.isdir(path): return  # don't make it if it already exists
    #print(f"  Making directory {path}")
    try:
        os.makedirs(path)  # recursively make all dirs named in path
    except:                # don't really care about errors
        pass


def blow_chunks(
    audio,          # long audio file to be chunked
    new_filename,   # stem of new filename(s) to be output as chunks
    chunk_size:int, # how big each audio chunk is, in samples
    sr=48000,       # audio sample rate in Hz
    overlap=0.5,    # fraction of each chunk to overlap between hops
    strip=False,     # strip silence: chunks with max power in dB below this value will not be saved to files
    thresh=-70      # threshold in dB for determining what counts as silence 
    ):
    "chunks up the audio and saves them with --{i} on the end of each chunk filename"
    chunk = torch.zeros(audio.shape[0], chunk_size)
    _, ext = os.path.splitext(new_filename)

    start, i = 0, 0
    while start < audio.shape[-1]:
        out_filename = new_filename.replace(ext, f'--{i}'+ext) 
        end = min(start + chunk_size, audio.shape[-1])
        if end-start < chunk_size:  # needs zero padding on end
            chunk = torch.zeros(audio.shape[0], chunk_size)
        chunk[:,0:end-start] = audio[:,start:end]
        if (not strip) or (not is_silence(chunk, thresh=thresh)):
            torchaudio.save(out_filename, chunk, sr)
        else:
            print(f"skipping chunk {out_filename} because it's 'silent' (below threhold of {thresh} dB).")
        start, i = start + int(overlap * chunk_size), i + 1
    return 


def process_one_file(filenames, args, file_ind):
    "this chunks up one file"
    filename = filenames[file_ind]  # this is actually input_path+/+filename
    output_path, input_paths = args.output_path, args.input_paths
    new_filename = None
    
    for ipath in input_paths: # set up the output filename & any folders it needs
        if ipath in filename:
            last_ipath = ipath.split('/')[-1]           # get the last part of ipath
            clean_filename = filename.replace(ipath,'') # remove all of ipath from the front of filename
            new_filename = f"{output_path}/{last_ipath}/{clean_filename}".replace('//','/') 
            makedir(os.path.dirname(new_filename))      # we might need to make a directory for the output file
            break
    
    if new_filename is None:
        print(f"ERROR: Something went wrong with name of input file {filename}. Skipping.") 
        return 
    try:
        audio = load_file(filename, sr=args.sr)
        blow_chunks(audio, new_filename, args.chunk_size, sr=args.sr, overlap=args.overlap, strip=args.strip, thresh=args.thresh)
    except Exception as e: 
        print(e)
        print(f"Error loading {filename} or writing chunks. Skipping.")

    return


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--chunk_size', type=int, default=2**17, help='Length of chunks')
    parser.add_argument('--sr', type=int, default=48000, help='Output sample rate')
    parser.add_argument('--overlap', type=float, default=0.5, help='Overlap factor')
    parser.add_argument('--strip', action='store_true', help='Strips silence: chunks with max dB below <thresh> are not outputted')
    parser.add_argument('--thresh', type=int, default=-70, help='threshold in dB for determining what constitutes silence')
    parser.add_argument('output_path', help='Path of output for chunkified data')
    parser.add_argument('input_paths', nargs='+', help='Path(s) of a file or a folder of files. (recursive)')
    args = parser.parse_args()

    print(f"  output_path = {args.output_path}")
    print(f"  chunk_size = {args.chunk_size}")

    torchaudio.set_audio_backend("sox_io")

    print("Getting list of input filenames")
    filenames = get_audio_filenames(args.input_paths)
    # for path in args.input_paths:
    #     for ext in ['wav','flac','ogg']:
    #         filenames += glob(f'{path}/**/*.{ext}', recursive=True)  
    n = len(filenames)   
    print(f"Got {n} input filenames") 

    # for i in range(n):
    #     process_one_file(filenames, args, i)

    print("Processing files (in parallel)")
    wrapper = partial(process_one_file, filenames, args)
    r = process_map(wrapper, range(0, n), chunksize=1, max_workers=48)  # different chunksize used by tqdm. max_workers is to avoid annoying other ppl

    print("Finished")
    

if __name__ == "__main__":
    main()
    
