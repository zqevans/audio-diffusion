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


def blow_chunks(audio, new_filename, chunk_size, sr=48000, overlap=0.5):
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
        torchaudio.save(out_filename, chunk, sr)
        start, i = start + int(overlap * chunk_size), i + 1

    return 


def process_one_file(filenames, args, file_ind):
    "this chunks up one file"
    filename = filenames[file_ind]  # this is actually input_path+/+filename
    chunk_size, sr, overlap, output_path, input_paths = args.chunk_size, args.sr, args.overlap, args.output_path, args.input_paths
    new_filename = None
    
    for ipath in input_paths: # set up the output filename & any folders it needs
        if ipath in filename:
            last_ipath = ipath.split('/')[-1]           # get the last part of ipath
            clean_filename = filename.replace(ipath,'') # remove all of ipath from the front of filename
            new_filename = f"{output_path}/{last_ipath}/{clean_filename}".replace('//','/') 
            makedir(os.path.dirname(new_filename))      # we might need to make a directory for the output file
            break
    
    if new_filename is None:
        print(f"ERROR: Something went wrong with input file {filename}") 
        return 

    audio = load_file(filename, sr=sr)
    blow_chunks(audio, new_filename, chunk_size, sr=sr, overlap=overlap)
    return


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--chunk_size', type=int, default=2**17, help='Length of chunks')
    parser.add_argument('--sr', type=int, default=48000, help='Output sample rate')
    parser.add_argument('--overlap', type=float, default=0.5, help='Overlap factor')
    parser.add_argument('output_path', help='Path of output for chunkified data')
    parser.add_argument('input_paths', nargs='+', help='Path(s) of a file or a folder of files. (recursive)')
    args = parser.parse_args()

    print(f"  output_path = {args.output_path}")
    print(f"  chunk_size = {args.chunk_size}")

    print("Getting list of input filenames")
    filenames = []
    for path in args.input_paths:
      for ext in ['wav','flac','ogg','aiff','aif','mp3']:
        filenames += glob(f'{path}/**/*.{ext}', recursive=True)  
    n = len(filenames)   
    print(f"  Got {n} input filenames") 

    print("Processing files (in parallel)")
    wrapper = partial(process_one_file, filenames, args)
    r = process_map(wrapper, range(0, n), chunksize=1, max_workers=24)  # different chunksize used by tqdm. max_workers is to avoid annoying other ppl

    print("Finished")

if __name__ == "__main__":
    main()
    
