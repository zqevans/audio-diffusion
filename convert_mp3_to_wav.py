#! /usr/bin/env python3

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
import subprocess

def process_one_file(filenames, args, file_ind):
    "this chunks up one file"
    filename = filenames[file_ind]  # this is actually input_path+/+filename
    input_paths = args.input_paths
    new_filename = None
    
    path, ext = os.path.splitext(filename)

    new_filename = f"{path}.wav"
    
    if new_filename is None:
        print(f"ERROR: Something went wrong with name of input file {filename}. Skipping.") 
        return 
    try:
        subprocess.call(['ffmpeg', '-i', filename,
                   new_filename])
    except Exception as e: 
        print(e)
        print(f"Error loading {filename} or writing chunks. Skipping.")

    return


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_paths', nargs='+', help='Path(s) of a file or a folder of files. (recursive)')
    args = parser.parse_args()

    #print(f"  output_path = {args.output_path}")
    #print(f"  chunk_size = {args.chunk_size}")

    torchaudio.set_audio_backend("sox_io")

    print("Getting list of input filenames")
    filenames = get_audio_filenames(args.input_paths, exts=[".mp3"])
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
    
