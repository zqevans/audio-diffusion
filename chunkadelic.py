__all__ = ['blow_chunks', 'set_bit_rate', 'chunk_one_file', 'main']

# %% ../03_chunkadelic.ipynb 5
import argparse 
import os 
from functools import partial
from tqdm.contrib.concurrent import process_map  
import torch
import torchaudio
import math
from aeiou.core import is_silence, load_audio, makedir, get_audio_filenames, normalize_audio, get_dbmax
import multiprocessing as mp
from multiprocessing import Pool, cpu_count, Barrier

# %% ../03_chunkadelic.ipynb 6
def blow_chunks(
    audio:torch.tensor,  # long audio file to be chunked
    new_filename:str,    # stem of new filename(s) to be output as chunks
    chunk_size:int,      # how big each audio chunk is, in samples
    sr=48000,            # audio sample rate in Hz
    norm='False',      # normalize input audio, based on the max of the absolute value ['global','channel', or anything else for None, e.g. False]
    spacing=0.5,         # fraction of each chunk to advance between hops
    strip=False,    # strip silence: chunks with max power in dB below this value will not be saved to files
    thresh=-70,      # threshold in dB for determining what counts as silence
    bits_per_sample=None, # kwarg for torchaudio.save, None means use defaults
    debug=False,     # print debugging information 
    ):
    "chunks up the audio and saves them with --{i} on the end of each chunk filename"
    if (debug): print(f"       blow_chunks: audio.shape = {audio.shape}",flush=True)
        
    chunk = torch.zeros(audio.shape[0], chunk_size)
    _, ext = os.path.splitext(new_filename)
    
    if norm in ['global','channel']:  audio = normalize_audio(audio, norm)     

    spacing = 0.5 if spacing == 0 else spacing # handle degenerate case as a request for the defaults
    
    start, i = 0, 0
    while start < audio.shape[-1]:
        out_filename = new_filename.replace(ext, f'--{i}'+ext) 
        end = min(start + chunk_size, audio.shape[-1])
        if end-start < chunk_size:  # needs zero padding on end
            chunk = torch.zeros(audio.shape[0], chunk_size)
        chunk[:,0:end-start] = audio[:,start:end]
        if (not strip) or (not is_silence(chunk, thresh=thresh)):
            if debug: print(f"     Saving output chunk {out_filename}, bits_per_sample={bits_per_sample}", flush=True)
            torchaudio.save(out_filename, chunk, sr, bits_per_sample=bits_per_sample)
        else:
            print(f"Skipping chunk {out_filename} because it's 'silent' (below threhold of {thresh} dB).",flush=True)
        start, i = start + int(spacing * chunk_size), i + 1
    return 

# %% ../03_chunkadelic.ipynb 7
def set_bit_rate(bits, filename, debug=False):
    if (bits is None) or isinstance(bits, int): bits_per_sample = bits
    elif bits.lower()=='none': 
        bits_per_sample = None  # use torchaudio default 
    elif bits.lower()=='match':
        try:
            bits_per_sample = torchaudio.info(filename).bits_per_sample
        except Exception as e:
            print("     Error with bits=match: Can't get audio medatadata. Choosing default=None")
            bits_per_sample=None
    else:
        bits_per_sample =  int(bits)
    if debug: print("     set_bit_rate: bits_per_sample =",bits_per_sample,flush=True)
    return bits_per_sample

# %% ../03_chunkadelic.ipynb 8
def chunk_one_file(
    filenames:list,      # list of filenames from which we'll pick one
    args,                # output of argparse
    file_ind             # index from filenames list to read from
    ):
    "this chunks up one file by setting things up and then calling blow_chunks"
    filename = filenames[file_ind]  # this is actually input_path+/+filename
    output_path, input_paths = args.output_path, args.input_paths
    new_filename = None
    if args.debug: print(f" --- process_one_file: filenames[{file_ind}] = {filename}\n", flush=True)
    
    for ipath in input_paths: # set up the output filename & any folders it needs
        if args.nomix and ('Mix' in ipath) and ('Audio Files' in ipath): return  # this is specific to the BDCT dataset, otherwise ignore
        if ipath in filename:
            last_ipath = ipath.split('/')[-1]           # get the last part of ipath
            clean_filename = filename.replace(ipath,'') # remove all of ipath from the front of filename
            new_filename = f"{output_path}/{last_ipath}/{clean_filename}".replace('//','/') 
            makedir(os.path.dirname(new_filename))      # we might need to make a directory for the output file
            break

    if new_filename is None:
        print(f"ERROR: Something went wrong with name of input file {filename}. Skipping.",flush=True) 
        return 
    
    try:  # try to load the audio file and chunk it up
        if args.debug: print(f"   About to load filenames[{file_ind}] = {filename}\n", flush=True)
        audio = load_audio(filename, sr=args.sr, verbose=args.debug)
        if args.debug: print(f"   We loaded the audio, audio.shape = {audio.shape}.  Setting bit rate.",flush=True)  
        bits_per_sample = set_bit_rate(args.bits, filename, debug=args.debug)
        if args.debug: print(f"   Bit rate set.  Calling blow_chunks...", flush=True)
        blow_chunks(audio, new_filename, args.chunk_size, sr=args.sr, spacing=args.spacing, 
                    strip=args.strip, thresh=args.thresh, bits_per_sample=bits_per_sample, debug=args.debug)
    except Exception as e: 
        print(f"Error '{e}' while loading {filename} or writing chunks. Skipping.", flush=True)

    if args.debug: print(f" --- File {file_ind}: {filename} completed.\n", flush=True)
    return

# %% ../03_chunkadelic.ipynb 12
def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--chunk_size', type=int, default=2**17, help='Length of chunks')
    parser.add_argument('--sr', type=int, default=48000, help='Output sample rate')
    parser.add_argument('--norm', default='False', const='False', nargs='?', choices=['False', 'global', 'channel'],
                   help='Normalize audio, based on the max of the absolute value [global/channel/False]')
    parser.add_argument('--spacing', type=float, default=0.5, help='Spacing factor, advance this fraction of a chunk per copy')
    parser.add_argument('--strip', action='store_true', help='Strips silence: chunks with max dB below <thresh> are not outputted')
    parser.add_argument('--thresh', type=int, default=-70, help='threshold in dB for determining what constitutes silence')
    parser.add_argument('--bits', type=str, default='None', help='Bit depth: "None" uses torchaudio default | "match"=match input audio files | or specify an int')
    parser.add_argument('--workers', type=int, default=min(32, os.cpu_count() + 4), help='Maximum number of workers to use (default: all)')
    parser.add_argument('--nomix', action='store_true',  help='(BDCT Dataset specific) exclude output of "*/Audio Files/*Mix*"')
    parser.add_argument('output_path', help='Path of output for chunkified data')
    parser.add_argument('input_paths', nargs='+', help='Path(s) of a file or a folder of files. (recursive)')
    parser.add_argument('--verbose', action='store_true',  help='Extra output logging')
    parser.add_argument('--debug', action='store_true',  help='Extra EXTRA output logging')
    args = parser.parse_args()
   
    if args.verbose: 
        print("chunkadelic: args = ",args)
        print("Getting list of input filenames")
    filenames = get_audio_filenames(args.input_paths)
    if args.verbose:
        print(f"  Got {len(filenames)} input filenames") 
        if not (args.norm in ['global','channel']): 
            print(f"Warning: since norm = {args.norm}, no normalizations will be performed.")
        print("Processing files (in parallel)...")
            
    wrapper = partial(chunk_one_file, filenames, args)
    r = process_map(wrapper, range(len(filenames)), chunksize=1, max_workers=args.workers)  # different chunksize used by tqdm. max_workers is to avoid annoying other ppl
  
    if args.verbose: print("Finished")      


if __name__ == '__main__': 
    main()