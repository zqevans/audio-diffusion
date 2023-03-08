#!/usr/bin/env python3

from prefigure.prefigure import get_all_args, push_wandb_config
from contextlib import contextmanager
from copy import deepcopy
import math
from pathlib import Path

import sys, re
import random
import torch
from torch import optim, nn
from torch.nn import functional as F
from torchaudio import transforms as T
from torch.utils import data
from tqdm import trange
from einops import rearrange
import numpy as np
import torchaudio

from functools import partial

import wandb

from dataset.dataset import get_all_s3_urls, get_s3_contents, get_wds_loader, wds_preprocess, log_and_continue, is_valid_sample
from prompts.prompters import get_prompt_from_jmann_metadata, get_prompt_from_fma_metadata, get_prompt_from_audio_file_metadata
import webdataset as wds
import time

def base_plus_ext(path):
    """Split off all file extensions.
    Returns base, allext.
    :param path: path with extensions
    :param returns: path with all extensions removed
    """
    match = re.match(r"^((?:.*/|)[^.]+)[.]([^/]*)$", path)
    if not match:
        return None, None
    return match.group(1), match.group(2)

def group_by_keys(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.
    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    print("Running new function")
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if wds.tariterators.trace:
            print(
                prefix,
                suffix,
                current_sample.keys() if isinstance(current_sample, dict) else None,
            )
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        if current_sample is None or prefix != current_sample["__key__"]:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffix in current_sample:
            raise ValueError(
                f"{fname}: duplicate file name in tar file {suffix} {current_sample.keys()}"
            )
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample

def valid_sample(sample):
    """Check whether a sample is valid.
    :param sample: sample to be checked
    """
    return (
        sample is not None
        and isinstance(sample, dict)
        and len(list(sample.keys())) > 0
        and not sample.get("__bad__", False)
    )


# Creates and returns a text prompt given a metadata object
def get_prompt_from_metadata(metadata):

    #print(metadata)

    return ""


def main():

    args = get_all_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    torch.manual_seed(args.seed)

    print("Creating data loader")

    # train_dl = iter(get_wds_loader(
    #     batch_size=args.batch_size, 
    #     s3_url_prefix=None, 
    #     sample_size=args.sample_size, 
    #     names=names, 
    #     sample_rate=args.sample_rate, 
    #     num_workers=args.num_workers, 
    #     recursive=True,
    #     random_crop=True
    # ))

    
    
    names = [
     
    ]

    metadata_prompt_funcs = {}
  

    urls = get_all_s3_urls(
        names=names, 
        #s3_url_prefix="",
        recursive=True,
    )

    preprocess_fn = partial(wds_preprocess, 
        sample_size=args.sample_size, 
        sample_rate=args.sample_rate, 
        random_crop=args.random_crop, 
        #verbose=True, 
        normalize_lufs=-12.0,
        metadata_prompt_funcs=metadata_prompt_funcs
    )


    def print_inputs(inputs):
        print(f"Sample: {inputs}")
        return inputs

    wds.tariterators.group_by_keys = group_by_keys

    dataset = wds.DataPipeline(
        wds.ResampledShards(urls), # Yields a single .tar URL
        wds.split_by_worker,
        wds.map(print_inputs),
        wds.tarfile_to_samples(handler=log_and_continue), # Opens up a stream to the TAR file, yields files grouped by keys
        #wds.shuffle(bufsize=100, initial=10), # Pulls from iterator until initial value
        wds.decode(wds.torch_audio, handler=log_and_continue),
        #wds.LMDBCached(f"/scratch/wds_lmdb_{time.time()}", mrm -ap_size=6e12),
        wds.map(preprocess_fn, handler=log_and_continue),
        wds.select(is_valid_sample),
        #wds.Cached(),
        #wds.shuffle(bufsize=100, initial=10, handler=log_and_continue), # Pulls from iterator until initial value
        wds.to_tuple("audio", "json", "timestamps", handler=log_and_continue),
        wds.batched(args.batch_size, partial=False)
    )

    train_dl = wds.WebLoader(dataset, num_workers=args.num_workers)

    print("Creating data loader")

    max_seconds_total = 0

    #for json in train_dl:
    for epoch_num in range(1):
        train_iter = iter(train_dl)
        print(f"Starting epoch {epoch_num}")
        start_time = time.time()
        for i, sample in enumerate(train_iter):
            #json = next(train_dl)
            audio, jsons, timestamps = sample
            print(f"Epoch {epoch_num} Batch {i}")
            for json in jsons:
                prompt = json["prompt"][0]
                seconds_total = json["seconds_total"][0].item()
                if seconds_total > max_seconds_total:
                    max_seconds_total = seconds_total
                    print(prompt)
                    print(max_seconds_total)

                print(max_seconds_total)
#            print(audio.shape)
            samples_per_sec = ((i+1) * args.batch_size) / (time.time() - start_time)
            #print(f"Samples/sec this epoch: {samples_per_sec}")
            #time.sleep(5.0)

if __name__ == '__main__':
    main()