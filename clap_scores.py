#!/usr/bin/env python3

from prefigure.prefigure import get_all_args, push_wandb_config
from contextlib import contextmanager
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
import laion_clap
from laion_clap.training.data import get_audio_features

from functools import partial

from prompts.prompters import get_prompt_from_audio_file_metadata

from dataset.dataset import get_all_s3_urls, get_s3_contents, get_wds_loader, wds_preprocess, log_and_continue, is_valid_sample
import webdataset as wds
import time
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import json

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

def collation_fn(batch):
    # Given input [[info1, prompt1], [info2, prompt2], ...]
    # Return [info1, info2, ...], [prompt1, prompt2, ...]
    return list(zip(*batch))
    #return batch

def main():

    args = get_all_args()

    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    torch.manual_seed(args.seed)

    print("Creating data loader")

    preprocess_fn = partial(wds_preprocess, 
        sample_size=None, 
        sample_rate=args.sample_rate, 
        random_crop=args.random_crop, 
        #verbose=True, 
        #normalize_lufs=-12.0,
        augment_phase=False,
        force_channels="mono"
    )
    
    names = [
    
    ]

    urls = get_all_s3_urls(
        names=names, 
        s3_url_prefix=None,
        recursive=True,
    )

    clap_model = laion_clap.CLAP_Module(enable_fusion=args.clap_fusion, device=device, amodel= args.clap_amodel).requires_grad_(False).eval()

    if args.clap_ckpt_path:
        clap_model.load_ckpt(ckpt=args.clap_ckpt_path)
    else:
        clap_model.load_ckpt(model_id=1)

    # Fix for webdataset not being able to handle multiple keys with the same name
    wds.tariterators.group_by_keys = group_by_keys

    dataset = wds.DataPipeline(
        wds.ResampledShards(urls), # Yields a single .tar URL
        wds.split_by_worker,
        wds.tarfile_to_samples(handler=log_and_continue), # Opens up a stream to the TAR file, yields files grouped by keys
        wds.decode(wds.torch_audio, handler=log_and_continue),
        wds.map(preprocess_fn),
        wds.select(is_valid_sample),
        wds.to_tuple("audio", "json", "timestamps", handler=log_and_continue),
        wds.batched(args.batch_size, collation_fn=collation_fn, partial=True)
    )

    data_loader = iter(wds.WebLoader(dataset, num_workers=args.num_workers))

    print("Creating data loader")

    start_time = time.time()

    cosine_similarities = []
    prompts = []
    audio_embeds_list = []

    for i in range(500):
        try:
            sample = next(data_loader)
            audios, jsons, _ = sample
            print(f"Batch {i}")
            prompts_batch = [json["prompt"][0] for json in jsons]
            audios_batch = [audio[0] for audio in audios]
            audio_embeds = clap_model.get_audio_embedding_from_data(audios_batch, use_tensor=True)
            text_embeds = clap_model.get_text_embedding(prompts_batch, use_tensor=True)
            cosine_sims = torch.nn.functional.cosine_similarity(audio_embeds, text_embeds, dim=1)
            cosine_similarities.extend(cosine_sims.tolist())
            prompts.extend(prompts_batch)
            audio_embeds_list.append(audio_embeds)
            samples_per_sec = ((i+1) * args.batch_size) / (time.time() - start_time)
            print(f"Samples/sec this epoch: {samples_per_sec}")
        except Exception as e:
            print(e)
            continue

    # Concatenate audio embeddings
    audio_embeds_all = torch.cat(audio_embeds_list, dim=0).cpu().numpy()
    audio_embeds_all = audio_embeds_all.astype(np.float32)

    # Perform PCA on audio embeddings
    pca = PCA(n_components=2)
    audio_embeds_pca = pca.fit_transform(audio_embeds_all)

    # Perform t-SNE on audio embeddings
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000)
    audio_embeds_tsne = tsne.fit_transform(audio_embeds_all)

    # Calculate and sort cosine similarities
    results = sorted(zip(prompts, cosine_similarities), key=lambda x: x[1], reverse=True)

    # Save sorted results to file
    with open("results_sorted.txt", "w") as f:
        for prompt, cosine_sim in results:
            f.write(f"Prompt: {prompt}, Cosine Similarity: {cosine_sim:.4f}\n")

    # Combine prompts and UMAP embeddings into a list of dictionaries
    data = []
    for i in range(len(prompts)):
        point = {
            "prompt": prompts[i], 
            "cosine_sim": float(cosine_similarities[i]), 
            "pca_x": float(audio_embeds_pca[i, 0]), 
            "pca_y": float(audio_embeds_pca[i, 1]), 
            #"pca_z": float(audio_embeds_pca[i, 2]),
            "tsne_x": float(audio_embeds_tsne[i, 0]), 
            "tsne_y": float(audio_embeds_tsne[i, 1]), 
            #"tsne_z": float(audio_embeds_tsne[i, 2])
        }
        data.append(point)

    # Write data to JSON file
    with open("data.json", "w") as f:
        json.dump(data, f)

if __name__ == '__main__':
    main()