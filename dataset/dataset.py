import posixpath
import torch
import torchaudio
from os import makedirs, path
from torchaudio import transforms as T
import random, re
from glob import glob
import os, time
from pedalboard.io import AudioFile
from diffusion.utils import RandPool, Stereo, Mono, PadCrop, PhaseFlipper, NormInputs, FillTheNoise, OneMinus, RandPool, RandomGain, PadCrop_Normalized_T
import tqdm
#import multiprocessing
from multiprocessing import Pool, cpu_count, Barrier
from functools import partial
from einops import rearrange
from udls import SimpleLMDBDataset

import webdataset as wds
from aeiou.core import is_silence
import audio_metadata
import pyloudnorm as pyln

import subprocess

def fast_scandir(
    dir:str,  # top-level directory at which to begin scanning
    ext:list,  # list of allowed file extensions,
    #max_size = 1 * 1000 * 1000 * 1000 # Only files < 1 GB
    ):
    "very fast `glob` alternative. from https://stackoverflow.com/a/59803793/4259243"
    subfolders, files = [], []
    ext = ['.'+x if x[0]!='.' else x for x in ext]  # add starting period to extensions if needed
    try: # hope to avoid 'permission denied' by this try
        for f in os.scandir(dir):
            try: # 'hope to avoid too many levels of symbolic links' error
                if f.is_dir():
                    subfolders.append(f.path)
                elif f.is_file():
                    file_ext = os.path.splitext(f.name)[1].lower()
                    #too_large = os.stat(f.path).st_size > max_size
                    bad_prefix = os.path.basename(f.path).startswith("._")

                    # if file_ext == ".wav" and os.path.exists(f.path.replace(".wav", ".mp3")): #There is a .mp3 in the sample folder with the same name
                    #     print(f"Avoiding WAV copy: {f.path}")
                    #     continue
                    #too_large = os.stat(f.path).st_size > max_size
                    if file_ext in ext and not bad_prefix: #and not too_large:
                        files.append(f.path)
            except:
                pass 
    except:
        pass

    for dir in list(subfolders):
        sf, f = fast_scandir(dir, ext)
        subfolders.extend(sf)
        files.extend(f)
    return subfolders, files

def keyword_scandir(
    dir:str,  # top-level directory at which to begin scanning
    ext:list,  # list of allowed file extensions
    keywords:list,  # list of keywords to search for in the file name
    ):
    "very fast `glob` alternative. from https://stackoverflow.com/a/59803793/4259243"
    subfolders, files = [], []
    keywords = [keyword.lower() for keyword in keywords]  # make keywords case insensitive
    ext = ['.'+x if x[0]!='.' else x for x in ext]  # add starting period to extensions if needed
    banned_words = ["paxheader", "__macosx"]
    try: # hope to avoid 'permission denied' by this try
        for f in os.scandir(dir):
            try: # 'hope to avoid too many levels of symbolic links' error
                if f.is_dir():
                    subfolders.append(f.path)
                elif f.is_file():
                    is_hidden = f.name.split("/")[-1][0] == '.'
                    has_ext = os.path.splitext(f.name)[1].lower() in ext
                    name_lower = f.name.lower()
                    has_keyword = any([keyword in name_lower for keyword in keywords])
                    has_banned = any([banned_word in name_lower for banned_word in banned_words])
                    if has_ext and has_keyword and not has_banned and not is_hidden and not os.path.basename(f.path).startswith("._"):
                        files.append(f.path)
            except:
                pass 
    except:
        pass

    for dir in list(subfolders):
        sf, f = keyword_scandir(dir, ext, keywords)
        subfolders.extend(sf)
        files.extend(f)
    return subfolders, files

def get_audio_filenames(
    paths:list,  # directories in which to search
    keywords = None,
    exts = ['.wav','.mp3','.flac','.ogg','.aif']
    ):
    "recursively get a list of audio filenames"
    filenames = []
    if type(paths) is str: paths = [paths]
    for path in paths:               # get a list of relevant filenames
        if keywords is not None:
          subfolders, files = keyword_scandir(path, exts, keywords)
        else:
          subfolders, files = fast_scandir(path, exts)
        filenames.extend(files)
    return filenames

class SampleDataset(torch.utils.data.Dataset):
  def __init__(self, paths, global_args, keywords = None, relpath=None, random_crop=True):
    super().__init__()
    self.filenames = []
    self.relpath = relpath

    print(f"Random crop: {random_crop}")
    self.augs = torch.nn.Sequential(
      PhaseFlipper(),
    )

    self.pad_crop = PadCrop_Normalized_T(global_args.sample_size, randomize=global_args.random_crop)

    self.encoding = torch.nn.Sequential(
      Stereo()
    )

    # for path in paths:
    #   for ext in ['wav','flac','ogg','aiff','aif','mp3']:
    #     self.filenames += fast_scandir(path, ext) #glob(f'{path}/**/*.{ext}', recursive=True)

    self.filenames = get_audio_filenames(paths, keywords)

    print(f'Found {len(self.filenames)} files')

    self.sr = global_args.sample_rate
    
    self.load_frac = 1.0
    self.num_gpus = global_args.num_gpus

  def load_file(self, filename):
    ext = filename.split(".")[-1]

    if ext == "mp3":
      with AudioFile(filename) as f:
        audio = f.read(f.frames)
        audio = torch.from_numpy(audio)
        in_sr = f.samplerate
    else:
      audio, in_sr = torchaudio.load(filename, format=ext)

    if in_sr != self.sr:
      resample_tf = T.Resample(in_sr, self.sr)
      audio = resample_tf(audio)

    return audio

  def load_file_ind(self, file_list,i): # used when caching training data
    return self.load_file(file_list[i]).cpu()

  def get_data_range(self): # for parallel runs, only grab part of the data
    start, stop = 0, len(self.filenames)
    try: 
      local_rank = int(os.environ["LOCAL_RANK"])
      world_size = int(os.environ["WORLD_SIZE"])
      interval = stop//world_size 
      start, stop = local_rank*interval, (local_rank+1)*interval
      print("local_rank, world_size, start, stop =",local_rank, world_size, start, stop)
      return start, stop
      #rank = os.environ["RANK"]
    except KeyError as e: # we're on GPU 0 and the others haven't been initialized yet
      start, stop = 0, len(self.filenames)//self.num_gpus
      return start, stop

  # def preload_files(self):
  #     n = int(len(self.filenames)*self.load_frac)
  #     print(f"Caching {n} input audio files:")
  #     wrapper = partial(self.load_file_ind, self.filenames)
  #     start, stop = self.get_data_range()
  #     with Pool(processes=cpu_count()) as p:   # //8 to avoid FS bottleneck and/or too many processes (b/c * num_gpus)
  #       self.audio_files = list(tqdm.tqdm(p.imap(wrapper, range(start,stop)), total=stop-start))

  def __len__(self):
    return len(self.filenames)

  def __getitem__(self, idx):
    audio_filename = self.filenames[idx]
    try:
      start_time = time.time()
      audio = self.load_file(audio_filename)

      # if audio.shape[-1] > 60 * self.sr * 1:
      #   print(f"Long file ({audio.shape[-1] / self.sr} seconds): {audio_filename} ")

      audio, t_start, t_end = self.pad_crop(audio)

      #Run augmentations on this sample (including random crop)
      if self.augs is not None:
        audio = self.augs(audio)

      audio = audio.clamp(-1, 1)

      #Encode the file to assist in prediction
      if self.encoding is not None:
        audio = self.encoding(audio)

      if self.relpath is not None:
        audio_path = path.relpath(audio_filename, self.relpath)
      else:
        audio_path = audio_filename

      info = {}

      info["path"] = audio_path

      info["timestamps"] = (t_start, t_end)

      end_time = time.time()

      info["load_time"] = end_time - start_time

      return (audio, info)
    except Exception as e:
      print(f'Couldn\'t load file {audio_filename}: {e}')
      return self[random.randrange(len(self))]

def group_by_keys(data, keys=wds.tariterators.base_plus_ext, lcase=True, suffixes=None, handler=None):
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
            if wds.tariterators.valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffix in current_sample:
            print(f"{fname}: duplicate file name in tar file {suffix} {current_sample.keys()}")
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if wds.tariterators.valid_sample(current_sample):
        yield current_sample

wds.tariterators.group_by_keys = group_by_keys

def get_s3_contents(dataset_path, s3_url_prefix=None, filter='', recursive=True, debug=False, profile='default'):
    """
    Returns a list of full S3 paths to files in a given S3 bucket and directory path.
    """
    # Ensure dataset_path ends with a trailing slash
    if dataset_path != '' and not dataset_path.endswith('/'):
        dataset_path += '/'
    # Use posixpath to construct the S3 URL path
    bucket_path = posixpath.join(s3_url_prefix or '', dataset_path)
    # Construct the `aws s3 ls` command
    cmd = ['aws', 's3', 'ls', bucket_path, '--profile', profile]
    if recursive:
        # Add the --recursive flag if requested
        cmd.append('--recursive')
    # Run the `aws s3 ls` command and capture the output
    run_ls = subprocess.run(cmd, capture_output=True, check=True)
    # Split the output into lines and strip whitespace from each line
    contents = run_ls.stdout.decode('utf-8').split('\n')
    contents = [x.strip() for x in contents if x]
    # Remove the timestamp from lines that begin with a timestamp
    contents = [re.sub(r'^\S+\s+\S+\s+\d+\s+', '', x) if re.match(r'^\S+\s+\S+\s+\d+\s+', x) else x for x in contents]
    # Construct a full S3 path for each file in the contents list
    contents = [posixpath.join(s3_url_prefix or '', x) for x in contents if not x.endswith('/')]
    # Apply the filter, if specified
    if filter:
        contents = [x for x in contents if filter in x]
    # Remove redundant directory names in the S3 URL
    if recursive:
        # Get the main directory name from the S3 URL
        main_dir = "/".join(bucket_path.split('/')[3:])
        # Remove the redundant directory names from each file path
        contents = [x.replace(f'{main_dir}', '').replace('//', '/') for x in contents]
    # Print debugging information, if requested
    if debug:
        print("contents = \n", contents)
    # Return the list of S3 paths to files
    return contents

def get_all_s3_urls(
    names=[],           # list of all valid [LAION AudioDataset] dataset names 
    subsets=[''],       # list of subsets you want from those datasets, e.g. ['train','valid']
    s3_url_prefix=None, # prefix for those dataset names
    recursive=True,     # recursively list all tar files in all subdirs
    filter_str='tar',   # only grab files with this substring
    debug=False,        # print debugging info -- note: info displayed likely to change at dev's whims
    profiles={},        # dictionary of profiles for each item in names, e.g. {'dataset1': 'profile1', 'dataset2': 'profile2'}
):
    "get urls of shards (tar files) for multiple datasets in one s3 bucket"
    urls = []
    for name in names:
        # If s3_url_prefix is not specified, assume the full S3 path is included in each element of the names list
        if s3_url_prefix is None:
            contents_str = name
        else:
            # Construct the S3 path using the s3_url_prefix and the current name value
            contents_str = posixpath.join(s3_url_prefix, name)
        if debug:
            print(f"get_all_s3_urls: {contents_str}:")
        for subset in subsets:
            subset_str = posixpath.join(contents_str, subset)
            if debug:
                print(f"subset_str = {subset_str}")
            # Get the list of tar files in the current subset directory
            profile = profiles.get(name, 'default')
            tar_list = get_s3_contents(subset_str, s3_url_prefix=None, recursive=recursive, filter=filter_str, debug=debug, profile=profile)
            for tar in tar_list:
                # Escape spaces and parentheses in the tar filename for use in the shell command
                tar = tar.replace(" ","\ ").replace("(","\(").replace(")","\)")
                # Construct the S3 path to the current tar file
                s3_path  = posixpath.join(name, subset, tar) + " -"
                # Construct the AWS CLI command to download the current tar file
                if s3_url_prefix is None:
                    request_str = f"pipe:aws s3 --cli-connect-timeout 0 cp {s3_path}"
                else:
                    request_str = f"pipe:aws s3 --cli-connect-timeout 0 cp {posixpath.join(s3_url_prefix, s3_path)}" 
                if profiles.get(name):
                    request_str += f" --profile {profiles.get(name)}"
                if debug:
                    print("request_str = ", request_str)
                # Add the constructed URL to the list of URLs
                urls.append(request_str)
    return urls

def wds_preprocess(
  sample, 
  sample_size=65536, 
  sample_rate=48000, 
  verbose=False, 
  random_crop=True, 
  normalize_lufs=None, 
  metadata_prompt_funcs=None,
  force_channels = "stereo",
  augment_phase = True,
):
    "utility routine for QuickWebDataLoader, below"
    audio_keys = ("flac", "wav", "mp3", "m4a", "ogg")

    found_key, rewrite_key = '', ''
    for k,v in sample.items():  # print the all entries in dict
        for akey in audio_keys:
            if k.endswith(akey): 
                found_key, rewrite_key = k, akey  # to rename long/weird key with its simpler counterpart
                break
        if '' != found_key: break 
    if '' == found_key:  # got no audio!   
        # print("  Error: No audio in this sample:")
        # for k,v in sample.items():  # print the all entries in dict
        #     print(f"    {k:20s} {repr(v)[:50]}")
        # print("       Skipping it.")
        return None  # try returning None to tell WebDataset to skip this one ?   
    
    audio, in_sr = sample[found_key]
    if in_sr != sample_rate:
        if in_sr < 8000:
          print(f"Very low SR ({in_sr}) for file {sample['url']}")
        if verbose: print(f"Resampling from {in_sr} Hz to {sample_rate} Hz",flush=True)
        resample_tf = T.Resample(in_sr, sample_rate)
        audio = resample_tf(audio)        

    if normalize_lufs is not None:
      # Loudness normalization to -12 LKFS, adapted from pyloudnorm
      meter = pyln.Meter(sample_rate)
      loudness = meter.integrated_loudness(audio.transpose(-2, -1).numpy())
      delta_loudness = (normalize_lufs - float(loudness))
      gain = 10.0 ** (delta_loudness/20.0)
      audio = gain * audio

    if sample_size is not None:
      # Pad/crop and get the relative timestamp
      pad_crop = PadCrop_Normalized_T(sample_size, randomize=random_crop, sample_rate=sample_rate)
      audio, t_start, t_end, seconds_start, seconds_total = pad_crop(audio)
      sample["json"]["seconds_start"] = seconds_start
      sample["json"]["seconds_total"] = seconds_total
    else:
      t_start, t_end = 0, 1

    #Check if audio is length zero, initialize to a single zero if so
    if audio.shape[-1] == 0:
      audio = torch.zeros(1, 1)

    # Make the audio stereo and augment by randomly inverting phase
    augs = torch.nn.Sequential(
      Stereo() if force_channels == "stereo" else torch.nn.Identity(), 
      Mono() if force_channels == "mono" else torch.nn.Identity(),
      PhaseFlipper() if augment_phase else torch.nn.Identity()
    )
    audio = augs(audio)

    sample["timestamps"] = (t_start, t_end)

    if "text" in sample["json"]:
      sample["json"]["prompt"] = sample["json"]["text"]

    if metadata_prompt_funcs is not None:
        for key, prompt_func in metadata_prompt_funcs.items():
            if key in sample["__url__"]:
                prompt = prompt_func(sample["json"])
                sample["json"]["prompt"] = prompt

    if found_key != rewrite_key:   # rename long/weird key with its simpler counterpart
        del sample[found_key]
    sample["audio"] = audio    
    return sample

def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, isssue a warning, and continue."""
    print(f"Handling webdataset error ({repr(exn)}). Ignoring.")
    rank, world_size, worker, num_workers = wds.utils.pytorch_worker_info()
    print(f"Rank: {rank}, worker: {worker}")
    return True

def get_laion_630k_loader(batch_size, sample_size, sample_rate=48000, num_workers=8):
    names = [
      "freesound_no_overlap", 
      "sonniss_game_effects", 
      #"wesoundeffects", # Restrictive license
      "paramount_motion", 
      #"audiostock", #Audible watermark
      "BBCSoundEffects", 
      "epidemic_sound_effects", 
      "audiocaps", 
      "free_to_use_sounds",
      "FMA"
    ]
    return get_wds_loader(batch_size=batch_size, sample_size=sample_size, names=names, s3_url_prefix="s3://s-laion-audio/webdataset_tar/", sample_rate=sample_rate, num_workers=num_workers)

def is_valid_sample(sample):
  return "json" in sample and "audio" in sample and not is_silence(sample["audio"])

def get_wds_loader(batch_size, sample_size, names, s3_url_prefix=None, sample_rate=48000, num_workers=8, recursive=True, profiles={}, epoch_steps=1000, random_crop=True, normalize_lufs=None, metadata_prompt_funcs=None, force_channels="stereo", augment_phase=True):

  preprocess_fn = partial(wds_preprocess, sample_size=sample_size, sample_rate=sample_rate, random_crop=random_crop, normalize_lufs=normalize_lufs, metadata_prompt_funcs=metadata_prompt_funcs, force_channels=force_channels, augment_phase=augment_phase)

  urls = get_all_s3_urls(
      names=names, 
      s3_url_prefix=s3_url_prefix,
      recursive=recursive,
      profiles=profiles
  )

  dataset = wds.DataPipeline(
      wds.ResampledShards(urls), # Yields a single .tar URL
      wds.tarfile_to_samples(handler=log_and_continue), # Opens up a stream to the TAR file, yields files grouped by keys
      wds.decode(wds.torch_audio, handler=log_and_continue),
      wds.map(preprocess_fn, handler=log_and_continue),
      #wds.shuffle(bufsize=100, initial=10, handler=log_and_continue), # Pulls from iterator until initial value
      wds.select(is_valid_sample),
      wds.to_tuple("audio", "json", "timestamps", handler=log_and_continue),
      wds.batched(batch_size, partial=False)
  ).with_epoch(epoch_steps//num_workers if num_workers > 0 else epoch_steps)

  return wds.WebLoader(dataset, num_workers=num_workers)
