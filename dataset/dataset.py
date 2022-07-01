import torch
import torchaudio
from os import makedirs
from torchaudio import transforms as T
import random
from glob import glob
import os
from diffusion.utils import RandPool, Stereo, PadCrop, PhaseFlipper, NormInputs, FillTheNoise, OneMinus, RandPool, RandomGain
import tqdm
#import multiprocessing
from multiprocessing import Pool, cpu_count, Barrier
from functools import partial

from udls import SimpleLMDBDataset


class SampleDataset(torch.utils.data.Dataset):
  def __init__(self, paths, global_args):
    super().__init__()
    self.filenames = []

    self.augs = torch.nn.Sequential(
      PadCrop(global_args.sample_size, randomize=global_args.random_crop),
      RandomGain(0.7, 1.0),
      #NormInputs(do_norm=global_args.norm_inputs),
      #OneMinus(), # this is crazy, reverse the signal rel. to +/-1
      #RandPool(),
      #FillTheNoise(),
      PhaseFlipper(),
      #NormInputs(do_norm=global_args.norm_inputs),
    )

    self.encoding = torch.nn.Sequential(
      Stereo()
    )

    for path in paths:
      for ext in ['wav','flac','ogg','aiff','aif','mp3']:
        self.filenames += glob(f'{path}/**/*.{ext}', recursive=True)

    self.sr = global_args.sample_rate
    if hasattr(global_args,'load_frac'):
      self.load_frac = global_args.load_frac
    else:
      self.load_frac = 1.0
    self.num_gpus = global_args.num_gpus

    self.cache_training_data = global_args.cache_training_data

    if self.cache_training_data: self.preload_files()


  def load_file(self, filename):
    audio, sr = torchaudio.load(filename)
    if sr != self.sr:
      resample_tf = T.Resample(sr, self.sr)
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

  def preload_files(self):
      n = int(len(self.filenames)*self.load_frac)
      print(f"Caching {n} input audio files:")
      wrapper = partial(self.load_file_ind, self.filenames)
      start, stop = self.get_data_range()
      with Pool(processes=cpu_count()//8) as p:   # //8 to avoid FS bottleneck and/or too many processes (b/c * num_gpus)
        self.audio_files = list(tqdm.tqdm(p.imap(wrapper, range(start,stop)), total=stop-start))

  def __len__(self):
    return len(self.filenames)

  def __getitem__(self, idx):
    audio_filename = self.filenames[idx]
    try:
      if self.cache_training_data:
        audio = self.audio_files[idx] # .copy()
      else:
        audio = self.load_file(audio_filename)

      #Run augmentations on this sample (including random crop)
      if self.augs is not None:
        audio = self.augs(audio)

      audio = audio.clamp(-1, 1)

      #Encode the file to assist in prediction
      if self.encoding is not None:
        audio = self.encoding(audio)

      return (audio, audio_filename)
    except Exception as e:
     # print(f'Couldn\'t load file {audio_filename}: {e}')
      return self[random.randrange(len(self))]

# A dataset that will return spectrograms alongside the audio data
class SpecDataset(torch.utils.data.Dataset):
  def __init__(self, paths, global_args):
    super().__init__()
    self.filenames = []

    self.augs = torch.nn.Sequential(
      PadCrop(global_args.sample_size),
    )

    self.encoding = torch.nn.Sequential(
      Stereo()
    )


    for path in paths:
      for ext in ['wav','flac','ogg','aiff','aif','mp3']:
        self.filenames += glob(f'{path}/**/*.{ext}', recursive=True)

    self.sr = global_args.sample_rate

    self.to_mel_spec = T.MelSpectrogram(sample_rate=self.sr, n_fft=1024, hop_length=256, n_mels=80, pad_mode='constant')

    self.cache_training_data = global_args.cache_training_data

    if self.cache_training_data:
      self.audio_files = [self.load_file(filename) for filename in self.filenames]

  def load_file(self, filename):
    audio, sr = torchaudio.load(filename)
    if sr != self.sr:
      resample_tf = T.Resample(sr, self.sr)
      audio = resample_tf(audio)
    return audio

  def __len__(self):
    return len(self.filenames)

  def __getitem__(self, idx):
    audio_filename = self.filenames[idx]
    try:
      if self.cache_training_data:
        audio = self.audio_files[idx]
      else:
        audio = self.load_file(audio_filename)
         
      audio = audio.clamp(-1, 1)

      #Run file-level augmentations
      if self.augs is not None:
        audio = self.augs(audio)

      #Encode the file to assist in prediction
      if self.encoding is not None:
        audio = self.encoding(audio)

      spec = self.to_mel_spec(audio) #(C, n_mels, T)

      #Get the mean spectrogram over the dimensions
      spec = torch.mean(spec, 0) #(n_mels, T)

      return (spec, audio, audio_filename)
    except Exception as e:
     # print(f'Couldn\'t load file {audio_filename}: {e}')
      return self[random.randrange(len(self))]


# A dataset that will return MFCCs alongside the audio data
class MFCCDataset(torch.utils.data.Dataset):
  def __init__(self, paths, global_args):
    super().__init__()
    self.filenames = []

    self.augs = torch.nn.Sequential(
      PadCrop(global_args.sample_size),
    )

    self.encoding = torch.nn.Sequential(
      Stereo()
    )

    for path in paths:
      self.filenames += glob(f'{path}/**/*.wav', recursive=True)
      self.filenames += glob(f'{path}/**/*.flac', recursive=True)
      self.filenames += glob(f'{path}/**/*.ogg', recursive=True)
      self.filenames += glob(f'{path}/**/*.aiff', recursive=True)
      self.filenames += glob(f'{path}/**/*.aif', recursive=True)
      self.filenames += glob(f'{path}/**/*.mp3', recursive=True)

    self.sr = global_args.sample_rate

    self.to_mfcc = T.MFCC(sample_rate=self.sr, n_mfcc=80, melkwargs={ "n_fft":1024, "hop_length":256, "n_mels":80, "norm": "slaney"})

    self.cache_training_data = global_args.cache_training_data

    # if self.cache_training_data:
    #   self.audio_files = [None for i in range(len(self.filenames))]

  def load_file(self, filename):
    audio, sr = torchaudio.load(filename)
    if sr != self.sr:
      resample_tf = T.Resample(sr, self.sr)
      audio = resample_tf(audio)
    return audio

  def __len__(self):
    return len(self.filenames)

  def __getitem__(self, idx):
    audio_filename = self.filenames[idx]
    try:
      # if self.cache_training_data and self.audio_files[idx] is not None:
      #   audio = self.audio_files[idx]
      # else:
      #   audio = self.load_file(audio_filename)
      #   self.audio_files[idx] = audio

      audio = self.load_file(audio_filename)

      audio = audio.clamp(-1, 1)

      #Run file-level augmentations
      if self.augs is not None:
        audio = self.augs(audio)

      #Encode the file to assist in prediction
      if self.encoding is not None:
        audio = self.encoding(audio)

      mfcc = self.to_mfcc(audio) #(C, n_mels, T)

      #Get the mean spectrogram over the dimensions
      mfcc = torch.mean(mfcc, 0) #(n_mels, T)

      return (mfcc, audio, audio_filename)
    except Exception as e:
     # print(f'Couldn\'t load file {audio_filename}: {e}')
      return self[random.randrange(len(self))]

def load_file(filename, sample_rate):
    audio, sr = torchaudio.load(filename)
    if sr != sample_rate:
      resample_tf = T.Resample(sr, sample_rate)
      audio = resample_tf(audio)
    return audio

def load_file_ind( file_list, sr, i): # used when caching training data
  return load_file(file_list[i], sr).cpu()
  

class DBDataset(torch.utils.data.Dataset):
  def __init__(self, preprocessed_path, data_paths, global_args, map_size=1e13):
    super().__init__()
    self.filenames = []

    assert data_paths is not None
    makedirs(preprocessed_path, exist_ok=True)

    self.env = SimpleLMDBDataset(preprocessed_path, map_size)

    self.augs = torch.nn.Sequential(
      PadCrop(global_args.sample_size, randomize=global_args.random_crop),
      RandomGain(0.7, 1.0),
      PhaseFlipper(),
    )

    self.encoding = torch.nn.Sequential(
      Stereo()
    )

    for path in data_paths:
      for ext in ['wav','flac','ogg','aiff','aif','mp3']:
        self.filenames += glob(f'{path}/**/*.{ext}', recursive=True)

    self.sr = global_args.sample_rate

    if hasattr(global_args,'load_frac'):
      self.load_frac = global_args.load_frac
    else:
      self.load_frac = 1.0

    self.num_gpus = global_args.num_gpus

     # IF NO DATA INSIDE DATASET: PREPROCESS
    self.len = len(self.env)

    if self.len == 0:
        self.preload_files()
        self.len = len(self.env)
  
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

  def preload_files(self):
      n = int(len(self.filenames)*self.load_frac)
      print(f"Caching {n} input audio files:")
      wrapper = partial(load_file_ind, self.filenames, self.sr)
      start, stop = self.get_data_range()
      with Pool(processes=cpu_count()//8) as p:   # //8 to avoid FS bottleneck and/or too many processes (b/c * num_gpus)
        audio_files = list(tqdm.tqdm(p.imap(wrapper, range(start,stop)), total=stop-start))

      print(f"Adding {len(audio_files)} files to database")
      for i, file in tqdm.tqdm(enumerate(audio_files)):
        self.env[i] = file
  def __len__(self):
        return self.len

  def __getitem__(self, idx):
    audio = self.env[idx]
    try:
      #Run augmentations on this sample (including random crop)
      if self.augs is not None:
        audio = self.augs(audio)

      audio = audio.clamp(-1, 1)

      #Encode the file to assist in prediction
      if self.encoding is not None:
        audio = self.encoding(audio)

      return (audio, self.filenames[idx])
    except Exception as e:
     # print(f'Couldn\'t load file {audio_filename}: {e}')
      return self[random.randrange(len(self))]