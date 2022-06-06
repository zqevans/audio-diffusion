import torch
import torchaudio
from torchaudio import transforms as T
import random
from glob import glob
from diffusion.utils import Stereo, PadCrop, PhaseFlipper, NormInputs

class SampleDataset(torch.utils.data.Dataset):
  def __init__(self, paths, global_args):
    super().__init__()
    self.filenames = []

    self.augs = torch.nn.Sequential(
      #RandomGain(0.9, 1.0),
      PadCrop(global_args.sample_size, randomize=global_args.random_crop),
      PhaseFlipper(),
      NormInputs(do_norm=global_args.norm_inputs)
    )

    self.encoding = torch.nn.Sequential(
      Stereo()
    )

    for path in paths:
      for ext in ['wav','flac','ogg','aiff','aif','mp3']:
        self.filenames += glob(f'{path}/**/*.{ext}', recursive=True)

    self.sr = global_args.sample_rate

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