import torch
import torchaudio
from torchaudio import transforms as T
import random
from glob import glob

from ..diffusion.utils import MidSideEncoding, Stereo, RandomGain, PadCrop

class SampleDataset(torch.utils.data.Dataset):
  def __init__(self, paths, global_args):
    super().__init__()
    self.filenames = []

    self.augs = torch.nn.Sequential(
      #RandomGain(0.9, 1.0),
      PadCrop(global_args.sample_size),
    )

    self.encoding = torch.nn.Sequential(
      Stereo(),
      MidSideEncoding()
    )

    for path in paths:
      self.filenames += glob(f'{path}/**/*.wav', recursive=True)
      self.filenames += glob(f'{path}/**/*.flac', recursive=True)
      self.filenames += glob(f'{path}/**/*.ogg', recursive=True)
      self.filenames += glob(f'{path}/**/*.aiff', recursive=True)
      self.filenames += glob(f'{path}/**/*.aif', recursive=True)
      self.filenames += glob(f'{path}/**/*.mp3', recursive=True)

    self.num_files = len(self.filenames)
    self.data_repeats = global_args.data_repeats
    
    self.sr = global_args.sample_rate

  def __len__(self):
    return self.num_files * self.data_repeats

  def __getitem__(self, idx):
    audio_filename = self.filenames[idx % self.num_files]
    try:
      audio, sr = torchaudio.load(audio_filename)
      if sr != self.sr:
          resample_tf = T.Resample(sr, self.sr)
          audio = resample_tf(audio)
          
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
