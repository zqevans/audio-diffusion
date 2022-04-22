import torch
import torchaudio
from torchaudio import transforms as T
import random
from glob import glob

from .utils import MidSideEncoding, Stereo, RandomGain, PadCrop

class SampleDataset(torch.utils.data.Dataset):
  def __init__(self, paths, global_args):
    super().__init__()
    self.filenames = []

    self.transform = torch.nn.Sequential(
        #augmentations and normalization
        RandomGain(0.5, 1.0),
        PadCrop(global_args.sample_size),

        #encoding
        Stereo(),
        #MidSideEncoding()
    )

    for path in paths:
      self.filenames += glob(f'{path}/**/*.wav', recursive=True)

    self.num_files = len(self.filenames)
    self.data_repeats = global_args.data_repeats

  def __len__(self):
    return self.num_files * self.data_repeats

  def __getitem__(self, idx):
    audio_filename = self.filenames[idx % self.num_files]
    try:
      audio, sr = torchaudio.load(audio_filename, normalize=True)
      if sr != 44100:
          resample_tf = T.Resample(sr, 44100)
          audio = resample_tf(audio)
      audio = audio.clamp(-1, 1)

      if self.transform is not None:
        audio = self.transform(audio)

      return (audio, audio_filename)
    except Exception as e:
     # print(f'Couldn\'t load file {audio_filename}: {e}')
      return self[random.randrange(len(self))]
