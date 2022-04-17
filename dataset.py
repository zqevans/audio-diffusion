import torch
import torchaudio
import random
from torchaudio import transforms as T
from glob import glob

class SampleDataset(torch.utils.data.Dataset):
  def __init__(self, paths, transform=None):
    super().__init__()
    self.filenames = []
    self.transform = transform
    for path in paths:
      self.filenames += glob(f'{path}/**/*.wav', recursive=True)

  def __len__(self):
    return len(self.filenames)

  def __getitem__(self, idx):
    audio_filename = self.filenames[idx]
    try:
      audio, sr = torchaudio.load(audio_filename, normalize=True)
      if sr != 44100:
          resample_tf = T.Resample(sr, 44100)
          audio = resample_tf(audio)
      audio = audio.clamp(-1, 1)

      if self.transform is not None:
        audio = self.transform(audio)

      return audio
    except Exception as e:
     # print(f'Couldn\'t load file {audio_filename}: {e}')
      return self[random.randrange(len(self))]
