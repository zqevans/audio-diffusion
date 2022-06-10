# Utilities from Ethan Manilows's TagBox: https://github.com/ethman/tagbox
# slightly modified by Scott H. Hawley @drscotthawley

import librosa
import numpy as np
import torch
from einops import rearrange


#JUKEBOX_SAMPLE_RATE = 44100  # ethan's original
JUKEBOX_SAMPLE_RATE = None

def init_jukebox_sample_rate(sr=44100): # will probably use 48000 in practice
    "SHH added this util to preserve rest of code minimall-modified"
    global JUKEBOX_SAMPLE_RATE
    JUKEBOX_SAMPLE_RATE = sr
    return



def audio_for_jbx(audio, trunc_sec=None, device=None):
    """Readies an audio TENSOR for Jukebox."""
    if audio.ndim == 1:
        audio = audio[None]
        audio = audio.mean(axis=0)

    # normalize audio
    norm_factor = torch.abs(audio).max()
    if norm_factor > 0:
        audio /= norm_factor

    #print("1 audio.size() = ",audio.size())
    #audio = audio.flatten()  #whahhh? # stero to mono maybe?
    #audio = rearrange(audio, 'b d n -> b (d n)')

    if trunc_sec is not None:  # truncate sequence
        audio = audio[: int(JUKEBOX_SAMPLE_RATE * trunc_sec)]

    #print("2 audio.size() = ",audio.size())
    audio = audio[:, :, None]  # add one more dimension on the end?
    #print("3 audio.size() = ",audio.size())
    return audio



def load_audio_for_jbx(path, offset=0.0, dur=None, trunc_sec=None, device=None):
    """Loads a path for use with Jukebox."""
    audio, sr = librosa.load(path, sr=None, offset=offset, duration=dur)

    if JUKEBOX_SAMPLE_RATE is None: init_jukebox_sample_rate()

    if sr != JUKEBOX_SAMPLE_RATE:
        audio = librosa.resample(audio, sr, JUKEBOX_SAMPLE_RATE)

    return audio_for_jbx(audio, trunc_sec, device=device)

