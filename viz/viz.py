
import math
from pathlib import Path
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.cm as cm
import matplotlib.pyplot as plt 
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
import numpy as np
from PIL import Image

import torch
from torch import optim, nn
from torch.nn import functional as F
import torchaudio
import torchaudio.transforms as T
import librosa 
from einops import rearrange

import wandb
import numpy as np
import pandas as pd



def embeddings_table(tokens):
    "make a table of embeddings for use with wandb"
    features, labels = [], []
    embeddings = rearrange(tokens, 'b d n -> b n d') # each demo sample is n vectors in d-dim space
    for i in range(embeddings.size()[0]):  # nested for's are slow but sure ;-) 
        for j in range(embeddings.size()[1]):
            features.append(embeddings[i,j].detach().cpu().numpy())
            labels.append([f'demo{i}'])    # labels does the grouping / color for each point
    features = np.array(features)
    #print("\nfeatures.shape = ",features.shape)
    labels = np.concatenate(labels, axis=0)
    cols = [f"dim_{i}" for i in range(features.shape[1])]
    df   = pd.DataFrame(features, columns=cols)
    df['LABEL'] = labels
    return wandb.Table(columns=df.columns.to_list(), data=df.values)


def proj_pca(tokens, proj_dims=3):
    "this projects via PCA, grabbing the first _3_ dimensions"
    A = rearrange(tokens, 'b d n -> (b n) d') # put all the vectors into the same d-dim space
    k = proj_dims
    (U, S, V) = torch.pca_lowrank(A)
    proj_data = torch.matmul(A, V[:, :k])  # this is the actual PCA projection step
    return torch.reshape(proj_data, (tokens.size()[0], -1, proj_dims)) # put it in shape [batch, n, 3]


def pca_point_cloud(tokens):
    "produces a 3D wandb point cloud of the tokens using PCA"
    data = proj_pca(tokens).cpu().numpy()
    points = []
    cmap = cm.tab20  # 20 farly distinct colors
    norm = Normalize(vmin=0, vmax=data.shape[0])
    for bi in range(data.shape[0]):  # batch index
        [r, g, b, _] = [int(255*x) for x in cmap(norm(bi))]
        for n in range(data.shape[1]):
            #points.append([data[b,n,0], data[b,n,1], data[b,n,2], color]) # only works for color=1 to 14
            points.append([data[bi,n,0], data[bi,n,1], data[bi,n,2], r, g, b])

    point_cloud = np.array(points)
    return wandb.Object3D(point_cloud)


def spectrogram_image(spec, title=None, ylabel='freq_bin', aspect='auto', xmax=None, db_range=[35,120]):
    """
    # cf. https://pytorch.org/tutorials/beginner/audio_feature_extractions_tutorial.html

    """
    fig = Figure(figsize=(5, 4), dpi=100)
    canvas = FigureCanvasAgg(fig)
    axs = fig.add_subplot()
    axs.set_title(title or 'Spectrogram (db)')
    axs.set_ylabel(ylabel)
    axs.set_xlabel('frame')
    im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect, vmin=db_range[0], vmax=db_range[1])
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    return Image.fromarray(rgba)


def audio_spectrogram_image(waveform, power=2.0, sample_rate=48000):
    """
    # cf. https://pytorch.org/tutorials/beginner/audio_feature_extractions_tutorial.html
    """
    n_fft = 1024
    win_length = None
    hop_length = 512
    n_mels = 80

    mel_spectrogram_op = T.MelSpectrogram(
        sample_rate=sample_rate, n_fft=n_fft, win_length=win_length, 
        hop_length=hop_length, center=True, pad_mode="reflect", power=power, 
        norm='slaney', onesided=True, n_mels=n_mels, mel_scale="htk")

    melspec = mel_spectrogram_op(waveform.float())
    melspec = melspec[0] # TODO: only left channel for now
    return spectrogram_image(melspec, title="MelSpectrogram", ylabel='mel bins (log freq)')


def tokens_spectrogram_image(tokens, aspect='auto', title='Embeddings', ylabel='index'):
    embeddings = rearrange(tokens, 'b d n -> (b n) d') 
    print(f"tokens_spectrogram_image: embeddings.shape = ",embeddings.shape)
    fig = Figure(figsize=(10, 4), dpi=100)
    canvas = FigureCanvasAgg(fig)
    axs = fig.add_subplot()
    axs.set_title(title or 'Embeddings')
    axs.set_ylabel(ylabel)
    axs.set_xlabel('time frame')
    im = axs.imshow(embeddings.cpu().numpy().T, origin='lower', aspect=aspect, interpolation='none') #.T because numpy is x/y 'backwards'
    fig.colorbar(im, ax=axs)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    return Image.fromarray(rgba)
