
from contextlib import contextmanager
from copy import deepcopy
import math
from pathlib import Path
import matplotlib.cm as cm
from matplotlib.colors import Normalize

import sys
import torch
from torch import optim, nn
from torch.nn import functional as F
from tqdm import trange
from pytorch_lightning.utilities.distributed import rank_zero_only
from einops import rearrange

from fairscale.nn import checkpoint_wrapper, auto_wrap, wrap

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
    for b in range(data.shape[0]):
        #color = b 
        [r, g, b, _] = [int(255*x) for x in cmap(norm(b))]
        for n in range(data.shape[1]):
            #points.append([data[b,n,0], data[b,n,1], data[b,n,2], color]) # only works for color=1 to 14
            points.append([data[b,n,0], data[b,n,1], data[b,n,2], r, g, b])

    point_cloud = np.array(points)
    return wandb.Object3D(point_cloud)
