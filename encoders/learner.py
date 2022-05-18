import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .encoders import SoundStreamXL
from pytorch_lightning import LightningModule
import auraloss

import torch
import torch.nn as nn
import torch.nn.functional as F

def collate_fn(batch):
    lengths = torch.tensor([elem.shape[-1] for elem in batch])
    return nn.utils.rnn.pad_sequence(batch, batch_first=True), lengths


#Taken from https://github.com/rishikksh20/TFGAN/blob/main/utils/timeloss.py
#Licensed under the Apache Licence 2.0

class TimeDomainLoss(nn.Module):
    """Time domain loss module."""
    def __init__(self, batch_size ,segment_size=3200, 
                 T_frame_sizes=[1, 240, 480, 960],
                 T_hop_sizes=[1, 120, 240, 480]):
        super(TimeDomainLoss, self).__init__()
        self.shapes = []
        self.strides = []
        self.seg_size = segment_size
        for i in range(len(T_frame_sizes)):
            no_over_lap = T_frame_sizes[i] - T_hop_sizes[i]
            self.shapes.append((batch_size,
                               (segment_size - no_over_lap)//T_hop_sizes[i],
                                T_frame_sizes[i]
                                ))
            self.strides.append((segment_size,
                                 T_hop_sizes[i],
                                 1
                                 ))
        self.len = len(self.shapes)
        
    def forward(self, y, y_hat):
        """Calculate time domain loss
        Args:
            y (Tensor): real waveform
            y_hat (Tensor): fake waveform
        Return: 
            total_loss (Tensor): total loss of time domain
            
        """

        # Energy loss & Time loss & Phase loss
        loss_e = torch.zeros(self.len).to(y)
        loss_t = torch.zeros(self.len).to(y)
        loss_p = torch.zeros(self.len).to(y)
        
        for i in range(self.len):
            y_tmp = torch.as_strided(y, self.shapes[i], self.strides[i])
            y_hat_tmp = torch.as_strided(y_hat, self.shapes[i], self.strides[i])
            
            loss_e[i] = F.l1_loss(torch.mean(y_tmp**2, dim=-1), torch.mean(y_hat_tmp**2, dim=-1))
            loss_t[i] = F.l1_loss(torch.mean(y_tmp, dim=-1), torch.mean(y_hat_tmp, dim=-1))
            if i == 0:
                y_phase = F.pad(y_tmp.transpose(1, 2), (1, 0), "constant", 0) - F.pad(y_tmp.transpose(1, 2), (0, 1), "constant", 0)
                y_hat_phase = F.pad(y_hat_tmp.transpose(1, 2), (1, 0), "constant", 0) - F.pad(y_hat_tmp.transpose(1, 2), (0, 1), "constant", 0)
            else:
                y_phase = F.pad(y_tmp, (1, 0), "constant", 0) - F.pad(y_tmp, (0, 1), "constant", 0)
                y_hat_phase = F.pad(y_hat_tmp, (1, 0), "constant", 0) - F.pad(y_hat_tmp, (0, 1), "constant", 0)
            loss_p[i] = F.l1_loss(y_phase, y_hat_phase)
        
        total_loss = torch.sum(loss_e) + torch.sum(loss_t) + torch.sum(loss_p)
        
        return total_loss

class SoundStreamXLLearner(LightningModule):
    def __init__(self, global_args):
        super().__init__()
        self.soundstream = SoundStreamXL(n_io_channels=2, n_feature_channels=16, latent_dim=128, n_quantizers=8, codebook_size=1024, )     

        self.stft_loss = auraloss.freq.MelSTFTLoss(global_args.sample_rate, w_phs=1.0, device=self.device) 

        self.time_loss = TimeDomainLoss(global_args.batch_size)

    def training_step(self, batch, batch_idx):
        inputs, _ = batch
        inputs = inputs.to(self.device)

        #Get the reconstructed signal
        reconstructed, _, vq_losses  = self.soundstream(inputs)

        #Sum the commit losses
        vq_loss = torch.sum(vq_losses, -1)

        freq_loss = self.stft_loss(inputs, reconstructed)

        time_loss = self.time_loss(inputs, reconstructed)

        loss = vq_loss + freq_loss + time_loss

        log_dict = {'train/loss': loss.detach(), 
                    'train/freq_loss': freq_loss.detach(), 
                    'train/time_loss': time_loss.detach(),
                    'train/vq_loss': vq_loss.detach()
                    }
        self.log_dict(log_dict, prog_bar=True, on_step=True)
        return {'loss': loss}

    def configure_optimizers(self):
        self.optimizer_g = optim.Adam(self.soundstream.parameters(), lr=1e-4, betas=(0.5, 0.9))
        self.optimizer_d = optim.Adam(list(self.wave_disc.parameters()) + list(self.stft_disc.parameters()), lr=1e-4, betas=(0.5, 0.9))
