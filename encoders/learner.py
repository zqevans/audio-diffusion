import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from einops import rearrange

from .encoders import SoundStreamXL, WaveDiscriminator, STFTDiscriminator
from .losses import adversarial_d_loss, adversarial_g_loss, spectral_reconstruction_loss, feature_loss
from pytorch_lightning import LightningModule
import auraloss

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoundStreamXLLearner(LightningModule):
    def __init__(self, global_args):
        super().__init__()
        self.soundstream = SoundStreamXL(n_io_channels=2, n_feature_channels=32, latent_dim=global_args.style_latent_size, n_quantizers=global_args.num_quantizers, codebook_size=global_args.codebook_size)     

        self.wave_disc = WaveDiscriminator(num_D=3, downsampling_factor=2)
        W = 1024
        self.stft_disc = STFTDiscriminator(C=1, F_bins=W//2)

        self.sample_rate = global_args.sample_rate

        LAMBDA_ADV = 1
        LAMBDA_FEAT = 100
        LAMBDA_REC = 1

        self.criterion_g = lambda x, G_x, features_stft_disc_x, features_wave_disc_x, features_stft_disc_G_x, features_wave_disc_G_x, lengths_wave, lengths_stft:\
            LAMBDA_ADV*adversarial_g_loss(features_stft_disc_G_x, features_wave_disc_G_x, lengths_stft, lengths_wave) + \
            LAMBDA_FEAT*feature_loss(features_stft_disc_x, features_wave_disc_x, features_stft_disc_G_x, features_wave_disc_G_x, lengths_wave, lengths_stft) +\
            LAMBDA_REC*spectral_reconstruction_loss(x, G_x)
        self.criterion_d = adversarial_d_loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, lengths_x = batch
        x = x.to(self.device)

        #Get the reconstructed signal
        G_x, _, vq_losses  = self.soundstream(x)

        #Sum the commit losses
        vq_loss = torch.sum(vq_losses, -1)

        loss = vq_loss

        #Reshape multi-channel audio for STFT loss
        x = rearrange(x, 'b c s -> b (c s)')
        G_x = rearrange(G_x, 'b c s -> b (c s)')

        # Get the STFTs of each signal in the batch
        s_x = torch.stft(x.squeeze(), n_fft=1024, hop_length=256, window=torch.hann_window(window_length=1024, device=self.device), return_complex=False).permute(0, 3, 1, 2)
        lengths_s_x = 1 + torch.div(lengths_x, 256, rounding_mode="floor")
        s_G_x = torch.stft(G_x.squeeze(), n_fft=1024, hop_length=256, window=torch.hann_window(window_length=1024, device=self.device), return_complex=False).permute(0, 3, 1, 2)

        lengths_stft = self.stft_disc.features_lengths(lengths_s_x)
        lengths_wave = self.wave_disc.features_lengths(lengths_x)

        features_stft_disc_x = self.stft_disc(s_x)
        features_wave_disc_x = self.wave_disc(x)
        
        features_stft_disc_G_x = self.stft_disc(s_G_x)
        features_wave_disc_G_x = self.wave_disc(G_x)

        loss_g = self.criterion_g(x, G_x, features_stft_disc_x, features_wave_disc_x, features_stft_disc_G_x, features_wave_disc_G_x, lengths_wave, lengths_stft)
        loss += loss_g.item()

        features_stft_disc_x = self.stft_disc(s_x)
        features_wave_disc_x = self.wave_disc(x)
        
        features_stft_disc_G_x_det = self.stft_disc(s_G_x.detach())
        features_wave_disc_G_x_det = self.wave_disc(G_x.detach())
        
        loss_d = self.criterion_d(features_stft_disc_x, features_wave_disc_x, features_stft_disc_G_x_det, features_wave_disc_G_x_det, lengths_stft, lengths_wave)

        loss += loss_d.item()

        log_dict = {'train/loss': loss.detach(), 
                    'train/vq_loss': vq_loss.detach(),
                    'train/loss_g': loss_g.detach(),
                    'train/loss_d': loss_d.detach()
                    }
        self.log_dict(log_dict, prog_bar=True, on_step=True)
        return {'loss': loss, 'loss_d': loss_d}

    def configure_optimizers(self):
        optim_g = optim.Adam(self.soundstream.parameters(), lr=1e-4, betas=(0.5, 0.9))
        optim_d = optim.Adam(list(self.wave_disc.parameters()) + list(self.stft_disc.parameters()), lr=1e-4, betas=(0.5, 0.9))
        return optim_g, optim_d
        

