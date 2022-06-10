import torch
import torch.optim as optim

from .encoders import SoundStreamXL
from pytorch_lightning import LightningModule


from auraloss.freq import MultiResolutionSTFTLoss


class SoundStreamXLLearner(LightningModule):
    def __init__(self, global_args):
        super().__init__()
        self.soundstream = SoundStreamXL(n_io_channels=2, n_feature_channels=32, latent_dim=global_args.style_latent_size, n_quantizers=global_args.num_quantizers, codebook_size=global_args.codebook_size)     
        self.mrstft = MultiResolutionSTFTLoss()

    def training_step(self, batch, batch_idx, optimizer_idx):
        audios, filenames = batch
        audios = audios.to(self.device)

        #Get the reconstructed signal
        reconstructions, _, vq_losses  = self.soundstream(audios)

        #Sum the commit losses
        vq_loss = torch.sum(vq_losses, -1)

        stft_loss = self.mrstft(audios, reconstructions)

        loss = vq_loss + stft_loss

        log_dict = {'train/loss': loss.detach(), 
                    'train/vq_loss': vq_loss.detach(),
                    'train/stft_loss': stft_loss.detach()
                    }

        self.log_dict(log_dict, prog_bar=True, on_step=True)

        return {'loss': loss}

    def configure_optimizers(self):
        optim_g = optim.Adam(self.soundstream.parameters(), lr=1e-4, betas=(0.5, 0.9))
        return optim_g