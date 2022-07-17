import torch

from diffusion.pqmf import CachedPQMF as PQMF

#Might work, but incredibly CPU-intensive

class MultiScalePQMFLoss(torch.nn.Module):
    def __init__(self,n_channels = 2, attenuation=70, band_counts=[8, 16, 32]):
        super(MultiScalePQMFLoss, self).__init__()
        self.pqmfs = []

        for band_count in band_counts:
            self.pqmfs.append(PQMF(n_channels, attenuation, band_count))
    
    def forward(self, input, target):
        loss = 0.0
        for pqmf in self.pqmfs:
            input_pqmf = pqmf(input.float().cpu())
            target_pqmf = pqmf(target.float().cpu())
            loss += torch.square(input_pqmf - target_pqmf).mean()

        return loss