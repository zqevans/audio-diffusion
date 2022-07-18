import torch
from torch import nn
from nwt_pytorch import Memcodes

from torch.nn import functional as F

def l2norm(t):
    return F.normalize(t, p = 2, dim = -1)

# class ResidualMemcodesLayer(nn.Module):
#     def __init__(
#         self,
#         *,
#         **kwargs
#     ):
#         super().__init__()

#         self.memcodes = Memcodes(**kwargs)

class ResidualMemcodes(nn.Module):
    def __init__(
        self,
        *,
        num_quantizers,
        shared_codebook = False,
        **kwargs
    ):
        super().__init__()
        self.layers = nn.ModuleList([Memcodes(**kwargs) for _ in range(num_quantizers)])

        if not shared_codebook:
            return

        first_vq, *rest_vq = self.layers
        codebook = first_vq._codebook

        for vq in rest_vq:
            vq._codebook = codebook

    def forward(self, x):
        quantized_out = 0.
        residual = x

        all_indices = []

        for layer in self.layers:
            quantized, indices = layer(residual)
            residual = residual - quantized
            quantized_out = quantized_out + quantized

            all_indices.append(indices)

        all_indices = torch.stack(all_indices, dim=-1)
        return quantized_out, all_indices

    # def forward(self, x):
    #     quantized_out = 0.
    #     residual = x

    #     all_indices = []

    #     for layer in self.layers:
    #         #Perform multi-head attention on the residual (or the input if first layer)
    #         quantized, indices = layer(residual)

            
    #         residual = residual - quantized
            
    #         #Add and normalize
    #         quantized_out = quantized_out + quantized
    #         quantized = l2norm(quantized)

            

    #         all_indices.append(indices)



    #     all_indices = torch.stack(all_indices, dim=-1)
    #     return quantized_out, all_indices