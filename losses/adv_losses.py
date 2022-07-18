import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as wn
import auraloss
import cached_conv as cc

class Discriminator(nn.Module):

    def __init__(self, in_size, capacity, multiplier, n_layers):
        super().__init__()

        net = [
            wn(cc.Conv1d(in_size, capacity, 15, padding=cc.get_padding(15)))
        ]
        net.append(nn.LeakyReLU(.2))

        for i in range(n_layers):
            net.append(
                wn(
                    cc.Conv1d(
                        capacity * multiplier**i,
                        min(1024, capacity * multiplier**(i + 1)),
                        41,
                        stride=multiplier,
                        padding=cc.get_padding(41, multiplier),
                        groups=multiplier**(i + 1),
                    )))
            net.append(nn.LeakyReLU(.2))

        net.append(
            wn(
                cc.Conv1d(
                    min(1024, capacity * multiplier**(i + 1)),
                    min(1024, capacity * multiplier**(i + 1)),
                    5,
                    padding=cc.get_padding(5),
                )))
        net.append(nn.LeakyReLU(.2))
        net.append(
            wn(cc.Conv1d(min(1024, capacity * multiplier**(i + 1)), 1, 1)))
        self.net = nn.ModuleList(net)

    def forward(self, x):
        feature = []
        for layer in self.net:
            x = layer(x)
            if isinstance(layer, nn.Conv1d):
                feature.append(x)
        return feature


class StackDiscriminators(nn.Module):

    def __init__(self, n_dis, *args, **kwargs):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [Discriminator(*args, **kwargs) for i in range(n_dis)], )

    def forward(self, x):
        features = []
        for layer in self.discriminators:
            features.append(layer(x))
            x = nn.functional.avg_pool1d(x, 2)
        return features

    def adversarial_combine(self, score_real, score_fake):
        loss_dis = torch.relu(1 - score_real) + torch.relu(1 + score_fake)
        loss_dis = loss_dis.mean()
        loss_gen = -score_fake.mean()
        return loss_dis, loss_gen

    def loss(self, x, y):
        feature_matching_distance = 0.
        feature_true = self.forward(x)
        feature_fake = self.forward(y)

        loss_dis = 0
        loss_adv = 0

        pred_true = 0
        pred_fake = 0

        for scale_true, scale_fake in zip(feature_true, feature_fake):
            feature_matching_distance = feature_matching_distance + sum(
                map(
                    lambda x, y: abs(x - y).mean(),
                    scale_true,
                    scale_fake,
                )) / len(scale_true)

            _dis, _adv = self.adversarial_combine(
                scale_true[-1],
                scale_fake[-1],
            )

            pred_true = pred_true + scale_true[-1].mean()
            pred_fake = pred_fake + scale_fake[-1].mean()

            loss_dis = loss_dis + _dis
            loss_adv = loss_adv + _adv

        return loss_dis, loss_adv, feature_matching_distance, pred_true, pred_fake


class ResidualUnit2d(nn.Module):
    def __init__(self, in_channels, N, m, stride_time, stride_freq):
        super().__init__()
        
        self.s_t = stride_time
        self.s_f = stride_freq

        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=N,
                kernel_size=(3, 3),
                padding="same"
            ),
            nn.ELU(),
            nn.Conv2d(
                in_channels=N,
                out_channels=m*N,
                kernel_size=(stride_freq+2, stride_time+2),
                stride=(stride_freq, stride_time)
            )
        )
        
        self.skip_connection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=m*N,
            kernel_size=(1, 1), stride=(stride_freq, stride_time)
        )

    def forward(self, x):
        return self.layers(F.pad(x, [self.s_t+1, 0, self.s_f+1, 0])) + self.skip_connection(x)


class STFTDiscriminator(nn.Module):
    def __init__(self, capacity, F_bins):
        super().__init__()

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=2, out_channels=32, kernel_size=(7, 7)),
                nn.ELU()
            ),
            nn.Sequential(
                ResidualUnit2d(in_channels=32,  N=capacity,   m=2, stride_time=1, stride_freq=2),
                nn.ELU()
            ),
            nn.Sequential(
                ResidualUnit2d(in_channels=2*capacity, N=2*capacity, m=2, stride_time=2, stride_freq=2),
                nn.ELU()
            ),
            nn.Sequential(
                ResidualUnit2d(in_channels=4*capacity, N=4*capacity, m=1, stride_time=1, stride_freq=2),
                nn.ELU()
            ),
            nn.Sequential(
                ResidualUnit2d(in_channels=4*capacity, N=4*capacity, m=2, stride_time=2, stride_freq=2),
                nn.ELU()
            ),
            nn.Sequential(
                ResidualUnit2d(in_channels=8*capacity, N=8*capacity, m=1, stride_time=1, stride_freq=2),
                nn.ELU()
            ),
            nn.Sequential(
                ResidualUnit2d(in_channels=8*capacity,  N=8*capacity, m=2, stride_time=2, stride_freq=2),
                nn.ELU()
            ),
            nn.Conv2d(in_channels=16*capacity, out_channels=1,
                      kernel_size=(F_bins//2**6, 1))
        ])
    
    def features_lengths(self, lengths):
        return [
            lengths-6,
            lengths-6,
            torch.div(lengths-5, 2, rounding_mode="floor"),
            torch.div(lengths-5, 2, rounding_mode="floor"),
            torch.div(lengths-3, 4, rounding_mode="floor"),
            torch.div(lengths-3, 4, rounding_mode="floor"),
            torch.div(lengths+1, 8, rounding_mode="floor"),
            torch.div(lengths+1, 8, rounding_mode="floor")
        ]

    def forward(self, x):
        feature_map = []
        for layer in self.layers:
            x = layer(x)
            feature_map.append(x)
        return feature_map


class SumAndDifferenceSTFTDiscriminator(torch.nn.Module):

    def __init__(
        self,
        capacity=32,
        fft_size=1024,
        hop_size=256,
        win_length=1024,
        window="hann_window",
        w_sum=1.0,
        w_diff=1.0,
        output="loss",
        sample_rate=48000,
        **stft_args
    ):
        super(SumAndDifferenceSTFTDiscriminator, self).__init__()
        self.sd = auraloss.perceptual.SumAndDifference()
        self.w_sum = 1.0
        self.w_diff = 1.0
        self.output = output
        self.window = getattr(torch, window)(win_length)
        self.stft = lambda x: torch.stft(x, n_fft=fft_size, hop_length=hop_size, win_length=win_length, window=self.window, return_complex=True)

        self.mono_discriminator = STFTDiscriminator(capacity, )
        

    def forward(self, x):
        features = []
        for layer in self.discriminators:
            features.append(layer(x))
            x = nn.functional.avg_pool1d(x, 2)
        return features

    def loss(self, input, target):
        input_sum, input_diff = self.sd(input)
        target_sum, target_diff = self.sd(target)

        #Get STFTs for sum and diff
        input_sum_stft = self.stft(input_sum)
        input_diff_stft = self.stft(input_diff)
        target_sum_stft = self.stft(target_sum)
        target_diff_stft = self.stft(target_diff)


        feature_matching_distance = 0.
        feature_sum_true = self.forward(input_sum_stft)
        feature_diff_true = self.forward(input_diff_stft)
        feature_sum_fake = self.forward(target_sum_stft)
        feature_diff_fake = self.forward(target_diff_stft)
        
        loss_dis = 0
        loss_adv = 0

        pred_true = 0
        pred_fake = 0

        # Feature matching on sum
        for scale_true, scale_fake in zip(feature_sum_true, feature_sum_fake):
            feature_matching_distance = feature_matching_distance + sum(
                map(
                    lambda x, y: abs(x - y).mean(),
                    scale_true,
                    scale_fake,
                )) / len(scale_true)

            _dis, _adv = self.adversarial_combine(
                scale_true[-1],
                scale_fake[-1],
            )

            pred_true = pred_true + scale_true[-1].mean()
            pred_fake = pred_fake + scale_fake[-1].mean()

            loss_dis = loss_dis + _dis
            loss_adv = loss_adv + _adv
        
        # Feature matching on diff
        for scale_true, scale_fake in zip(feature_diff_true, feature_diff_fake):
            feature_matching_distance = feature_matching_distance + sum(
                map(
                    lambda x, y: abs(x - y).mean(),
                    scale_true,
                    scale_fake,
                )) / len(scale_true)

            _dis, _adv = self.adversarial_combine(
                scale_true[-1],
                scale_fake[-1],
            )

            pred_true = pred_true + scale_true[-1].mean()
            pred_fake = pred_fake + scale_fake[-1].mean()

            loss_dis = loss_dis + _dis
            loss_adv = loss_adv + _adv

        return loss_dis, loss_adv, feature_matching_distance, pred_true, pred_fake