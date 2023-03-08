import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as wn
import cached_conv as cc
from encodec.msstftd import MultiScaleSTFTDiscriminator

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

class EncodecDiscriminator(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.discriminators = MultiScaleSTFTDiscriminator(*args, **kwargs)

    def forward(self, x):
        logits, features = self.discriminators(x)
        return logits, features

    def adversarial_combine(self, score_real, score_fake):
        loss_dis = torch.relu(1 - score_real) + torch.relu(1 + score_fake)
        loss_dis = loss_dis.mean()
        loss_gen = -score_fake.mean()
        return loss_dis, loss_gen

    def loss(self, x, y):
        feature_matching_distance = 0.
        logits_true, feature_true = self.forward(x)
        logits_fake, feature_fake = self.forward(y)

        loss_dis = 0
        loss_adv = 0

        pred_true = 0
        pred_fake = 0

        for i, (scale_true, scale_fake) in enumerate(zip(feature_true, feature_fake)):

            feature_matching_distance = feature_matching_distance + sum(
                map(
                    lambda x, y: abs(x - y).mean(),
                    scale_true,
                    scale_fake,
                )) / len(scale_true)

            _dis, _adv = self.adversarial_combine(
                logits_true[i],
                logits_fake[i],
            )

            pred_true = pred_true + logits_true[i].mean()
            pred_fake = pred_fake + logits_fake[i].mean()

            loss_dis = loss_dis + _dis
            loss_adv = loss_adv + _adv

        return loss_dis, loss_adv, feature_matching_distance, pred_true, pred_fake