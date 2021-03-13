import torch
import torch.nn as nn
import torch.nn.functional as F

def downsample(in_feat, out_feat, normalize=True):
    layers = [nn.Conv2d(in_feat, out_feat, 4, stride=2, padding=1)]
    if normalize:
        layers.append(nn.BatchNorm2d(out_feat, 0.8))
    layers.append(nn.LeakyReLU(0.2))
    return layers

def upsample(in_feat, out_feat, normalize=True):
    layers = [nn.ConvTranspose2d(in_feat, out_feat, 4, stride=2, padding=1)]
    if normalize:
        layers.append(nn.BatchNorm2d(out_feat, 0.8))
    layers.append(nn.LeakyReLU(0.2))
    return layers

class Generator(nn.Module):
    def __init__(self, channels=3):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            *downsample(channels, 64, normalize=False),
            *downsample(64, 128),
            *downsample(128, 256),
            *downsample(256, 512),
            nn.Conv2d(512, 4000, 1),
            *downsample(4000, 512),
            *downsample(512, 256),
            *downsample(256, 128),
            *downsample(128, 64),
            nn.Conv2d(64, channels, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)


def discriminator_block(in_filters, out_filters, stride, normalize):
    """return layers of each discriminator block"""
    layers = [nn.Conv2d(in_filters, out_filters, 3, stride, 1)]
    if normalize:
        layers.append(nn.InstanceNorm2d(out_filters))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return layers

class Discriminator(nn.Module):
    def __init__(self, channels=3):
        super(Discriminator, self).__init__()

        layers = []
        in_filters = channels
        for out_filters, stride, normalize in [(64, 2, False), (128, 2, True), (256, 2, True), (512, 1, True)]:
            layers.extend(discriminator_block(in_filters, out_filters, stride, normalize))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, 3, 1, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)

