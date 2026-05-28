import numpy as np
import torch.nn as nn

# Ported from megvii-research/HomoGAN:
# https://github.com/megvii-research/HomoGAN
# Source files: model/swin_multi.py and model/net.py.

__all__ = ["FeatureExtractor", "feature_extractor"]


class FeatureExtractor(nn.Module):
    """Multi-scale feature pyramid extractor used by HomoGAN's transformer."""

    def __init__(self, embed_dim, num_layers, activation):
        super(FeatureExtractor, self).__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.activation = activation
        self.convs = nn.ModuleList()
        for i_layer in range(self.num_layers):
            in_channel = int(
                (self.embed_dim * 2 ** (i_layer - 1)) ** np.heaviside(i_layer, 0))  # 1, embed_dim, 2*embed_dim ...
            out_channel = int((self.embed_dim * 2 ** i_layer) ** np.heaviside(i_layer + 1,
                                                                              0))  # embed_dim, 2*embed_dim, 4*embed_dim...
            layer = nn.Sequential(
                nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=2, padding=1),
                self.activation(),
                nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1),
                self.activation())
            self.convs.append(layer)

    def forward(self, x):
        feature_pyramid = []
        for conv in self.convs:
            x = conv(x)
            feature_pyramid.append(x)

        return feature_pyramid[::-1]


def feature_extractor(input_channels, out_channles, kernel_size=3, padding=1):
    """Shallow feature extractor used by HomoGAN before homography regression."""
    layers = []
    channels = [input_channels // 2, 4, 8, out_channles]
    for i in range(len(channels) - 1):
        layers.append(nn.Conv2d(channels[i], channels[i + 1], kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.BatchNorm2d(channels[i + 1]))
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)
