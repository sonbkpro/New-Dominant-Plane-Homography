"""CDPC-v4: small Siamese feature extractor for the triplet loss.

Mirrors `Oneline-DLTv1.resnet.ResNet.ShareFeature`. A tiny 3-block conv tower
that maps a 1-channel patch to a 1-channel "matching" feature map at the same
resolution. Used by the unsupervised triplet loss exactly as in v1:

    d^+_i = |ShareFeature(I_b_patch)_i - ShareFeature(warp(I_a_full, H))_i|
    d^-_i = |ShareFeature(I_b_patch)_i - ShareFeature(I_a_patch)_i|
    L_trip = max(margin + d^+_i - d^-_i, 0), averaged over a mask.

This network is jointly trained with the geometry trunk -- the triplet loss
backpropagates through it AND through H (via the warp). It is deliberately
shallow (= no semantic specialization, low parameter count, full-resolution
features) so the triplet supervision stays geometric.
"""

import torch
import torch.nn as nn


class ShareFeature(nn.Module):
    def __init__(self, in_channels: int = 1, hidden: int = 8, out_channels: int = 1):
        super().__init__()
        self.tower = nn.Sequential(
            nn.Conv2d(in_channels, 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),

            nn.Conv2d(4, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tower(x)
