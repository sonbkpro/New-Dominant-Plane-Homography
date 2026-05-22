"""planv3 §3.2: Siamese 3-level feature pyramid.

ResNet-18 stem + layer1 + layer2 + layer3, producing features at strides
1/4 (64-ch), 1/8 (128-ch), 1/16 (256-ch). Same network is shared between
I_a and I_b (Siamese -- the homography pyramid consumes (F_a, F_b) at each
level, so the trunk must produce features in a single common embedding).

The v2 joint 2-channel backbone is removed; pairwise cues now come from
local correlation + pre-warping inside the homography pyramid.

conv1 is adapted to 1-channel input by averaging the original 3-channel
kernel.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torchvision.models as tv_models


class FeaturePyramid(nn.Module):
    """ResNet-18 trunk producing F^(1/4), F^(1/8), F^(1/16)."""

    def __init__(
        self,
        pretrained: bool = True,
        in_channels: int = 1,
        out_channels_quarter:  int = 64,
        out_channels_eighth:   int = 128,
        out_channels_sixteenth: int = 256,
    ):
        super().__init__()
        try:
            weights = tv_models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            net = tv_models.resnet18(weights=weights)
        except (AttributeError, TypeError):
            net = tv_models.resnet18(pretrained=pretrained)

        orig_conv1 = net.conv1
        new_conv1 = nn.Conv2d(
            in_channels, orig_conv1.out_channels,
            kernel_size=orig_conv1.kernel_size, stride=orig_conv1.stride,
            padding=orig_conv1.padding, bias=False,
        )
        if pretrained:
            with torch.no_grad():
                mean_w = orig_conv1.weight.mean(dim=1, keepdim=True)
                new_conv1.weight.copy_(mean_w.expand(-1, in_channels, -1, -1))
        net.conv1 = new_conv1

        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1       # 1/4, 64-ch
        self.layer2 = net.layer2       # 1/8, 128-ch
        self.layer3 = net.layer3       # 1/16, 256-ch

        self.proj_quarter = (
            nn.Conv2d(64, out_channels_quarter, kernel_size=1, bias=False)
            if out_channels_quarter != 64 else nn.Identity()
        )
        self.proj_eighth = (
            nn.Conv2d(128, out_channels_eighth, kernel_size=1, bias=False)
            if out_channels_eighth != 128 else nn.Identity()
        )
        self.proj_sixteenth = (
            nn.Conv2d(256, out_channels_sixteenth, kernel_size=1, bias=False)
            if out_channels_sixteenth != 256 else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """x: (B, in_channels, H, W). Returns features at strides 1/4, 1/8, 1/16."""
        s = self.stem(x)             # 1/4
        f4  = self.layer1(s)         # 1/4
        f8  = self.layer2(f4)        # 1/8
        f16 = self.layer3(f8)        # 1/16
        return {
            "stride4":  self.proj_quarter(f4),
            "stride8":  self.proj_eighth(f8),
            "stride16": self.proj_sixteenth(f16),
        }
