"""Multi-scale ResNet-18 feature extractor for 1-channel patches.

Outputs features at strides 1/4 and 1/8 of the input patch resolution.
ImageNet-pretrained weights are loaded for layers >= conv1; conv1 is
adapted to 1-channel input by averaging the original 3-channel weights."""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torchvision.models as tv_models


class MultiScaleBackbone(nn.Module):
    """ResNet-18 stem + layer1 + layer2 producing F^(1/4) and F^(1/8).

    Note: torchvision's resnet18 stride pattern is:
        conv1: stride 2   -> 1/2
        maxpool: stride 2 -> 1/4
        layer1: stride 1  -> 1/4   (channels = 64)
        layer2: stride 2  -> 1/8   (channels = 128)
        layer3: stride 2  -> 1/16  (unused)
        layer4: stride 2  -> 1/32  (unused)
    """

    def __init__(
        self,
        pretrained: bool = True,
        out_channels_quarter: int = 64,
        out_channels_eighth: int = 128,
    ):
        super().__init__()
        try:
            weights = tv_models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            net = tv_models.resnet18(weights=weights)
        except (AttributeError, TypeError):
            net = tv_models.resnet18(pretrained=pretrained)

        # Adapt conv1 to 1-channel input by averaging the 3-channel kernel.
        orig_conv1 = net.conv1
        new_conv1 = nn.Conv2d(
            1, orig_conv1.out_channels,
            kernel_size=orig_conv1.kernel_size, stride=orig_conv1.stride,
            padding=orig_conv1.padding, bias=False,
        )
        if pretrained:
            with torch.no_grad():
                new_conv1.weight.copy_(orig_conv1.weight.mean(dim=1, keepdim=True))
        net.conv1 = new_conv1

        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1                                 # 1/4, 64-ch
        self.layer2 = net.layer2                                 # 1/8, 128-ch

        self.proj_quarter = nn.Conv2d(64, out_channels_quarter, kernel_size=1, bias=False) \
            if out_channels_quarter != 64 else nn.Identity()
        self.proj_eighth = nn.Conv2d(128, out_channels_eighth, kernel_size=1, bias=False) \
            if out_channels_eighth != 128 else nn.Identity()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """x: (B, 1, H, W)
        Returns:
            {"quarter": (B, C, H/4, W/4), "eighth": (B, C, H/8, W/8)}
        """
        s = self.stem(x)             # 1/4
        f4 = self.layer1(s)          # 1/4
        f8 = self.layer2(f4)         # 1/8
        f4_out = self.proj_quarter(f4)
        f8_out = self.proj_eighth(f8)
        return {"quarter": f4_out, "eighth": f8_out}
