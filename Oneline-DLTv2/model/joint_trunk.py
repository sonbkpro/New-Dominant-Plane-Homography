"""CDPC-v4: v1-style joint 2-channel ResNet-34 geometry trunk.

This restores the pairwise inductive bias that v3's per-image Siamese trunk
discarded. The full geometry path is:

    x_a = ShareFeature(I_a_patch)              # (B, 1, ph, pw)
    x_b = ShareFeature(I_b_patch)              # (B, 1, ph, pw)
    x   = cat(x_a, x_b)                        # (B, 2, ph, pw)
    f   = ResNet34_2ch_stem_layer1_4(x)        # (B, 512, ph/32, pw/32)
    p   = AdaptiveAvgPool2d(1)(f)              # (B, 512, 1, 1)
    off = rho_init * tanh(FC(flatten(p)))      # (B, 8)  corner offsets, bounded
    H   = DLT_normalized(canonical_corners, off)

Why:
  * conv1 sees both images concatenated channel-wise, so pairwise gradient
    enters at the first conv layer (v1's strongest inductive bias).
  * ResNet-34 layer4 gives a 1/32 final feature map: large receptive field
    and global context for a one-shot 8-DoF regression.
  * Trained from random init (no ImageNet); the 2-channel conv1 has no
    pretrained counterpart and the homography task is not class-discriminative.
  * Output is bounded by tanh so the trunk cannot emit pathological corner
    offsets even on early-iteration noise.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

from model.share_feature import ShareFeature
from utils.dlt_normalized import DLT_solve_normalized
from utils.dlt import DLT_solve


def _make_resnet34_2ch(pretrained: bool) -> nn.Module:
    """ResNet-34 with conv1 widened to accept a 2-channel input.

    If `pretrained=True`, the conv1 weights are initialized by averaging the
    pretrained 3-channel kernel and tiling across the 2 input channels
    (cheap but at least preserves the spatial filter shape). The rest of the
    backbone keeps its ImageNet weights. Default (and recommended) is False.
    """
    try:
        weights = tv_models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        net = tv_models.resnet34(weights=weights)
    except (AttributeError, TypeError):
        net = tv_models.resnet34(pretrained=pretrained)

    orig_conv1 = net.conv1
    new_conv1 = nn.Conv2d(
        2, orig_conv1.out_channels,
        kernel_size=orig_conv1.kernel_size, stride=orig_conv1.stride,
        padding=orig_conv1.padding, bias=False,
    )
    if pretrained:
        with torch.no_grad():
            mean_w = orig_conv1.weight.mean(dim=1, keepdim=True)
            new_conv1.weight.copy_(mean_w.expand(-1, 2, -1, -1))
    net.conv1 = new_conv1
    return net


class JointGeometryTrunk(nn.Module):
    """v1-style joint 2-channel ResNet-34 H regressor.

    Produces (H_patch, offset_8) given a pair of patches (I_a_patch, I_b_patch).
    Also exposes the ShareFeature outputs so the triplet loss can use them.
    """

    def __init__(
        self,
        patch_h: int = 315,
        patch_w: int = 560,
        pretrained: bool = False,
        rho_init: float = 32.0,
        share_feature_channels: int = 1,
        use_normalized_dlt: bool = True,
    ):
        super().__init__()
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.rho_init = float(rho_init)
        self.use_normalized_dlt = use_normalized_dlt

        # Shared 1-channel "matching feature" extractor (v1: ShareFeature).
        self.share_feature = ShareFeature(
            in_channels=1, hidden=8, out_channels=share_feature_channels,
        )

        # Joint 2-channel ResNet-34 trunk.
        net = _make_resnet34_2ch(pretrained=pretrained)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1       # 1/4,  64-ch
        self.layer2 = net.layer2       # 1/8, 128-ch
        self.layer3 = net.layer3       # 1/16,256-ch
        self.layer4 = net.layer4       # 1/32,512-ch
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 8),
        )
        # Init scale chosen so initial raw is O(0.5) and tanh is in linear
        # regime; bias 0 -> initial mean offset 0.
        nn.init.normal_(self.fc[-1].weight, std=0.05)
        nn.init.zeros_(self.fc[-1].bias)

    @staticmethod
    def _canonical_corners(B: int, ph: int, pw: int,
                           device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Flat 8-vector in canonical TL, BL, BR, TR order."""
        h4p = torch.tensor(
            [0.0, 0.0,
             0.0, float(ph),
             float(pw), float(ph),
             float(pw), 0.0],
            device=device, dtype=dtype,
        )
        return h4p.unsqueeze(0).expand(B, -1).contiguous()

    def _solve(self, h4p: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        if self.use_normalized_dlt:
            return DLT_solve_normalized(h4p, offset)
        return DLT_solve(h4p, offset)

    def forward(self, I_a_patch: torch.Tensor, I_b_patch: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            I_a_patch, I_b_patch: (B, 1, patch_h, patch_w) normalized grayscale.
        Returns dict with:
            H_patch:  (B, 3, 3)
            offset:   (B, 8)
            sf_a:     (B, C_sf, ph, pw) ShareFeature of I_a (for triplet loss)
            sf_b:     (B, C_sf, ph, pw) ShareFeature of I_b
        """
        B = I_a_patch.shape[0]
        device, dtype = I_a_patch.device, I_a_patch.dtype

        sf_a = self.share_feature(I_a_patch)
        sf_b = self.share_feature(I_b_patch)

        x = torch.cat([sf_a, sf_b], dim=1)                      # (B, 2, ph, pw)
        x = self.stem(x)                                        # 1/4
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)                                      # 1/32
        x = self.pool(x)                                        # (B, 512, 1, 1)
        raw = self.fc(x)                                        # (B, 8)
        offset = self.rho_init * torch.tanh(raw)                # bounded corner offsets (px)

        h4p = self._canonical_corners(B, self.patch_h, self.patch_w, device, dtype)
        H_patch = self._solve(h4p, offset)

        return {
            "H_patch": H_patch,
            "offset":  offset,
            "sf_a":    sf_a,
            "sf_b":    sf_b,
        }
