"""Prediction heads for the CDPC network.

- HomographyHead:  pooled features -> 8-DoF corner offset, BOUNDED by rho*tanh.
- PosteriorHead:   per-pixel features -> q in [0, 1]
- UncertaintyHead: per-pixel features -> log sigma via sigma_min + softplus(u)
- ReliabilityHead: pooled summary statistics -> s in [0, 1]
"""

from math import log

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_bn_relu(in_c: int, out_c: int, k: int = 3) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=k, padding=k // 2, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class HomographyHead(nn.Module):
    """Consumes pooled cross-image features and regresses 8 corner offsets.

    Output is bounded: offset = rho * tanh(raw). The tanh bound caps the
    maximum corner displacement at `rho` px regardless of feature magnitude,
    which prevents fold-over of the warp grid and any single step from
    collapsing the homography to a degenerate state.
    """

    def __init__(self, in_channels: int, rho: float = 16.0):
        super().__init__()
        self.rho = rho
        self.tower = nn.Sequential(
            _conv_bn_relu(in_channels, 256),
            _conv_bn_relu(256, 256),
            _conv_bn_relu(256, 256, k=3),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 8),
        )
        # Init scale chosen so that initial raw is O(0.5) and the tanh
        # is in its linear regime; initial offset is then ~rho * 0.5 ~= 8 px
        # for rho=16. Bias stays zero so initial *mean* offset is zero.
        nn.init.normal_(self.fc[-1].weight, std=0.05)
        nn.init.zeros_(self.fc[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, in_channels, H, W). Returns (B, 8) corner offset in pixels."""
        h = self.tower(x)
        h = self.pool(h)
        raw = self.fc(h)
        return self.rho * torch.tanh(raw)


class PosteriorHead(nn.Module):
    """Per-pixel posterior P(z=1 | I_a, I_b). Final bias is initialized so
    that q ~= init_prob at init."""

    def __init__(self, in_channels: int, hidden: int = 64, init_prob: float = 0.7):
        super().__init__()
        self.tower = nn.Sequential(
            _conv_bn_relu(in_channels, hidden),
            _conv_bn_relu(hidden, hidden),
            _conv_bn_relu(hidden, hidden),
        )
        self.head = nn.Conv2d(hidden, 1, kernel_size=1)

        with torch.no_grad():
            nn.init.normal_(self.head.weight, std=1e-3)
            init_prob = float(min(max(init_prob, 1e-3), 1.0 - 1e-3))
            self.head.bias.fill_(log(init_prob / (1.0 - init_prob)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, in_channels, H, W). Returns (B, 1, H, W) in (0, 1)."""
        h = self.tower(x)
        logits = self.head(h)
        return torch.sigmoid(logits)


class UncertaintyHead(nn.Module):
    """Per-pixel uncertainty parameterized as sigma = sigma_min + softplus(u).

    This is strictly better than clamp(log_sigma): softplus is differentiable
    everywhere, lower-bounded by sigma_min, and grows linearly with u so the
    Kendall-Gal denominator can adapt without dead gradient zones. Without an
    upper clamp the value can in principle grow large; the L_sigma_reg term
    in train.py keeps it near 1.
    """

    def __init__(self, in_channels: int, hidden: int = 64, sigma_min: float = 0.05):
        super().__init__()
        self.tower = nn.Sequential(
            _conv_bn_relu(in_channels, hidden),
            _conv_bn_relu(hidden, hidden),
            _conv_bn_relu(hidden, hidden),
        )
        self.head = nn.Conv2d(hidden, 1, kernel_size=1)
        self.sigma_min = sigma_min

        with torch.no_grad():
            nn.init.normal_(self.head.weight, std=1e-3)
            # u=0 -> sigma = sigma_min + softplus(0) = sigma_min + log(2) ~= 0.74
            # for sigma_min=0.05, matching the prior clamp lower bound.
            self.head.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.tower(x)
        u = self.head(h)
        sigma = self.sigma_min + F.softplus(u)
        return torch.log(sigma)


class ReliabilityHead(nn.Module):
    """Maps a small statistics vector to a scalar s in (0, 1)."""

    def __init__(self, feat_dim: int, hidden: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        """phi: (B, feat_dim). Returns (B,) scalar in (0, 1)."""
        return torch.sigmoid(self.mlp(phi)).squeeze(-1)
