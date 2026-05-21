"""Prediction heads for the CDPC network.

- HomographyHead:  pooled features -> 8-DoF corner offset
- PosteriorHead:   per-pixel features -> q in [0, 1]
- UncertaintyHead: per-pixel features -> log sigma (clamped)
- ReliabilityHead: pooled summary statistics -> s in [0, 1]"""

from math import log

import torch
import torch.nn as nn


def _conv_bn_relu(in_c: int, out_c: int, k: int = 3) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=k, padding=k // 2, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class HomographyHead(nn.Module):
    """Consumes [F_a^(1/8), F_b^(1/8), correlation_c] concatenated channelwise
    and regresses 8 corner offsets via global average pooling + MLP.

    Output is in pixels, scaled by `rho` so the network outputs are roughly
    unit-magnitude at initialization.
    """

    def __init__(self, in_channels: int, rho: float = 32.0):
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
        # Small weight init + zero bias keeps the initial offset close to zero
        # (so the predicted homography is near identity) WITHOUT killing the
        # upstream gradient — fully zero weights cut off correlation/backbone
        # on the first step.
        nn.init.normal_(self.fc[-1].weight, std=1e-3)
        nn.init.zeros_(self.fc[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, in_channels, H, W). Returns (B, 8) corner offset in pixels."""
        h = self.tower(x)
        h = self.pool(h)
        offset = self.fc(h)
        return offset * self.rho


class PosteriorHead(nn.Module):
    """Per-pixel posterior P(z=1 | I_a, I_b). Final bias is initialized so
    that q ≈ 0.7 at init, which prevents the early-training collapse the
    user observed in v1 (mask all-zero around iter 65k).
    """

    def __init__(self, in_channels: int, hidden: int = 64, init_prob: float = 0.7):
        super().__init__()
        self.tower = nn.Sequential(
            _conv_bn_relu(in_channels, hidden),
            _conv_bn_relu(hidden, hidden),
            _conv_bn_relu(hidden, hidden),
        )
        self.head = nn.Conv2d(hidden, 1, kernel_size=1)

        with torch.no_grad():
            # Small weight + biased toward init_prob. Pure zero weight would
            # kill upstream gradient on the first step.
            nn.init.normal_(self.head.weight, std=1e-3)
            init_prob = float(min(max(init_prob, 1e-3), 1.0 - 1e-3))
            self.head.bias.fill_(log(init_prob / (1.0 - init_prob)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, in_channels, H, W). Returns (B, 1, H, W) in (0, 1)."""
        h = self.tower(x)
        logits = self.head(h)
        return torch.sigmoid(logits)


class UncertaintyHead(nn.Module):
    """Per-pixel log sigma. Clamped to [-5, 5] to prevent numerical blow-up
    inside the Kendall-Gal likelihood."""

    def __init__(self, in_channels: int, hidden: int = 64,
                 log_sigma_min: float = -5.0, log_sigma_max: float = 5.0):
        super().__init__()
        self.tower = nn.Sequential(
            _conv_bn_relu(in_channels, hidden),
            _conv_bn_relu(hidden, hidden),
            _conv_bn_relu(hidden, hidden),
        )
        self.head = nn.Conv2d(hidden, 1, kernel_size=1)
        self.lo, self.hi = log_sigma_min, log_sigma_max

        with torch.no_grad():
            nn.init.normal_(self.head.weight, std=1e-3)
            self.head.bias.zero_()                   # init sigma = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.tower(x)
        log_sigma = self.head(h)
        return log_sigma.clamp(self.lo, self.hi)


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
