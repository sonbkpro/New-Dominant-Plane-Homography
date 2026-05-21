"""Local correlation block: per-pixel cosine similarity between F_a and a
shifted F_b within a small radius R. Output is then reduced with a small
1x1 conv stack to a compact per-pixel feature `c`."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalCorrelation(nn.Module):
    def __init__(self, radius: int = 4, out_channels: int = 32):
        super().__init__()
        self.radius = radius
        self.cost_dim = (2 * radius + 1) ** 2
        self.reduce = nn.Sequential(
            nn.Conv2d(self.cost_dim, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, F_a: torch.Tensor, F_b: torch.Tensor) -> torch.Tensor:
        """F_a, F_b: (B, C, H, W). Returns (B, out_channels, H, W)."""
        B, C, H, W = F_a.shape
        R = self.radius

        # L2-normalize for cosine similarity (numerically nicer than raw dot).
        Fa_n = F.normalize(F_a, dim=1, eps=1e-6)
        Fb_n = F.normalize(F_b, dim=1, eps=1e-6)

        # Pad F_b so we can extract shifted versions.
        Fb_pad = F.pad(Fb_n, [R, R, R, R], mode="replicate")     # (B, C, H+2R, W+2R)

        costs = []
        # Iterate over the (2R+1)^2 shifts. The volume is small for R<=4.
        for dy in range(-R, R + 1):
            for dx in range(-R, R + 1):
                Fb_shift = Fb_pad[:, :, R + dy : R + dy + H, R + dx : R + dx + W]
                cost = (Fa_n * Fb_shift).sum(dim=1, keepdim=True)   # (B, 1, H, W)
                costs.append(cost)
        cost_volume = torch.cat(costs, dim=1)                       # (B, (2R+1)^2, H, W)

        return self.reduce(cost_volume)
