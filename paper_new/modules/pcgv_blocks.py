"""Reusable neural blocks for Progressive Correlation-Guided Voting."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EvidenceEncoder(nn.Module):
    """Encode per-point geometric/matching evidence into a hidden token."""

    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, evidence: torch.Tensor) -> torch.Tensor:
        return self.net(evidence)


class VoteGRU(nn.Module):
    """GRUCell applied independently to every feature location."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.cell = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        batch, num_points, hidden_dim = z.shape
        h_next = self.cell(z.reshape(batch * num_points, hidden_dim),
                           h.reshape(batch * num_points, hidden_dim))
        return h_next.reshape(batch, num_points, hidden_dim)


class GeometryAwareTransformer(nn.Module):
    """Optional lightweight token transformer with geometric positional input."""

    def __init__(self, hidden_dim: int, num_heads: int = 4, depth: int = 1):
        super().__init__()
        self.pos = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)

    def forward(self, h: torch.Tensor, xy: torch.Tensor,
                delta: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        geom = torch.cat((xy, delta, residual), dim=-1)
        return self.encoder(h + self.pos(geom))


class PlaneTokenConsensus(nn.Module):
    """Compute and broadcast a global dominant-plane token."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        attn = torch.softmax(self.score(h), dim=1)
        token = (attn * h).sum(dim=1, keepdim=True)
        token = self.proj(token)
        return token.expand_as(h)


class VoteHead(nn.Module):
    """Predict correspondence vote logits."""

    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UncertaintyHead(nn.Module):
    """Predict positive per-correspondence uncertainty."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.net(h)) + 1e-4


class MaskUpsampler(nn.Module):
    """Upsample a feature-grid mask to a requested image/patch resolution."""

    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, mask: torch.Tensor, size) -> torch.Tensor:
        up = F.interpolate(mask, size=size, mode="bilinear", align_corners=True)
        return self.refine(up)

