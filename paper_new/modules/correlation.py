"""H-aware local feature correlation for PCGV."""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from paper_new.geometry import torch_make_pixel_grid, torch_pixel_to_norm


def make_pixel_grid(batch: int, height: int, width: int, device=None, dtype=None) -> torch.Tensor:
    """Alias used by the PCGV plan."""
    return torch_make_pixel_grid(batch, height, width, device=device, dtype=dtype)


def build_offsets(radius: int, device=None, dtype=None) -> torch.Tensor:
    """Return ``[(2r+1)^2, 2]`` local offsets in ``(dx, dy)`` order."""
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    dy, dx = torch.meshgrid(coords, coords, indexing="ij")
    return torch.stack((dx, dy), dim=-1).reshape(-1, 2)


def sample_features_at_points(feat: torch.Tensor, points: torch.Tensor,
                              padding_mode: str = "zeros") -> torch.Tensor:
    """Bilinearly sample ``feat`` at ``[B, N, 2]`` pixel points.

    Returns ``[B, N, C]``.
    """
    batch, channels, height, width = feat.shape
    grid = torch_pixel_to_norm(points.to(dtype=feat.dtype), height, width).view(batch, -1, 1, 2)
    sampled = F.grid_sample(
        feat,
        grid,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=True,
    )
    return sampled.squeeze(-1).transpose(1, 2).contiguous()


def extract_local_windows(feat_b: torch.Tensor, center_points: torch.Tensor,
                          radius: int, padding_mode: str = "zeros") -> torch.Tensor:
    """Sample local windows around ``center_points`` from ``feat_b``.

    Args:
        feat_b: ``[B, C, H, W]``.
        center_points: ``[B, N, 2]`` feature-scale pixel centers.

    Returns:
        ``[B, C, N, K]`` where ``K=(2r+1)^2``.
    """
    batch, _, height, width = feat_b.shape
    offsets = build_offsets(radius, feat_b.device, feat_b.dtype)
    points = center_points.to(dtype=feat_b.dtype).unsqueeze(2) + offsets.view(1, 1, -1, 2)
    norm = torch_pixel_to_norm(points, height, width)
    return F.grid_sample(
        feat_b,
        norm,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=True,
    )


def local_correlation(feat_a: torch.Tensor, feat_b: torch.Tensor,
                      center_points: torch.Tensor, radius: int,
                      grid_a: Optional[torch.Tensor] = None,
                      padding_mode: str = "zeros") -> torch.Tensor:
    """Compute local dot-product correlation around H-projected centers.

    Returns raw correlation logits with shape ``[B, N, K]``.
    """
    batch, channels, height, width = feat_a.shape
    if grid_a is None:
        grid_a = make_pixel_grid(batch, height, width, feat_a.device, feat_a.dtype)
    feat_a_n = F.normalize(feat_a, dim=1)
    feat_b_n = F.normalize(feat_b, dim=1)
    tokens_a = sample_features_at_points(feat_a_n, grid_a, padding_mode="border").transpose(1, 2)
    windows_b = extract_local_windows(feat_b_n, center_points, radius, padding_mode=padding_mode)
    return (tokens_a.unsqueeze(-1) * windows_b).sum(dim=1) / math.sqrt(float(channels))


def soft_argmax_corr(corr_volume: torch.Tensor, radius: int,
                     temperature: float = 0.1) -> Dict[str, torch.Tensor]:
    """Soft-argmax a local correlation volume into a subpixel offset."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    offsets = build_offsets(radius, corr_volume.device, corr_volume.dtype)
    probs = torch.softmax(corr_volume / temperature, dim=-1)
    delta = torch.matmul(probs, offsets)

    top2 = torch.topk(probs, k=min(2, probs.shape[-1]), dim=-1).values
    peak = top2[..., :1]
    gap = top2[..., :1] - (top2[..., 1:2] if top2.shape[-1] > 1 else torch.zeros_like(top2[..., :1]))
    entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1, keepdim=True)
    return {
        "delta": delta,
        "probs": probs,
        "peak": peak,
        "gap": gap,
        "entropy": entropy,
    }


def correlation_match(feat_a: torch.Tensor, feat_b: torch.Tensor,
                      center_points: torch.Tensor, radius: int,
                      grid_a: Optional[torch.Tensor] = None,
                      temperature: float = 0.1,
                      padding_mode: str = "zeros") -> Dict[str, torch.Tensor]:
    """Run local correlation and return matched points plus diagnostics."""
    corr = local_correlation(
        feat_a,
        feat_b,
        center_points,
        radius=radius,
        grid_a=grid_a,
        padding_mode=padding_mode,
    )
    stats = soft_argmax_corr(corr, radius=radius, temperature=temperature)
    matches = center_points + stats["delta"]
    return {
        "corr": corr,
        "matches": matches,
        "delta": stats["delta"],
        "probs": stats["probs"],
        "peak": stats["peak"],
        "gap": stats["gap"],
        "entropy": stats["entropy"],
    }

