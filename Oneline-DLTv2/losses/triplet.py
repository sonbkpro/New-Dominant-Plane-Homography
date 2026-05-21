"""L_triplet (q-weighted hinge), anti-collapse for the feature extractor."""

import torch
import torch.nn.functional as F


def triplet_loss(
    F_b: torch.Tensor,
    F_a_warped: torch.Tensor,
    F_a: torch.Tensor,
    q: torch.Tensor,
    valid_mask: torch.Tensor,
    margin: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """All inputs are (B, C, H, W) or (B, 1, H, W) at the same resolution.

    d^+_i = ||F_b(i) - F_a_warped(i)||_1     (positive: should be small)
    d^-_i = ||F_b(i) - F_a(i)||_1             (negative: should be larger)
    L = sum_i w_i max(0, m + d^+ - d^-) / sum_i w_i,  where w_i = q_i * valid_i
    """
    d_pos = (F_b - F_a_warped).abs().sum(dim=1, keepdim=True)
    d_neg = (F_b - F_a).abs().sum(dim=1, keepdim=True)
    hinge = (margin + d_pos - d_neg).clamp(min=0.0)

    w = q * valid_mask
    num = (hinge * w).sum(dim=(1, 2, 3))
    den = w.sum(dim=(1, 2, 3)).clamp(min=eps)
    return (num / den).mean()
