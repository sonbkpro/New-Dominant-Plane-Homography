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
    use_q_weighting: bool = False,
) -> torch.Tensor:
    """All inputs are (B, C, H, W) or (B, 1, H, W) at the same resolution.

    d^+_i = ||F_b(i) - F_a_warped(i)||_1     (positive: should be small)
    d^-_i = ||F_b(i) - F_a(i)||_1             (negative: should be larger)

    By default L uses UNIFORM per-pixel weighting (valid_mask only). The
    plan's q-weighted formulation was empirically self-defeating: when q
    collapses, the triplet gradient vanishes and H stops being learned,
    which then makes q-collapse worse. Decoupling triplet from q keeps H
    learning robust even when the posterior is transiently degenerate.
    `use_q_weighting=True` restores the plan-literal behavior for ablation.
    """
    # Channel-MEAN (not sum) so per-pixel distances are O(1) regardless of C.
    # With sum-over-channels the margin is effectively zero relative to typical
    # distances and the hinge stays pinned at m, preventing H from learning.
    d_pos = (F_b - F_a_warped).abs().mean(dim=1, keepdim=True)
    d_neg = (F_b - F_a).abs().mean(dim=1, keepdim=True)
    hinge = (margin + d_pos - d_neg).clamp(min=0.0)

    w = (q * valid_mask) if use_q_weighting else valid_mask
    num = (hinge * w).sum(dim=(1, 2, 3))
    den = w.sum(dim=(1, 2, 3)).clamp(min=eps)
    return (num / den).mean()
