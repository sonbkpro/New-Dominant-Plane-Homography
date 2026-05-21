"""L_cycle: analytic-inverse cycle consistency in feature space.

L = mean over valid pixels of || F_a - W(W(F_a, H_ab), H_ab^{-1}) ||_1.

Samples with ill-conditioned H_ab (cond_valid == 0) are excluded."""

import torch


def cycle_loss(
    F_a: torch.Tensor,                  # (B, C, H, W)
    F_a_recovered: torch.Tensor,        # (B, C, H, W) = W(W(F_a, H), H^{-1})
    cycle_valid: torch.Tensor,          # (B, 1, H, W) per-pixel validity (in both warps)
    cond_valid: torch.Tensor,           # (B,) 1.0 if H well-conditioned, else 0
    eps: float = 1.0,
) -> torch.Tensor:
    err = (F_a - F_a_recovered).abs().sum(dim=1, keepdim=True)     # (B, 1, H, W)
    denom = cycle_valid.sum(dim=(1, 2, 3)).clamp(min=eps)
    per_sample = (err * cycle_valid).sum(dim=(1, 2, 3)) / denom    # (B,)
    per_sample = per_sample * cond_valid
    # Average over the well-conditioned samples; if none, return zero.
    n_valid = cond_valid.sum().clamp(min=1.0)
    return per_sample.sum() / n_valid
