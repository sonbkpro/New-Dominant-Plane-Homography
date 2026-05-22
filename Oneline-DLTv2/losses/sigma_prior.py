"""planv3 §4.4 / B3: data-adaptive log-residual prior on sigma.

Old:
    L_sigma_reg = mean( (log sigma)^2 ),
pulling sigma toward 1. This is independent of the actual residual scale,
so as residuals shrink during training sigma stays at 1 and the Kendall-Gal
denominator is mis-calibrated.

New:
    L_sigma = mean_v( (log sigma_i - mu_sigma)^2 ),
    mu_sigma = sg( median_i log (r_i + eps) ),
where mu_sigma is the (stop-graded) median log-residual of the batch. This
pulls sigma toward the residual scale, which together with L_align_het's
log-sigma term gives an equilibrium sigma_i ≈ r_i on inlier pixels.
"""

import torch


def sigma_prior_loss(
    log_sigma: torch.Tensor,       # (B, 1, H, W)
    residual: torch.Tensor,        # (B, 1, H, W)
    valid_mask: torch.Tensor,      # (B, 1, H, W)
    eps: float = 1e-3,
    weight_eps: float = 1e-6,
) -> torch.Tensor:
    """Per-batch log-residual-anchored regularizer on sigma."""
    with torch.no_grad():
        # Median log r over valid pixels of the whole batch (NOT per-sample,
        # since per-sample median is noisy with small batches).
        r_clamped = residual.clamp(min=eps)
        v = (valid_mask > 0.5)
        if v.any():
            log_r = torch.log(r_clamped[v])
            mu_sigma = log_r.median()
        else:
            mu_sigma = torch.zeros((), device=residual.device, dtype=residual.dtype)

    diff = log_sigma - mu_sigma
    diff2 = diff * diff
    num = (diff2 * valid_mask).sum(dim=(1, 2, 3))
    den = valid_mask.sum(dim=(1, 2, 3)).clamp(min=weight_eps)
    return (num / den).mean()
