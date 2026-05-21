"""L_align: Kendall-Gal heteroscedastic likelihood with Charbonnier residual,
weighted by the dominant-plane posterior q."""

import torch


def alignment_loss(
    residual: torch.Tensor,        # (B, 1, H, W)  r_i = ||F_b - F_a_warped||
    log_sigma: torch.Tensor,       # (B, 1, H, W)
    q: torch.Tensor,               # (B, 1, H, W)
    valid_mask: torch.Tensor,      # (B, 1, H, W)
    charbonnier_eps: float = 1e-3,
    weight_eps: float = 1e-6,
) -> torch.Tensor:
    """
    L = sum_i w_i [ rho(r_i / sigma_i) + log sigma_i ] / sum_i w_i,
    w_i = q_i * valid_i, rho(x) = sqrt(x^2 + eps^2).
    """
    sigma = log_sigma.exp()
    x = residual / (sigma + 1e-8)
    rho = torch.sqrt(x * x + charbonnier_eps * charbonnier_eps)
    per_pixel = rho + log_sigma

    w = q * valid_mask
    num = (per_pixel * w).sum(dim=(1, 2, 3))
    den = w.sum(dim=(1, 2, 3)).clamp(min=weight_eps)
    return (num / den).mean()
