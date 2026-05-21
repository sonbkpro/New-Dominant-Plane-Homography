"""L_support: floor on the average posterior to prevent collapse."""

import torch


def support_loss(
    q: torch.Tensor,
    valid_mask: torch.Tensor,
    alpha: float = 0.10,
) -> torch.Tensor:
    """Returns max(0, alpha - q_bar)^2 averaged over batch, where q_bar is
    the mean of q over valid pixels."""
    denom = valid_mask.sum(dim=(1, 2, 3)).clamp(min=1.0)
    q_bar = (q * valid_mask).sum(dim=(1, 2, 3)) / denom
    deficit = (alpha - q_bar).clamp(min=0.0)
    return (deficit ** 2).mean()
