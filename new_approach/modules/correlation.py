import math

import torch
import torch.nn.functional as F

__all__ = ["local_correlation"]


def local_correlation(f_ref: torch.Tensor, f_tgt: torch.Tensor, radius: int = 4) -> torch.Tensor:
    """Local dot-product correlation volume around each target pixel."""
    if f_ref.shape != f_tgt.shape:
        raise ValueError(f"feature shapes must match, got {tuple(f_ref.shape)} and {tuple(f_tgt.shape)}")
    if f_ref.ndim != 4:
        raise ValueError("features must be [B, C, H, W]")

    radius = int(radius)
    if radius < 0:
        raise ValueError("radius must be non-negative")

    batch_size, channels, height, width = f_ref.shape
    if radius == 0:
        return (f_ref * f_tgt).sum(dim=1, keepdim=True) / math.sqrt(max(channels, 1))

    kernel = 2 * radius + 1
    padded = F.pad(f_tgt, (radius, radius, radius, radius), mode="replicate")
    patches = F.unfold(padded, kernel_size=kernel)
    patches = patches.view(batch_size, channels, kernel * kernel, height, width)
    corr = (f_ref.unsqueeze(2) * patches).sum(dim=1)
    return corr / math.sqrt(max(channels, 1))
