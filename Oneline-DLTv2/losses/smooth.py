"""L_smooth: edge-aware total variation on q. Allows discontinuities at
image edges; suppresses noise on flat regions."""

import torch
import torch.nn.functional as F


def edge_aware_smoothness(
    q: torch.Tensor,                 # (B, 1, H, W)
    image: torch.Tensor,             # (B, 1, H, W) reference for edges (use I_b patch resized)
    gamma: float = 10.0,
) -> torch.Tensor:
    """L = mean( |dq/dx| * exp(-gamma * |dI/dx|) + |dq/dy| * exp(-gamma * |dI/dy|) )."""
    # Resize image to match q's spatial extent if needed.
    if image.shape[-2:] != q.shape[-2:]:
        image = F.interpolate(image, size=q.shape[-2:], mode="bilinear", align_corners=True)

    dq_dx = (q[..., :, 1:] - q[..., :, :-1]).abs()
    dq_dy = (q[..., 1:, :] - q[..., :-1, :]).abs()
    di_dx = (image[..., :, 1:] - image[..., :, :-1]).abs().mean(dim=1, keepdim=True)
    di_dy = (image[..., 1:, :] - image[..., :-1, :]).abs().mean(dim=1, keepdim=True)

    sx = dq_dx * torch.exp(-gamma * di_dx)
    sy = dq_dy * torch.exp(-gamma * di_dy)
    return sx.mean() + sy.mean()
