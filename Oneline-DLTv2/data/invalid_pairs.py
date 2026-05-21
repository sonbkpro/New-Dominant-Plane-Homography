"""Invalid-pair construction for the reliability head.

We do NOT have video-id metadata, so two strategies are used:

1. Intra-batch shuffle: pair I_a^{(k)} with I_b^{(pi(k))} under a random
   permutation pi. With high probability the two patches do not share a
   homography. Label y = 0.

2. Patch reshuffle within I_a: extract a second, non-overlapping crop from
   I_a alone and present (crop1, crop2) as a pair. Label y = 0.

Both negatives are concatenated to the natural batch before the forward pass."""

from typing import Tuple

import torch


@torch.no_grad()
def build_invalid_batch(
    I_a: torch.Tensor,          # (B, 1, H, W) natural pair side a
    I_b: torch.Tensor,          # (B, 1, H, W) natural pair side b
    shuffle_frac: float = 0.5,
    reshuffle_frac: float = 0.25,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (I_a_neg, I_b_neg, y_neg) where:
        I_a_neg : (B_neg, 1, H, W)
        I_b_neg : (B_neg, 1, H, W)
        y_neg   : (B_neg,) all zeros
    `shuffle_frac` and `reshuffle_frac` are fractions of the original batch
    size B used for each negative strategy."""
    B, C, H, W = I_a.shape
    device = I_a.device

    n_shuffle = max(1, int(round(shuffle_frac * B)))
    n_reshuf  = max(1, int(round(reshuffle_frac * B)))

    # --- (1) intra-batch shuffle ---
    perm = torch.randperm(B, device=device)
    while (perm == torch.arange(B, device=device)).all():
        perm = torch.randperm(B, device=device)
    idx_a = torch.arange(B, device=device)[:n_shuffle]
    idx_b = perm[:n_shuffle]
    I_a_sh = I_a[idx_a]
    I_b_sh = I_b[idx_b]

    # --- (2) patch reshuffle within a single image ---
    # We split the spatial extent in half horizontally and pair the two halves.
    # If H or W is too small for two halves, fall back to flipping.
    if W >= 2 * 16:
        half = W // 2
        left  = I_a[:n_reshuf, :, :, :half]
        right = I_a[:n_reshuf, :, :, half:half + half]
        # Both halves resized back to (H, W) so the model sees the same shape.
        left  = torch.nn.functional.interpolate(left,  size=(H, W), mode="bilinear", align_corners=True)
        right = torch.nn.functional.interpolate(right, size=(H, W), mode="bilinear", align_corners=True)
        I_a_rs = left
        I_b_rs = right
    else:
        I_a_rs = I_a[:n_reshuf]
        I_b_rs = torch.flip(I_a[:n_reshuf], dims=[-1])

    I_a_neg = torch.cat([I_a_sh, I_a_rs], dim=0)
    I_b_neg = torch.cat([I_b_sh, I_b_rs], dim=0)
    y_neg = torch.zeros(I_a_neg.shape[0], device=device)
    return I_a_neg, I_b_neg, y_neg
