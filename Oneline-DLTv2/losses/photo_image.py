"""planv3 B5 / §5: image-space photometric Charbonnier anchor.

The legacy `L_photo` in train.py reused the feature-space residual that L_align
already consumes -- so it added no independent signal. This module computes
photometric reconstruction error directly on the raw normalized grayscale
patch:

    I_b^p(p) ≈ I_a_full( H_full^{-1} @ (p + crop_xy) )   (sampled by grid_sample)

The Charbonnier penalty rho_eps( I_b - W(I_a, H) ) is robust to outliers and
provides an anchor that is independent of feature drift -- crucial in the
h_only stage before the posterior is trained.

Output is a scalar averaged over valid pixels (where the warp drew from
inside the full image).
"""

import torch
import torch.nn.functional as F

from utils.warping import warp_by_homography, make_validity_mask


def _charbonnier(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt(x * x + eps * eps)


def photometric_image_loss(
    I_a_full: torch.Tensor,        # (B, 1, H_full, W_full) normalized grayscale
    I_b_patch: torch.Tensor,       # (B, 1, ph, pw)
    H_full: torch.Tensor,          # (B, 3, 3) full-image-coords homography
    crop_xy: torch.Tensor,         # (B, 2) patch origin in full-image pixel coords
    charbonnier_eps: float = 1e-3,
) -> torch.Tensor:
    """Sample I_a_full at H_full^{-1}(p + crop_xy) for each patch pixel p,
    compare against I_b_patch, return Charbonnier(diff) averaged over the
    valid mask.

    The validity mask is derived from the warp itself: pixels where the
    backward-mapped source coordinate lies outside I_a_full's extent are
    excluded so border zeros do not bias the loss downward.
    """
    B, _, ph, pw = I_b_patch.shape
    H_full_in = I_a_full.shape[-2]
    W_full_in = I_a_full.shape[-1]

    I_a_warped = warp_by_homography(
        I_a_full, H_full,
        out_size=(ph, pw),
        out_origin_xy=crop_xy,
        padding_mode="zeros",
    )
    valid = make_validity_mask(
        H_full,
        out_size=(ph, pw),
        in_size=(H_full_in, W_full_in),
        out_origin_xy=crop_xy,
    )

    err = _charbonnier(I_b_patch - I_a_warped, eps=charbonnier_eps)
    num = (err * valid).sum(dim=(1, 2, 3))
    den = valid.sum(dim=(1, 2, 3)).clamp(min=1.0)
    return (num / den).mean()
