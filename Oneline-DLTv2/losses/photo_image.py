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


def photometric_image_q_weighted_loss(
    I_a_full: torch.Tensor,        # (B, 1, H_full, W_full)
    I_b_patch: torch.Tensor,       # (B, 1, ph, pw)
    H_full: torch.Tensor,          # (B, 3, 3) full-image-coords homography
    crop_xy: torch.Tensor,         # (B, 2)
    q_full: torch.Tensor,          # (B, 1, ph_q, pw_q) posterior at 1/4 scale
    charbonnier_eps: float = 1e-3,
    weight_eps: float = 1e-6,
) -> torch.Tensor:
    """q-weighted image-space photometric Charbonnier (Fix B).

    Identical to photometric_image_loss except each pixel's contribution is
    weighted by q (upsampled from its native 1/4 scale to the patch scale).
    Denominator is sum_i v_i (not sum_i q_i v_i), so the model cannot
    suppress cost by lowering q uniformly — only spatially selective q
    variation pays off.

    Collapse-immune: this loss operates on raw normalized grayscale, not on
    backbone features. q can suppress non-planar pixels without inducing any
    feature shrinkage. This is the H-side counterpart to the q-gating that
    planv3 intended for `align_soft`.

    Why q_full is upsampled rather than q being predicted at patch resolution:
    the CDPCNet emits q at 1/4 scale; upsampling preserves the spatial
    structure q learned during q_only without requiring an architectural
    change to the posterior head.
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

    # Upsample q to patch resolution. Gradient flows back to the posterior
    # head — this is intentional: photometric residual on selected pixels
    # gives q a real geometric supervision signal, complementary to EM.
    q_up = F.interpolate(q_full, size=(ph, pw), mode="bilinear", align_corners=True)

    err = _charbonnier(I_b_patch - I_a_warped, eps=charbonnier_eps)
    w = q_up * valid
    num = (err * w).sum(dim=(1, 2, 3))
    den = valid.sum(dim=(1, 2, 3)).clamp(min=weight_eps)
    return (num / den).mean()
