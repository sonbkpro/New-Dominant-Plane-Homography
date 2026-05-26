"""Top-level Calibrated Dominant-Plane Consensus network -- planv3 redesign.

Architecture (see planv3 §3):

  Per-image features:
      backbone = FeaturePyramid   (Siamese trunk, ResNet-18 stem + layer1-3)
      F_a_full = backbone(I_a_full) -> {stride4, stride8, stride16}
      F_b_full = backbone(I_b_full) -> {stride4, stride8, stride16}

  Patch-cropped features (differentiable crop at crop_xy):
      F_a_patch_t = crop(F_a_full_t, crop_xy / stride_t)
      F_b_patch_t = crop(F_b_full_t, crop_xy / stride_t)

  Homography pyramid:
      H_patch, per_level_offsets = HomographyPyramid(F_a_patch, F_b_patch, patch_h, patch_w)

  Image-coord conversion:
      H_full = T(+crop) @ H_patch @ T(-crop)

  Warping for the dominant-plane block (FULL-image source -> patch destination):
      F_a_warped(p_b) = sample(F_a_full_quarter, H_full^{-1}(p_b + crop_xy))

  Per-pixel heads at 1/4:
      r       = mean_c |F_b_patch_quarter - F_a_warped|
      q       = PosteriorHead([F_b_patch_quarter, F_a_warped, r]) * valid_mask
      log_sigma = UncertaintyHead([same input])

  Reliability:
      phi = pooled statistics of (q, r, sigma, cycle_residual, offset, kappa(H), A_v)
      s   = ReliabilityHead(sg(phi))

v2's joint 2-channel backbone is removed. Pairwise cues come from the
correlation block (now per-level inside HomographyPyramid) and from the
F_b pre-warp at each level.
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.feature_pyramid import FeaturePyramid
from model.homography_pyramid import HomographyPyramid
from model.heads import PosteriorHead, UncertaintyHead, ReliabilityHead
from utils.warping import (
    warp_by_homography, warp_patch_by_homography, make_validity_mask,
)
from utils.inverse import safe_inverse_3x3, condition_number_3x3


# ---------------------------------------------------------------------------
# Geometric helpers
# ---------------------------------------------------------------------------

def _rescale_homography(H: torch.Tensor, scale_in_to_out: float) -> torch.Tensor:
    B = H.shape[0]
    device, dtype = H.device, H.dtype
    s = scale_in_to_out
    S = torch.tensor([[s, 0, 0], [0, s, 0], [0, 0, 1]], device=device, dtype=dtype)
    Si = torch.tensor([[1.0 / s, 0, 0], [0, 1.0 / s, 0], [0, 0, 1]], device=device, dtype=dtype)
    Sb  = S.unsqueeze(0).expand(B, -1, -1)
    Sib = Si.unsqueeze(0).expand(B, -1, -1)
    return torch.bmm(torch.bmm(Sb, H), Sib)


def _patch_to_full(H_patch: torch.Tensor, crop_xy: torch.Tensor) -> torch.Tensor:
    """H_full = T(+crop) @ H_patch @ T(-crop)."""
    B = H_patch.shape[0]
    device, dtype = H_patch.device, H_patch.dtype
    T = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(B, 1, 1)
    T[:, 0, 2] = crop_xy[:, 0]
    T[:, 1, 2] = crop_xy[:, 1]
    Ti = T.clone()
    Ti[:, 0, 2] = -crop_xy[:, 0]
    Ti[:, 1, 2] = -crop_xy[:, 1]
    return torch.bmm(torch.bmm(T, H_patch), Ti)


def _crop_feature(feat: torch.Tensor, origin_xy: torch.Tensor,
                  out_h: int, out_w: int) -> torch.Tensor:
    """Differentiable bilinear crop of a feature map at a per-batch origin."""
    B, C, H, W = feat.shape
    device, dtype = feat.device, feat.dtype
    ys = torch.arange(out_h, device=device, dtype=dtype).view(1, out_h, 1).expand(B, out_h, out_w)
    xs = torch.arange(out_w, device=device, dtype=dtype).view(1, 1, out_w).expand(B, out_h, out_w)
    ox = origin_xy[:, 0].view(B, 1, 1).to(dtype)
    oy = origin_xy[:, 1].view(B, 1, 1).to(dtype)
    sx = xs + ox
    sy = ys + oy
    nx = 2.0 * sx / max(W - 1, 1) - 1.0
    ny = 2.0 * sy / max(H - 1, 1) - 1.0
    grid = torch.stack([nx, ny], dim=-1)
    return F.grid_sample(feat, grid, mode="bilinear",
                         padding_mode="zeros", align_corners=True)


# ---------------------------------------------------------------------------
# Top-level network
# ---------------------------------------------------------------------------

class CDPCNet(nn.Module):
    def __init__(
        self,
        patch_h: int = 315,
        patch_w: int = 560,
        backbone_pretrained: bool = True,
        # Per-image trunk channels per scale.
        bb_quarter_channels:   int = 64,
        bb_eighth_channels:    int = 128,
        bb_sixteenth_channels: int = 256,
        # Correlation block (per-level inside the pyramid).
        corr_radius: int = 4,
        corr_out_channels: int = 32,
        # Homography pyramid bounds.
        rho_per_level: tuple = (32.0, 16.0, 8.0),
        homography_levels: int = 3,
        use_normalized_dlt: bool = True,
        # Pixel heads.
        post_init_prob: float = 0.7,
        sigma_min: float = 0.5,
        # Postmortem fix: detach the q/sigma head inputs from the backbone
        # graph. Without this, in joint stage the backbone gradient via
        # (q-head input = F_b, F_a_warped, r) drives features to collapse
        # (the backbone learns to produce features that minimize EM loss
        # by shrinking r, not by encoding useful structure). With detach,
        # q and sigma heads still train against current features but cannot
        # reshape the backbone -- restoring the planv3-intended decoupling.
        detach_head_inputs: bool = False,
        # Legacy kwargs (ignored, retained so older configs still load).
        homography_rho: float = None,
        log_sigma_min: float = None,
        log_sigma_max: float = None,
    ):
        super().__init__()
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.detach_head_inputs = detach_head_inputs

        # ---- Per-image Siamese trunk ---------------------------------------
        self.backbone = FeaturePyramid(
            pretrained=backbone_pretrained,
            in_channels=1,
            out_channels_quarter=bb_quarter_channels,
            out_channels_eighth=bb_eighth_channels,
            out_channels_sixteenth=bb_sixteenth_channels,
        )

        # ---- Coarse-to-fine homography regressor ---------------------------
        self.homography_pyramid = HomographyPyramid(
            in_channels_per_level=(bb_sixteenth_channels,
                                   bb_eighth_channels,
                                   bb_quarter_channels),
            corr_radius=corr_radius,
            corr_out_channels=corr_out_channels,
            rho_per_level=rho_per_level,
            n_levels=homography_levels,
            use_normalized_dlt=use_normalized_dlt,
        )

        # ---- Pixel heads ---------------------------------------------------
        # Input is [F_b_patch_1/4, F_a_warped_1/4, residual] -- the v2 design
        # also pasted in a 1/4-upsampled correlation channel; planv3 §3.4
        # removes that upsample because correlation is now computed per-level
        # inside the H pyramid (and isn't needed by q/sigma -- the warped
        # residual is the relevant signal).
        post_in = 2 * bb_quarter_channels + 1
        self.posterior_head   = PosteriorHead(post_in, init_prob=post_init_prob)
        self.uncertainty_head = UncertaintyHead(post_in, sigma_min=sigma_min)

        # ---- Reliability calibrator ----------------------------------------
        # phi is 10-d: q_mean, q_var, q_area, r_mean, qr_mean, sigma_mean,
        # cycle_mean, offset_norm, kappa_log, area_valid.
        self.reliability_head = ReliabilityHead(feat_dim=10)

    # ----- pooled-stat helper ----------------------------------------------

    @staticmethod
    def _pool_stat(x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        if mask is None:
            return x.mean(dim=(1, 2, 3))
        denom = mask.sum(dim=(1, 2, 3)).clamp(min=1.0)
        return (x * mask).sum(dim=(1, 2, 3)) / denom

    # ----- forward ---------------------------------------------------------

    def forward(
        self,
        I_a_full: torch.Tensor,
        I_b_full: torch.Tensor,
        I_a_patch: torch.Tensor,         # kept for API back-compat; unused for backbone
        I_b_patch: torch.Tensor,         # kept for API back-compat; unused for backbone
        crop_xy: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            I_a_full, I_b_full: (B, 1, H_full, W_full) normalized grayscale.
            I_a_patch, I_b_patch: (B, 1, patch_h, patch_w) cropped patches at crop_xy.
              Not consumed by the trunk anymore (Siamese trunk runs on full images);
              still accepted so the dataloader interface is unchanged.
            crop_xy: (B, 2) [x, y] crop origin in full-image pixel coords.
        """
        B = I_a_full.shape[0]
        device = I_a_full.device
        dtype  = I_a_full.dtype
        ph, pw = self.patch_h, self.patch_w

        # ---- Per-image features on the FULL images -------------------------
        fa_full = self.backbone(I_a_full)
        fb_full = self.backbone(I_b_full)

        # ---- Crop patch-located features at all three pyramid scales -------
        # Derive per-scale patch sizes from the corresponding full-image
        # feature map sizes so any odd-rounding mismatch is consistent.
        crop4  = crop_xy / 4.0
        crop8  = crop_xy / 8.0
        crop16 = crop_xy / 16.0

        # Patch-shape at each scale: derived from feature-map dims (matches
        # the v2 behavior of the joint backbone running on the patch).
        # We use ceil-division here because torch ResNet rounds odd inputs UP.
        ph4  = (ph  + 3) // 4
        pw4  = (pw  + 3) // 4
        ph8  = (ph  + 7) // 8
        pw8  = (pw  + 7) // 8
        ph16 = (ph + 15) // 16
        pw16 = (pw + 15) // 16

        F_a4   = _crop_feature(fa_full["stride4"],   crop4,  ph4,  pw4)
        F_b4   = _crop_feature(fb_full["stride4"],   crop4,  ph4,  pw4)
        F_a8   = _crop_feature(fa_full["stride8"],   crop8,  ph8,  pw8)
        F_b8   = _crop_feature(fb_full["stride8"],   crop8,  ph8,  pw8)
        F_a16  = _crop_feature(fa_full["stride16"],  crop16, ph16, pw16)
        F_b16  = _crop_feature(fb_full["stride16"],  crop16, ph16, pw16)

        features_a = {"stride4": F_a4, "stride8": F_a8, "stride16": F_a16}
        features_b = {"stride4": F_b4, "stride8": F_b8, "stride16": F_b16}

        # ---- Coarse-to-fine homography regression --------------------------
        homo_out = self.homography_pyramid(features_a, features_b, ph, pw)
        H_patch  = homo_out["H_patch"]
        offset   = homo_out["offset"]
        H_full   = _patch_to_full(H_patch, crop_xy)

        # ---- Full-image warp of F_a at 1/4, output cropped at patch origin -
        # See ARCHITECTURE / planv2 P0: warp samples from the full-image
        # source feature so out-of-patch validity only fires when H sends
        # the source pixel outside the FULL image (not the patch).
        H_full_q = _rescale_homography(H_full, 0.25)
        F_a_warped = warp_by_homography(
            fa_full["stride4"], H_full_q,
            out_size=(ph4, pw4),
            out_origin_xy=crop4,
            padding_mode="zeros",
        )
        valid_mask = make_validity_mask(
            H_full_q,
            out_size=(ph4, pw4),
            in_size=fa_full["stride4"].shape[-2:],
            out_origin_xy=crop4,
        )

        # ---- Residual map at 1/4 -------------------------------------------
        r = (F_b4 - F_a_warped).abs().mean(dim=1, keepdim=True)

        # ---- Posterior + uncertainty (planv3 §3.4: single 1/4 scale, no corr) -
        if self.detach_head_inputs:
            # Backbone gradient does NOT flow into q-head / sigma-head losses.
            # Prevents the joint-stage feature-collapse feedback where the
            # backbone shrinks features to make r small (EM target trivially
            # inlier-everywhere). q and sigma still update from the current
            # features; they just can't update the features.
            post_in = torch.cat(
                [F_b4.detach(), F_a_warped.detach(), r.detach()], dim=1,
            )
        else:
            post_in = torch.cat([F_b4, F_a_warped, r], dim=1)
        q = self.posterior_head(post_in)
        log_sigma = self.uncertainty_head(post_in)
        q = q * valid_mask                                    # mask invalid pixels

        # ---- Cycle residual (patch coords, self-consistency only) ----------
        H_patch_q = _rescale_homography(H_patch, 0.25)
        H_patch_q_inv, cond_valid = safe_inverse_3x3(H_patch_q)
        F_a_recovered = warp_patch_by_homography(F_a_warped, H_patch_q_inv,
                                                 padding_mode="zeros")
        cycle_valid = valid_mask * make_validity_mask(
            H_patch_q_inv,
            out_size=F_a4.shape[-2:],
            in_size=F_a_warped.shape[-2:],
        )
        cycle_r = (F_a4 - F_a_recovered).abs().mean(dim=1, keepdim=True)

        # ---- Reliability features (10-dim phi) -----------------------------
        tau = 0.5
        q_above = (q > tau).float()
        q_mean = self._pool_stat(q)
        q_var  = self._pool_stat((q - q_mean.view(B, 1, 1, 1)) ** 2)
        q_area = self._pool_stat(q_above)
        r_mean    = self._pool_stat(r,       valid_mask)
        qr_mean   = self._pool_stat(q * r,   valid_mask)
        sigma_mean = self._pool_stat(log_sigma.exp(), valid_mask)
        # planv3 B7: replace the magic 10.0 outlier value with the batch
        # median residual so phi stays on a bounded scale.
        with torch.no_grad():
            r_med = float(r.detach().median())
        cycle_valid_pool = self._pool_stat(cycle_r, cycle_valid)
        cycle_mean = cycle_valid_pool * cond_valid + (1.0 - cond_valid) * r_med
        offset_norm = offset.abs().mean(dim=1)
        kappa_H = condition_number_3x3(H_full).to(dtype)
        log_kappa = torch.log(kappa_H.clamp(min=1.0))
        area_valid = valid_mask.mean(dim=(1, 2, 3))

        phi = torch.stack(
            [q_mean, q_var, q_area, r_mean, qr_mean, sigma_mean,
             cycle_mean, offset_norm, log_kappa, area_valid],
            dim=1,
        )

        # Detached: L_rel never reshapes geometry features.
        s = self.reliability_head(phi.detach())

        H_ab_inv, _ = safe_inverse_3x3(H_patch)

        return {
            # ---- primary geometry outputs ----
            "H_ab":         H_patch,
            "H_ab_inv":     H_ab_inv,
            "H_full":       H_full,
            "offset_ab":    offset,
            # ---- posterior / uncertainty / reliability ----
            "q_ab":         q,
            "log_sigma_ab": log_sigma,
            "reliability_score": s,
            "residual_map_ab":   r,
            # ---- features for triplet etc. ----
            "feature_a":    F_a4,
            "feature_b":    F_b4,
            # ---- internals ----
            "H_patch":         H_patch,
            "H_patch_q":       H_patch_q,
            "H_patch_q_inv":   H_patch_q_inv,
            "F_a_warped":      F_a_warped,
            "F_a_recovered":   F_a_recovered,
            "valid_mask":      valid_mask,
            "cycle_valid":     cycle_valid,
            "cond_valid":      cond_valid,
            "cycle_residual":  cycle_r,
            "phi":             phi,
            "kappa_H":         kappa_H,
            # ---- per-pyramid-level outputs (for diagnostics + multi-level supervision) ----
            "per_level_offset":     homo_out["per_level_offset"],
            "per_level_delta":      homo_out["per_level_delta"],
            "per_level_H":          homo_out["per_level_H"],
            # ---- back-compat aliases used throughout the training loop ----
            "offset":       offset,
            "q":            q,
            "log_sigma":    log_sigma,
            "s":            s,
            "residual":     r,
            "F_a4":         F_a4,
            "F_b4":         F_b4,
        }
