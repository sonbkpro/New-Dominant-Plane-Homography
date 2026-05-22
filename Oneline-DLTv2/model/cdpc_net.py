"""Top-level Calibrated Dominant-Plane Consensus network -- v2 overhaul.

Key differences from the earlier v2:

(1) Full-image warping (planv2 P0). The per-image backbone runs on the FULL
    image, not the patch. The H regression branch still consumes the cropped
    pair (joint 2-channel backbone); but the per-pixel branch's residual is
    computed as

        F_a_warped(p_b) = sample( F_a_full, H_full^{-1} (p_b + crop_xy_b) )

    by giving warp_by_homography an out_origin_xy = crop_xy/scale. This
    exactly replicates v1's transform-then-crop pipeline without ever
    materializing the warped full image.

(2) Bounded corner offset via rho * tanh in HomographyHead (heads.py).
(3) Sigma via sigma_min + softplus(u), not clamp (heads.py).
(4) Coordinate channels on the H input so the 1/8 GAP-pooled feature stays
    aware of absolute spatial layout.
(5) phi (reliability features) extended with kappa(H) and A_v, and the
    reliability head receives phi.detach() so L_rel never distorts geometry.
(6) Optional Hartley-normalized DLT (utils/dlt_normalized.py).

The forward signature now requires both the full images and the cropped
patches; the dataset returns both.
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.backbone import MultiScaleBackbone
from model.correlation import LocalCorrelation
from model.heads import HomographyHead, PosteriorHead, UncertaintyHead, ReliabilityHead
from utils.dlt import DLT_solve
from utils.dlt_normalized import DLT_solve_normalized
from utils.warping import (
    warp_by_homography, warp_patch_by_homography, make_validity_mask,
)
from utils.inverse import safe_inverse_3x3, condition_number_3x3


def _rescale_homography(H: torch.Tensor, scale_in_to_out: float) -> torch.Tensor:
    """If H operates on pixel coordinates at scale `1`, return H' that
    operates on pixel coordinates at scale `scale_in_to_out` (e.g. 0.25)."""
    B = H.shape[0]
    device, dtype = H.device, H.dtype
    s = scale_in_to_out
    S = torch.tensor([[s, 0, 0], [0, s, 0], [0, 0, 1]], device=device, dtype=dtype)
    S_inv = torch.tensor([[1.0 / s, 0, 0], [0, 1.0 / s, 0], [0, 0, 1]], device=device, dtype=dtype)
    S_b = S.unsqueeze(0).expand(B, -1, -1)
    Si_b = S_inv.unsqueeze(0).expand(B, -1, -1)
    return torch.bmm(torch.bmm(S_b, H), Si_b)


def _patch_to_full(H_patch: torch.Tensor, crop_xy: torch.Tensor) -> torch.Tensor:
    """H_full = T(+crop) @ H_patch @ T(-crop). See train.py.patch_to_full_homography."""
    B = H_patch.shape[0]
    device, dtype = H_patch.device, H_patch.dtype
    T = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(B, 1, 1)
    T[:, 0, 2] = crop_xy[:, 0]
    T[:, 1, 2] = crop_xy[:, 1]
    T_inv = T.clone()
    T_inv[:, 0, 2] = -crop_xy[:, 0]
    T_inv[:, 1, 2] = -crop_xy[:, 1]
    return torch.bmm(torch.bmm(T, H_patch), T_inv)


def _add_coord_channels(x: torch.Tensor) -> torch.Tensor:
    """Append two channels (x/W in [-1,1], y/H in [-1,1]) to a (B,C,H,W) tensor.
    Lets a pooled regressor know absolute spatial location."""
    B, C, H, W = x.shape
    device, dtype = x.device, x.dtype
    ys = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype).view(1, 1, H, 1).expand(B, 1, H, W)
    xs = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype).view(1, 1, 1, W).expand(B, 1, H, W)
    return torch.cat([x, xs, ys], dim=1)


def _crop_feature(feat: torch.Tensor, origin_xy: torch.Tensor,
                  out_h: int, out_w: int) -> torch.Tensor:
    """Differentiable crop of a per-image feature map at a per-batch origin.
    Used to extract patch-located features from a feature map computed over
    the FULL image. origin_xy is in feature-map pixel coords."""
    B, C, H, W = feat.shape
    device, dtype = feat.device, feat.dtype
    # Build a destination grid that just translates by origin_xy.
    ys = torch.arange(out_h, device=device, dtype=dtype).view(1, out_h, 1).expand(B, out_h, out_w)
    xs = torch.arange(out_w, device=device, dtype=dtype).view(1, 1, out_w).expand(B, out_h, out_w)
    ox = origin_xy[:, 0].view(B, 1, 1).to(dtype)
    oy = origin_xy[:, 1].view(B, 1, 1).to(dtype)
    sx = xs + ox
    sy = ys + oy
    nx = 2.0 * sx / max(W - 1, 1) - 1.0
    ny = 2.0 * sy / max(H - 1, 1) - 1.0
    sample_grid = torch.stack([nx, ny], dim=-1)                    # (B, out_h, out_w, 2)
    return F.grid_sample(feat, sample_grid, mode="bilinear",
                         padding_mode="zeros", align_corners=True)


class CDPCNet(nn.Module):
    def __init__(
        self,
        patch_h: int = 315,
        patch_w: int = 560,
        backbone_pretrained: bool = True,
        corr_radius: int = 4,
        corr_out_channels: int = 32,
        bb_quarter_channels: int = 64,
        bb_eighth_channels: int = 128,
        homography_rho: float = 16.0,
        post_init_prob: float = 0.7,
        sigma_min: float = 0.05,
        use_normalized_dlt: bool = True,
        # Legacy kwargs ignored; retained so older configs still load.
        log_sigma_min: float = None,
        log_sigma_max: float = None,
    ):
        super().__init__()
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.use_normalized_dlt = use_normalized_dlt

        # Per-image backbone runs on FULL images.
        self.backbone = MultiScaleBackbone(
            pretrained=backbone_pretrained,
            in_channels=1,
            out_channels_quarter=bb_quarter_channels,
            out_channels_eighth=bb_eighth_channels,
        )

        # Joint 2-channel backbone on the cropped patch pair, for H regression.
        self.joint_backbone = MultiScaleBackbone(
            pretrained=backbone_pretrained,
            in_channels=2,
            out_channels_quarter=bb_quarter_channels,
            out_channels_eighth=bb_eighth_channels,
        )

        self.correlation = LocalCorrelation(
            radius=corr_radius, out_channels=corr_out_channels,
        )

        # +2 for the coord channels appended in forward().
        homo_in = bb_eighth_channels + corr_out_channels + 2
        self.homography_head = HomographyHead(homo_in, rho=homography_rho)

        # Per-pixel heads input: [F_b^{1/4}, F_a_warped^{1/4}, c^{1/4}, r].
        post_in = 2 * bb_quarter_channels + corr_out_channels + 1
        self.posterior_head = PosteriorHead(post_in, init_prob=post_init_prob)
        self.uncertainty_head = UncertaintyHead(post_in, sigma_min=sigma_min)

        # phi now has 10 dims: q_mean, q_var, q_area, r_mean, qr_mean,
        # sigma_mean, cycle_mean, offset_norm, kappa_H, area_valid.
        self.reliability_head = ReliabilityHead(feat_dim=10)

        h4p = torch.tensor(
            [0, 0,
             0, patch_h,
             patch_w, patch_h,
             patch_w, 0],
            dtype=torch.float32,
        )
        self.register_buffer("h4p_patch", h4p.unsqueeze(0), persistent=False)

    # ----- helpers ----------------------------------------------------------

    @staticmethod
    def _pool_stat(x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        if mask is None:
            return x.mean(dim=(1, 2, 3))
        denom = mask.sum(dim=(1, 2, 3)).clamp(min=1.0)
        return (x * mask).sum(dim=(1, 2, 3)) / denom

    def _solve_dlt(self, h4p_batch: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        if self.use_normalized_dlt:
            return DLT_solve_normalized(h4p_batch, offset)
        return DLT_solve(h4p_batch, offset)

    # ----- forward ----------------------------------------------------------

    def forward(
        self,
        I_a_full: torch.Tensor,
        I_b_full: torch.Tensor,
        I_a_patch: torch.Tensor,
        I_b_patch: torch.Tensor,
        crop_xy: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            I_a_full, I_b_full: (B, 1, H_full, W_full) normalized grayscale.
            I_a_patch, I_b_patch: (B, 1, patch_h, patch_w) cropped patches at crop_xy.
            crop_xy: (B, 2) [x, y] crop origin in full-image pixel coords.
        Returns: dict with all geometry / posterior / reliability outputs.
        """
        B = I_a_patch.shape[0]
        device = I_a_patch.device
        dtype = I_a_patch.dtype
        ph, pw = self.patch_h, self.patch_w
        H_full_img, W_full_img = I_a_full.shape[-2:]

        # --- per-image features on the FULL images --------------------------
        fa_full = self.backbone(I_a_full)
        fb_full = self.backbone(I_b_full)
        F_a_full4, F_a_full8 = fa_full["quarter"], fa_full["eighth"]
        F_b_full4, F_b_full8 = fb_full["quarter"], fb_full["eighth"]

        # --- joint features on patches -- predicts H in patch-local coords --
        joint_in = torch.cat([I_a_patch, I_b_patch], dim=1)         # (B, 2, ph, pw)
        f_joint = self.joint_backbone(joint_in)
        F_joint8_patch = f_joint["eighth"]

        # --- crop patch features from FULL feature maps for correlation
        #     and for the per-pixel heads. The correlation must be on the
        #     patch features (not the full image) because that is what is
        #     being matched.
        # Derive crop sizes from the joint backbone's actual 1/8 output --
        # ResNet's stem (stride 2) + maxpool (stride 2) + layer2 (stride 2)
        # rounds odd inputs UP (e.g. patch_h=315 -> 158 -> 79 -> 40), so
        # ph//8 = 39 would be off by 1 from the joint output (40). Using
        # the actual output dims keeps every per-image / correlation /
        # joint tensor at the same spatial resolution.
        crop_xy_8 = crop_xy / 8.0
        crop_xy_4 = crop_xy / 4.0
        ph_8 = F_joint8_patch.shape[-2]
        pw_8 = F_joint8_patch.shape[-1]
        ph_4 = ph_8 * 2
        pw_4 = pw_8 * 2

        F_a8_patch = _crop_feature(F_a_full8, crop_xy_8, ph_8, pw_8)
        F_b8_patch = _crop_feature(F_b_full8, crop_xy_8, ph_8, pw_8)
        F_b4_patch = _crop_feature(F_b_full4, crop_xy_4, ph_4, pw_4)
        F_a4_patch = _crop_feature(F_a_full4, crop_xy_4, ph_4, pw_4)

        # --- correlation on patch features at 1/8 ---------------------------
        c8 = self.correlation(F_a8_patch, F_b8_patch)

        # --- homography prediction ------------------------------------------
        homo_in = torch.cat([F_joint8_patch, c8], dim=1)
        homo_in = _add_coord_channels(homo_in)                      # +2 channels
        offset = self.homography_head(homo_in)                      # (B, 8) px, bounded

        h4p_batch = self.h4p_patch.expand(B, -1).to(device)
        H_patch = self._solve_dlt(h4p_batch, offset)                # (B, 3, 3) patch coords
        H_full = _patch_to_full(H_patch, crop_xy)                   # full-image pixel coords

        # --- FULL-IMAGE warp at quarter scale, output cropped to patch ------
        # This is v1's transform-then-crop: warp samples from anywhere in
        # F_a_full4, gated only by the *full image* extent rather than the
        # patch extent. Out-of-bounds occurs only when H sends the source
        # pixel outside the full image, not the patch.
        H_full_q = _rescale_homography(H_full, 0.25)
        F_a_warped = warp_by_homography(
            F_a_full4, H_full_q,
            out_size=(ph_4, pw_4),
            out_origin_xy=crop_xy_4,
            padding_mode="zeros",
        )
        valid_mask = make_validity_mask(
            H_full_q,
            out_size=(ph_4, pw_4),
            in_size=F_a_full4.shape[-2:],
            out_origin_xy=crop_xy_4,
        )

        # --- residual map ---------------------------------------------------
        r = (F_b4_patch - F_a_warped).abs().mean(dim=1, keepdim=True)

        # --- posterior + uncertainty ----------------------------------------
        c4 = F.interpolate(c8, size=F_b4_patch.shape[-2:],
                           mode="bilinear", align_corners=True)
        post_in = torch.cat([F_b4_patch, F_a_warped, c4, r], dim=1)
        q = self.posterior_head(post_in)
        log_sigma = self.uncertainty_head(post_in)

        q = q * valid_mask

        # --- cycle residual (in-patch, patch coords) ------------------------
        # Cycle uses H_patch (already in patch coords) and patch-only warp,
        # since the cycle is a self-consistency check, not the alignment loss.
        H_patch_q = _rescale_homography(H_patch, 0.25)
        H_patch_q_inv, cond_valid = safe_inverse_3x3(H_patch_q)
        F_a_recovered = warp_patch_by_homography(F_a_warped, H_patch_q_inv,
                                                 padding_mode="zeros")
        cycle_valid = valid_mask * make_validity_mask(
            H_patch_q_inv,
            out_size=F_a4_patch.shape[-2:],
            in_size=F_a_warped.shape[-2:],
        )
        cycle_r = (F_a4_patch - F_a_recovered).abs().mean(dim=1, keepdim=True)

        # --- reliability features (10-dim phi) ------------------------------
        tau = 0.5
        q_above = (q > tau).float()
        q_mean = self._pool_stat(q)
        q_var = self._pool_stat((q - q_mean.view(B, 1, 1, 1)) ** 2)
        q_area = self._pool_stat(q_above)
        r_mean = self._pool_stat(r, valid_mask)
        qr_mean = self._pool_stat(q * r, valid_mask)
        sigma_mean = self._pool_stat(log_sigma.exp(), valid_mask)
        cycle_mean = self._pool_stat(cycle_r, cycle_valid) * cond_valid \
            + (1.0 - cond_valid) * 10.0
        offset_norm = offset.abs().mean(dim=1)
        kappa_H = condition_number_3x3(H_full).to(dtype)
        # Squash kappa to log-scale so the regressor input stays bounded.
        log_kappa = torch.log(kappa_H.clamp(min=1.0))
        area_valid = valid_mask.mean(dim=(1, 2, 3))

        phi = torch.stack(
            [q_mean, q_var, q_area, r_mean, qr_mean, sigma_mean,
             cycle_mean, offset_norm, log_kappa, area_valid],
            dim=1,
        )

        # DETACHED reliability: L_rel never reshapes geometry features.
        s = self.reliability_head(phi.detach())

        # Full-scale closed-form inverse for downstream eval / cycle losses.
        H_ab_inv, _ = safe_inverse_3x3(H_patch)

        return {
            "H_ab":       H_patch,
            "H_ab_inv":   H_ab_inv,
            "H_full":     H_full,
            "offset_ab":  offset,
            "q_ab":       q,
            "log_sigma_ab": log_sigma,
            "reliability_score": s,
            "residual_map_ab":   r,
            "feature_a":  F_a4_patch,
            "feature_b":  F_b4_patch,
            # internals
            "H_patch":      H_patch,
            "H_patch_q":    H_patch_q,
            "H_patch_q_inv": H_patch_q_inv,
            "F_a_warped":   F_a_warped,
            "F_a_recovered": F_a_recovered,
            "valid_mask":   valid_mask,
            "cycle_valid":  cycle_valid,
            "cond_valid":   cond_valid,
            "cycle_residual": cycle_r,
            "phi":          phi,
            "kappa_H":      kappa_H,
            # back-compat aliases
            "offset":       offset,
            "q":            q,
            "log_sigma":    log_sigma,
            "s":            s,
            "residual":     r,
            "F_a4":         F_a4_patch,
            "F_b4":         F_b4_patch,
        }
