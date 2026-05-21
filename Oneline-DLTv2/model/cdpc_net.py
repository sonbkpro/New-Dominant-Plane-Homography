"""Top-level Calibrated Dominant-Plane Consensus network.

Pipeline (one direction, predicts H_ab only):

    (I_a_patch, I_b_patch)
        -> multi-scale features F_a^{1/4}, F_a^{1/8}, F_b^{1/4}, F_b^{1/8}
        -> local correlation c^{1/8}(F_a, F_b)
        -> HomographyHead([F_a^{1/8}, F_b^{1/8}, c^{1/8}]) -> 4-pt offset
        -> H_ab in patch-local pixel coordinates (via DLT_solve)

        -> warp F_a^{1/4} by H_ab (rescaled to 1/4) -> F_a_warped^{1/4}
        -> residual r = ||F_b^{1/4} - F_a_warped^{1/4}||_1 (channel-summed)
        -> upsample c^{1/8} -> c^{1/4}
        -> PosteriorHead([F_b^{1/4}, F_a_warped^{1/4}, c^{1/4}, r]) -> q
        -> UncertaintyHead([F_b^{1/4}, F_a_warped^{1/4}, c^{1/4}, r]) -> log sigma

        -> optional analytic-inverse cycle for the reliability vector
        -> ReliabilityHead(phi) -> s in (0, 1)

Outputs are returned as a dictionary so train.py can apply individual losses.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.backbone import MultiScaleBackbone
from model.correlation import LocalCorrelation
from model.heads import HomographyHead, PosteriorHead, UncertaintyHead, ReliabilityHead
from utils.dlt import DLT_solve
from utils.warping import warp_patch_by_homography, make_validity_mask
from utils.inverse import safe_inverse_3x3


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
        homography_rho: float = 32.0,
        post_init_prob: float = 0.7,
        log_sigma_min: float = -5.0,
        log_sigma_max: float = 5.0,
    ):
        super().__init__()
        self.patch_h = patch_h
        self.patch_w = patch_w

        self.backbone = MultiScaleBackbone(
            pretrained=backbone_pretrained,
            out_channels_quarter=bb_quarter_channels,
            out_channels_eighth=bb_eighth_channels,
        )
        self.correlation = LocalCorrelation(
            radius=corr_radius, out_channels=corr_out_channels,
        )

        homo_in = 2 * bb_eighth_channels + corr_out_channels
        self.homography_head = HomographyHead(homo_in, rho=homography_rho)

        # Posterior / uncertainty heads consume [F_b^{1/4}, F_a_warped^{1/4},
        # c^{1/4}_upsampled, residual] — c is part of the input per planv1.txt
        # §5.3, ensuring q and sigma depend on inter-image correspondence quality.
        post_in = 2 * bb_quarter_channels + corr_out_channels + 1
        self.posterior_head = PosteriorHead(post_in, init_prob=post_init_prob)
        self.uncertainty_head = UncertaintyHead(
            post_in,
            log_sigma_min=log_sigma_min,
            log_sigma_max=log_sigma_max,
        )

        # Reliability features: 8-dim vector defined in forward().
        self.reliability_head = ReliabilityHead(feat_dim=8)

        # Reusable patch-corner template for the DLT.
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
        """Returns (B,) mean over spatial dims. Optional mask of same shape."""
        if mask is None:
            return x.mean(dim=(1, 2, 3))
        denom = mask.sum(dim=(1, 2, 3)).clamp(min=1.0)
        return (x * mask).sum(dim=(1, 2, 3)) / denom

    # ----- forward ----------------------------------------------------------

    def forward(self, I_a: torch.Tensor, I_b: torch.Tensor) -> Dict[str, torch.Tensor]:
        """I_a, I_b: (B, 1, patch_h, patch_w) normalized grayscale patches."""
        B = I_a.shape[0]
        device = I_a.device

        # --- feature extraction ---------------------------------------------
        fa = self.backbone(I_a)
        fb = self.backbone(I_b)
        F_a4, F_a8 = fa["quarter"], fa["eighth"]
        F_b4, F_b8 = fb["quarter"], fb["eighth"]

        # --- correlation at 1/8 ---------------------------------------------
        c8 = self.correlation(F_a8, F_b8)                           # (B, Cc, H/8, W/8)

        # --- homography prediction ------------------------------------------
        homo_in = torch.cat([F_a8, F_b8, c8], dim=1)
        offset = self.homography_head(homo_in)                      # (B, 8) in pixels

        h4p_batch = self.h4p_patch.expand(B, -1).to(device)
        H_ab = DLT_solve(h4p_batch, offset)                         # (B, 3, 3) in patch coords

        # --- warp F_a^{1/4} onto F_b^{1/4} frame ----------------------------
        H_quarter = _rescale_homography(H_ab, 0.25)
        F_a_warped = warp_patch_by_homography(F_a4, H_quarter, padding_mode="zeros")
        valid_mask = make_validity_mask(
            H_quarter,
            out_size=F_b4.shape[-2:],
            in_size=F_a4.shape[-2:],
            out_origin_xy=(0, 0),
        )

        # --- residual map ---------------------------------------------------
        # Channel-MEAN (not sum) so per-pixel residual stays in ~[0, 5] regardless
        # of channel count. With channel-sum, residual ~ O(C) makes the triplet
        # margin m=1 effectively zero and lets L_align numerics misbehave.
        r = (F_b4 - F_a_warped).abs().mean(dim=1, keepdim=True)     # (B, 1, H/4, W/4)

        # --- posterior + uncertainty ----------------------------------------
        # Upsample correlation feature from 1/8 to 1/4 so it can feed the
        # per-pixel heads (planv1.txt §5.3).
        c4 = F.interpolate(c8, size=F_b4.shape[-2:],
                           mode="bilinear", align_corners=True)
        post_in = torch.cat([F_b4, F_a_warped, c4, r], dim=1)
        q = self.posterior_head(post_in)                            # (B, 1, H/4, W/4)
        log_sigma = self.uncertainty_head(post_in)                  # (B, 1, H/4, W/4)

        # Mask out invalid (out-of-bounds) pixels so the loss does not chase them.
        q = q * valid_mask
        # log_sigma stays as-is; loss multiplies by q so invalid regions vanish.

        # --- cycle residual for reliability features ------------------------
        H_inv_quarter, cond_valid = safe_inverse_3x3(H_quarter)
        F_a_recovered = warp_patch_by_homography(F_a_warped, H_inv_quarter, padding_mode="zeros")
        cycle_valid = valid_mask * make_validity_mask(
            H_inv_quarter,
            out_size=F_a4.shape[-2:],
            in_size=F_a_warped.shape[-2:],
        )
        cycle_r = (F_a4 - F_a_recovered).abs().mean(dim=1, keepdim=True)

        # --- reliability features -------------------------------------------
        tau = 0.5
        q_above = (q > tau).float()
        q_mean = self._pool_stat(q)
        q_var = self._pool_stat((q - q_mean.view(B, 1, 1, 1)) ** 2)
        q_area = self._pool_stat(q_above)
        r_mean = self._pool_stat(r, valid_mask)
        qr_mean = self._pool_stat(q * r, valid_mask)
        sigma_mean = self._pool_stat(log_sigma.exp(), valid_mask)
        cycle_mean = self._pool_stat(cycle_r, cycle_valid) * cond_valid \
            + (1.0 - cond_valid) * 10.0    # penalize ill-conditioned H
        offset_norm = offset.abs().mean(dim=1)

        phi = torch.stack(
            [q_mean, q_var, q_area, r_mean, qr_mean, sigma_mean, cycle_mean, offset_norm],
            dim=1,
        )
        s = self.reliability_head(phi)                              # (B,)

        # Full-scale closed-form inverse (planv1 §5.1: not a learned head).
        H_ab_inv, _ = safe_inverse_3x3(H_ab)

        return {
            # ----- plan §7.3 canonical names -----
            "H_ab": H_ab,
            "H_ab_inv": H_ab_inv,
            "offset_ab": offset,
            "q_ab": q,
            "log_sigma_ab": log_sigma,
            "reliability_score": s,
            "residual_map_ab": r,
            "feature_a": F_a4,
            "feature_b": F_b4,
            # ----- additional internals used by losses / training ----------
            "H_quarter": H_quarter,
            "H_inv_quarter": H_inv_quarter,
            "F_a_warped": F_a_warped,
            "F_a_recovered": F_a_recovered,
            "valid_mask": valid_mask,
            "cycle_valid": cycle_valid,
            "cond_valid": cond_valid,
            "cycle_residual": cycle_r,
            "phi": phi,
            # ----- back-compat aliases (no _ab suffix) ---------------------
            "offset": offset,
            "q": q,
            "log_sigma": log_sigma,
            "s": s,
            "residual": r,
            "F_a4": F_a4,
            "F_b4": F_b4,
        }
