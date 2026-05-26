"""Top-level CDPC-v4 network.

Architecture (replaces planv3's Siamese-pyramid design):

    Geometry path (joint, v1-style):
        out_trunk   = JointGeometryTrunk(I_a_patch, I_b_patch)
                      -> H_init_patch  (joint 2-ch ResNet-34 + GAP + FC + DLT)

    Sub-pixel refinement (correlation soft-argmax at 1/4):
        out_refine  = SubpixelRefiner(I_a_patch, I_b_patch, H_init_patch)
                      -> H_patch       (residual-update ΔH composed onto H_init)
                      -> F_a_quarter, F_b_quarter, F_b_warped, confidence

    Image-coord conversion:
        H_full = T(+crop) @ H_patch @ T(-crop)

    CDPC heads (downstream-only, on DETACHED features):
        F_a_warped = warp(F_a_quarter, H_patch @ 0.25)   [computed under no_grad
                     for the heads' input feature so trunk gradient does not
                     flow through the q/sigma path]
        r          = mean_c |F_b_quarter - F_a_warped|.detach()
        q          = PosteriorHead([F_b_quarter.detach(), F_a_warped.detach(), r])
        log_sigma  = UncertaintyHead(same input)
        s          = ReliabilityHead(detach(phi))

The CDPC heads CANNOT collapse the geometry features (the entire postmortem
B5/Fix-A/B/C failure mode from v3): their input is detached, so the only
backward path into the trunks is through L_geom = triplet + photometric +
synth-supervision.

Output dict keeps all keys consumed by train.py / eval.py so the existing
training loop can be retargeted without API surgery.
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.joint_trunk import JointGeometryTrunk
from model.subpixel_refiner import SubpixelRefiner
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


# ---------------------------------------------------------------------------
# Top-level network
# ---------------------------------------------------------------------------

class CDPCNet(nn.Module):
    def __init__(
        self,
        patch_h: int = 315,
        patch_w: int = 560,
        # Joint trunk
        trunk_pretrained: bool = False,
        trunk_rho_init: float = 32.0,
        share_feature_channels: int = 1,
        # Sub-pixel refiner
        refiner_feat_channels: int = 64,
        refiner_radius: int = 6,
        refiner_pretrained: bool = False,
        refiner_corner_window: int = 7,
        refiner_temperature: float = 1.0,
        refiner_rho_max: float = 16.0,
        # DLT
        use_normalized_dlt: bool = True,
        # Pixel heads
        post_init_prob: float = 0.7,
        sigma_min: float = 0.5,
        # ------ Back-compat kwargs (silently accepted, possibly ignored) -----
        backbone_pretrained: bool = None,
        bb_quarter_channels: int = None,
        bb_eighth_channels: int = None,
        bb_sixteenth_channels: int = None,
        corr_radius: int = None,
        corr_out_channels: int = None,
        rho_per_level: tuple = None,
        homography_levels: int = None,
        detach_head_inputs: bool = None,
        homography_rho: float = None,
        log_sigma_min: float = None,
        log_sigma_max: float = None,
    ):
        super().__init__()
        self.patch_h = patch_h
        self.patch_w = patch_w

        # Honor a v3-style `backbone_pretrained` override if the caller passes it.
        if backbone_pretrained is not None and not trunk_pretrained:
            trunk_pretrained = bool(backbone_pretrained)

        # Geometry trunk (joint, v1-style ResNet-34, 2-channel input).
        self.trunk = JointGeometryTrunk(
            patch_h=patch_h, patch_w=patch_w,
            pretrained=trunk_pretrained,
            rho_init=trunk_rho_init,
            share_feature_channels=share_feature_channels,
            use_normalized_dlt=use_normalized_dlt,
        )

        # Sub-pixel correlation refiner.
        self.refiner = SubpixelRefiner(
            patch_h=patch_h, patch_w=patch_w,
            feat_channels=refiner_feat_channels,
            radius=refiner_radius,
            feat_pretrained=refiner_pretrained,
            corner_window=refiner_corner_window,
            confidence_temperature=refiner_temperature,
            rho_max=refiner_rho_max,
            use_normalized_dlt=use_normalized_dlt,
        )

        # CDPC heads. Input is [F_b_quarter, F_a_warped, residual] -- all DETACHED.
        post_in = 2 * refiner_feat_channels + 1
        self.posterior_head   = PosteriorHead(post_in, init_prob=post_init_prob)
        self.uncertainty_head = UncertaintyHead(post_in, sigma_min=sigma_min)

        # phi has 10 fields: q_mean, q_var, q_area, r_mean, qr_mean,
        # sigma_mean, cycle_mean, offset_norm, log_kappa, area_valid.
        self.reliability_head = ReliabilityHead(feat_dim=10)

    # ----- pooled-stat helper ----------------------------------------------

    @staticmethod
    def _pool_stat(x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        if mask is None:
            return x.mean(dim=(1, 2, 3))
        denom = mask.sum(dim=(1, 2, 3)).clamp(min=1.0)
        return (x * mask).sum(dim=(1, 2, 3)) / denom

    # ----- module-group accessors (used by train.py for staged freezing) ----

    @property
    def geometry_modules(self):
        """The geometry path that train.py freezes/unfreezes as a unit."""
        return [self.trunk, self.refiner]

    @property
    def cdpc_modules(self):
        """The downstream CDPC heads (excluding reliability)."""
        return [self.posterior_head, self.uncertainty_head]

    # ----- forward ---------------------------------------------------------

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
            I_a_full, I_b_full:   (B, 1, H_full, W_full) normalized grayscale.
                                  Used for image-space photometric loss only.
            I_a_patch, I_b_patch: (B, 1, patch_h, patch_w) cropped patches.
            crop_xy: (B, 2) [x, y] crop origin in full-image pixel coords.
        """
        B = I_a_patch.shape[0]
        device = I_a_patch.device
        dtype  = I_a_patch.dtype
        ph, pw = self.patch_h, self.patch_w

        # ===== 1. Joint trunk: coarse H_init from concat'd patches =========
        trunk_out = self.trunk(I_a_patch, I_b_patch)
        H_init_patch = trunk_out["H_patch"]
        offset_init  = trunk_out["offset"]
        sf_a         = trunk_out["sf_a"]
        sf_b         = trunk_out["sf_b"]

        # ===== 2. Sub-pixel refiner: residual ΔH at 1/4 ====================
        refine_out = self.refiner(I_a_patch, I_b_patch, H_init_patch)
        H_patch        = refine_out["H_refined_patch"]
        offset_resid   = refine_out["offset_residual"]
        F_a4           = refine_out["F_a_quarter"]
        F_b4           = refine_out["F_b_quarter"]
        F_b_warped_pre = refine_out["F_b_warped"]
        conf_map_q     = refine_out["confidence_map"]

        # Sum the two corner-offset contributions for diagnostics / fold loss.
        offset_total = offset_init + offset_resid

        # ===== 3. Image-coord H ============================================
        H_full = _patch_to_full(H_patch, crop_xy)

        # ===== 3b. ShareFeature warp at PATCH scale (drives triplet loss) ==
        # Gradient flows through H_patch (-> trunk + refiner) and through
        # ShareFeature(I_a_patch). This is the v1-style supervision signal
        # that drives geometry training in the `geom` stage.
        sf_a_warped = warp_patch_by_homography(sf_a, H_patch, padding_mode="zeros")
        valid_mask_patch = make_validity_mask(
            H_patch,
            out_size=(ph, pw),
            in_size=(ph, pw),
        )

        # ===== 4. Warp F_a at 1/4 for CDPC residual map (DETACHED) ========
        # The CDPC heads do not backprop into the refinement trunk: their
        # input is the warped-then-detached F_a_quarter.
        H_patch_q = _rescale_homography(H_patch, 0.25)
        ph4 = F_a4.shape[-2]
        pw4 = F_a4.shape[-1]
        with torch.no_grad():
            F_a_warped = warp_patch_by_homography(
                F_a4, H_patch_q, padding_mode="zeros",
            )
            valid_mask = make_validity_mask(
                H_patch_q,
                out_size=(ph4, pw4),
                in_size=(ph4, pw4),
            )
            F_b4_d         = F_b4.detach()
            F_a_warped_d   = F_a_warped.detach()
            r              = (F_b4_d - F_a_warped_d).abs().mean(dim=1, keepdim=True)

        post_in = torch.cat([F_b4_d, F_a_warped_d, r], dim=1)
        q = self.posterior_head(post_in)
        log_sigma = self.uncertainty_head(post_in)
        q = q * valid_mask

        # ===== 5. Cycle residual (patch coords, used only by phi/cycle loss)
        H_patch_q_inv, cond_valid = safe_inverse_3x3(H_patch_q)
        with torch.no_grad():
            F_a_recovered = warp_patch_by_homography(
                F_a_warped_d, H_patch_q_inv, padding_mode="zeros",
            )
            cycle_valid = valid_mask * make_validity_mask(
                H_patch_q_inv,
                out_size=F_a4.shape[-2:],
                in_size=F_a_warped_d.shape[-2:],
            )
            cycle_r = (F_a4.detach() - F_a_recovered).abs().mean(dim=1, keepdim=True)

        # ===== 6. Reliability features (phi, 10-d) =========================
        tau = 0.5
        q_above = (q > tau).float()
        q_mean = self._pool_stat(q)
        q_var  = self._pool_stat((q - q_mean.view(B, 1, 1, 1)) ** 2)
        q_area = self._pool_stat(q_above)
        r_mean     = self._pool_stat(r,           valid_mask)
        qr_mean    = self._pool_stat(q * r,       valid_mask)
        sigma_mean = self._pool_stat(log_sigma.exp(), valid_mask)
        with torch.no_grad():
            r_med = float(r.detach().median())
        cycle_valid_pool = self._pool_stat(cycle_r, cycle_valid)
        cycle_mean = cycle_valid_pool * cond_valid + (1.0 - cond_valid) * r_med
        offset_norm = offset_total.abs().mean(dim=1)
        kappa_H = condition_number_3x3(H_full).to(dtype)
        log_kappa = torch.log(kappa_H.clamp(min=1.0))
        area_valid = valid_mask.mean(dim=(1, 2, 3))

        phi = torch.stack(
            [q_mean, q_var, q_area, r_mean, qr_mean, sigma_mean,
             cycle_mean, offset_norm, log_kappa, area_valid],
            dim=1,
        )

        # Reliability backprop only into the reliability head.
        s = self.reliability_head(phi.detach())

        H_ab_inv, _ = safe_inverse_3x3(H_patch)

        # Single dict; back-compat keys point at the same tensors so train.py
        # and eval.py work without modification.
        return {
            # ---- primary geometry outputs ----
            "H_ab":         H_patch,
            "H_ab_inv":     H_ab_inv,
            "H_full":       H_full,
            "H_init_patch": H_init_patch,
            "offset_ab":    offset_total,
            "offset_init":  offset_init,
            "offset_residual": offset_resid,
            # ---- posterior / uncertainty / reliability ----
            "q_ab":         q,
            "log_sigma_ab": log_sigma,
            "reliability_score": s,
            "residual_map_ab":   r,
            # ---- ShareFeature outputs for triplet loss ----
            "sf_a":         sf_a,
            "sf_b":         sf_b,
            "sf_a_warped":  sf_a_warped,
            "valid_mask_patch": valid_mask_patch,
            # ---- features for legacy losses (cycle/triplet on F4) ----
            "feature_a":    F_a4,
            "feature_b":    F_b4,
            # ---- internals ----
            "H_patch":         H_patch,
            "H_patch_q":       H_patch_q,
            "H_patch_q_inv":   H_patch_q_inv,
            "F_a_warped":      F_a_warped,
            "F_b_warped":      F_b_warped_pre,
            "F_a_recovered":   F_a_recovered,
            "valid_mask":      valid_mask,
            "cycle_valid":     cycle_valid,
            "cond_valid":      cond_valid,
            "cycle_residual":  cycle_r,
            "phi":             phi,
            "kappa_H":         kappa_H,
            "confidence_map_quarter": conf_map_q,
            # ---- v3 back-compat aliases (so older code still works) ----
            "per_level_offset":     [offset_init, offset_resid],
            "per_level_delta":      [offset_init, offset_total],
            "per_level_H":          [H_init_patch, H_patch],
            "offset":       offset_total,
            "q":            q,
            "log_sigma":    log_sigma,
            "s":            s,
            "residual":     r,
            "F_a4":         F_a4,
            "F_b4":         F_b4,
        }
