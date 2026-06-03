"""PCGV homography model built beside the stable no-mask baseline."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Dict, Optional

import torch
import torch.nn as nn

from new_approach.model_baseline import CROP_H, CROP_W, FULL_H, FULL_W, build_baseline, make_params
from new_approach.modules.featureHomo import FeatureExtractor, feature_extractor

from paper_new.geometry import (
    torch_convert_homography_scale,
    torch_flow_to_homography_from_corners,
    torch_homography_to_flow,
    torch_normalize_homography,
    torch_warp_full_with_patch_flow,
    torch_warp_tensor_with_flow,
)
from paper_new.modules.pcgv import PCGVModule
from paper_new.modules.pcgv_blocks import MaskUpsampler


def _getattr(params: SimpleNamespace, name: str, default):
    return getattr(params, name, default)


def _copy_matching_state(dst: nn.Module, src: nn.Module) -> Dict[str, int]:
    """Copy state entries whose names and shapes match exactly."""
    dst_state = dst.state_dict()
    src_state = src.state_dict()
    load_state = {}
    skipped = 0
    for key, dst_value in dst_state.items():
        src_value = src_state.get(key)
        if src_value is None or tuple(src_value.shape) != tuple(dst_value.shape):
            skipped += 1
            continue
        load_state[key] = src_value.detach().to(device=dst_value.device, dtype=dst_value.dtype)
    dst_state.update(load_state)
    dst.load_state_dict(dst_state, strict=True)
    return {"copied": len(load_state), "skipped": skipped}


class PCGVFeatureBackbone(nn.Module):
    """Feature wrapper that reuses the existing ``featureHomo`` components."""

    def __init__(self,
                 shallow_out: int = 1,
                 pyramid_embed_dim: int = 24,
                 pyramid_layers: int = 3,
                 pcgv_dim: int = 64,
                 pcgv_level: int = 0,
                 activation=nn.GELU):
        super().__init__()
        self.pcgv_level = pcgv_level
        self.shallow = feature_extractor(2, shallow_out)
        self.pyramid = FeatureExtractor(
            embed_dim=pyramid_embed_dim,
            num_layers=pyramid_layers,
            activation=activation,
        )
        # FeatureExtractor returns the pyramid reversed: coarsest first.
        channels = [pyramid_embed_dim * (2 ** (pyramid_layers - 1 - i))
                    for i in range(pyramid_layers)]
        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, pcgv_dim, kernel_size=1, bias=False),
                nn.GroupNorm(num_groups=8 if pcgv_dim % 8 == 0 else 1, num_channels=pcgv_dim),
                nn.GELU(),
            )
            for ch in channels
        ])

    def forward_one(self, img: torch.Tensor) -> Dict[str, torch.Tensor]:
        shallow = self.shallow(img)
        pyramid = self.pyramid(shallow)
        pcgv_pyramid = [proj(feat) for proj, feat in zip(self.proj, pyramid)]
        return {
            "shallow": shallow,
            "pyramid": pyramid,
            "pcgv_pyramid": pcgv_pyramid,
            "pcgv": pcgv_pyramid[self.pcgv_level],
        }

    def forward_shallow(self, img: torch.Tensor) -> torch.Tensor:
        return self.shallow(img)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor):
        return self.forward_one(img1), self.forward_one(img2)

    def init_from_coarse(self, coarse: nn.Module) -> Dict[str, int]:
        report = {}
        if hasattr(coarse, "fea_extra"):
            copied = _copy_matching_state(self.shallow, coarse.fea_extra)
            report["shallow_copied"] = copied["copied"]
            report["shallow_skipped"] = copied["skipped"]
        else:
            report["shallow_copied"] = 0
            report["shallow_skipped"] = len(self.shallow.state_dict())

        pyramid_src = getattr(getattr(coarse, "h_net", None), "feature_pyramid_extractor", None)
        if pyramid_src is not None:
            copied = _copy_matching_state(self.pyramid, pyramid_src)
            report["pyramid_copied"] = copied["copied"]
            report["pyramid_skipped"] = copied["skipped"]
        else:
            report["pyramid_copied"] = 0
            report["pyramid_skipped"] = len(self.pyramid.state_dict())
        return report


def make_pcgv_params(crop_h: int = CROP_H, crop_w: int = CROP_W, **overrides):
    """Create a params namespace for the PCGV model."""
    base = make_params(crop_h=crop_h, crop_w=crop_w, pretrain_phase=True)
    pcgv_defaults = dict(
        pcgv_enabled=True,
        pcgv_feat_dim=64,
        pcgv_hidden_dim=128,
        pcgv_num_iters=4,
        pcgv_radius=4,
        pcgv_temperature=0.1,
        pcgv_min_vote=1e-4,
        pcgv_use_transformer=False,
        pcgv_use_plane_token=True,
        pcgv_use_uncertainty=True,
        pcgv_use_leverage=True,
        pcgv_damped_update=True,
        pcgv_update_alpha=0.7,
        pcgv_refine_blend_init=0.05,
        pcgv_learn_refine_blend=True,
        pcgv_freeze_coarse=False,
        pcgv_init_mode="coarse_flow_corners",
        pcgv_override_baseline_keys=True,
        pcgv_pyramid_embed_dim=24,
        pcgv_pyramid_layers=3,
        pcgv_level=0,
    )
    pcgv_defaults.update(overrides)
    for key, value in pcgv_defaults.items():
        setattr(base, key, value)
    return base


def build_pcgv(params: Optional[SimpleNamespace] = None):
    if params is None:
        params = make_pcgv_params()
    return PCGVHomoNet(params)


class PCGVHomoNet(nn.Module):
    """Coarse HomoGAN initializer plus PCGV refinement."""

    def __init__(self, params: SimpleNamespace):
        super().__init__()
        self.params = params
        self.coarse = build_baseline(params)
        self.features = PCGVFeatureBackbone(
            pyramid_embed_dim=_getattr(params, "pcgv_pyramid_embed_dim", 24),
            pyramid_layers=_getattr(params, "pcgv_pyramid_layers", 3),
            pcgv_dim=_getattr(params, "pcgv_feat_dim", 64),
            pcgv_level=_getattr(params, "pcgv_level", 0),
        )
        self.pcgv = PCGVModule(
            feat_dim=_getattr(params, "pcgv_feat_dim", 64),
            hidden_dim=_getattr(params, "pcgv_hidden_dim", 128),
            num_iters=_getattr(params, "pcgv_num_iters", 4),
            radius=_getattr(params, "pcgv_radius", 4),
            temperature=_getattr(params, "pcgv_temperature", 0.1),
            min_vote=_getattr(params, "pcgv_min_vote", 1e-4),
            use_transformer=_getattr(params, "pcgv_use_transformer", False),
            use_plane_token=_getattr(params, "pcgv_use_plane_token", True),
            use_uncertainty=_getattr(params, "pcgv_use_uncertainty", True),
            use_leverage=_getattr(params, "pcgv_use_leverage", True),
            damped_update=_getattr(params, "pcgv_damped_update", True),
            update_alpha=_getattr(params, "pcgv_update_alpha", 0.7),
            refine_blend_init=_getattr(params, "pcgv_refine_blend_init", 0.05),
            learn_refine_blend=_getattr(params, "pcgv_learn_refine_blend", True),
        )
        self.mask_upsampler = MaskUpsampler()
        if _getattr(params, "pcgv_freeze_coarse", False):
            self.freeze_coarse()

    def freeze_coarse(self):
        for param in self.coarse.parameters():
            param.requires_grad = False

    def unfreeze_coarse(self):
        for param in self.coarse.parameters():
            param.requires_grad = True

    def init_pcgv_features_from_coarse(self) -> Dict[str, int]:
        return self.features.init_from_coarse(self.coarse)

    def set_refine_blend(self, blend: float) -> float:
        return self.pcgv.set_refine_blend(blend)

    def get_refine_blend(self) -> float:
        return self.pcgv.get_refine_blend()

    def freeze_pcgv_shallow(self):
        for param in self.features.shallow.parameters():
            param.requires_grad = False
        self.features.shallow.eval()

    def freeze_pcgv_pyramid(self):
        for param in self.features.pyramid.parameters():
            param.requires_grad = False
        self.features.pyramid.eval()

    def unfreeze_pcgv_shallow(self):
        for param in self.features.shallow.parameters():
            param.requires_grad = True
        self.features.shallow.train(self.training)

    def unfreeze_pcgv_pyramid(self):
        for param in self.features.pyramid.parameters():
            param.requires_grad = True
        self.features.pyramid.train(self.training)

    def _identity_h(self, batch: int, device, dtype) -> torch.Tensor:
        return torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(batch, 1, 1)

    def _initial_h(self, coarse_out: Dict[str, torch.Tensor], key: str,
                   batch: int, device, dtype) -> torch.Tensor:
        mode = _getattr(self.params, "pcgv_init_mode", "coarse_flow_corners")
        if mode == "identity" or key not in coarse_out:
            return self._identity_h(batch, device, dtype)
        if mode != "coarse_flow_corners":
            raise ValueError(f"unknown pcgv_init_mode: {mode}")
        H = torch_flow_to_homography_from_corners(coarse_out[key].detach())
        finite = torch.isfinite(H).flatten(1).all(dim=1).view(batch, 1, 1)
        return torch.where(finite, torch_normalize_homography(H), self._identity_h(batch, device, dtype))

    def _pcgv_outputs(self, img1_patch: torch.Tensor, img2_patch: torch.Tensor,
                      coarse_out: Dict[str, torch.Tensor],
                      img1_full: Optional[torch.Tensor] = None,
                      img2_full: Optional[torch.Tensor] = None,
                      start: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        batch, _, h_patch, w_patch = img1_patch.shape
        feat1, feat2 = self.features(img1_patch, img2_patch)
        pcgv1 = feat1["pcgv"]
        pcgv2 = feat2["pcgv"]
        _, _, h_feat, w_feat = pcgv1.shape

        H0_f_patch = self._initial_h(coarse_out, "flow_f_patch", batch, img1_patch.device, img1_patch.dtype)
        H0_b_patch = self._initial_h(coarse_out, "flow_b_patch", batch, img1_patch.device, img1_patch.dtype)
        H0_f_feat = torch_convert_homography_scale(H0_f_patch, (h_patch, w_patch), (h_feat, w_feat))
        H0_b_feat = torch_convert_homography_scale(H0_b_patch, (h_patch, w_patch), (h_feat, w_feat))

        pcgv_f = self.pcgv(pcgv1, pcgv2, H0_f_feat)
        pcgv_b = self.pcgv(pcgv2, pcgv1, H0_b_feat)

        H_f = torch_convert_homography_scale(pcgv_f["H"], (h_feat, w_feat), (h_patch, w_patch))
        H_b = torch_convert_homography_scale(pcgv_b["H"], (h_feat, w_feat), (h_patch, w_patch))
        flow_f_patch = torch_homography_to_flow(H_f, h_patch, w_patch)
        flow_b_patch = torch_homography_to_flow(H_b, h_patch, w_patch)

        img1_fea = feat1["shallow"]
        img2_fea = feat2["shallow"]
        if img1_full is not None and img2_full is not None:
            img1_full_fea = self.features.forward_shallow(img1_full)
            img2_full_fea = self.features.forward_shallow(img2_full)
            warp_img2_patch = torch_warp_full_with_patch_flow(img2_full, flow_f_patch, start)
            warp_img1_patch = torch_warp_full_with_patch_flow(img1_full, flow_b_patch, start)
            warp_img2_patch_fea = torch_warp_full_with_patch_flow(img2_full_fea, flow_f_patch, start)
            warp_img1_patch_fea = torch_warp_full_with_patch_flow(img1_full_fea, flow_b_patch, start)
        else:
            warp_img2_patch = torch_warp_tensor_with_flow(img2_patch, flow_f_patch)
            warp_img1_patch = torch_warp_tensor_with_flow(img1_patch, flow_b_patch)
            warp_img2_patch_fea = torch_warp_tensor_with_flow(img2_fea, flow_f_patch)
            warp_img1_patch_fea = torch_warp_tensor_with_flow(img1_fea, flow_b_patch)
        img2_patch_warp_fea = self.features.forward_shallow(warp_img2_patch)
        img1_patch_warp_fea = self.features.forward_shallow(warp_img1_patch)

        mask_f_patch = self.mask_upsampler(pcgv_f["mask"], size=(h_patch, w_patch))
        mask_b_patch = self.mask_upsampler(pcgv_b["mask"], size=(h_patch, w_patch))
        warp_img2_patch_mask = torch_warp_tensor_with_flow(mask_b_patch, flow_f_patch)
        warp_img1_patch_mask = torch_warp_tensor_with_flow(mask_f_patch, flow_b_patch)

        return {
            "H_f": H_f,
            "H_b": H_b,
            "H0_f": H0_f_patch,
            "H0_b": H0_b_patch,
            "pcgv_H_feat_f": pcgv_f["H"],
            "pcgv_H_feat_b": pcgv_b["H"],
            "pcgv_H_raw_feat_f": pcgv_f["H_refined"],
            "pcgv_H_raw_feat_b": pcgv_b["H_refined"],
            "pcgv_H_start_feat_f": pcgv_f["H_start"],
            "pcgv_H_start_feat_b": pcgv_b["H_start"],
            "flow_f": flow_f_patch,
            "flow_b": flow_b_patch,
            "flow_f_patch": flow_f_patch,
            "flow_b_patch": flow_b_patch,
            "img1_patch_fea": img1_fea,
            "img2_patch_fea": img2_fea,
            "warp_img1_patch_fea": warp_img1_patch_fea,
            "warp_img2_patch_fea": warp_img2_patch_fea,
            "img1_patch_warp_fea": img1_patch_warp_fea,
            "img2_patch_warp_fea": img2_patch_warp_fea,
            "warp_img1_patch": warp_img1_patch,
            "warp_img2_patch": warp_img2_patch,
            "img1_patch_mask": mask_f_patch,
            "img2_patch_mask": mask_b_patch,
            "warp_img1_patch_mask": warp_img1_patch_mask,
            "warp_img2_patch_mask": warp_img2_patch_mask,
            "pcgv_mask_f": pcgv_f["mask"],
            "pcgv_mask_b": pcgv_b["mask"],
            "pcgv_mask_f_patch": mask_f_patch,
            "pcgv_mask_b_patch": mask_b_patch,
            "pcgv_uncertainty_f": pcgv_f["uncertainty"],
            "pcgv_uncertainty_b": pcgv_b["uncertainty"],
            "pcgv_uncertainty_tokens_f": pcgv_f["uncertainty_tokens"],
            "pcgv_uncertainty_tokens_b": pcgv_b["uncertainty_tokens"],
            "pcgv_corr_stats_f": pcgv_f["stats"],
            "pcgv_corr_stats_b": pcgv_b["stats"],
            "pcgv_refine_blend_f": pcgv_f["refine_blend"],
            "pcgv_refine_blend_b": pcgv_b["refine_blend"],
            "pcgv_residuals_f": pcgv_f["residuals"],
            "pcgv_residuals_b": pcgv_b["residuals"],
            "pcgv_votes_f": pcgv_f["votes"],
            "pcgv_votes_b": pcgv_b["votes"],
            "pcgv_matches_f": pcgv_f["matches"],
            "pcgv_matches_b": pcgv_b["matches"],
            "pcgv_grid": pcgv_f["grid"],
            "solver_cond_f": pcgv_f["solver_cond"],
            "solver_cond_b": pcgv_b["solver_cond"],
            "pcgv_warp_img1_patch": warp_img1_patch,
            "pcgv_warp_img2_patch": warp_img2_patch,
            "pcgv_warp_img1_patch_fea": warp_img1_patch_fea,
            "pcgv_warp_img2_patch_fea": warp_img2_patch_fea,
        }

    def forward(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        freeze_coarse = _getattr(self.params, "pcgv_freeze_coarse", False)
        if freeze_coarse:
            self.coarse.eval()
            with torch.no_grad():
                coarse_out = self.coarse(data_batch)
        else:
            coarse_out = self.coarse(data_batch)

        if not _getattr(self.params, "pcgv_enabled", True):
            return coarse_out

        img1_patch = data_batch["imgs_gray_patch"][:, :1]
        img2_patch = data_batch["imgs_gray_patch"][:, 1:]
        img1_full = data_batch.get("imgs_gray_full", None)
        img2_full = None
        if img1_full is not None:
            img1_full, img2_full = img1_full[:, :1], img1_full[:, 1:]
        pcgv_out = self._pcgv_outputs(
            img1_patch,
            img2_patch,
            coarse_out,
            img1_full=img1_full,
            img2_full=img2_full,
            start=data_batch.get("start"),
        )

        out = dict(coarse_out)
        out.update({
            "coarse_flow_f_patch": coarse_out["flow_f_patch"],
            "coarse_flow_b_patch": coarse_out["flow_b_patch"],
            "coarse_warp_img1_patch_fea": coarse_out["warp_img1_patch_fea"],
            "coarse_warp_img2_patch_fea": coarse_out["warp_img2_patch_fea"],
        })
        if _getattr(self.params, "pcgv_override_baseline_keys", True):
            out.update(pcgv_out)
        else:
            for key, value in pcgv_out.items():
                out[f"pcgv_{key}" if not key.startswith("pcgv_") else key] = value
            out.update({
                "H_f": pcgv_out["H_f"],
                "H_b": pcgv_out["H_b"],
                "pcgv_flow_f_patch": pcgv_out["flow_f_patch"],
                "pcgv_flow_b_patch": pcgv_out["flow_b_patch"],
            })
        return out
