"""Losses for PCGV training.

The default curriculum starts with the existing no-mask baseline objective:
alignment triplet + Feature Identity Loss.  PCGV-specific losses are implemented
here but default to zero-weight so they can be activated gradually.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from new_approach.losses import baseline_loss

from paper_new.geometry import torch_warp_points_h


def _scalar(value: torch.Tensor) -> float:
    return float(value.detach().mean().cpu())


def _zero_like_loss(out: Dict[str, torch.Tensor]) -> torch.Tensor:
    for value in out.values():
        if torch.is_tensor(value):
            return value.new_tensor(0.0)
    return torch.tensor(0.0)


def masked_triplet_align(fb: torch.Tensor, fa_warp: torch.Tensor, fa: torch.Tensor,
                         mask: torch.Tensor, margin: float = 1.0,
                         eps: float = 1e-6) -> torch.Tensor:
    """Mask-weighted version of the existing DeepHomography triplet."""
    loss = F.relu(margin + (fb - fa_warp).abs() - (fb - fa).abs())
    if mask.shape[-2:] != loss.shape[-2:]:
        mask = F.interpolate(mask, size=loss.shape[-2:], mode="bilinear", align_corners=True)
    return (loss * mask).sum() / (mask.sum() * loss.shape[1] + eps)


def masked_baseline_loss(out: Dict[str, torch.Tensor], lambda_align: float = 1.0,
                         lambda_fil: float = 0.5, margin: float = 1.0) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Existing alignment/FIL objective with PCGV masks on the triplet terms."""
    f1 = out["img1_patch_fea"]
    f2 = out["img2_patch_fea"]
    f2_warp = out["warp_img2_patch_fea"]
    f1_warp = out["warp_img1_patch_fea"]
    f_warp2 = out["img2_patch_warp_fea"]
    f_warp1 = out["img1_patch_warp_fea"]

    mask_f = out.get("pcgv_mask_f_patch", out.get("img1_patch_mask"))
    mask_b = out.get("pcgv_mask_b_patch", out.get("img2_patch_mask"))
    if mask_f is None:
        mask_f = torch.ones_like(f1[:, :1])
    if mask_b is None:
        mask_b = torch.ones_like(f2[:, :1])

    align_f = masked_triplet_align(f1, f2_warp, f2, mask_f, margin)
    align_b = masked_triplet_align(f2, f1_warp, f1, mask_b, margin)
    fil_f = (f2_warp - f_warp2).abs().mean()
    fil_b = (f1_warp - f_warp1).abs().mean()
    align = align_f + align_b
    fil = fil_f + fil_b
    total = lambda_align * align + lambda_fil * fil
    return total, {
        "total": _scalar(total),
        "align": _scalar(align),
        "align_f": _scalar(align_f),
        "align_b": _scalar(align_b),
        "fil": _scalar(fil),
        "fil_f": _scalar(fil_f),
        "fil_b": _scalar(fil_b),
    }


def weighted_reprojection_loss(out: Dict[str, torch.Tensor], robust: str = "charbonnier",
                               eps: float = 1e-6) -> torch.Tensor:
    losses = []
    for suffix in ("f", "b"):
        H = out.get(f"pcgv_H_feat_{suffix}")
        grid = out.get("pcgv_grid")
        matches = out.get(f"pcgv_matches_{suffix}")
        votes = out.get(f"pcgv_votes_{suffix}")
        if H is None or grid is None or matches is None or votes is None:
            continue
        pred = torch_warp_points_h(grid, H)
        residual = (pred - matches).square().sum(dim=-1, keepdim=True)
        if robust == "charbonnier":
            residual = (residual + eps).sqrt()
        else:
            residual = residual.sqrt()
        losses.append((votes * residual).sum() / (votes.sum() + eps))
    return sum(losses) / len(losses) if losses else _zero_like_loss(out)


def coarse_flow_anchor_loss(out: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Keep early PCGV flows close to the strong frozen coarse baseline."""
    losses = []
    for suffix in ("f", "b"):
        flow = out.get(f"flow_{suffix}_patch")
        coarse = out.get(f"coarse_flow_{suffix}_patch")
        if flow is None or coarse is None:
            continue
        losses.append(F.smooth_l1_loss(flow.float(), coarse.detach().float(), beta=1.0))
    return sum(losses) / len(losses) if losses else _zero_like_loss(out)


def cycle_consistency_loss(out: Dict[str, torch.Tensor]) -> torch.Tensor:
    H_f = out.get("H_f")
    H_b = out.get("H_b")
    if H_f is None or H_b is None:
        return _zero_like_loss(out)
    eye = torch.eye(3, device=H_f.device, dtype=H_f.dtype).unsqueeze(0)
    cycle_fb = H_b @ H_f
    cycle_bf = H_f @ H_b
    return (cycle_fb - eye).abs().mean() + (cycle_bf - eye).abs().mean()


def vote_pseudo_label_loss(out: Dict[str, torch.Tensor], tau: float = 2.0) -> torch.Tensor:
    losses = []
    for suffix in ("f", "b"):
        votes = out.get(f"pcgv_votes_{suffix}")
        residuals = out.get(f"pcgv_residuals_{suffix}")
        if votes is None or residuals is None:
            continue
        pseudo = torch.exp(-residuals.detach().square() / max(tau, 1e-6)).clamp(0.0, 1.0)
        v = votes.float().clamp(1e-6, 1.0 - 1e-6)
        p = pseudo.float()
        losses.append(-(p * v.log() + (1.0 - p) * (1.0 - v).log()).mean())
    return sum(losses) / len(losses) if losses else _zero_like_loss(out)


def mask_tv_loss(mask: torch.Tensor):
    if mask is None:
        return None
    dx = (mask[..., :, 1:] - mask[..., :, :-1]).abs().mean()
    dy = (mask[..., 1:, :] - mask[..., :-1, :]).abs().mean()
    return dx + dy


def area_prior_loss(out: Dict[str, torch.Tensor], target_area: float = 0.35) -> torch.Tensor:
    masks = [out.get("pcgv_mask_f_patch"), out.get("pcgv_mask_b_patch")]
    masks = [m for m in masks if m is not None]
    if not masks:
        return _zero_like_loss(out)
    return sum((m.mean() - target_area).square() for m in masks) / len(masks)


def entropy_regularization(out: Dict[str, torch.Tensor], eps: float = 1e-6) -> torch.Tensor:
    votes = [out.get("pcgv_votes_f"), out.get("pcgv_votes_b")]
    votes = [v for v in votes if v is not None]
    if not votes:
        return _zero_like_loss(out)
    ent = []
    for v in votes:
        v = v.clamp(eps, 1.0 - eps)
        ent.append(-(v * v.log() + (1.0 - v) * (1.0 - v).log()).mean())
    return sum(ent) / len(ent)


def conditioning_loss(out: Dict[str, torch.Tensor]) -> torch.Tensor:
    conds = [out.get("solver_cond_f"), out.get("solver_cond_b")]
    conds = [c for c in conds if torch.is_tensor(c)]
    if not conds:
        return _zero_like_loss(out)
    vals = [torch.nan_to_num(c.float(), nan=1e6, posinf=1e6).clamp_min(1.0).log().mean()
            for c in conds]
    return sum(vals) / len(vals)


def uncertainty_nll_loss(out: Dict[str, torch.Tensor], eps: float = 1e-6) -> torch.Tensor:
    losses = []
    for suffix in ("f", "b"):
        residuals = out.get(f"pcgv_residuals_{suffix}")
        sigma = out.get(f"pcgv_uncertainty_tokens_{suffix}")
        if residuals is None or sigma is None:
            continue
        var = sigma.square().clamp_min(eps)
        losses.append(0.5 * (residuals.detach().square() / var + var.log()).mean())
    return sum(losses) / len(losses) if losses else _zero_like_loss(out)


def pcgv_loss(out: Dict[str, torch.Tensor],
              lambda_align: float = 1.0,
              lambda_fil: float = 0.5,
              lambda_coarse_flow: float = 0.0,
              lambda_reproj: float = 0.0,
              lambda_cycle: float = 0.0,
              lambda_vote: float = 0.0,
              lambda_tv: float = 0.0,
              lambda_area: float = 0.0,
              lambda_entropy: float = 0.0,
              lambda_cond: float = 0.0,
              lambda_uncertainty: float = 0.0,
              target_area: float = 0.35,
              vote_tau: float = 2.0,
              margin: float = 1.0,
              use_masked_align: bool = False) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Total PCGV loss.

    By default this is exactly the existing baseline objective applied to the
    model's current baseline-compatible outputs.  Extra terms activate only when
    their lambdas are non-zero.
    """
    if use_masked_align:
        total, logs = masked_baseline_loss(out, lambda_align=lambda_align,
                                           lambda_fil=lambda_fil, margin=margin)
    else:
        total, logs = baseline_loss(out, lambda_align=lambda_align,
                                    lambda_fil=lambda_fil, margin=margin)

    zero = _zero_like_loss(out)
    coarse_flow = coarse_flow_anchor_loss(out) if lambda_coarse_flow else zero
    reproj = weighted_reprojection_loss(out) if lambda_reproj else zero
    cycle = cycle_consistency_loss(out) if lambda_cycle else zero
    vote = vote_pseudo_label_loss(out, tau=vote_tau) if lambda_vote else zero
    if lambda_tv:
        tv_terms = [v for v in (
            mask_tv_loss(out.get("pcgv_mask_f_patch")),
            mask_tv_loss(out.get("pcgv_mask_b_patch")),
        ) if torch.is_tensor(v)]
        tv = sum(tv_terms) if tv_terms else zero
    else:
        tv = zero
    area = area_prior_loss(out, target_area=target_area) if lambda_area else zero
    entropy = entropy_regularization(out) if lambda_entropy else zero
    cond = conditioning_loss(out) if lambda_cond else zero
    uncertainty = uncertainty_nll_loss(out) if lambda_uncertainty else zero

    total = (
        total
        + lambda_coarse_flow * coarse_flow
        + lambda_reproj * reproj
        + lambda_cycle * cycle
        + lambda_vote * vote
        + lambda_tv * tv
        + lambda_area * area
        + lambda_entropy * entropy
        + lambda_cond * cond
        + lambda_uncertainty * uncertainty
    )

    logs.update({
        "total": _scalar(total),
        "coarse_flow": _scalar(coarse_flow),
        "reproj": _scalar(reproj),
        "cycle": _scalar(cycle),
        "vote": _scalar(vote),
        "tv": _scalar(tv),
        "area": _scalar(area),
        "entropy": _scalar(entropy),
        "cond": _scalar(cond),
        "uncertainty": _scalar(uncertainty),
    })
    for key in ("pcgv_votes_f", "pcgv_votes_b", "pcgv_residuals_f", "pcgv_residuals_b"):
        value = out.get(key)
        if torch.is_tensor(value):
            logs[key.replace("pcgv_", "mean_")] = _scalar(value)
    for key in ("solver_cond_f", "solver_cond_b"):
        value = out.get(key)
        if torch.is_tensor(value):
            logs[key] = _scalar(torch.nan_to_num(value.float(), nan=0.0, posinf=0.0))
    return total, logs
