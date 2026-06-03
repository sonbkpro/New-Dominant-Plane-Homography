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


def coverage_loss(mask: torch.Tensor, mode: str = "hinge",
                  floor: float = 0.35, eps: float = 1e-6) -> torch.Tensor:
    """Prevent mask collapse without pinning the mask to a fixed area."""
    mean_mask = mask.float().mean().clamp_min(eps)
    if mode == "hinge":
        return F.relu(mask.new_tensor(float(floor)) - mean_mask)
    if mode == "log":
        return -mean_mask.log()
    raise ValueError(f"unknown coverage mode: {mode}")


def edge_aware_tv(mask: torch.Tensor, guide_img: torch.Tensor | None) -> torch.Tensor:
    """Total variation weighted down across strong image edges."""
    if guide_img is None:
        return mask_tv_loss(mask)
    if guide_img.shape[-2:] != mask.shape[-2:]:
        guide_img = F.interpolate(guide_img, size=mask.shape[-2:], mode="bilinear", align_corners=True)
    guide_img = guide_img.float()
    mask_f = mask.float()
    wx = torch.exp(-(guide_img[..., :, 1:] - guide_img[..., :, :-1]).abs().mean(dim=1, keepdim=True))
    wy = torch.exp(-(guide_img[..., 1:, :] - guide_img[..., :-1, :]).abs().mean(dim=1, keepdim=True))
    dx = ((mask_f[..., :, 1:] - mask_f[..., :, :-1]).abs() * wx).mean()
    dy = ((mask_f[..., 1:, :] - mask_f[..., :-1, :]).abs() * wy).mean()
    return dx + dy


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


def _mean_terms(terms, zero: torch.Tensor) -> torch.Tensor:
    if not terms:
        return zero
    total = zero
    for term in terms:
        total = total + term
    return total / len(terms)


def _last_corr_stat(out: Dict[str, torch.Tensor], suffix: str, key: str):
    stats = out.get(f"pcgv_corr_stats_{suffix}")
    if not stats:
        return None
    value = stats[-1].get(key)
    return value if torch.is_tensor(value) else None


def _quantile_score(value: torch.Tensor, low_q: float, high_q: float,
                    high_is_good: bool, eps: float = 1e-6) -> torch.Tensor:
    x = value.detach().float()
    lo = torch.quantile(x, low_q, dim=1, keepdim=True)
    hi = torch.quantile(x, high_q, dim=1, keepdim=True)
    denom = (hi - lo).abs().clamp_min(eps)
    score = (x - lo) / denom if high_is_good else (hi - x) / denom
    return score.clamp(0.0, 1.0)


def _sharpen_probability(prob: torch.Tensor, gamma: float,
                         eps: float = 1e-6) -> torch.Tensor:
    if abs(gamma - 1.0) < eps:
        return prob
    p = prob.clamp(eps, 1.0 - eps)
    pos = p.pow(gamma)
    neg = (1.0 - p).pow(gamma)
    return pos / (pos + neg + eps)


def _masked_mean(value: torch.Tensor, mask: torch.Tensor,
                 fallback: torch.Tensor) -> torch.Tensor:
    mask_f = mask.float()
    total = mask_f.sum()
    if float(total.detach().cpu()) < 1.0:
        return fallback
    return (value.float() * mask_f).sum() / total.clamp_min(1.0)


def _vote_basic_diagnostics(votes: torch.Tensor, residuals: torch.Tensor,
                            low_thresh: float, high_thresh: float,
                            prefix: str) -> Dict[str, float]:
    votes_f = votes.detach().float()
    residuals_f = residuals.detach().float()
    high_mask = votes_f >= high_thresh
    low_mask = votes_f <= low_thresh
    fallback = residuals_f.mean()
    return {
        f"vote_std_{prefix}": _scalar(votes_f.std(unbiased=False)),
        f"vote_hi_{prefix}": _scalar(high_mask.float().mean()),
        f"vote_lo_{prefix}": _scalar(low_mask.float().mean()),
        f"vote_inlier_residual_{prefix}": _scalar(_masked_mean(residuals_f, high_mask, fallback)),
        f"vote_outlier_residual_{prefix}": _scalar(_masked_mean(residuals_f, low_mask, fallback)),
    }


def weighted_reprojection_loss(out: Dict[str, torch.Tensor], robust: str = "charbonnier",
                               use_raw_h: bool = True, eps: float = 1e-6) -> torch.Tensor:
    losses = []
    for suffix in ("f", "b"):
        H = out.get(f"pcgv_H_raw_feat_{suffix}") if use_raw_h else None
        if H is None:
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


def robust_vote_pseudo_label_loss(out: Dict[str, torch.Tensor],
                                  residual_low_q: float = 0.25,
                                  residual_high_q: float = 0.75,
                                  corr_weight: float = 0.35,
                                  gamma: float = 1.5,
                                  min_confidence_weight: float = 0.25,
                                  low_thresh: float = 0.30,
                                  high_thresh: float = 0.70,
                                  eps: float = 1e-6) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Train votes from robust residual ranks and local-correlation confidence.

    Residuals define geometric consistency relative to the current local match.
    Correlation peak/gap/entropy modulate that residual target so flat or
    ambiguous matches are less likely to become high-confidence inliers.
    """
    losses = []
    logs: Dict[str, float] = {}
    corr_weight = min(max(float(corr_weight), 0.0), 1.0)
    for suffix in ("f", "b"):
        votes = out.get(f"pcgv_votes_{suffix}")
        residuals = out.get(f"pcgv_residuals_{suffix}")
        if votes is None or residuals is None:
            continue
        residual_score = _quantile_score(
            residuals,
            residual_low_q,
            residual_high_q,
            high_is_good=False,
            eps=eps,
        )
        corr_scores = []
        peak = _last_corr_stat(out, suffix, "corr_peak")
        gap = _last_corr_stat(out, suffix, "corr_gap")
        entropy = _last_corr_stat(out, suffix, "corr_entropy")
        if peak is not None:
            corr_scores.append(_quantile_score(peak, residual_low_q, residual_high_q, True, eps))
        if gap is not None:
            corr_scores.append(_quantile_score(gap, residual_low_q, residual_high_q, True, eps))
        if entropy is not None:
            corr_scores.append(_quantile_score(entropy, residual_low_q, residual_high_q, False, eps))
        if corr_scores:
            corr_score = sum(corr_scores) / len(corr_scores)
            pseudo = (1.0 - corr_weight) * residual_score + corr_weight * corr_score
        else:
            pseudo = residual_score
        pseudo = _sharpen_probability(pseudo.detach(), gamma, eps=eps)

        v = votes.float().clamp(eps, 1.0 - eps)
        p = pseudo.float().clamp(eps, 1.0 - eps)
        confidence = (p - 0.5).abs() * 2.0
        sample_weight = min_confidence_weight + (1.0 - min_confidence_weight) * confidence
        loss = -(p * v.log() + (1.0 - p) * (1.0 - v).log())
        losses.append((loss * sample_weight).sum() / sample_weight.sum().clamp_min(eps))

        high_pseudo = p >= high_thresh
        low_pseudo = p <= low_thresh
        fallback = residuals.detach().float().mean()
        logs.update({
            f"pseudo_mean_{suffix}": _scalar(p.mean()),
            f"pseudo_std_{suffix}": _scalar(p.std(unbiased=False)),
            f"pseudo_hi_{suffix}": _scalar(high_pseudo.float().mean()),
            f"pseudo_lo_{suffix}": _scalar(low_pseudo.float().mean()),
            f"pseudo_inlier_residual_{suffix}": _scalar(
                _masked_mean(residuals.detach(), high_pseudo, fallback)
            ),
            f"pseudo_outlier_residual_{suffix}": _scalar(
                _masked_mean(residuals.detach(), low_pseudo, fallback)
            ),
        })
        if peak is not None:
            logs[f"corr_peak_{suffix}"] = _scalar(peak)
        if gap is not None:
            logs[f"corr_gap_{suffix}"] = _scalar(gap)
        if entropy is not None:
            logs[f"corr_entropy_{suffix}"] = _scalar(entropy)
    return (sum(losses) / len(losses) if losses else _zero_like_loss(out)), logs


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


def dominant_plane_loss(out: Dict[str, torch.Tensor],
                        lambda_align: float = 1.0,
                        lambda_cov: float = 0.1,
                        lambda_fil: float = 0.5,
                        lambda_tv: float = 0.0,
                        lambda_cycle: float = 0.05,
                        lambda_entropy: float = 0.0,
                        coverage_floor: float = 0.35,
                        coverage_mode: str = "hinge",
                        margin: float = 1.0) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Dominant-plane v1 objective using current PCGV patch masks."""
    zero = _zero_like_loss(out)

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
    align = align_f + align_b

    fil_f = (f2_warp - f_warp2).abs().mean()
    fil_b = (f1_warp - f_warp1).abs().mean()
    fil = fil_f + fil_b

    coverage = _mean_terms(
        [
            coverage_loss(mask_f, mode=coverage_mode, floor=coverage_floor),
            coverage_loss(mask_b, mode=coverage_mode, floor=coverage_floor),
        ],
        zero,
    )

    if lambda_tv:
        tv = _mean_terms(
            [
                edge_aware_tv(mask_f, out.get("img1_patch")),
                edge_aware_tv(mask_b, out.get("img2_patch")),
            ],
            zero,
        )
    else:
        tv = zero
    cycle = cycle_consistency_loss(out) if lambda_cycle else zero
    entropy = entropy_regularization(out) if lambda_entropy else zero

    total = (
        lambda_align * align
        + lambda_cov * coverage
        + lambda_fil * fil
        + lambda_tv * tv
        + lambda_cycle * cycle
        + lambda_entropy * entropy
    )

    logs: Dict[str, float] = {
        "total": _scalar(total),
        "align": _scalar(align),
        "align_f": _scalar(align_f),
        "align_b": _scalar(align_b),
        "coverage": _scalar(coverage),
        "fil": _scalar(fil),
        "fil_f": _scalar(fil_f),
        "fil_b": _scalar(fil_b),
        "tv": _scalar(tv),
        "cycle": _scalar(cycle),
        "entropy": _scalar(entropy),
        "mask_area_f": _scalar(mask_f.float().mean()),
        "mask_area_b": _scalar(mask_b.float().mean()),
    }
    for suffix in ("f", "b"):
        votes = out.get(f"pcgv_votes_{suffix}")
        if torch.is_tensor(votes):
            logs[f"mean_vote_{suffix}"] = _scalar(votes.float().mean())
            logs[f"vote_std_{suffix}"] = _scalar(votes.detach().float().std(unbiased=False))
        fallbacks = out.get(f"dlt_fallbacks_{suffix}")
        attempts = out.get(f"dlt_attempts_{suffix}")
        if torch.is_tensor(fallbacks):
            logs[f"dlt_fallbacks_{suffix}"] = _scalar(fallbacks)
            if torch.is_tensor(attempts):
                rate = fallbacks.float() / attempts.float().clamp_min(1.0)
                logs[f"dlt_fallback_rate_{suffix}"] = _scalar(rate)
    for key in ("pcgv_refine_blend_f", "pcgv_refine_blend_b"):
        value = out.get(key)
        if torch.is_tensor(value):
            logs[key.replace("pcgv_", "")] = _scalar(value)
    return total, logs


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
              vote_target_mode: str = "robust",
              vote_residual_low_q: float = 0.25,
              vote_residual_high_q: float = 0.75,
              vote_corr_weight: float = 0.35,
              vote_gamma: float = 1.5,
              vote_min_confidence_weight: float = 0.25,
              vote_low_thresh: float = 0.30,
              vote_high_thresh: float = 0.70,
              margin: float = 1.0,
              use_masked_align: bool = False) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Total PCGV loss.

    By default this is exactly the existing baseline objective applied to the
    model's current baseline-compatible outputs.  Extra terms activate only when
    their lambdas are non-zero.
    """
    if lambda_align == 0.0 and lambda_fil == 0.0:
        total = _zero_like_loss(out)
        zero_float = _scalar(total)
        logs = {
            "total": zero_float,
            "align": zero_float,
            "align_f": zero_float,
            "align_b": zero_float,
            "fil": zero_float,
            "fil_f": zero_float,
            "fil_b": zero_float,
        }
    elif use_masked_align:
        total, logs = masked_baseline_loss(out, lambda_align=lambda_align,
                                           lambda_fil=lambda_fil, margin=margin)
    else:
        total, logs = baseline_loss(out, lambda_align=lambda_align,
                                    lambda_fil=lambda_fil, margin=margin)

    zero = _zero_like_loss(out)
    coarse_flow = coarse_flow_anchor_loss(out) if lambda_coarse_flow else zero
    reproj = weighted_reprojection_loss(out) if lambda_reproj else zero
    cycle = cycle_consistency_loss(out) if lambda_cycle else zero
    vote_logs: Dict[str, float] = {}
    if lambda_vote:
        if vote_target_mode == "exp":
            vote = vote_pseudo_label_loss(out, tau=vote_tau)
        elif vote_target_mode == "robust":
            vote, vote_logs = robust_vote_pseudo_label_loss(
                out,
                residual_low_q=vote_residual_low_q,
                residual_high_q=vote_residual_high_q,
                corr_weight=vote_corr_weight,
                gamma=vote_gamma,
                min_confidence_weight=vote_min_confidence_weight,
                low_thresh=vote_low_thresh,
                high_thresh=vote_high_thresh,
            )
        else:
            raise ValueError(f"unknown vote_target_mode: {vote_target_mode}")
    else:
        vote = zero
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
    logs.update(vote_logs)
    for key in ("pcgv_votes_f", "pcgv_votes_b", "pcgv_residuals_f", "pcgv_residuals_b"):
        value = out.get(key)
        if torch.is_tensor(value):
            logs[key.replace("pcgv_", "mean_")] = _scalar(value)
    for suffix in ("f", "b"):
        mask = out.get(f"pcgv_mask_{suffix}_patch")
        if torch.is_tensor(mask):
            logs[f"mask_area_{suffix}"] = _scalar(mask.float().mean())
        votes = out.get(f"pcgv_votes_{suffix}")
        if torch.is_tensor(votes):
            logs[f"mean_vote_{suffix}"] = _scalar(votes.float().mean())
        fallbacks = out.get(f"dlt_fallbacks_{suffix}")
        attempts = out.get(f"dlt_attempts_{suffix}")
        if torch.is_tensor(fallbacks):
            logs[f"dlt_fallbacks_{suffix}"] = _scalar(fallbacks)
            if torch.is_tensor(attempts):
                rate = fallbacks.float() / attempts.float().clamp_min(1.0)
                logs[f"dlt_fallback_rate_{suffix}"] = _scalar(rate)
    for suffix in ("f", "b"):
        votes = out.get(f"pcgv_votes_{suffix}")
        residuals = out.get(f"pcgv_residuals_{suffix}")
        if torch.is_tensor(votes) and torch.is_tensor(residuals):
            logs.update(_vote_basic_diagnostics(
                votes,
                residuals,
                low_thresh=vote_low_thresh,
                high_thresh=vote_high_thresh,
                prefix=suffix,
            ))
    for key in ("pcgv_refine_blend_f", "pcgv_refine_blend_b"):
        value = out.get(key)
        if torch.is_tensor(value):
            logs[key.replace("pcgv_", "")] = _scalar(value)
    for key in ("solver_cond_f", "solver_cond_b"):
        value = out.get(key)
        if torch.is_tensor(value):
            logs[key] = _scalar(torch.nan_to_num(value.float(), nan=0.0, posinf=0.0))
    return total, logs
