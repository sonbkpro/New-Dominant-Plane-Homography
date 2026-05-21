"""Evaluation metrics: point-reprojection error, AUC/inlier@thresh, AUROC,
AUPRC, ECE, NLL, and risk-coverage curve."""

from typing import List, Tuple

import numpy as np
import torch


@torch.no_grad()
def point_reprojection_error(
    H_ab: torch.Tensor,            # (B, 3, 3)  source -> destination
    pts_a: torch.Tensor,           # (B, K, 2)  source-image points
    pts_b: torch.Tensor,           # (B, K, 2)  destination-image targets
) -> torch.Tensor:
    """Returns (B, K) per-point reprojection error in pixels."""
    B, K, _ = pts_a.shape
    ones = torch.ones(B, K, 1, device=pts_a.device, dtype=pts_a.dtype)
    pa_h = torch.cat([pts_a, ones], dim=-1)                      # (B, K, 3)
    proj = torch.bmm(pa_h, H_ab.transpose(1, 2))                 # (B, K, 3)
    proj_xy = proj[..., :2] / (proj[..., 2:3] + 1e-8)
    err = torch.linalg.norm(proj_xy - pts_b, dim=-1)             # (B, K)
    return err


def auc_at_thresholds(errors: np.ndarray, thresholds: List[float]) -> List[float]:
    """Continuous AUC at each threshold: integral of inlier-ratio curve
    from 0 to threshold, normalized to [0, 1]."""
    errors = np.asarray(errors).reshape(-1)
    out = []
    for t in thresholds:
        xs = np.linspace(0.0, t, 200)
        ys = np.array([(errors <= x).mean() for x in xs])
        out.append(float(np.trapz(ys, xs) / t))
    return out


def inlier_ratio_at_thresholds(errors: np.ndarray, thresholds: List[float]) -> List[float]:
    errors = np.asarray(errors).reshape(-1)
    return [float((errors <= t).mean()) for t in thresholds]


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Score is "higher = more positive". For failure detection, pass
    1 - reliability as the score and 1 for failures."""
    from sklearn.metrics import roc_auc_score
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def auprc(scores: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(average_precision_score(labels, scores))


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Standard ECE with equal-width bins."""
    probs = np.clip(np.asarray(probs).reshape(-1), 0.0, 1.0)
    labels = np.asarray(labels).reshape(-1).astype(float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    N = len(probs)
    for b in range(n_bins):
        in_bin = (probs >= bin_edges[b]) & (probs < bin_edges[b + 1] if b < n_bins - 1 else probs <= bin_edges[b + 1])
        if in_bin.sum() == 0:
            continue
        avg_conf = probs[in_bin].mean()
        avg_acc = labels[in_bin].mean()
        ece += (in_bin.sum() / N) * abs(avg_conf - avg_acc)
    return float(ece)


def risk_coverage_curve(
    errors: np.ndarray,
    reliability: np.ndarray,
    coverages: List[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
) -> Tuple[List[float], List[float]]:
    """Sort by reliability descending; for each coverage c, compute mean
    error on the top-c fraction. Returns (coverages, risks)."""
    errors = np.asarray(errors).reshape(-1)
    reliability = np.asarray(reliability).reshape(-1)
    order = np.argsort(-reliability)
    sorted_err = errors[order]
    risks = []
    N = len(errors)
    for c in coverages:
        k = max(1, int(round(c * N)))
        risks.append(float(sorted_err[:k].mean()))
    return list(coverages), risks
