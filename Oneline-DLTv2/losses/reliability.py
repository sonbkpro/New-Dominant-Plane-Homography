"""L_rel: pair-level binary cross-entropy with hard-negative mining.

Invalid pairs are produced by either of:
  (a) intra-batch shuffle    -> label 0
  (b) patch reshuffle        -> label 0  (constructed by the dataloader)
  (c) hard-negative mining   -> label 0.5  (in-batch natural pairs whose mean
                                            patch residual is above a running
                                            percentile threshold)
Natural pairs that are not flagged by (c) get label 1.

`build_invalid_pair_labels` produces the y_target vector consistent with the
ordering used by train.py (natural batch first, then shuffled-copy batch)."""

from typing import Tuple

import torch
import torch.nn.functional as F


def build_invalid_pair_labels(
    natural_residual_mean: torch.Tensor,    # (B,)  per-sample mean residual
    hard_negative_percentile: float = 0.85,
) -> torch.Tensor:
    """Returns (B,) target labels for the natural-pair half of the batch.

    A natural pair whose residual is above the running batch percentile is
    relabeled to 0.5 (label smoothing for hard negatives); everyone else is 1.
    The shuffled half (always label 0) is added by the caller.
    """
    threshold = torch.quantile(natural_residual_mean.detach(),
                               hard_negative_percentile)
    is_hard = (natural_residual_mean >= threshold).float()
    return 1.0 - 0.5 * is_hard               # 1.0 if easy, 0.5 if hard


def reliability_loss(
    s_pred: torch.Tensor,                   # (B_total,)
    y_target: torch.Tensor,                 # (B_total,) in {0, 0.5, 1}
    eps: float = 1e-6,
) -> torch.Tensor:
    s = s_pred.clamp(eps, 1.0 - eps)
    return -(y_target * torch.log(s) + (1.0 - y_target) * torch.log(1.0 - s)).mean()
