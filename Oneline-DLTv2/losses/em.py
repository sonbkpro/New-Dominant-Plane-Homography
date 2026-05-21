"""L_em: trains the predicted posterior q against an E-step target derived
from a two-component generative residual model.

Inlier:  p(r | z=1) propto exp(-r / sigma) / sigma       (Laplacian, predicted sigma)
Outlier: p(r | z=0) = 1 / r_max                           (broad uniform)

E-step:  q_target_i = pi * p(r|1) / [pi * p(r|1) + (1-pi) * p(r|0)]
         (all stop-gradient on the right-hand side)

Loss:    BCE(q_pred, q_target)  weighted by valid_mask."""

import torch
import torch.nn.functional as F


def em_posterior_loss(
    q_pred: torch.Tensor,          # (B, 1, H, W) in (0, 1)
    residual: torch.Tensor,        # (B, 1, H, W) non-negative
    log_sigma: torch.Tensor,       # (B, 1, H, W)
    valid_mask: torch.Tensor,      # (B, 1, H, W)
    pi: float = 0.5,
    r_max: float = 4.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    with torch.no_grad():
        sigma = log_sigma.exp().clamp(min=1e-3)
        log_p_in  = -residual / sigma - torch.log(sigma)
        log_p_out = torch.full_like(log_p_in, -float(torch.tensor(r_max).log()))

        # Numerically stable posterior via log-sum-exp.
        log_pi_in  = torch.log(torch.tensor(pi,       device=q_pred.device))
        log_pi_out = torch.log(torch.tensor(1.0 - pi, device=q_pred.device))
        num = log_pi_in + log_p_in
        den = torch.logaddexp(log_pi_in + log_p_in, log_pi_out + log_p_out)
        q_target = (num - den).exp().clamp(eps, 1.0 - eps)

    q_pred_c = q_pred.clamp(eps, 1.0 - eps)
    bce = -(q_target * torch.log(q_pred_c) + (1.0 - q_target) * torch.log(1.0 - q_pred_c))

    w = valid_mask
    num = (bce * w).sum(dim=(1, 2, 3))
    den = w.sum(dim=(1, 2, 3)).clamp(min=1.0)
    return (num / den).mean()
