"""L_em: trains the predicted posterior q against an E-step target derived
from a two-component generative residual model.

Inlier:  p(r | z=1) ∝ exp(-r / sg(sigma)) / sg(sigma)   (Laplacian, predicted sigma w/ stop-grad)
Outlier: p(r | z=0) = 1 / r_max                          (broad uniform)

E-step:  q_target_i = pi * p(r|1) / [pi * p(r|1) + (1-pi) * p(r|0)]
         (all stop-gradient on the right-hand side)

Loss:    BCE(q_pred, q_target)  weighted by valid_mask.

planv3 §4.3 changes:
  - sigma in the inlier likelihood is stop-graded so EM cannot co-train
    sigma toward a broad inlier prior at the same time L_align_het is
    trying to shrink it. Sigma is updated ONLY by L_align_het and the
    sigma_prior; EM uses whatever sigma currently is.
  - Optional self-paced gate: multiply the loss by sg(exp(-beta * med(r)/r_max))
    so EM smoothly fades when residuals are large, instead of the v2 hard
    `em_residual_gate` cutoff.
"""

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
    self_paced_beta: float = 0.0,  # 0 = disabled; >0 enables exp gate
) -> torch.Tensor:
    with torch.no_grad():
        sigma_sg = log_sigma.exp().clamp(min=1e-3).detach()
        log_p_in  = -residual / sigma_sg - torch.log(sigma_sg)
        # log_p_out = log(1/r_max), as a scalar broadcast to log_p_in's shape.
        log_p_out = torch.full_like(
            log_p_in, float(-torch.log(torch.tensor(r_max))))

        # Numerically stable posterior via log-sum-exp.
        device = q_pred.device
        log_pi_in  = torch.log(torch.tensor(pi,       device=device))
        log_pi_out = torch.log(torch.tensor(1.0 - pi, device=device))
        num = log_pi_in + log_p_in
        den = torch.logaddexp(log_pi_in + log_p_in, log_pi_out + log_p_out)
        q_target = (num - den).exp().clamp(eps, 1.0 - eps)

        # Optional self-paced gate -- decays the loss smoothly with median residual.
        if self_paced_beta > 0.0:
            med_r = residual[valid_mask > 0.5].median() if (valid_mask > 0.5).any() \
                else torch.zeros((), device=device)
            gate = torch.exp(-self_paced_beta * med_r / r_max)
        else:
            gate = torch.ones((), device=device)

    q_pred_c = q_pred.clamp(eps, 1.0 - eps)
    bce = -(q_target * torch.log(q_pred_c) + (1.0 - q_target) * torch.log(1.0 - q_pred_c))

    w = valid_mask
    num = (bce * w).sum(dim=(1, 2, 3))
    den = w.sum(dim=(1, 2, 3)).clamp(min=1.0)
    loss = (num / den).mean()
    return loss * gate
