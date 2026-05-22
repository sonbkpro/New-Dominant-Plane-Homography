"""planv3 §4.2: split alignment loss into soft and heteroscedastic terms.

Why split: in the v2 single loss
    L_align = sum_i q_i v_i [rho(r_i/sigma_i) + log sigma_i] / sum_i q_i v_i,
the same gradient drives both q (via the weighting) and sigma (via the
1/sigma scaling). The q-in-denominator lets q suppress hard regions; the
1/sigma amplifies H gradient by 1/sigma_min when sigma collapses. Both
shortcuts close the loop on each other and produce the q-collapse + sigma-
collapse failure mode.

The fix:

  L_align_soft = (1 / sum_i v_i) * sum_i q_i v_i rho_eps(r_i)
       - drives H, q, and the trunk
       - NO sigma involvement, NO normalization by q (the sum sum_i v_i is
         independent of q, so the model cannot reduce cost by lowering q
         uniformly -- only spatially-selective q variation pays off)
       - q here is NOT detached: the posterior learns to upweight pixels
         that already align (which is the cdpc story).

  L_align_het = (1 / |S|) * sum_{i in S} [rho_eps(r_i / sg(sigma_i)) + log sigma_i]
       - drives sigma only; S = {i : q_i > tau_q & v_i = 1} is a hard
         dominant-plane selection set (stop-gradient through the selection)
       - sg(sigma_i) inside the numerator removes the 1/sigma amplification
         path on H; the only sigma-gradient is from log sigma_i and the
         implicit dependence in the denominator (which is bounded).
"""

from typing import Tuple

import torch


def _charbonnier(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt(x * x + eps * eps)


def align_soft_loss(
    residual: torch.Tensor,        # (B, 1, H, W)  r_i = ||F_b - F_a_warped||
    q: torch.Tensor,               # (B, 1, H, W)
    valid_mask: torch.Tensor,      # (B, 1, H, W)
    charbonnier_eps: float = 1e-3,
    weight_eps: float = 1e-6,
) -> torch.Tensor:
    """Soft q-weighted Charbonnier on the residual, normalized by Σ v (not Σ q·v).

    Returns the batch-mean loss. Sigma is NOT used here.
    """
    rho = _charbonnier(residual, eps=charbonnier_eps)
    w = q * valid_mask
    num = (rho * w).sum(dim=(1, 2, 3))
    den = valid_mask.sum(dim=(1, 2, 3)).clamp(min=weight_eps)
    return (num / den).mean()


def align_het_loss(
    residual: torch.Tensor,        # (B, 1, H, W)
    log_sigma: torch.Tensor,       # (B, 1, H, W)
    q: torch.Tensor,               # (B, 1, H, W)
    valid_mask: torch.Tensor,      # (B, 1, H, W)
    tau_q: float = 0.5,
    charbonnier_eps: float = 1e-3,
    weight_eps: float = 1e-6,
) -> torch.Tensor:
    """Kendall-Gal heteroscedastic loss restricted to the q-selected set S.

    S = {i : q_i > tau_q AND v_i = 1}. The membership is computed under
    stop-gradient so the selection itself is gradient-free; only the values
    of (residual, log_sigma) contribute to the loss within S.

    Residual is divided by sg(sigma) so the only sigma-gradient path is
    through `log sigma`. This breaks the H-amplification-via-1/sigma loop.
    """
    sigma = log_sigma.exp()
    sigma_sg = sigma.detach()

    # Hard selection mask, stop-gradient on q for selection purposes.
    with torch.no_grad():
        sel = ((q > tau_q) & (valid_mask > 0.5)).float()

    x = residual / (sigma_sg + 1e-8)
    rho = _charbonnier(x, eps=charbonnier_eps)
    per_pixel = rho + log_sigma

    num = (per_pixel * sel).sum(dim=(1, 2, 3))
    den = sel.sum(dim=(1, 2, 3)).clamp(min=weight_eps)
    # If a batch sample has no selected pixels, its contribution is 0 (den=1
    # via clamp -> num/den=0 since num=0 also).
    return (num / den).mean()


# ---------- Diagnostic helper (not a loss) -----------------------------------

def selection_set_stats(
    q: torch.Tensor,
    valid_mask: torch.Tensor,
    tau_q: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (|S|/|v|, mean q in S) as scalars per batch sample for logging."""
    with torch.no_grad():
        sel = ((q > tau_q) & (valid_mask > 0.5)).float()
        v_size = valid_mask.sum(dim=(1, 2, 3)).clamp(min=1.0)
        s_size = sel.sum(dim=(1, 2, 3))
        frac = s_size / v_size
        q_in_s = ((q * sel).sum(dim=(1, 2, 3)) /
                  s_size.clamp(min=1.0))
    return frac, q_in_s
