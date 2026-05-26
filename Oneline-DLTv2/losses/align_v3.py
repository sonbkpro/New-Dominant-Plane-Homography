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

    KNOWN LIMITATION (planv3 §B-postmortem): this L1-style loss on raw feature
    residuals rewards the backbone for producing small-magnitude features
    (||F|| → 0 minimizes |F_b − F_a_warped| trivially). When run as part of
    h_only or joint with the backbone unfrozen, features collapse and H stops
    being learned. Prefer `align_soft_cosine_loss` below for any stage that
    trains the backbone.
    """
    rho = _charbonnier(residual, eps=charbonnier_eps)
    w = q * valid_mask
    num = (rho * w).sum(dim=(1, 2, 3))
    den = valid_mask.sum(dim=(1, 2, 3)).clamp(min=weight_eps)
    return (num / den).mean()


def align_soft_cosine_loss(
    F_b: torch.Tensor,             # (B, C, H, W)
    F_a_warped: torch.Tensor,      # (B, C, H, W)
    q: torch.Tensor,               # (B, 1, H, W)
    valid_mask: torch.Tensor,      # (B, 1, H, W)
    weight_eps: float = 1e-6,
    cos_eps: float = 1e-6,
) -> torch.Tensor:
    """Scale-invariant q-weighted cosine alignment loss (Fix A).

    Per-pixel:  L_i = 1 - cos(F_b(i), F_a_warped(i))
                    = 1 - <F_b(i), F_a_warped(i)> / (||F_b(i)|| ||F_a_warped(i)||)

    Loss aggregation: L = sum_i q_i v_i L_i / sum_i v_i.

    Why this fixes feature collapse: cosine measures *direction* only, not
    magnitude. Setting F → 0 gives 0/0 (undefined, clamped via eps to 0
    similarity → loss 1, not 0). The only way to minimize the loss is for
    F_b and F_a_warped to *point in the same direction*, which requires
    accurate H (geometric content) AND non-degenerate features. Magnitude
    is free to be whatever the backbone wants; collapse provides no benefit.

    Like align_soft, denominator is sum_i v_i (NOT sum_i q_i v_i) so q cannot
    suppress hard regions by going to zero — it can only emphasize/de-emphasize
    relative to a fixed denominator.
    """
    # Normalize along the channel axis.
    # F.normalize handles the eps internally; result has ||·||_2 = 1 per pixel.
    Fb_n = torch.nn.functional.normalize(F_b,        dim=1, eps=cos_eps)
    Fa_n = torch.nn.functional.normalize(F_a_warped, dim=1, eps=cos_eps)
    cos = (Fb_n * Fa_n).sum(dim=1, keepdim=True)                  # (B, 1, H, W)
    per_pixel = 1.0 - cos                                          # in [0, 2]
    w = q * valid_mask
    num = (per_pixel * w).sum(dim=(1, 2, 3))
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
