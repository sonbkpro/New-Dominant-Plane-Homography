"""
Loss functions for Calibrated Dominant-Plane Consensus Homography Estimation.

Active losses (Milestones 2-4):
  L_align   – robust uncertainty-weighted feature alignment (bidirectional)
  L_triplet – consensus-weighted triplet margin loss (bidirectional)
  L_inv     – inverse consistency  ||H_ab H_ba - I||_F^2
  L_support – prevent posterior collapse:  max(0, alpha - mean(q))^2
  L_smooth  – edge-aware TV on consensus map q
  L_calib   – uncertainty calibration via stop-gradient residuals

Deferred (Milestones 5-6):
  L_rel     – reliability BCE with synthetic invalid pairs
  L_temp    – temporal composition consistency
"""

import torch

_EPS = 1e-6


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def charbonnier(x, eps=1e-3):
    return torch.sqrt(x.pow(2) + eps ** 2)


# ---------------------------------------------------------------------------
# L_align  (Section 6.1)
# ---------------------------------------------------------------------------

def _align_single(r, log_sigma, q, eps=_EPS):
    """
    r, log_sigma, q : [B, 1, Ph, Pw]
    Returns scalar.
    """
    sigma = torch.exp(log_sigma).clamp(min=eps)
    robust = charbonnier(r / sigma)
    per_pixel = q * (robust + log_sigma)
    return per_pixel.sum() / (q.sum() + eps)


def loss_align(r_ab, log_sigma_ab, q_ab,
               r_ba, log_sigma_ba, q_ba, eps=_EPS):
    return _align_single(r_ab, log_sigma_ab, q_ab, eps) + \
           _align_single(r_ba, log_sigma_ba, q_ba, eps)


# ---------------------------------------------------------------------------
# L_triplet  (Section 6.2)
# ---------------------------------------------------------------------------

def _triplet_single(d_pos, d_neg, q, m=1.0, eps=_EPS):
    """
    d_pos = |F_b - W(F_a, H_ab)|  (residual after alignment)
    d_neg = |F_b - F_a|           (difference without alignment)
    q     : [B, 1, Ph, Pw]
    """
    hinge = torch.clamp(m + d_pos - d_neg, min=0.0)
    return (q * hinge).sum() / (q.sum() + eps)


def loss_triplet(r_ab, d_neg_ab, q_ab,
                 r_ba, d_neg_ba, q_ba, m=1.0, eps=_EPS):
    return _triplet_single(r_ab, d_neg_ab, q_ab, m, eps) + \
           _triplet_single(r_ba, d_neg_ba, q_ba, m, eps)


# ---------------------------------------------------------------------------
# L_inv  (Section 6.3)
# ---------------------------------------------------------------------------

def loss_inv(H_ab, H_ba):
    """||H_ab H_ba - I||_F^2 averaged over batch."""
    B = H_ab.size(0)
    I = torch.eye(3, device=H_ab.device, dtype=H_ab.dtype) \
             .unsqueeze(0).expand(B, -1, -1)
    diff = torch.bmm(H_ab, H_ba) - I
    return (diff ** 2).sum() / B


# ---------------------------------------------------------------------------
# L_support  (Section 6.4)
# ---------------------------------------------------------------------------

def loss_support(q_ab, q_ba, alpha=0.05):
    """Prevent q from collapsing to zero support."""
    q_mean = (q_ab.mean() + q_ba.mean()) * 0.5
    return torch.clamp(alpha - q_mean, min=0.0) ** 2


# ---------------------------------------------------------------------------
# L_smooth  (Section 6.5)
# ---------------------------------------------------------------------------

def loss_smooth(q_ab, q_ba, img_patch_b, img_patch_a, gamma=5.0):
    """
    Edge-aware TV regularizer on the consensus maps.
    q_ab, q_ba    : [B, 1, Ph, Pw]
    img_patch_b/a : [B, 1, Ph, Pw]  – used for edge weighting
    """
    def _tv(q, img):
        dq_x = torch.abs(q[:, :, :, 1:] - q[:, :, :, :-1])
        dq_y = torch.abs(q[:, :, 1:, :] - q[:, :, :-1, :])
        dI_x = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1])
        dI_y = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :])
        return (dq_x * torch.exp(-gamma * dI_x)).mean() + \
               (dq_y * torch.exp(-gamma * dI_y)).mean()
    return _tv(q_ab, img_patch_b) + _tv(q_ba, img_patch_a)


# ---------------------------------------------------------------------------
# L_calib  (Section 6.6)
# ---------------------------------------------------------------------------

def loss_calib(log_sigma_ab, r_ab, log_sigma_ba, r_ba, eps=_EPS):
    """Weak self-supervised calibration via stop-gradient residual targets."""
    def _single(log_sigma, r):
        log_r_sg = torch.log(r.detach() + eps)
        return torch.abs(log_sigma - log_r_sg).mean()
    return _single(log_sigma_ab, r_ab) + _single(log_sigma_ba, r_ba)


# ---------------------------------------------------------------------------
# Total loss  (Section 6.9)
# ---------------------------------------------------------------------------

def compute_total_loss(la, lt, li, ls, lsm, lc,
                       lam1=1.0, lam2=0.01, lam3=0.01,
                       lam4=0.001, lam5=0.05):
    """
    la  = L_align
    lt  = L_triplet  (weight lam1)
    li  = L_inv      (weight lam2)
    ls  = L_support  (weight lam3)
    lsm = L_smooth   (weight lam4)
    lc  = L_calib    (weight lam5)
    """
    return la + lam1 * lt + lam2 * li + lam3 * ls + lam4 * lsm + lam5 * lc
