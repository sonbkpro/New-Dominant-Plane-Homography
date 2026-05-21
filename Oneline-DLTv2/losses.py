"""
Loss functions for Calibrated Dominant-Plane Consensus Homography Estimation.

Active losses (Milestones 2-4):
  L_align   – robust heteroscedastic NLL with built-in σ calibration
              (Charbonnier(r/σ) + log σ).  Replaces the prior split between
              L_align (using log1p σ) and a separate L_calib head; that split
              over-determined σ and pinned it at the clamp floor.
  L_triplet – consensus-weighted triplet margin loss (Oneline H_ab)
  L_support – prevent posterior collapse:  max(0, alpha - mean(q))^2
  L_smooth  – edge-aware TV on consensus map q

Active optional losses (Milestones 5-6):
  L_rel     – reliability BCE with synthetic invalid pairs
  L_temp    – temporal composition consistency
"""

import torch
import torch.nn.functional as F

_EPS = 1e-6
_LOG_SIGMA_MIN = -5.0
_LOG_SIGMA_MAX = 5.0


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def charbonnier(x, eps=1e-3):
    return torch.sqrt(x.pow(2) + eps ** 2)


def clamp_log_sigma(log_sigma):
    """Keep heteroscedastic likelihood numerically bounded."""
    return torch.clamp(log_sigma, _LOG_SIGMA_MIN, _LOG_SIGMA_MAX)


def positive_sigma(log_sigma):
    """Map the raw uncertainty head output to a positive, well-behaved sigma."""
    return F.softplus(clamp_log_sigma(log_sigma)) + _EPS


# ---------------------------------------------------------------------------
# L_align  (Section 6.1)
# ---------------------------------------------------------------------------

def _align_single(r, log_sigma, q, eps=_EPS):
    """
    Heteroscedastic alignment NLL with a Charbonnier likelihood surrogate:

        per_pixel = q * ( Charbonnier(r / sigma) + log(sigma) )

    The +log(sigma) term is the standard self-calibration penalty whose
    minimum w.r.t. sigma occurs near |r|; it removes the need for a
    separate L_calib head.  sigma is bounded by clamp_log_sigma so the
    loss is bounded both above and below.

    r, log_sigma, q : [B, 1, Ph, Pw]   →  scalar
    """
    sigma = positive_sigma(log_sigma)
    robust = charbonnier(r / sigma)
    per_pixel = q * (robust + torch.log(sigma))
    return per_pixel.sum() / (q.sum() + eps)


def loss_align(r_ab, log_sigma_ab, q_ab, eps=_EPS):
    return _align_single(r_ab, log_sigma_ab, q_ab, eps)


# ---------------------------------------------------------------------------
# L_triplet  (Section 6.2)
# ---------------------------------------------------------------------------

def _triplet_hinge(d_pos, d_neg, m):
    return torch.clamp(m + d_pos - d_neg, min=0.0)


def _triplet_single(d_pos, d_neg, q, m=0.2, eps=_EPS):
    """
    d_pos = |F_b - W(F_a, H_ab)|  (residual after alignment)
    d_neg = |F_b - F_a|           (difference without alignment)
    q     : [B, 1, Ph, Pw]
    """
    hinge = _triplet_hinge(d_pos, d_neg, m)
    return (q * hinge).sum() / (q.sum() + eps)


def loss_triplet(r_ab, d_neg_ab, q_ab, m=0.2, eps=_EPS):
    return _triplet_single(r_ab, d_neg_ab, q_ab, m, eps)


def triplet_diagnostics(r_ab, d_neg_ab, q_ab, m=0.2, eps=_EPS):
    """Detached scalar diagnostics for checking whether the triplet is saturated."""
    with torch.no_grad():
        q_sum = q_ab.sum() + eps
        d_pos_mean = (q_ab * r_ab).sum() / q_sum
        d_neg_mean = (q_ab * d_neg_ab).sum() / q_sum
        raw_gap = d_pos_mean - d_neg_mean
        hinge = _triplet_hinge(r_ab, d_neg_ab, m)
        active = (hinge > 0.0).to(dtype=q_ab.dtype)
        active_ratio = (q_ab * active).sum() / q_sum
        margin_gap = (q_ab * (m + r_ab - d_neg_ab)).sum() / q_sum
        q_mean = q_ab.mean()
        q_std = q_ab.std()
    return {
        'triplet_d_pos': d_pos_mean,
        'triplet_d_neg': d_neg_mean,
        'triplet_gap': raw_gap,
        'triplet_margin_gap': margin_gap,
        'triplet_active': active_ratio,
        'q_mean': q_mean,
        'q_std': q_std,
    }


# ---------------------------------------------------------------------------
# L_offset
# ---------------------------------------------------------------------------

def loss_offset(offset_ab):
    """Small stabilizer that discourages wild early DLT corner offsets."""
    return offset_ab.pow(2).mean()


# ---------------------------------------------------------------------------
# L_support  (Section 6.4)
# ---------------------------------------------------------------------------

def loss_support(q_ab, alpha=0.05):
    """Prevent q from collapsing to zero support."""
    return torch.clamp(alpha - q_ab.mean(), min=0.0) ** 2


# ---------------------------------------------------------------------------
# L_smooth  (Section 6.5)
# ---------------------------------------------------------------------------

def loss_smooth(q_ab, img_patch_b, gamma=5.0):
    """
    Edge-aware TV regularizer on the consensus map.
    q_ab        : [B, 1, Ph, Pw]
    img_patch_b : [B, 1, Ph, Pw]  – used for edge weighting
    """
    def _tv(q, img):
        dq_x = torch.abs(q[:, :, :, 1:] - q[:, :, :, :-1])
        dq_y = torch.abs(q[:, :, 1:, :] - q[:, :, :-1, :])
        dI_x = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1])
        dI_y = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :])
        return (dq_x * torch.exp(-gamma * dI_x)).mean() + \
               (dq_y * torch.exp(-gamma * dI_y)).mean()
    return _tv(q_ab, img_patch_b)


# ---------------------------------------------------------------------------
# L_calib  (Section 6.6)
# ---------------------------------------------------------------------------
# Removed.  σ self-calibration is now performed inside L_align via the
# +log(σ) term; running both losses pinned log σ at the clamp floor and
# decoupled σ from the residual scale (see Section 6 review notes).


# ---------------------------------------------------------------------------
# L_rel  (Section 6.7)
# ---------------------------------------------------------------------------

def loss_reliability(reliability_score, rel_label):
    """
    Binary reliability loss.

    reliability_score : [B, 1], high means a valid/reliable homography pair.
    rel_label         : [B] or [B, 1], 1=valid pair, 0=invalid generated pair.
    """
    rel_label = rel_label.to(
        device=reliability_score.device,
        dtype=reliability_score.dtype,
    ).reshape_as(reliability_score)
    score = reliability_score.clamp(min=_EPS, max=1.0 - _EPS)
    return F.binary_cross_entropy(score, rel_label)


# ---------------------------------------------------------------------------
# L_temp  (Section 6.8)
# ---------------------------------------------------------------------------

def loss_temporal(H_t_t2, H_t1_t2, H_t_t1):
    """||H_t,t+2 - H_t+1,t+2 H_t,t+1||_F^2 averaged over batch."""
    composed = torch.bmm(H_t1_t2, H_t_t1)
    diff = H_t_t2 - composed
    return (diff ** 2).reshape(diff.size(0), -1).sum(1).mean()


# ---------------------------------------------------------------------------
# Total loss  (Section 6.9)
# ---------------------------------------------------------------------------

def compute_total_loss(la, lt, ls, lsm, lo=None, lr=None, ltmp=None,
                       lam_triplet=0.1, lam_support=0.01,
                       lam_smooth=0.001,
                       lam_offset=1e-4, lam_rel=0.1, lam_temp=0.1,
                       include_geometric=True):
    """
    la  = L_align    (carries the +log σ calibration term internally)
    lt  = L_triplet  (weight lam_triplet)
    ls  = L_support  (weight lam_support)
    lsm = L_smooth   (weight lam_smooth)
    lo  = L_offset   (weight lam_offset)
    lr  = L_rel      (weight lam_rel)
    ltmp= L_temp     (weight lam_temp)
    """
    total = la.new_tensor(0.0)
    if include_geometric:
        total = total + la + lam_triplet * lt + lam_support * ls + lam_smooth * lsm
        if lo is not None:
            total = total + lam_offset * lo
    if lr is not None:
        total = total + lam_rel * lr
    if ltmp is not None:
        total = total + lam_temp * ltmp
    return total
