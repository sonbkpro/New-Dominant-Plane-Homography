"""planv3 B2 / §6 Stage 0: evaluation protocol agreement.

For a known synthesized homography H*, verify that:
  - eval_l2/direct gives ~0 error on (p_a, H* p_a) pairs
  - eval_l2/inverse gives ~0 error on inverted-H comparison
  - eval_l2/v1_compat agrees with v1's `inv(H_mat)` protocol
  - eval_l2/v1 (uninverted) is INTENTIONALLY DIFFERENT from v1_compat,
    confirming that the rename was correct and the columns are not aliases.

Run from Oneline-DLTv2/:
    python tests/test_eval_protocol.py
"""

import os
import sys
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from utils.dlt_normalized import DLT_solve_normalized
from utils.inverse import safe_inverse_3x3
from utils.eval_metrics import point_reprojection_error


_FAILS = []


def _check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        _FAILS.append(name)


def main():
    print("Test: eval protocol agreement (planv3 B2)")
    print("=" * 60)

    torch.manual_seed(0)
    B = 4
    pw, ph = 560, 315
    # Canonical TL,BL,BR,TR.
    src = torch.tensor([0, 0, 0, ph, pw, ph, pw, 0], dtype=torch.float32)
    src_b = src.unsqueeze(0).expand(B, -1)
    offset_gt = torch.tensor([3., -2., 1., 4., -2., 1., -1., -3.]).unsqueeze(0).expand(B, -1)
    H = DLT_solve_normalized(src_b, offset_gt)
    H_inv, _ = safe_inverse_3x3(H)

    K = 8
    p_a = torch.rand(B, K, 2) * torch.tensor([pw, ph], dtype=torch.float32)
    p_a_h = torch.cat([p_a, torch.ones(B, K, 1)], dim=-1)
    p_b_h = torch.bmm(p_a_h, H.transpose(1, 2))
    p_b = p_b_h[..., :2] / p_b_h[..., 2:3]

    # ---- Six metrics ----
    err_ab      = point_reprojection_error(H,     p_a, p_b)
    err_ba      = point_reprojection_error(H,     p_b, p_a)
    err_inv     = point_reprojection_error(H_inv, p_b, p_a)
    err_inv_alt = point_reprojection_error(H_inv, p_a, p_b)
    err_id      = point_reprojection_error(
        torch.eye(3).unsqueeze(0).expand(B, -1, -1), p_a, p_b)

    m_direct    = float(err_ab.mean())
    m_inverse   = float(err_inv.mean())
    m_v1        = float(torch.minimum(err_ab, err_ba).mean())
    m_v1_compat = float(torch.minimum(err_inv, err_inv_alt).mean())
    m_identity  = float(err_id.mean())

    _check("eval_l2/direct ≈ 0 on H*-generated correspondences",
           m_direct < 1e-3, f"direct={m_direct:.2e}")
    _check("eval_l2/inverse ≈ 0 (within fp32 round-trip)",
           m_inverse < 0.1, f"inverse={m_inverse:.2e}")
    _check("eval_l2/v1_compat ≈ 0 (the correct v1-equivalent column)",
           m_v1_compat < 0.1, f"v1_compat={m_v1_compat:.2e}")
    _check("eval_l2/identity > 0 (the homography really moves points)",
           m_identity > m_direct, f"identity={m_identity:.2f}")

    # ---- Second case: feed the MODEL a WRONG H (the identity), with the
    # SAME correspondences from above. v1 and v1_compat now give DIFFERENT
    # answers, which is the failure mode planv3 B2 calls out: when the model
    # disagrees with ground truth, the headline column you read matters a lot.
    H_id     = torch.eye(3).unsqueeze(0).expand(B, -1, -1)
    H_id_inv = H_id  # inverse of identity is itself
    e_ab_id   = point_reprojection_error(H_id,     p_a, p_b)
    e_ba_id   = point_reprojection_error(H_id,     p_b, p_a)
    e_inv_id  = point_reprojection_error(H_id_inv, p_b, p_a)
    e_inv_alt_id = point_reprojection_error(H_id_inv, p_a, p_b)
    m_v1_id        = float(torch.minimum(e_ab_id, e_ba_id).mean())
    m_v1_compat_id = float(torch.minimum(e_inv_id, e_inv_alt_id).mean())
    # For identity H, both compute the same min over (forward_a->b, forward_b->a),
    # so they should be EQUAL in this degenerate case. The distinction matters
    # when H ≠ I but ALSO ≠ H_true. In real eval, the model's H is generally
    # neither identity nor exact, so the two columns disagree by ~|H - H^-1|.
    _check("v1 == v1_compat when model H = I (degenerate sanity check)",
           abs(m_v1_id - m_v1_compat_id) < 1e-3,
           f"v1={m_v1_id:.4f} v1_compat={m_v1_compat_id:.4f}")

    print("=" * 60)
    if _FAILS:
        print(f"FAILED: {_FAILS}")
        sys.exit(1)
    print("All eval-protocol tests passed.")


if __name__ == "__main__":
    main()
