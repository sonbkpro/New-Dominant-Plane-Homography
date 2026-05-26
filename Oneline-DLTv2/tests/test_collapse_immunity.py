"""Proof tests for Fix A (cosine align_soft) and Fix B (q-weighted photo_img).

Fix A: Verify that scaling features by 1/k (k>>1) does NOT reduce
  align_soft_cosine_loss (it's scale-invariant), while it DOES reduce the
  L1-based align_soft_loss (collapse vulnerability).

Fix B: Verify that lowering q on a pixel reduces its contribution to the
  q-weighted photo_img loss (q has a real gating role), while uniform-q
  photo_img is unchanged.

Run from Oneline-DLTv2/:
    python tests/test_collapse_immunity.py
"""

import os
import sys
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from losses.align_v3 import align_soft_loss, align_soft_cosine_loss
from losses.photo_image import (
    photometric_image_loss, photometric_image_q_weighted_loss,
)
from utils.dlt_normalized import DLT_solve_normalized


_FAILS = []


def _check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        _FAILS.append(name)


# ---------------------------------------------------------------------------
# Fix A: scale invariance of cosine align_soft
# ---------------------------------------------------------------------------

def test_fix_a_scale_invariance():
    print("Fix A: cosine align_soft is scale-invariant; L1 is not")
    torch.manual_seed(0)
    B, C, H, W = 2, 32, 16, 24
    F_b = torch.randn(B, C, H, W)
    F_a_warped = F_b + 0.05 * torch.randn_like(F_b)
    q = 0.7 * torch.ones(B, 1, H, W)
    v = torch.ones(B, 1, H, W)
    residual = (F_b - F_a_warped).abs().mean(dim=1, keepdim=True)

    L_l1_normal  = float(align_soft_loss(residual, q, v))
    L_cos_normal = float(align_soft_cosine_loss(F_b, F_a_warped, q, v))

    # Now shrink BOTH features by 100x. Cosine should be unchanged
    # (direction preserved); L1 collapses to ~0.
    F_b_small = F_b / 100.0
    F_aw_small = F_a_warped / 100.0
    residual_small = (F_b_small - F_aw_small).abs().mean(dim=1, keepdim=True)

    L_l1_small  = float(align_soft_loss(residual_small, q, v))
    L_cos_small = float(align_soft_cosine_loss(F_b_small, F_aw_small, q, v))

    print(f"        L1  normal={L_l1_normal:.4f}  shrunk={L_l1_small:.4f}  "
          f"ratio={L_l1_small/L_l1_normal:.3e}")
    print(f"        cos normal={L_cos_normal:.4f}  shrunk={L_cos_small:.4f}  "
          f"ratio={L_cos_small/L_cos_normal:.3f}")

    # Note: Charbonnier eps=1e-3 floors the loss, so 100x feature shrink
    # gives ~30x loss reduction (not 100x). Still a strong collapse signal.
    _check("L1 align_soft is reduced 20x+ when features shrink 100x",
           L_l1_small < L_l1_normal / 20.0,
           f"normal={L_l1_normal:.4f} shrunk={L_l1_small:.4f}")
    _check("Cosine align_soft is invariant to feature scaling (ratio in [0.95, 1.05])",
           0.95 < (L_cos_small / max(L_cos_normal, 1e-9)) < 1.05,
           f"ratio={L_cos_small/L_cos_normal:.3f}")


def test_fix_a_gradient_does_not_shrink_features():
    print("Fix A: cosine align_soft gradient does NOT push features toward 0")
    torch.manual_seed(0)
    B, C, H, W = 2, 32, 16, 24
    F_b_param = torch.nn.Parameter(torch.randn(B, C, H, W))
    F_aw_param = torch.nn.Parameter(F_b_param.detach() + 0.05 * torch.randn(B, C, H, W))
    q = 0.7 * torch.ones(B, 1, H, W)
    v = torch.ones(B, 1, H, W)

    L = align_soft_cosine_loss(F_b_param, F_aw_param, q, v)
    L.backward()

    # The gradient on each feature should be orthogonal to that feature
    # (cosine of normalized vectors has zero gradient in the radial
    # direction). Inner product <F, dF> per pixel should be ~0.
    inner_b = (F_b_param * F_b_param.grad).sum(dim=1)              # (B, H, W)
    rel_inner = inner_b.abs().mean().item() / (F_b_param.norm(dim=1).mean().item() *
                                               F_b_param.grad.norm(dim=1).mean().item() + 1e-9)
    print(f"        |<F, dF>| / (||F|| ||dF||) (mean over pixels) = {rel_inner:.4f}")
    _check("Cosine gradient is approximately orthogonal to F (no radial shrink)",
           rel_inner < 0.1, f"relative inner product = {rel_inner:.3f}")


# ---------------------------------------------------------------------------
# Fix B: q gates the image-space photometric loss
# ---------------------------------------------------------------------------

def test_fix_b_q_gates_photometric():
    print("Fix B: q-weighted photo_img is reduced when q is suppressed")
    torch.manual_seed(0)
    B, ph, pw = 2, 64, 80
    Hf, Wf = ph + 32, pw + 32
    I_a_full = torch.randn(B, 1, Hf, Wf)
    # Generate I_b as a slightly translated copy of I_a so the photometric
    # residual is nonzero but bounded.
    I_b_patch = I_a_full[:, :, 16:16 + ph, 18:18 + pw].contiguous()
    crop_xy = torch.tensor([[16, 16]] * B, dtype=torch.float32)
    # Identity H -> warp pulls (16,16) into the patch; residual = small shift.
    H_full = torch.eye(3).unsqueeze(0).expand(B, -1, -1).contiguous()

    # Uniform q.
    q_uniform = 0.7 * torch.ones(B, 1, ph // 4, pw // 4)
    # Spatially suppressed q (zero on right half).
    q_suppressed = q_uniform.clone()
    q_suppressed[..., :, q_suppressed.shape[-1] // 2:] = 0.05

    L_unweighted = float(photometric_image_loss(I_a_full, I_b_patch, H_full, crop_xy))
    L_uniform_q  = float(photometric_image_q_weighted_loss(
        I_a_full, I_b_patch, H_full, crop_xy, q_uniform))
    L_supp_q     = float(photometric_image_q_weighted_loss(
        I_a_full, I_b_patch, H_full, crop_xy, q_suppressed))

    print(f"        L unweighted     = {L_unweighted:.4f}")
    print(f"        L q-uniform 0.7  = {L_uniform_q:.4f}   (≈ 0.7 * unweighted)")
    print(f"        L q-suppressed   = {L_supp_q:.4f}      (right half down-weighted)")

    _check("uniform-q-weighted ≈ 0.7 * unweighted",
           abs(L_uniform_q - 0.7 * L_unweighted) < 0.05 * L_unweighted,
           f"got {L_uniform_q:.4f} vs 0.7*{L_unweighted:.4f}={0.7*L_unweighted:.4f}")
    _check("q-suppressed < q-uniform (suppression actually reduces the loss)",
           L_supp_q < L_uniform_q,
           f"suppressed={L_supp_q:.4f} uniform={L_uniform_q:.4f}")


def test_fix_b_q_gradient():
    print("Fix B: q-weighted photo_img produces gradient on q")
    torch.manual_seed(0)
    B, ph, pw = 2, 64, 80
    Hf, Wf = ph + 32, pw + 32
    I_a_full = torch.randn(B, 1, Hf, Wf)
    I_b_patch = torch.randn(B, 1, ph, pw)
    crop_xy = torch.tensor([[16, 16]] * B, dtype=torch.float32)
    H_full = torch.eye(3).unsqueeze(0).expand(B, -1, -1).contiguous()
    q_param = torch.nn.Parameter(0.7 * torch.ones(B, 1, ph // 4, pw // 4))

    L = photometric_image_q_weighted_loss(
        I_a_full, I_b_patch, H_full, crop_xy, q_param,
    )
    L.backward()

    grad_norm = float(q_param.grad.norm())
    print(f"        ||dL/dq|| = {grad_norm:.4f}  L = {float(L):.4f}")
    _check("q receives nonzero gradient from photo_img",
           grad_norm > 1e-6,
           f"grad norm = {grad_norm:.2e}")
    # All gradient signs should be the same as the loss is q * positive_per_pixel.
    sign_agree = (q_param.grad > 0).float().mean().item()
    print(f"        fraction of positive grad entries = {sign_agree:.3f}")
    _check("q-gradient is consistent in sign (loss is q * positive)",
           sign_agree > 0.95 or sign_agree < 0.05,
           f"fraction positive = {sign_agree:.3f}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    print("planv3.1 collapse-immunity tests (Fix A + Fix B)")
    print("=" * 60)
    test_fix_a_scale_invariance()
    test_fix_a_gradient_does_not_shrink_features()
    test_fix_b_q_gates_photometric()
    test_fix_b_q_gradient()
    print("=" * 60)
    if _FAILS:
        print(f"FAILED ({len(_FAILS)}):")
        for n in _FAILS:
            print(f"  - {n}")
        sys.exit(1)
    print("All Fix-A/B tests passed.")


if __name__ == "__main__":
    main()
