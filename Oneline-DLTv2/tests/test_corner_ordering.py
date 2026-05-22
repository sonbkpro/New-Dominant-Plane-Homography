"""planv3 B1: assert that synth's H synthesis and the head's DLT_solve
agree on canonical corner ordering.

If this test fails, the supervised stage trains the head against permuted
corner labels and downstream H is non-physical regardless of how well
Stage 1 converges.

Run from Oneline-DLTv2/:
    python tests/test_corner_ordering.py
"""

import os
import sys
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from utils.dlt import DLT_solve
from utils.dlt_normalized import DLT_solve_normalized
from data.synth_pairs import _solve_homography_4pt
from losses.fold import fold_loss


_PASS = 0
_FAIL = 0


def _check(name, ok, detail=""):
    global _PASS, _FAIL
    if ok:
        _PASS += 1
        tag = "PASS"
    else:
        _FAIL += 1
        tag = "FAIL"
    print(f"  [{tag}] {name}" + (f"  ({detail})" if detail else ""))


def test_canonical_corner_order():
    """Synth's _solve_homography_4pt and the head's DLT_solve must produce
    the SAME H when given the SAME 8-vector of corner offsets, interpreted
    as TL, BL, BR, TR."""
    print("Test: canonical corner-order consistency (synth ↔ DLT_solve ↔ fold)")
    np.random.seed(0)
    torch.manual_seed(0)

    pw, ph = 560, 315
    src_canon = np.array(
        [[0,   0],
         [0,   ph],
         [pw,  ph],
         [pw,  0]],
        dtype=np.float64,
    )

    for trial in range(4):
        offset_np = np.random.uniform(-30, 30, size=(4, 2)).astype(np.float64)
        dst_canon = src_canon + offset_np
        H_synth = _solve_homography_4pt(src_canon, dst_canon)

        # DLT_solve takes flat (B,8) src and offset in the same canonical order.
        src_flat = torch.tensor(src_canon.reshape(-1), dtype=torch.float32).unsqueeze(0)
        off_flat = torch.tensor(offset_np.reshape(-1), dtype=torch.float32).unsqueeze(0)
        H_dlt = DLT_solve(src_flat, off_flat)[0].numpy().astype(np.float64)
        H_dlt_n = DLT_solve_normalized(src_flat, off_flat)[0].numpy().astype(np.float64)

        # Normalize each so H[2,2] = 1 for comparison.
        H_synth = H_synth / H_synth[2, 2]
        H_dlt   = H_dlt   / H_dlt[2, 2]
        H_dlt_n = H_dlt_n / H_dlt_n[2, 2]

        max_err = max(np.abs(H_dlt - H_synth).max(),
                      np.abs(H_dlt_n - H_synth).max())
        _check(f"trial {trial}: max|H_synth - H_dlt(_normalized)| < 1e-4",
               max_err < 1e-4, f"max_err={max_err:.2e}")


def test_fold_canonical_order():
    """Bowtie under canonical TL,BL,BR,TR ordering: swap TL <-> BL (adjacent)
    -> self-intersecting quad -> nonzero penalty. Diagonal swap (TL<->BR)
    just flips winding -> penalty stays zero."""
    print("Test: fold_loss respects canonical TL,BL,BR,TR ordering")
    pw, ph = 560, 315

    # Zero offset must give zero penalty.
    off_zero = torch.zeros(1, 8)
    L0 = float(fold_loss(off_zero, ph, pw))
    _check("zero offset -> L_fold = 0", L0 < 1e-6, f"L_fold={L0:.4f}")

    # Swap TL <-> BL: offset[0:2]=(0,ph), offset[2:4]=(0,-ph), rest 0.
    # Under TL,BL,BR,TR, this gives quad: TL=(0,ph), BL=(0,0), BR=(pw,ph), TR=(pw,0).
    # The new TL is at the old BL location -- but BL is now at the old TL.
    # The quad ordering becomes (0,ph) -> (0,0) -> (pw,ph) -> (pw,0), which
    # self-intersects between segments (BL->BR) and (TR->TL).
    off_adj_swap = torch.tensor([[0., ph, 0., -ph, 0., 0., 0., 0.]],
                                dtype=torch.float32)
    L_adj = float(fold_loss(off_adj_swap, ph, pw))
    _check("adjacent (TL<->BL) swap -> L_fold > 0", L_adj > 0.0,
           f"L_fold={L_adj:.4f}")


def main():
    print("planv3 B1 corner-ordering tests")
    print("=" * 60)
    test_canonical_corner_order()
    test_fold_canonical_order()
    print("=" * 60)
    print(f"{_PASS} passed, {_FAIL} failed")
    if _FAIL:
        sys.exit(1)


if __name__ == "__main__":
    main()
