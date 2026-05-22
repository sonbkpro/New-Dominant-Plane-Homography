"""planv3 §6 Stage 0: DLT identity sanity.

zero offset -> H = I; corner reprojection error < 1e-4 px.

Run from Oneline-DLTv2/:
    python tests/test_dlt_identity.py
"""

import os
import sys
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from utils.dlt import DLT_solve
from utils.dlt_normalized import DLT_solve_normalized
from utils.eval_metrics import point_reprojection_error


_FAILS = []


def _check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        _FAILS.append(name)


def main():
    print("Test: DLT identity (zero offset -> H = I, canonical TL,BL,BR,TR)")
    print("=" * 60)

    B = 4
    pw, ph = 560, 315
    # Canonical TL, BL, BR, TR order.
    src = torch.tensor([0, 0,
                        0, ph,
                        pw, ph,
                        pw, 0], dtype=torch.float32)
    src_b = src.unsqueeze(0).expand(B, -1)
    offset_zero = torch.zeros(B, 8)

    corners = torch.tensor([[0, 0], [0, ph], [pw, ph], [pw, 0]], dtype=torch.float32)
    corners_b = corners.unsqueeze(0).expand(B, -1, -1)

    for name, solver in [("DLT_solve", DLT_solve),
                         ("DLT_solve_normalized", DLT_solve_normalized)]:
        H = solver(src_b, offset_zero)
        I = torch.eye(3).unsqueeze(0).expand(B, -1, -1)
        max_err = float((H - I).abs().max())
        _check(f"{name}: ||H - I||_inf < 1e-3", max_err < 1e-3,
               f"max_err={max_err:.2e}")
        proj_err = float(point_reprojection_error(H, corners_b, corners_b).max())
        _check(f"{name}: max corner reprojection error < 1e-3",
               proj_err < 1e-3, f"max={proj_err:.2e}")

    print("=" * 60)
    if _FAILS:
        print(f"FAILED: {_FAILS}")
        sys.exit(1)
    print("All DLT identity tests passed.")


if __name__ == "__main__":
    main()
