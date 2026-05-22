"""planv3 §6 Stage 0: known-translation warp test.

Set Delta_p = (tx, ty) for all 4 corners; expect H = translation matrix and
warp_by_homography to move a checkerboard pattern by exactly (tx, ty).

Run from Oneline-DLTv2/:
    python tests/test_warp_translation.py
"""

import os
import sys
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from utils.dlt_normalized import DLT_solve_normalized
from utils.warping import warp_by_homography


_FAILS = []


def _check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        _FAILS.append(name)


def main():
    print("Test: known-translation DLT + warp")
    print("=" * 60)

    B = 2
    pw, ph = 80, 60
    tx, ty = 10.0, 5.0

    # Canonical TL, BL, BR, TR ordering.
    src = torch.tensor([0, 0, 0, ph, pw, ph, pw, 0], dtype=torch.float32)
    src_b = src.unsqueeze(0).expand(B, -1)
    offset = torch.tensor([tx, ty, tx, ty, tx, ty, tx, ty],
                          dtype=torch.float32).unsqueeze(0).expand(B, -1)
    H = DLT_solve_normalized(src_b, offset)

    T_expected = torch.tensor([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=torch.float32)
    err = float((H[0] - T_expected).abs().max())
    _check("DLT recovers pure translation matrix", err < 1e-3, f"err={err:.2e}")

    # Place an impulse at (y=20, x=30); after H = T(+tx,+ty), the warped peak
    # should be at (y+ty=25, x+tx=40) since warp_by_homography samples
    # destination output (i.e., it applies the forward warp).
    img = torch.zeros(B, 1, ph, pw)
    img[:, :, 20, 30] = 1.0
    warped = warp_by_homography(img, H, out_size=(ph, pw),
                                out_origin_xy=(0, 0), padding_mode="zeros")
    nz = (warped[0, 0] > 0.5).nonzero(as_tuple=False)
    if nz.shape[0] == 0:
        _check("impulse appears in warped image", False, "no peak found")
    else:
        y_peak, x_peak = nz[0].tolist()
        _check("warp translates impulse by (+tx,+ty)",
               y_peak == 25 and x_peak == 40,
               f"peak=({y_peak},{x_peak}) expected=(25,40)")

    print("=" * 60)
    if _FAILS:
        print(f"FAILED: {_FAILS}")
        sys.exit(1)
    print("All translation-warp tests passed.")


if __name__ == "__main__":
    main()
