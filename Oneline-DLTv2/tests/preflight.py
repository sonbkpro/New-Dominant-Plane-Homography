"""Stage 0 preflight driver (planv3 §6).

Runs every test in tests/test_*.py + a CDPCNet forward smoke test.

    python tests/preflight.py
    python tests/preflight.py --long       # also runs identity-overfit (slow)
    python tests/preflight.py --no-data    # skip identity-H baseline (no real data)
"""

import argparse
import os
import subprocess
import sys

import numpy as np
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_DIR = os.path.dirname(_THIS_DIR)
_REPO_ROOT = os.path.dirname(_REPO_DIR)
sys.path.insert(0, _REPO_DIR)

from configs.default import Config
from model.cdpc_net import CDPCNet
from utils.eval_metrics import point_reprojection_error


_FAILS = []


def _check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        _FAILS.append(name)


def _run_test_script(rel_path: str) -> bool:
    """Run a sibling test_*.py script in a subprocess and return True if it exits 0."""
    script = os.path.join(_THIS_DIR, rel_path)
    r = subprocess.run([sys.executable, script], capture_output=True, text=True)
    print(r.stdout, end="" if r.stdout.endswith("\n") else "\n")
    if r.returncode != 0:
        if r.stderr:
            print(r.stderr, end="")
        return False
    return True


def test_cdpcnet_forward():
    """Smoke test: CDPCNet v3 forward produces sane shapes and no NaNs."""
    print("Test: CDPCNet v3 forward smoke")
    cfg = Config()
    cfg.patch_h, cfg.patch_w = 128, 160
    cfg.img_h,   cfg.img_w   = 192, 256
    B = 2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = CDPCNet(
        patch_h=cfg.patch_h, patch_w=cfg.patch_w,
        backbone_pretrained=False,
        corr_radius=2, corr_out_channels=8,
        bb_quarter_channels=32, bb_eighth_channels=64,
        bb_sixteenth_channels=128,
        rho_per_level=(16.0, 8.0, 4.0),
        homography_levels=3,
        post_init_prob=cfg.post_init_prob,
        sigma_min=cfg.sigma_min,
        use_normalized_dlt=cfg.use_normalized_dlt,
    ).to(device)
    I_a_full = torch.randn(B, 1, cfg.img_h, cfg.img_w, device=device)
    I_b_full = torch.randn(B, 1, cfg.img_h, cfg.img_w, device=device)
    x0, y0 = 8, 8
    I_a_patch = I_a_full[:, :, y0:y0+cfg.patch_h, x0:x0+cfg.patch_w].contiguous()
    I_b_patch = I_b_full[:, :, y0:y0+cfg.patch_h, x0:x0+cfg.patch_w].contiguous()
    crop_xy = torch.tensor([[x0, y0]] * B, dtype=torch.float32, device=device)

    out = net(I_a_full, I_b_full, I_a_patch, I_b_patch, crop_xy)
    nan_any = any(
        bool(v.isnan().any())
        for v in out.values()
        if isinstance(v, torch.Tensor)
    )
    _check("forward produces no NaNs", not nan_any, "")
    max_off = float(out["offset"].abs().max())
    rho_sum = sum((16.0, 8.0, 4.0))
    _check("offset within total rho budget",
           max_off <= rho_sum + 1e-2,
           f"max|off|={max_off:.2f} <= sum_rho={rho_sum}")
    _check("phi has 10 dims", out["phi"].shape[1] == 10,
           f"phi shape={tuple(out['phi'].shape)}")
    _check("per_level_offset has homography_levels entries",
           len(out["per_level_offset"]) == 3,
           f"got {len(out['per_level_offset'])}")


def test_identity_baseline(repo_root: str):
    """Identity-H reprojection baseline on the real test set (skippable)."""
    print("Test: identity-H baseline on real test set")
    from data.test_dataset import TestDataset
    cfg = Config()
    try:
        ds = TestDataset(data_root=repo_root,
                         patch_h=cfg.patch_h, patch_w=cfg.patch_w,
                         img_h=cfg.img_h, img_w=cfg.img_w)
    except FileNotFoundError as e:
        _check("TestDataset available", False, f"{e}")
        return
    errs, scenes = [], []
    SCENES = ("RE", "LT", "LL", "SF", "LF")
    n_skipped = 0
    for i in range(len(ds)):
        try:
            batch = ds[i]
        except FileNotFoundError:
            n_skipped += 1
            continue
        pts = batch["points"].unsqueeze(0)[:, :6, :, :]
        pts_a = pts[:, :, 0, :]
        pts_b = pts[:, :, 1, :]
        I = torch.eye(3).unsqueeze(0)
        e = point_reprojection_error(I, pts_a, pts_b).mean(dim=1)
        errs.append(float(e.item()))
        scenes.append(batch["scene"])
    if n_skipped > 0:
        print(f"        [warn] skipped {n_skipped}/{len(ds)} pairs (missing .npy)")
    errs = np.array(errs)
    scenes = np.array(scenes)
    per_scene = {s: float(errs[scenes == s].mean()) if (scenes == s).any() else float("nan")
                 for s in SCENES}
    overall = float(np.mean([v for v in per_scene.values() if not np.isnan(v)]))
    print("        identity-H reprojection (px): " +
          "  ".join(f"{s}={v:.2f}" for s, v in per_scene.items()) +
          f"  Avg={overall:.2f}")
    _check("identity-H baseline reads cleanly", np.isfinite(overall), "")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--long", action="store_true")
    p.add_argument("--no-data", action="store_true")
    p.add_argument("--repo_root", type=str, default=_REPO_ROOT)
    args = p.parse_args()
    print(f"planv3 preflight driver  (repo_root={args.repo_root})")
    print("=" * 60)

    # B1: corner ordering MUST pass before anything else.
    ok = _run_test_script("test_corner_ordering.py")
    _check("tests/test_corner_ordering.py exited 0", ok)

    ok = _run_test_script("test_dlt_identity.py")
    _check("tests/test_dlt_identity.py exited 0", ok)

    ok = _run_test_script("test_warp_translation.py")
    _check("tests/test_warp_translation.py exited 0", ok)

    ok = _run_test_script("test_eval_protocol.py")
    _check("tests/test_eval_protocol.py exited 0", ok)

    test_cdpcnet_forward()
    if not args.no_data:
        test_identity_baseline(args.repo_root)
    if args.long:
        ok = _run_test_script("test_identity_overfit.py")
        _check("tests/test_identity_overfit.py exited 0", ok)

    print()
    print("=" * 60)
    if _FAILS:
        print(f"FAILED ({len(_FAILS)}):")
        for n in _FAILS:
            print(f"  - {n}")
        sys.exit(1)
    print("All preflight tests passed.")


if __name__ == "__main__":
    main()
