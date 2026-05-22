"""Stage 0 preflight tests (planv2 §3 Phase 0).

Run from the Oneline-DLTv2/ directory:

    python tests/preflight.py
    python tests/preflight.py --long        # also runs overfit tests
    python tests/preflight.py --no-data     # skip identity-H baseline (no real data)

Each test prints PASS / FAIL with the relevant numbers. The script exits with
non-zero status if any test fails so it can be wired into CI."""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_DIR = os.path.dirname(_THIS_DIR)
_REPO_ROOT = os.path.dirname(_REPO_DIR)
sys.path.insert(0, _REPO_DIR)

from configs.default import Config
from model.cdpc_net import CDPCNet
from utils.dlt import DLT_solve
from utils.dlt_normalized import DLT_solve_normalized
from utils.warping import warp_by_homography, make_validity_mask
from utils.inverse import safe_inverse_3x3
from utils.eval_metrics import point_reprojection_error
from losses.fold import fold_loss


_FAILS = []


def _check(name: str, ok: bool, detail: str = ""):
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        _FAILS.append(name)


# ---------------------------------------------------------------------------
# Test 1: DLT identity
# ---------------------------------------------------------------------------

def test_dlt_identity():
    print("Test 1: DLT identity (zero offset -> H = I)")
    B = 4
    pw, ph = 560, 315
    src = torch.tensor([0, 0, pw, 0, pw, ph, 0, ph], dtype=torch.float32)
    src_b = src.unsqueeze(0).expand(B, -1)
    offset_zero = torch.zeros(B, 8)

    for name, solver in [("DLT_solve", DLT_solve),
                         ("DLT_solve_normalized", DLT_solve_normalized)]:
        H = solver(src_b, offset_zero)
        I = torch.eye(3).unsqueeze(0).expand(B, -1, -1)
        max_err = float((H - I).abs().max())
        _check(f"{name}: ||H - I||_inf < 1e-3", max_err < 1e-3, f"max_err={max_err:.2e}")

        # And that corners are exactly fixed under H.
        corners = torch.tensor([[0, 0], [pw, 0], [pw, ph], [0, ph]], dtype=torch.float32)
        corners_b = corners.unsqueeze(0).expand(B, -1, -1)
        proj_err = point_reprojection_error(H, corners_b, corners_b).max().item()
        _check(f"{name}: max corner reprojection error < 1e-3",
               proj_err < 1e-3, f"max={proj_err:.2e}")


# ---------------------------------------------------------------------------
# Test 2: Known-translation DLT + warp
# ---------------------------------------------------------------------------

def test_known_translation_warp():
    print("Test 2: known-translation DLT + warp")
    B = 2
    pw, ph = 80, 60
    tx, ty = 10.0, 5.0

    src = torch.tensor([0, 0, pw, 0, pw, ph, 0, ph], dtype=torch.float32)
    src_b = src.unsqueeze(0).expand(B, -1)
    offset = torch.tensor([tx, ty, tx, ty, tx, ty, tx, ty], dtype=torch.float32).unsqueeze(0).expand(B, -1)
    H = DLT_solve_normalized(src_b, offset)

    # Sanity: H should be a pure translation matrix.
    T_expected = torch.tensor([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=torch.float32)
    err = float((H[0] - T_expected).abs().max())
    _check("DLT recovers pure translation", err < 1e-3, f"err={err:.2e}")

    # Make a known pattern and warp it. With H = T(+tx,+ty) and the convention
    # "H maps source to destination", a source pixel at (x,y) lands at
    # (x+tx, y+ty) in destination. warp_by_homography samples destination
    # output, so the IMPULSE moves to (+tx, +ty). For an impulse originally
    # at (y=20, x=30), the warped peak is at (y=20+ty=25, x=30+tx=40).
    img = torch.zeros(B, 1, ph, pw)
    img[:, :, 20, 30] = 1.0
    warped = warp_by_homography(img, H, out_size=(ph, pw),
                                out_origin_xy=(0, 0), padding_mode="zeros")
    y_peak, x_peak = (warped[0, 0] > 0.5).nonzero(as_tuple=False)[0].tolist()
    _check("warp translates impulse by (+tx,+ty)",
           y_peak == 25 and x_peak == 40, f"peak=({y_peak}, {x_peak})")


# ---------------------------------------------------------------------------
# Test 3: warp round-trip
# ---------------------------------------------------------------------------

def test_warp_roundtrip():
    print("Test 3: warp round-trip with H and H^{-1}")
    torch.manual_seed(0)
    B = 2
    pw, ph = 80, 60
    src = torch.tensor([0, 0, pw, 0, pw, ph, 0, ph], dtype=torch.float32)
    src_b = src.unsqueeze(0).expand(B, -1)
    offset = torch.tensor([2., 1., -1., 2., 1., -2., -2., -1.]).unsqueeze(0).expand(B, -1)
    H = DLT_solve_normalized(src_b, offset)
    H_inv, _ = safe_inverse_3x3(H)

    # A smooth image so bilinear interpolation introduces less noise.
    yy, xx = torch.meshgrid(
        torch.arange(ph, dtype=torch.float32),
        torch.arange(pw, dtype=torch.float32),
        indexing="ij",
    )
    img = (torch.sin(0.1 * xx) + torch.cos(0.07 * yy)).unsqueeze(0).unsqueeze(0)
    img = img.expand(B, 1, -1, -1).contiguous()

    warped = warp_by_homography(img, H, (ph, pw), (0, 0), padding_mode="border")
    recovered = warp_by_homography(warped, H_inv, (ph, pw), (0, 0), padding_mode="border")

    # Only score the interior so we don't compare against extrapolated border.
    crop = (slice(None), slice(None), slice(5, -5), slice(5, -5))
    rel_err = float((img[crop] - recovered[crop]).abs().mean()
                    / img[crop].abs().mean().clamp(min=1e-6))
    _check("interior cycle error < 5% of signal scale",
           rel_err < 0.05, f"rel_err={rel_err:.3%}")


# ---------------------------------------------------------------------------
# Test 4: fold-over penalty sanity
# ---------------------------------------------------------------------------

def test_fold_penalty():
    print("Test 4: fold-over penalty sanity")
    pw, ph = 560, 315
    # Zero offset: convex, penalty should be ~0.
    off_zero = torch.zeros(1, 8)
    L0 = float(fold_loss(off_zero, ph, pw))
    _check("convex (zero offset) has zero fold-over", L0 < 1e-6, f"L_fold={L0:.4f}")

    # True bowtie: swap two ADJACENT corners (TL <-> TR) so the quad
    # self-intersects. (Swapping diagonals TL <-> BR just reverses winding,
    # which is convex and correctly not penalized.)
    off_flip = torch.tensor([[pw, 0,  -pw, 0,  0, 0,  0, 0]], dtype=torch.float32)
    Lf = float(fold_loss(off_flip, ph, pw))
    _check("self-intersecting quad has nonzero fold-over", Lf > 0.0,
           f"L_fold={Lf:.4f}")


# ---------------------------------------------------------------------------
# Test 5: eval convention sanity (planv2 §2.5)
# ---------------------------------------------------------------------------

def test_eval_convention():
    print("Test 5: eval convention agreement on a known H*")
    torch.manual_seed(0)
    B = 4

    # Generate a known small-perturbation homography in patch coords.
    pw, ph = 560, 315
    src = torch.tensor([0, 0, pw, 0, pw, ph, 0, ph], dtype=torch.float32)
    src_b = src.unsqueeze(0).expand(B, -1)
    offset_gt = torch.tensor([3., -2., 1., 4., -2., 1., -1., -3.]).unsqueeze(0).expand(B, -1)
    H = DLT_solve_normalized(src_b, offset_gt)
    H_inv, _ = safe_inverse_3x3(H)

    # Choose a set of "correspondences" by picking p_a, computing p_b = H p_a.
    K = 8
    p_a = torch.rand(B, K, 2) * torch.tensor([pw, ph], dtype=torch.float32)
    p_a_h = torch.cat([p_a, torch.ones(B, K, 1)], dim=-1)
    p_b_h = torch.bmm(p_a_h, H.transpose(1, 2))
    p_b = p_b_h[..., :2] / p_b_h[..., 2:3]

    err_direct = float(point_reprojection_error(H, p_a, p_b).max())
    err_inverse = float(point_reprojection_error(H_inv, p_b, p_a).max())
    err_id = float(point_reprojection_error(
        torch.eye(3).unsqueeze(0).expand(B, -1, -1), p_a, p_b).mean())

    _check("direct error ~0 on H @ p_a == p_b",   err_direct < 1e-3,
           f"max={err_direct:.2e}")
    # The inverse path goes through fp32 matrix inversion of H, so the round
    # trip H^-1 H carries fp32-eps * cond(H) * coord-scale error. On a 560-px
    # patch that's ~0.01 px in pixel-space tolerance.
    _check("inverse error ~0 on H^-1 @ p_b == p_a", err_inverse < 0.1,
           f"max={err_inverse:.2e}")
    # Identity baseline should be nonzero (the homography really moves points).
    _check("identity-H baseline error > direct error", err_id > err_direct,
           f"identity={err_id:.2f} > direct={err_direct:.2e}")


# ---------------------------------------------------------------------------
# Test 6: identity-H baseline on real test set (skippable)
# ---------------------------------------------------------------------------

def test_identity_baseline(repo_root: str):
    print("Test 6: identity-H baseline on real test set")
    from data.test_dataset import TestDataset
    from torch.utils.data import DataLoader
    cfg = Config()
    try:
        ds = TestDataset(data_root=repo_root,
                         patch_h=cfg.patch_h, patch_w=cfg.patch_w,
                         img_h=cfg.img_h, img_w=cfg.img_w)
    except FileNotFoundError as e:
        _check("TestDataset available", False, f"{e}")
        return
    loader = DataLoader(ds, batch_size=1, num_workers=0, shuffle=False)

    errs = []
    scenes = []
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
        I = torch.eye(3).unsqueeze(0).expand(1, -1, -1)
        e = point_reprojection_error(I, pts_a, pts_b).mean(dim=1)
        errs.append(float(e.item()))
        scenes.append(batch["scene"])
    if n_skipped > 0:
        print(f"        [warn] skipped {n_skipped}/{len(ds)} pairs with missing .npy files")

    errs = np.array(errs)
    scenes = np.array(scenes)
    per_scene_mean = {}
    for s in SCENES:
        m = scenes == s
        per_scene_mean[s] = float(errs[m].mean()) if m.any() else float("nan")
    overall = float(np.mean([v for v in per_scene_mean.values() if not np.isnan(v)]))
    detail = "  ".join(f"{s}={v:.2f}" for s, v in per_scene_mean.items()) + f"  Avg={overall:.2f}"
    print(f"        identity-H reprojection (px): {detail}")
    _check("identity-H baseline reads cleanly", np.isfinite(overall), "")


# ---------------------------------------------------------------------------
# Test 7: CDPCNet forward smoke
# ---------------------------------------------------------------------------

def test_cdpcnet_forward():
    print("Test 7: CDPCNet forward (full-image warp signature)")
    cfg = Config()
    cfg.patch_h, cfg.patch_w = 128, 160
    cfg.img_h,   cfg.img_w   = 144, 192
    B = 2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = CDPCNet(
        patch_h=cfg.patch_h, patch_w=cfg.patch_w,
        backbone_pretrained=False,
        corr_radius=2, corr_out_channels=8,
        bb_quarter_channels=32, bb_eighth_channels=64,
        homography_rho=cfg.homography_rho,
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
    nan_any = any(bool(v.isnan().any()) for v in out.values() if isinstance(v, torch.Tensor))
    _check("forward produces no NaNs", not nan_any, "")
    _check("offset is bounded by rho", float(out["offset"].abs().max()) <= cfg.homography_rho + 1e-3,
           f"max|offset|={float(out['offset'].abs().max()):.2f} <= rho={cfg.homography_rho}")
    _check("phi has 10 dims", out["phi"].shape[1] == 10,
           f"phi shape={tuple(out['phi'].shape)}")


# ---------------------------------------------------------------------------
# Test 8 (long): synthetic-H supervised overfit on a mini-batch
# ---------------------------------------------------------------------------

def test_synthetic_overfit():
    print("Test 8 (long): synthetic-H supervised overfit on a mini-batch")
    import cv2
    torch.manual_seed(0)
    np.random.seed(0)
    cfg = Config()
    cfg.patch_h, cfg.patch_w = 128, 160
    cfg.img_h,   cfg.img_w   = 192, 256
    B = 4

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = CDPCNet(
        patch_h=cfg.patch_h, patch_w=cfg.patch_w,
        backbone_pretrained=False,
        corr_radius=2, corr_out_channels=8,
        bb_quarter_channels=32, bb_eighth_channels=64,
        homography_rho=cfg.homography_rho,
        post_init_prob=cfg.post_init_prob,
        sigma_min=cfg.sigma_min,
        use_normalized_dlt=cfg.use_normalized_dlt,
    ).to(device)

    pw, ph = cfg.patch_w, cfg.patch_h
    x0, y0 = 16, 16

    # B independent random source images so the H head can DISTINGUISH the
    # samples (otherwise identical I_a + per-sample offset_gt is unlearnable).
    rng = np.random.RandomState(0)
    yy, xx = np.meshgrid(np.arange(cfg.img_h, dtype=np.float32),
                         np.arange(cfg.img_w, dtype=np.float32),
                         indexing="ij")
    src_corners = np.array([[0, 0], [pw, 0], [pw, ph], [0, ph]], dtype=np.float32)

    offset_gt_np = rng.uniform(-5, 5, size=(B, 4, 2)).astype(np.float32)
    I_a_full_list, I_b_full_list = [], []
    for b in range(B):
        # Distinct sinusoidal mixture so each sample's appearance differs.
        a_phi = rng.uniform(0, 2 * np.pi, size=2)
        I_a = (np.sin(0.07 * xx + a_phi[0]) +
               np.cos(0.05 * yy + a_phi[1])).astype(np.float32)
        # Build the patch-coord H and conjugate to full-image coords.
        dst_corners = src_corners + offset_gt_np[b]
        H_patch = cv2.getPerspectiveTransform(src_corners, dst_corners)
        T = np.array([[1.0, 0.0, x0], [0.0, 1.0, y0], [0.0, 0.0, 1.0]])
        T_inv = np.array([[1.0, 0.0, -x0], [0.0, 1.0, -y0], [0.0, 0.0, 1.0]])
        H_full = T @ H_patch @ T_inv
        I_b = cv2.warpPerspective(I_a, H_full, dsize=(cfg.img_w, cfg.img_h),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE)
        I_a_full_list.append(I_a)
        I_b_full_list.append(I_b)

    I_a_full = torch.from_numpy(np.stack(I_a_full_list)).unsqueeze(1).to(device)
    I_b_full = torch.from_numpy(np.stack(I_b_full_list)).unsqueeze(1).to(device)
    I_a_patch = I_a_full[:, :, y0:y0+ph, x0:x0+pw].contiguous()
    I_b_patch = I_b_full[:, :, y0:y0+ph, x0:x0+pw].contiguous()
    crop_xy = torch.tensor([[x0, y0]] * B, dtype=torch.float32, device=device)
    offset_gt = torch.from_numpy(offset_gt_np.reshape(B, 8)).to(device)

    params = list(net.joint_backbone.parameters()) + list(net.homography_head.parameters())
    opt = torch.optim.Adam(params, lr=5e-4)

    last_err = None
    for it in range(400):
        out = net(I_a_full, I_b_full, I_a_patch, I_b_patch, crop_xy)
        L = F.smooth_l1_loss(out["offset"], offset_gt)
        opt.zero_grad(set_to_none=True)
        L.backward()
        opt.step()
        if (it + 1) % 100 == 0:
            err = float((out["offset"] - offset_gt).abs().max())
            print(f"        iter {it+1:4d}  L={float(L):.4f}  max_corner_err={err:.3f} px")
            last_err = err
    _check("max corner error < 2 px after 400 supervised iters",
           last_err is not None and last_err < 2.0,
           f"max_err={last_err}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--long", action="store_true",
                   help="Also run the supervised-overfit test (slower).")
    p.add_argument("--no-data", action="store_true",
                   help="Skip the identity-H baseline test (no real test set).")
    p.add_argument("--repo_root", type=str, default=_REPO_ROOT)
    args = p.parse_args()

    print(f"Preflight tests (repo_root={args.repo_root})")
    print("=" * 60)

    test_dlt_identity()
    test_known_translation_warp()
    test_warp_roundtrip()
    test_fold_penalty()
    test_eval_convention()
    test_cdpcnet_forward()
    if not args.no_data:
        test_identity_baseline(args.repo_root)
    if args.long:
        test_synthetic_overfit()

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
