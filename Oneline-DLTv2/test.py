"""Point-reprojection test, planv3 §6: six metrics per pair, averaged
per scene (RE/LT/LL/SF/LF) and overall:

  direct     : || H * p_a - p_b ||                         per-point, then mean
  inverse    : || H^-1 * p_b - p_a ||                      per-point, then mean
  symmetric  : 0.5 * (direct + inverse)                    per-point, then mean
  identity   : || p_a - p_b ||         (H = I baseline)    per-point, then mean
  v1         : min(|| H p_a - p_b ||, || H p_b - p_a ||)   legacy (UNINVERTED)
  v1_compat  : min(|| H^-1 p_b - p_a ||, || H^-1 p_a - p_b ||)
                 planv3 B2: the actual v1-equivalent column (v1's test.py
                 INVERTS the model's H_mat before measuring error and then
                 takes a per-point min over the two correspondence orderings,
                 because the dataset's `matche_pts` is not consistently
                 oriented). The legacy `v1` row above does NOT invert and is
                 generally NOT comparable to v1's headline number.
"""

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.path.pardir))
sys.path.insert(0, _THIS_DIR)

from configs.default import Config
from model.cdpc_net import CDPCNet
from data.test_dataset import TestDataset
from utils.eval_metrics import point_reprojection_error
from utils.inverse import safe_inverse_3x3
from train import patch_to_full_homography
from eval import _load_state


SCENES = ("RE", "LT", "LL", "SF", "LF")
METRICS = ("direct", "inverse", "symmetric", "identity", "v1", "v1_compat")


def _eye_like(H: torch.Tensor) -> torch.Tensor:
    B = H.shape[0]
    I = torch.eye(3, device=H.device, dtype=H.dtype)
    return I.unsqueeze(0).expand(B, -1, -1).contiguous()


@torch.no_grad()
def test(ckpt_path: str):
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = CDPCNet(
        patch_h=cfg.patch_h, patch_w=cfg.patch_w,
        backbone_pretrained=False,
        corr_radius=cfg.corr_radius,
        corr_out_channels=cfg.corr_out_channels,
        bb_quarter_channels=cfg.bb_quarter_channels,
        bb_eighth_channels=cfg.bb_eighth_channels,
        bb_sixteenth_channels=cfg.bb_sixteenth_channels,
        rho_per_level=cfg.rho_per_level,
        homography_levels=cfg.homography_levels,
        post_init_prob=cfg.post_init_prob,
        sigma_min=cfg.sigma_min,
        use_normalized_dlt=cfg.use_normalized_dlt,
    ).to(device)
    _load_state(net, ckpt_path)
    net.eval()

    ds = TestDataset(data_root=_REPO_ROOT,
                     patch_h=cfg.patch_h, patch_w=cfg.patch_w,
                     img_h=cfg.img_h, img_w=cfg.img_w)
    loader = DataLoader(ds, batch_size=1, num_workers=0, shuffle=False)

    buckets = {m: {s: [] for s in SCENES} for m in METRICS}
    n_illcond = 0

    header = f"{'idx':>8s} [{'sc':>2s}]  " + "  ".join(f"{m:>9s}" for m in METRICS)
    print(header)
    print("-" * len(header))

    for i, batch in enumerate(loader):
        I_a_full  = batch["I_a_full"].to(device)
        I_b_full  = batch["I_b_full"].to(device)
        I_a_patch = batch["I_a_patch"].to(device)
        I_b_patch = batch["I_b_patch"].to(device)
        crop_xy   = batch["crop_xy"].to(device)
        out = net(I_a_full, I_b_full, I_a_patch, I_b_patch, crop_xy)

        H_full = patch_to_full_homography(out["H_ab"], crop_xy)
        H_full_inv, cond_valid = safe_inverse_3x3(H_full)
        if float(cond_valid.item()) < 0.5:
            n_illcond += 1

        pts = batch["points"].to(device)[:, :6, :, :]
        pts_a = pts[:, :, 0, :]
        pts_b = pts[:, :, 1, :]

        err_ab      = point_reprojection_error(H_full,            pts_a, pts_b)
        err_ba      = point_reprojection_error(H_full,            pts_b, pts_a)
        err_inv     = point_reprojection_error(H_full_inv,        pts_b, pts_a)
        err_inv_alt = point_reprojection_error(H_full_inv,        pts_a, pts_b)
        err_id      = point_reprojection_error(_eye_like(H_full), pts_a, pts_b)

        per_pair = {
            "direct":    float(err_ab.mean(dim=1).item()),
            "inverse":   float(err_inv.mean(dim=1).item()),
            "symmetric": float(((err_ab + err_inv) / 2.0).mean(dim=1).item()),
            "identity":  float(err_id.mean(dim=1).item()),
            "v1":        float(torch.minimum(err_ab, err_ba).mean(dim=1).item()),
            "v1_compat": float(torch.minimum(err_inv, err_inv_alt).mean(dim=1).item()),
        }

        scene = batch["scene"][0]
        if scene in SCENES:
            for m in METRICS:
                buckets[m][scene].append(per_pair[m])

        print(f"{i:08d} [{scene:>2s}]  " +
              "  ".join(f"{per_pair[m]:9.4f}" for m in METRICS))

    print()
    print("=" * 80)
    print(f"{'metric':>10s}  " +
          "  ".join(f"{s:>9s}" for s in SCENES + ("Avg",)))
    print("-" * 80)

    out_dict = {}
    for m in METRICS:
        per_scene = {s: float(np.mean(buckets[m][s])) if buckets[m][s] else float("nan")
                     for s in SCENES}
        scene_vals = [per_scene[s] for s in SCENES if not np.isnan(per_scene[s])]
        avg = float(np.mean(scene_vals)) if scene_vals else float("nan")
        print(f"{m:>10s}  " +
              "  ".join(f"{per_scene[s]:9.4f}" for s in SCENES) +
              f"  {avg:9.4f}")
        out_dict[m] = {**per_scene, "Avg": avg}

    if n_illcond > 0:
        print()
        print(f"[warn] {n_illcond} pair(s) had ill-conditioned H (cond > 1e4); "
              f"H_inv fell back to identity for those, so the `inverse` and "
              f"`symmetric` rows include identity-H behavior on those pairs.")

    return out_dict


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    args = p.parse_args()
    test(args.ckpt)


if __name__ == "__main__":
    main()
