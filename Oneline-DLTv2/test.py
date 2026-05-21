"""Legacy v1-style point-reprojection test. Prints per-category L2 in the
same RE/LT/LL/SF/LF format as v1's test.py so numbers are directly
comparable. For the full trustworthiness protocol use eval.py."""

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
from train import patch_to_full_homography
from eval import _load_state


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
        homography_rho=cfg.homography_rho,
        post_init_prob=cfg.post_init_prob,
        log_sigma_min=cfg.log_sigma_min,
        log_sigma_max=cfg.log_sigma_max,
    ).to(device)
    _load_state(net, ckpt_path)
    net.eval()

    ds = TestDataset(data_root=_REPO_ROOT,
                     patch_h=cfg.patch_h, patch_w=cfg.patch_w,
                     img_h=cfg.img_h, img_w=cfg.img_w)
    loader = DataLoader(ds, batch_size=1, num_workers=0, shuffle=False)

    bucket = {"RE": [], "LT": [], "LL": [], "SF": [], "LF": []}
    for i, batch in enumerate(loader):
        I_a = batch["I_a_patch"].to(device)
        I_b = batch["I_b_patch"].to(device)
        crop_xy = batch["crop_xy"].to(device)
        out = net(I_a, I_b)
        H_full = patch_to_full_homography(out["H_ab"], crop_xy)

        # v1 hardcodes the first 6 manual correspondences per pair
        # (test.py:169 `for j in range(6)`). Match it so per-pair errors
        # are directly comparable across the two codebases.
        pts = batch["points"].to(device)[:, :6, :, :]
        pts_a = pts[:, :, 0, :]
        pts_b = pts[:, :, 1, :]
        # v1 takes min(err_LR, err_RL) PER POINT, then averages — the
        # annotator's (A, B) ordering can flip independently per point.
        # Mean-then-min (the previous v2 form) is a different metric and
        # underestimates error on pairs with mixed annotation order.
        err_ab = point_reprojection_error(H_full, pts_a, pts_b)         # (B, 6)
        err_ba = point_reprojection_error(H_full, pts_b, pts_a)         # (B, 6)
        err_per_point = torch.minimum(err_ab, err_ba)                   # (B, 6)
        err = float(err_per_point.mean(dim=1).item())

        scene = batch["scene"][0]
        if scene in bucket:
            bucket[scene].append(err)
        print(f"{i:08d} [{scene}]: {err:.4f}")

    res = {k: float(np.mean(v)) if v else float("nan") for k, v in bucket.items()}
    res["Avg"] = float(np.mean([res[k] for k in ("RE", "LT", "LL", "SF", "LF")
                                if not np.isnan(res[k])]))
    print()
    print("=" * 50)
    for k, v in res.items():
        print(f"  {k}: {v:.4f}")
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    args = p.parse_args()
    test(args.ckpt)


if __name__ == "__main__":
    main()
