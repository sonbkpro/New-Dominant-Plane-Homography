"""Full trustworthiness evaluation for CDPC v2.

Reports per-category and overall:
  - L2 reprojection: mean, median, AUC@{1,3,5}, inlier@{1,3,5}
  - failure detection: AUROC, AUPRC (failure = err > tau)
  - calibration: ECE, NLL of s_ab against the failure label
  - risk-coverage curve driven by reliability score s_ab
"""

import argparse
import json
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
from utils.eval_metrics import (
    point_reprojection_error, auc_at_thresholds, inlier_ratio_at_thresholds,
    auroc, auprc, expected_calibration_error, risk_coverage_curve,
)
from train import patch_to_full_homography


def _load_state(net, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    cleaned = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    missing, unexpected = net.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"[load] missing  ({len(missing)}): {missing[:6]}...")
    if unexpected:
        print(f"[load] unexpected ({len(unexpected)}): {unexpected[:6]}...")
    return net


@torch.no_grad()
def evaluate(ckpt_path: str, tau_list=(3.0, 5.0)) -> dict:
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

    test_ds = TestDataset(data_root=_REPO_ROOT,
                          patch_h=cfg.patch_h, patch_w=cfg.patch_w,
                          img_h=cfg.img_h, img_w=cfg.img_w)
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=0, shuffle=False)

    errs, scenes, reliabilities = [], [], []
    for batch in test_loader:
        I_a = batch["I_a_patch"].to(device)
        I_b = batch["I_b_patch"].to(device)
        crop_xy = batch["crop_xy"].to(device)
        out = net(I_a, I_b)
        H_full = patch_to_full_homography(out["H_ab"], crop_xy)
        # Match v1's protocol: first 6 correspondences only, and per-point
        # min over the (A,B) / (B,A) swap, then mean. Mean-then-min is a
        # different metric — and every downstream number here (AUC, inlier,
        # AUROC, AUPRC, ECE, NLL, risk-coverage) is built from this per-pair
        # err, so the v1↔v2 trustworthiness table is only comparable when
        # the per-pair scalar is computed v1's way.
        pts = batch["points"].to(device)[:, :6, :, :]   # (B, 6, 2, 2)
        pts_a = pts[:, :, 0, :]
        pts_b = pts[:, :, 1, :]
        err_ab = point_reprojection_error(H_full, pts_a, pts_b)         # (B, 6)
        err_ba = point_reprojection_error(H_full, pts_b, pts_a)         # (B, 6)
        err = torch.minimum(err_ab, err_ba).mean(dim=1)                 # (B,)
        for j in range(I_a.shape[0]):
            errs.append(float(err[j]))
            scenes.append(batch["scene"][j])
            reliabilities.append(float(out["s"][j]))

    errs = np.array(errs, dtype=np.float64)
    rel  = np.array(reliabilities, dtype=np.float64)
    scenes = np.array(scenes)

    def _per_scene(arr: np.ndarray):
        out = {"overall": float(arr.mean()) if len(arr) else float("nan")}
        for s in sorted(set(scenes)):
            mask = scenes == s
            out[s] = float(arr[mask].mean()) if mask.any() else float("nan")
        return out

    summary = {
        "n_pairs": int(len(errs)),
        "L2_mean":   _per_scene(errs),
        "L2_median": {k: float(np.median(errs[scenes == k]) if (scenes == k).any() else float("nan"))
                      for k in sorted(set(scenes))},
        "AUC@1":     auc_at_thresholds(errs, [1.0])[0],
        "AUC@3":     auc_at_thresholds(errs, [3.0])[0],
        "AUC@5":     auc_at_thresholds(errs, [5.0])[0],
        "inlier@1":  inlier_ratio_at_thresholds(errs, [1.0])[0],
        "inlier@3":  inlier_ratio_at_thresholds(errs, [3.0])[0],
        "inlier@5":  inlier_ratio_at_thresholds(errs, [5.0])[0],
    }
    summary["L2_median"]["overall"] = float(np.median(errs))

    # Failure-detection trustworthiness metrics.
    trust = {}
    for tau in tau_list:
        labels = (errs > tau).astype(np.int32)
        # Score is "higher = more likely failure": use 1 - reliability.
        fail_score = 1.0 - rel
        trust[f"AUROC_tau={tau}"] = auroc(fail_score, labels)
        trust[f"AUPRC_tau={tau}"] = auprc(fail_score, labels)
        trust[f"ECE_tau={tau}"]   = expected_calibration_error(fail_score, labels)
        # NLL of the failure label under the reliability score.
        eps = 1e-6
        s = np.clip(rel, eps, 1.0 - eps)
        nll = -(labels * np.log(1.0 - s) + (1 - labels) * np.log(s)).mean()
        trust[f"NLL_tau={tau}"] = float(nll)
    summary["trustworthiness"] = trust

    # Risk-coverage curve.
    cov, risk = risk_coverage_curve(errs, rel)
    summary["risk_coverage"] = {f"cov={c:.2f}": r for c, r in zip(cov, risk)}

    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--out_json", type=str, default=None)
    args = p.parse_args()

    summary = evaluate(args.ckpt)
    print(json.dumps(summary, indent=2))
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[eval] wrote {args.out_json}")


if __name__ == "__main__":
    main()
