"""Full trustworthiness evaluation for CDPC v2 (full-image-warp pipeline).

Reports per-category and overall:
  - L2 reprojection: mean, median, AUC@{1,3,5}, inlier@{1,3,5}
    for direct / inverse / symmetric / identity / v1 (planv2 §2.5)
  - failure detection: AUROC, AUPRC (failure = err > tau, against the v1 err)
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
from utils.inverse import safe_inverse_3x3
from train import patch_to_full_homography


SCENES = ("RE", "LT", "LL", "SF", "LF")
# planv3 B2: `v1` is min(forward(a->b), forward(b->a)) using H itself; v1's
# actual test.py inverts H before measuring. The "v1-equivalent" column is
# `v1_compat`. See train.run_eval_l2 for the same change.
METRICS = ("direct", "inverse", "symmetric", "identity", "v1", "v1_compat")


def _eye_like(H: torch.Tensor) -> torch.Tensor:
    B = H.shape[0]
    I = torch.eye(3, device=H.device, dtype=H.dtype)
    return I.unsqueeze(0).expand(B, -1, -1).contiguous()


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
        bb_sixteenth_channels=cfg.bb_sixteenth_channels,
        rho_per_level=cfg.rho_per_level,
        homography_levels=cfg.homography_levels,
        post_init_prob=cfg.post_init_prob,
        sigma_min=cfg.sigma_min,
        use_normalized_dlt=cfg.use_normalized_dlt,
    ).to(device)
    _load_state(net, ckpt_path)
    net.eval()

    test_ds = TestDataset(data_root=_REPO_ROOT,
                          patch_h=cfg.patch_h, patch_w=cfg.patch_w,
                          img_h=cfg.img_h, img_w=cfg.img_w)
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=0, shuffle=False)

    # Per-pair lists for each of the 5 metrics; reliability score; scenes.
    per_metric = {m: [] for m in METRICS}
    reliabilities, scenes = [], []
    n_illcond = 0

    for batch in test_loader:
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

        per_metric["direct"].append(   float(err_ab.mean(dim=1).item()))
        per_metric["inverse"].append(  float(err_inv.mean(dim=1).item()))
        per_metric["symmetric"].append(float(((err_ab + err_inv) / 2.0).mean(dim=1).item()))
        per_metric["identity"].append( float(err_id.mean(dim=1).item()))
        per_metric["v1"].append(       float(torch.minimum(err_ab, err_ba).mean(dim=1).item()))
        per_metric["v1_compat"].append(
            float(torch.minimum(err_inv, err_inv_alt).mean(dim=1).item()))

        scenes.append(batch["scene"][0])
        reliabilities.append(float(out["s"][0]))

    scenes = np.array(scenes)
    rel = np.array(reliabilities, dtype=np.float64)

    def _per_scene(errs: np.ndarray):
        d = {"overall": float(errs.mean()) if len(errs) else float("nan")}
        for s in sorted(set(scenes)):
            mask = scenes == s
            d[s] = float(errs[mask].mean()) if mask.any() else float("nan")
        return d

    # The v1-compat (inverted-H, min over orderings) series is the
    # authoritative one for failure-detection labels (planv3 B2).
    errs_v1 = np.array(per_metric["v1_compat"], dtype=np.float64)

    summary = {"n_pairs": int(len(errs_v1)), "n_illcond_H": n_illcond}
    for m in METRICS:
        e = np.array(per_metric[m], dtype=np.float64)
        summary[f"L2_mean_{m}"] = _per_scene(e)
        summary[f"L2_median_{m}"] = {
            **{s: float(np.median(e[scenes == s]) if (scenes == s).any() else float("nan"))
               for s in sorted(set(scenes))},
            "overall": float(np.median(e)),
        }
        summary[f"AUC@1_{m}"]    = auc_at_thresholds(e, [1.0])[0]
        summary[f"AUC@3_{m}"]    = auc_at_thresholds(e, [3.0])[0]
        summary[f"AUC@5_{m}"]    = auc_at_thresholds(e, [5.0])[0]
        summary[f"inlier@1_{m}"] = inlier_ratio_at_thresholds(e, [1.0])[0]
        summary[f"inlier@3_{m}"] = inlier_ratio_at_thresholds(e, [3.0])[0]
        summary[f"inlier@5_{m}"] = inlier_ratio_at_thresholds(e, [5.0])[0]

    # Failure-detection trustworthiness metrics (labels from `v1_compat`).
    trust = {}
    for tau in tau_list:
        labels = (errs_v1 > tau).astype(np.int32)
        fail_score = 1.0 - rel
        trust[f"AUROC_tau={tau}"] = auroc(fail_score, labels)
        trust[f"AUPRC_tau={tau}"] = auprc(fail_score, labels)
        trust[f"ECE_tau={tau}"]   = expected_calibration_error(fail_score, labels)
        eps = 1e-6
        s = np.clip(rel, eps, 1.0 - eps)
        nll = -(labels * np.log(1.0 - s) + (1 - labels) * np.log(s)).mean()
        trust[f"NLL_tau={tau}"] = float(nll)
    summary["trustworthiness"] = trust

    cov, risk = risk_coverage_curve(errs_v1, rel)
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
