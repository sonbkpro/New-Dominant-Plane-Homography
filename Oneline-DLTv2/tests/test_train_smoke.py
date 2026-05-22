"""End-to-end smoke test: build a small CDPCNet, run forward + backward
through each stage's loss combination, assert no NaNs and that gradients
flow only into the stage-appropriate modules.

Run from Oneline-DLTv2/:
    python tests/test_train_smoke.py
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from configs.default import Config
from model.cdpc_net import CDPCNet
from losses.triplet import triplet_loss
from losses.align_v3 import align_soft_loss, align_het_loss
from losses.em import em_posterior_loss
from losses.support import support_loss
from losses.smooth import edge_aware_smoothness
from losses.cycle import cycle_loss
from losses.fold import fold_loss
from losses.photo_image import photometric_image_loss
from losses.sigma_prior import sigma_prior_loss
from train import patch_to_full_homography, _freeze_for_stage


_FAILS = []


def _check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        _FAILS.append(name)


def _build_net(device):
    cfg = Config()
    cfg.patch_h, cfg.patch_w = 64, 96
    cfg.img_h, cfg.img_w = 96, 128
    net = CDPCNet(
        patch_h=cfg.patch_h, patch_w=cfg.patch_w,
        backbone_pretrained=False,
        corr_radius=2, corr_out_channels=8,
        bb_quarter_channels=16, bb_eighth_channels=32, bb_sixteenth_channels=64,
        rho_per_level=(8.0, 4.0, 2.0),
        homography_levels=3,
        post_init_prob=cfg.post_init_prob,
        sigma_min=cfg.sigma_min,
    ).to(device)
    return net, cfg


def _make_batch(B, cfg, device):
    torch.manual_seed(0)
    I_a_full = torch.randn(B, 1, cfg.img_h, cfg.img_w, device=device)
    I_b_full = torch.randn(B, 1, cfg.img_h, cfg.img_w, device=device)
    x0, y0 = 8, 8
    I_a_patch = I_a_full[:, :, y0:y0+cfg.patch_h, x0:x0+cfg.patch_w].contiguous()
    I_b_patch = I_b_full[:, :, y0:y0+cfg.patch_h, x0:x0+cfg.patch_w].contiguous()
    crop_xy = torch.tensor([[x0, y0]] * B, dtype=torch.float32, device=device)
    offset_gt = torch.randn(B, 8, device=device) * 2.0
    return I_a_full, I_b_full, I_a_patch, I_b_patch, crop_xy, offset_gt


def _module_grad_norm(mod):
    """Sum of grad norms over parameters that have grads. Returns 0 if none."""
    g2 = 0.0
    n = 0
    for p in mod.parameters():
        if p.grad is not None:
            g2 += float(p.grad.norm() ** 2)
            n += 1
    return g2 ** 0.5 if n > 0 else 0.0


def _zero_grads(net):
    for p in net.parameters():
        p.grad = None


def test_stage(stage_name, net, cfg, device, expect_active_grads):
    print(f"Stage: {stage_name}")
    _freeze_for_stage(net, stage_name)
    B = 2
    I_a_full, I_b_full, I_a_patch, I_b_patch, crop_xy, offset_gt = _make_batch(B, cfg, device)
    out = net(I_a_full, I_b_full, I_a_patch, I_b_patch, crop_xy)

    F_b   = out["F_b4"].float()
    F_aw  = out["F_a_warped"].float()
    F_a   = out["F_a4"].float()
    F_arec = out["F_a_recovered"].float()
    q     = out["q"].float()
    log_s = out["log_sigma"].float()
    r     = out["residual"].float()
    v     = out["valid_mask"].float()
    cv    = out["cycle_valid"].float()
    cond  = out["cond_valid"].float()
    off   = out["offset"].float()
    Hpatch = out["H_patch"].float()
    Hfull = patch_to_full_homography(Hpatch, crop_xy.float())

    # Build the stage's loss combination identical to train.py.
    L = torch.zeros((), device=device)
    if stage_name == "synth":
        L = L + F.smooth_l1_loss(off, offset_gt) + fold_loss(off, cfg.patch_h, cfg.patch_w)
    elif stage_name == "h_only":
        L = L + triplet_loss(F_b, F_aw, F_a, q, v, margin=1.0)
        L = L + photometric_image_loss(I_a_full, I_b_patch, Hfull, crop_xy.float())
        L = L + fold_loss(off, cfg.patch_h, cfg.patch_w)
    elif stage_name == "q_only":
        L = L + em_posterior_loss(q, r, log_s, v, self_paced_beta=2.0)
        L = L + support_loss(q, v, alpha=0.5)
        L = L + edge_aware_smoothness(q, I_b_patch.float(), gamma=10.0)
    elif stage_name == "sigma_only":
        L = L + align_het_loss(r, log_s, q, v, tau_q=0.5)
        L = L + sigma_prior_loss(log_s, r, v)
    elif stage_name == "joint":
        L = L + triplet_loss(F_b, F_aw, F_a, q, v, margin=1.0)
        L = L + photometric_image_loss(I_a_full, I_b_patch, Hfull, crop_xy.float())
        q_dag = torch.clamp(q, min=0.25).detach()
        L = L + align_soft_loss(r, q_dag, v)
        L = L + align_het_loss(r, log_s, q, v, tau_q=0.5)
        L = L + em_posterior_loss(q, r, log_s, v, self_paced_beta=2.0)
        L = L + support_loss(q, v, alpha=0.5)
        L = L + edge_aware_smoothness(q, I_b_patch.float(), gamma=10.0)
        L = L + sigma_prior_loss(log_s, r, v)
        L = L + cycle_loss(F_a, F_arec, cv, cond)
        L = L + fold_loss(off, cfg.patch_h, cfg.patch_w)

    _check(f"{stage_name}: loss is finite",
           torch.isfinite(L).item(),
           f"L={float(L):.4f}")

    _zero_grads(net)
    L.backward()

    grads = {
        "backbone":           _module_grad_norm(net.backbone),
        "homography_pyramid": _module_grad_norm(net.homography_pyramid),
        "posterior_head":     _module_grad_norm(net.posterior_head),
        "uncertainty_head":   _module_grad_norm(net.uncertainty_head),
        "reliability_head":   _module_grad_norm(net.reliability_head),
    }

    for mod_name in expect_active_grads:
        _check(f"{stage_name}: {mod_name} has gradient",
               grads[mod_name] > 0.0,
               f"grad_norm={grads[mod_name]:.2e}")
    inactive = set(grads.keys()) - set(expect_active_grads)
    for mod_name in inactive:
        _check(f"{stage_name}: {mod_name} grad is zero (frozen)",
               grads[mod_name] == 0.0,
               f"grad_norm={grads[mod_name]:.2e}")


def main():
    print("Test: train.py loss wiring smoke test")
    print("=" * 60)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net, cfg = _build_net(device)

    test_stage("synth",      net, cfg, device,
               expect_active_grads=["backbone", "homography_pyramid"])
    test_stage("h_only",     net, cfg, device,
               expect_active_grads=["backbone", "homography_pyramid"])
    test_stage("q_only",     net, cfg, device,
               expect_active_grads=["posterior_head"])
    test_stage("sigma_only", net, cfg, device,
               expect_active_grads=["uncertainty_head"])
    test_stage("joint",      net, cfg, device,
               expect_active_grads=["backbone", "homography_pyramid",
                                    "posterior_head", "uncertainty_head"])

    print("=" * 60)
    if _FAILS:
        print(f"FAILED ({len(_FAILS)}):")
        for n in _FAILS:
            print(f"  - {n}")
        sys.exit(1)
    print("All loss-wiring smoke tests passed.")


if __name__ == "__main__":
    main()
