"""End-to-end smoke test: forward + backward on a dummy batch.

Run from inside Oneline-DLTv2/:
    python smoke_test.py
"""

import os
import sys

import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

from configs.default import Config
from model.cdpc_net import CDPCNet
from losses.triplet import triplet_loss
from losses.align import alignment_loss
from losses.em import em_posterior_loss
from losses.support import support_loss
from losses.smooth import edge_aware_smoothness
from losses.reliability import reliability_loss, build_invalid_pair_labels
from losses.cycle import cycle_loss
from data.invalid_pairs import build_invalid_batch


def main():
    torch.manual_seed(0)
    cfg = Config()
    # Use a small patch to keep the smoke test fast on any machine.
    cfg.patch_h = 128
    cfg.patch_w = 160
    cfg.batch_size = 2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[smoke] device={device}")

    net = CDPCNet(
        patch_h=cfg.patch_h, patch_w=cfg.patch_w,
        backbone_pretrained=False,
        corr_radius=2,
        corr_out_channels=8,
        bb_quarter_channels=32,
        bb_eighth_channels=64,
        homography_rho=cfg.homography_rho,
        post_init_prob=cfg.post_init_prob,
        log_sigma_min=cfg.log_sigma_min,
        log_sigma_max=cfg.log_sigma_max,
    ).to(device)
    print(f"[smoke] params: {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M")

    B = cfg.batch_size
    I_a = torch.randn(B, 1, cfg.patch_h, cfg.patch_w, device=device)
    I_b = torch.randn(B, 1, cfg.patch_h, cfg.patch_w, device=device)

    # Invalid pairs.
    I_a_neg, I_b_neg, y_neg = build_invalid_batch(
        I_a, I_b, shuffle_frac=0.5, reshuffle_frac=0.5,
    )
    I_a_all = torch.cat([I_a, I_a_neg], dim=0)
    I_b_all = torch.cat([I_b, I_b_neg], dim=0)
    print(f"[smoke] forward batch: {I_a_all.shape}")

    out = net(I_a_all, I_b_all)
    print(f"[smoke] outputs:")
    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            print(f"   {k:18s} shape={tuple(v.shape)}  dtype={v.dtype}  "
                  f"has_nan={bool(v.isnan().any())}")

    # Compute every loss.
    F_b_nat = out["F_b4"][:B]
    F_a_warped_nat = out["F_a_warped"][:B]
    F_a_nat = out["F_a4"][:B]
    F_a_rec_nat = out["F_a_recovered"][:B]
    q_nat = out["q"][:B]
    log_sigma_nat = out["log_sigma"][:B]
    residual_nat = out["residual"][:B]
    valid_nat = out["valid_mask"][:B]
    cycle_valid_nat = out["cycle_valid"][:B]
    cond_valid_nat = out["cond_valid"][:B]
    s_all = out["s"]

    L_triplet = triplet_loss(F_b_nat, F_a_warped_nat, F_a_nat, q_nat, valid_nat)
    L_align   = alignment_loss(residual_nat, log_sigma_nat, q_nat, valid_nat)
    L_em      = em_posterior_loss(q_nat, residual_nat, log_sigma_nat, valid_nat,
                                  pi=cfg.em_prior_pi, r_max=cfg.em_r_max)
    L_support = support_loss(q_nat, valid_nat, alpha=cfg.alpha_support)
    L_smooth  = edge_aware_smoothness(q_nat, I_b, gamma=cfg.smoothness_gamma)
    r_mean_nat = residual_nat.mean(dim=(1, 2, 3))
    y_nat = build_invalid_pair_labels(r_mean_nat,
                                      hard_negative_percentile=cfg.rel_hard_neg_percentile)
    y_target = torch.cat([y_nat, y_neg], dim=0)
    L_rel = reliability_loss(s_all, y_target)
    L_cycle = cycle_loss(F_a_nat, F_a_rec_nat, cycle_valid_nat, cond_valid_nat)

    losses = {
        "L_triplet": L_triplet, "L_align": L_align, "L_em": L_em,
        "L_support": L_support, "L_smooth": L_smooth, "L_rel": L_rel,
        "L_cycle": L_cycle,
    }
    print(f"[smoke] losses:")
    for k, v in losses.items():
        finite = bool(torch.isfinite(v))
        print(f"   {k:12s} = {float(v):.6f}   finite={finite}")
        assert finite, f"{k} is not finite"

    L_total = (cfg.lambda_triplet * L_triplet + cfg.lambda_align * L_align
               + cfg.lambda_em * L_em + cfg.lambda_support * L_support
               + cfg.lambda_smooth * L_smooth + cfg.lambda_rel * L_rel
               + cfg.lambda_cycle * L_cycle)
    print(f"[smoke] L_total = {float(L_total):.6f}")

    L_total.backward()
    # Check that gradients propagated to all major parameter groups.
    grad_norms = {}
    for name, module in [
        ("backbone", net.backbone),
        ("correlation", net.correlation),
        ("homography_head", net.homography_head),
        ("posterior_head", net.posterior_head),
        ("uncertainty_head", net.uncertainty_head),
        ("reliability_head", net.reliability_head),
    ]:
        gnorm = 0.0
        n = 0
        for p in module.parameters():
            if p.grad is not None:
                gnorm += float(p.grad.norm() ** 2)
                n += 1
        grad_norms[name] = (gnorm ** 0.5, n)
    print(f"[smoke] grad norms:")
    for k, (g, n) in grad_norms.items():
        print(f"   {k:18s} ||g||={g:.6f}  over {n} tensors")
        assert g > 0.0 or n == 0, f"{k} got zero gradient"

    print("[smoke] OK")


if __name__ == "__main__":
    main()
