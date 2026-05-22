"""End-to-end smoke test: forward + backward on a dummy batch (planv3 API).

Run from inside Oneline-DLTv2/:
    python smoke_test.py

Exercises the v3 forward signature (Siamese trunk + 3-level homography
pyramid) and all the planv3 loss functions, including the soft/het split
and the image-space photometric anchor.
"""

import os
import sys

import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

from configs.default import Config
from model.cdpc_net import CDPCNet
from losses.triplet import triplet_loss
from losses.align_v3 import align_soft_loss, align_het_loss
from losses.em import em_posterior_loss
from losses.support import support_loss
from losses.smooth import edge_aware_smoothness
from losses.reliability import reliability_loss, build_invalid_pair_labels
from losses.cycle import cycle_loss
from losses.fold import fold_loss
from losses.photo_image import photometric_image_loss
from losses.sigma_prior import sigma_prior_loss
from data.invalid_pairs import build_invalid_batch
from train import patch_to_full_homography


def main():
    torch.manual_seed(0)
    cfg = Config()
    cfg.patch_h, cfg.patch_w = 128, 160
    cfg.img_h, cfg.img_w = 144, 192
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
        bb_sixteenth_channels=128,
        rho_per_level=(16.0, 8.0, 4.0),
        homography_levels=3,
        post_init_prob=cfg.post_init_prob,
        sigma_min=cfg.sigma_min,
    ).to(device)
    print(f"[smoke] params: {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M")

    B = cfg.batch_size
    I_a_full = torch.randn(B, 1, cfg.img_h, cfg.img_w, device=device)
    I_b_full = torch.randn(B, 1, cfg.img_h, cfg.img_w, device=device)
    x0, y0 = 8, 8
    I_a_patch = I_a_full[:, :, y0:y0 + cfg.patch_h, x0:x0 + cfg.patch_w].contiguous()
    I_b_patch = I_b_full[:, :, y0:y0 + cfg.patch_h, x0:x0 + cfg.patch_w].contiguous()
    crop_xy = torch.tensor([[x0, y0]] * B, dtype=torch.float32, device=device)

    # Construct an invalid-pair batch for the reliability head path.
    I_a_neg, I_b_neg, y_neg = build_invalid_batch(
        I_a_patch, I_b_patch, shuffle_frac=0.5, reshuffle_frac=0.5,
    )
    I_a_full_neg = I_a_full[:I_a_neg.shape[0]]
    I_b_full_neg = I_b_full[:I_b_neg.shape[0]]
    crop_xy_neg = crop_xy[:I_a_neg.shape[0]]
    I_a_full_all  = torch.cat([I_a_full,  I_a_full_neg], dim=0)
    I_b_full_all  = torch.cat([I_b_full,  I_b_full_neg], dim=0)
    I_a_patch_all = torch.cat([I_a_patch, I_a_neg], dim=0)
    I_b_patch_all = torch.cat([I_b_patch, I_b_neg], dim=0)
    crop_xy_all   = torch.cat([crop_xy,   crop_xy_neg], dim=0)
    print(f"[smoke] forward batch: full {tuple(I_a_full_all.shape)}, "
          f"patch {tuple(I_a_patch_all.shape)}")

    out = net(I_a_full_all, I_b_full_all, I_a_patch_all, I_b_patch_all, crop_xy_all)
    print("[smoke] outputs:")
    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            print(f"   {k:18s} shape={tuple(v.shape)}  dtype={v.dtype}  "
                  f"has_nan={bool(v.isnan().any())}")

    F_b_nat        = out["F_b4"][:B]
    F_a_warped_nat = out["F_a_warped"][:B]
    F_a_nat        = out["F_a4"][:B]
    F_a_rec_nat    = out["F_a_recovered"][:B]
    q_nat          = out["q"][:B]
    log_sigma_nat  = out["log_sigma"][:B]
    residual_nat   = out["residual"][:B]
    valid_nat      = out["valid_mask"][:B]
    cycle_valid_nat = out["cycle_valid"][:B]
    cond_valid_nat  = out["cond_valid"][:B]
    offset_nat     = out["offset"][:B]
    H_patch_nat    = out["H_patch"][:B]
    s_all          = out["s"]

    H_full_nat = patch_to_full_homography(H_patch_nat, crop_xy.float())

    L_triplet    = triplet_loss(F_b_nat, F_a_warped_nat, F_a_nat, q_nat, valid_nat)
    L_align_soft = align_soft_loss(residual_nat, q_nat, valid_nat)
    L_align_het  = align_het_loss(residual_nat, log_sigma_nat, q_nat, valid_nat,
                                  tau_q=cfg.q_select_tau)
    L_em         = em_posterior_loss(q_nat, residual_nat, log_sigma_nat, valid_nat,
                                     pi=cfg.em_prior_pi, r_max=cfg.em_r_max,
                                     self_paced_beta=cfg.em_self_paced_beta)
    L_support    = support_loss(q_nat, valid_nat, alpha=cfg.alpha_support)
    L_smooth     = edge_aware_smoothness(q_nat, I_b_patch.float(), gamma=cfg.smoothness_gamma)
    L_cycle      = cycle_loss(F_a_nat, F_a_rec_nat, cycle_valid_nat, cond_valid_nat)
    L_fold       = fold_loss(offset_nat, cfg.patch_h, cfg.patch_w)
    L_photo_img  = photometric_image_loss(I_a_full, I_b_patch, H_full_nat, crop_xy.float())
    L_sigma      = sigma_prior_loss(log_sigma_nat, residual_nat, valid_nat)

    r_mean_nat = residual_nat.mean(dim=(1, 2, 3))
    y_nat = build_invalid_pair_labels(r_mean_nat,
                                      hard_negative_percentile=cfg.rel_hard_neg_percentile)
    y_target = torch.cat([y_nat, y_neg], dim=0)
    L_rel = reliability_loss(s_all, y_target)

    losses = {
        "L_triplet":    L_triplet,
        "L_align_soft": L_align_soft,
        "L_align_het":  L_align_het,
        "L_em":         L_em,
        "L_support":    L_support,
        "L_smooth":     L_smooth,
        "L_rel":        L_rel,
        "L_cycle":      L_cycle,
        "L_fold":       L_fold,
        "L_photo_img":  L_photo_img,
        "L_sigma":      L_sigma,
    }
    print("[smoke] losses:")
    for k, v in losses.items():
        finite = bool(torch.isfinite(v))
        print(f"   {k:14s} = {float(v):.6f}   finite={finite}")
        assert finite, f"{k} is not finite"

    L_total = (cfg.lambda_triplet     * L_triplet
               + cfg.lambda_align_soft * L_align_soft
               + cfg.lambda_align_het  * L_align_het
               + cfg.lambda_em         * L_em
               + cfg.lambda_support    * L_support
               + cfg.lambda_smooth     * L_smooth
               + cfg.lambda_rel        * L_rel
               + cfg.lambda_cycle      * L_cycle
               + cfg.lambda_sigma      * L_sigma
               + cfg.lambda_fold       * L_fold
               + cfg.lambda_photo_img  * L_photo_img)
    print(f"[smoke] L_total = {float(L_total):.6f}")

    L_total.backward()

    grad_norms = {}
    for name, module in [
        ("backbone (Siamese FULL)", net.backbone),
        ("homography_pyramid",      net.homography_pyramid),
        ("posterior_head",          net.posterior_head),
        ("uncertainty_head",        net.uncertainty_head),
        ("reliability_head",        net.reliability_head),
    ]:
        gnorm = 0.0
        n = 0
        for p in module.parameters():
            if p.grad is not None:
                gnorm += float(p.grad.norm() ** 2)
                n += 1
        grad_norms[name] = (gnorm ** 0.5, n)
    print("[smoke] grad norms:")
    for k, (g, n) in grad_norms.items():
        print(f"   {k:28s} ||g||={g:.6f}  over {n} tensors")

    rel_grad = grad_norms.get("reliability_head", (0.0, 0))[0]
    assert rel_grad > 0.0, "reliability head got zero gradient -- did detach break the signal?"

    print("[smoke] OK")


if __name__ == "__main__":
    main()
