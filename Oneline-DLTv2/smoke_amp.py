"""Smoke test the AMP path the v3 train.py actually uses."""

import os
import sys

import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[amp] device={device}")

    patch_h, patch_w = 128, 160
    img_h,  img_w  = 144, 192
    net = CDPCNet(
        patch_h=patch_h, patch_w=patch_w, backbone_pretrained=False,
        corr_radius=2, corr_out_channels=8,
        bb_quarter_channels=32, bb_eighth_channels=64, bb_sixteenth_channels=128,
        rho_per_level=(16.0, 8.0, 4.0), homography_levels=3,
    ).to(device)

    B = 2
    I_a_full = torch.randn(B, 1, img_h, img_w, device=device)
    I_b_full = torch.randn(B, 1, img_h, img_w, device=device)
    x0, y0 = 8, 8
    I_a_patch = I_a_full[:, :, y0:y0+patch_h, x0:x0+patch_w].contiguous()
    I_b_patch = I_b_full[:, :, y0:y0+patch_h, x0:x0+patch_w].contiguous()
    crop_xy = torch.tensor([[x0, y0]] * B, dtype=torch.float32, device=device)

    I_a_neg, I_b_neg, y_neg = build_invalid_batch(I_a_patch, I_b_patch, 0.5, 0.5)
    I_a_full_neg = I_a_full[:I_a_neg.shape[0]]
    I_b_full_neg = I_b_full[:I_b_neg.shape[0]]
    crop_xy_neg = crop_xy[:I_a_neg.shape[0]]
    I_a_full_all  = torch.cat([I_a_full, I_a_full_neg], 0)
    I_b_full_all  = torch.cat([I_b_full, I_b_full_neg], 0)
    I_a_patch_all = torch.cat([I_a_patch, I_a_neg], 0)
    I_b_patch_all = torch.cat([I_b_patch, I_b_neg], 0)
    crop_xy_all   = torch.cat([crop_xy,  crop_xy_neg], 0)

    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4)

    opt.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", enabled=device == "cuda"):
        out = net(I_a_full_all, I_b_full_all,
                  I_a_patch_all, I_b_patch_all, crop_xy_all)
    # All losses computed in fp32 (matching train.py's autocast(enabled=False)).
    with torch.amp.autocast("cuda", enabled=False):
        F_b      = out["F_b4"][:B].float()
        F_aw     = out["F_a_warped"][:B].float()
        F_a      = out["F_a4"][:B].float()
        F_arec   = out["F_a_recovered"][:B].float()
        q        = out["q"][:B].float()
        ls       = out["log_sigma"][:B].float()
        r        = out["residual"][:B].float()
        vm       = out["valid_mask"][:B].float()
        cv       = out["cycle_valid"][:B].float()
        cond     = out["cond_valid"][:B].float()
        off      = out["offset"][:B].float()
        H_patch  = out["H_patch"][:B].float()
        s_all    = out["s"].float()
        H_full   = patch_to_full_homography(H_patch, crop_xy.float())

        L_t   = triplet_loss(F_b, F_aw, F_a, q, vm)
        L_as  = align_soft_loss(r, q, vm)
        L_ah  = align_het_loss(r, ls, q, vm, tau_q=0.5)
        L_em  = em_posterior_loss(q, r, ls, vm, self_paced_beta=2.0)
        L_su  = support_loss(q, vm, alpha=0.5)
        L_sm  = edge_aware_smoothness(q, I_b_patch.float())
        L_cy  = cycle_loss(F_a, F_arec, cv, cond)
        L_fl  = fold_loss(off, patch_h, patch_w)
        L_pi  = photometric_image_loss(I_a_full, I_b_patch, H_full, crop_xy.float())
        L_sg  = sigma_prior_loss(ls, r, vm)
        y_nat = build_invalid_pair_labels(r.mean(dim=(1, 2, 3)))
        L_r   = reliability_loss(s_all, torch.cat([y_nat, y_neg], 0))

        L = (L_t + L_as + 0.3 * L_ah + 0.5 * L_em + 0.1 * L_su +
             1e-3 * L_sm + 0.1 * L_r + 0.05 * L_cy + 0.1 * L_fl +
             0.5 * L_pi + 0.1 * L_sg)

    print(f"[amp] L_total={float(L):.4f}  scale={scaler.get_scale():.0f}")
    scaler.scale(L).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    scaler.step(opt)
    scaler.update()
    print("[amp] step ok")


if __name__ == "__main__":
    main()
