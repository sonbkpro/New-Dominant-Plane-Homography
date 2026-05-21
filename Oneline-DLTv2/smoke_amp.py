"""Smoke test the AMP path that train.py actually uses."""

import os
import sys

import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = CDPCNet(patch_h=128, patch_w=160, backbone_pretrained=False,
                  corr_radius=2, corr_out_channels=8,
                  bb_quarter_channels=32, bb_eighth_channels=64).to(device)
    B = 2
    I_a = torch.randn(B, 1, 128, 160, device=device)
    I_b = torch.randn(B, 1, 128, 160, device=device)
    I_a_neg, I_b_neg, y_neg = build_invalid_batch(I_a, I_b, 0.5, 0.5)
    I_a_all = torch.cat([I_a, I_a_neg], 0)
    I_b_all = torch.cat([I_b, I_b_neg], 0)

    scaler = torch.amp.GradScaler("cuda", enabled=True)
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)

    opt.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", enabled=True):
        out = net(I_a_all, I_b_all)
        F_b_nat = out["F_b4"][:B]; F_a_w = out["F_a_warped"][:B]
        F_a_nat = out["F_a4"][:B]; F_a_rec = out["F_a_recovered"][:B]
        q = out["q"][:B]; ls = out["log_sigma"][:B]
        r = out["residual"][:B]; vm = out["valid_mask"][:B]
        cv = out["cycle_valid"][:B]; cond = out["cond_valid"][:B]
        s_all = out["s"]

        L_t  = triplet_loss(F_b_nat, F_a_w, F_a_nat, q, vm)
        L_a  = alignment_loss(r, ls, q, vm)
        L_em = em_posterior_loss(q, r, ls, vm)
        L_su = support_loss(q, vm)
        L_sm = edge_aware_smoothness(q, I_b)
        y_nat = build_invalid_pair_labels(r.mean(dim=(1, 2, 3)))
        L_r  = reliability_loss(s_all, torch.cat([y_nat, y_neg], 0))
        L_c  = cycle_loss(F_a_nat, F_a_rec, cv, cond)
        L = L_t + L_a + 0.5 * L_em + 0.01 * L_su + 1e-3 * L_sm + 0.1 * L_r + 0.05 * L_c

    print(f"[amp] L_total={float(L):.4f}  scale={scaler.get_scale():.0f}")
    scaler.scale(L).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    scaler.step(opt)
    scaler.update()
    print("[amp] step ok")


if __name__ == "__main__":
    main()
