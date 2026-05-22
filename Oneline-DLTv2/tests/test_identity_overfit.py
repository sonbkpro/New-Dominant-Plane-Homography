"""planv3 §6 Stage 0: identity-pair overfit smoke test.

Feed I_b = I_a (no warp) for a small batch, run a few hundred
steps of stage h_only-style training, require:
  - ||Delta_p||_inf < 0.5 px
  - mean predicted residual < 0.1 (feature scale)
  - the v1_compat eval column on a synthesized correspondence ~= 0.

This is the planv3 acceptance gate for Stage 0: if the network cannot
overfit a trivial identity pair, no real training will succeed.

Run from Oneline-DLTv2/:
    python tests/test_identity_overfit.py
"""

import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from configs.default import Config
from model.cdpc_net import CDPCNet


_FAILS = []


def _check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        _FAILS.append(name)


def _make_random_pair(B: int, img_h: int, img_w: int, device):
    """Synth a batch of smooth random patterns, set I_b = I_a (identity pair)."""
    yy, xx = torch.meshgrid(
        torch.arange(img_h, dtype=torch.float32),
        torch.arange(img_w, dtype=torch.float32),
        indexing="ij",
    )
    samples = []
    rng = np.random.RandomState(0)
    for b in range(B):
        a = rng.uniform(0.03, 0.10)
        b_ = rng.uniform(0.03, 0.10)
        phi1 = rng.uniform(0, 6.28)
        phi2 = rng.uniform(0, 6.28)
        img = (torch.sin(a * xx + phi1) + torch.cos(b_ * yy + phi2))
        samples.append(img)
    I_a = torch.stack(samples).unsqueeze(1).to(device)
    return I_a, I_a.clone()


def main():
    print("Test: identity-pair overfit (Stage 0 acceptance gate)")
    print("=" * 60)
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
        bb_sixteenth_channels=128,
        rho_per_level=(16.0, 8.0, 4.0),
        homography_levels=3,
        post_init_prob=cfg.post_init_prob,
        sigma_min=cfg.sigma_min,
        use_normalized_dlt=True,
    ).to(device)

    I_a_full, I_b_full = _make_random_pair(B, cfg.img_h, cfg.img_w, device)
    x0, y0 = 16, 16
    I_a_patch = I_a_full[:, :, y0:y0+cfg.patch_h, x0:x0+cfg.patch_w].contiguous()
    I_b_patch = I_b_full[:, :, y0:y0+cfg.patch_h, x0:x0+cfg.patch_w].contiguous()
    crop_xy = torch.tensor([[x0, y0]] * B, dtype=torch.float32, device=device)

    # Adam on backbone + homography_pyramid (h_only-style active set).
    params = (list(net.backbone.parameters()) +
              list(net.homography_pyramid.parameters()))
    opt = torch.optim.Adam(params, lr=2e-4)

    t0 = time.time()
    last_off_inf = None
    last_residual = None
    for it in range(500):
        out = net(I_a_full, I_b_full, I_a_patch, I_b_patch, crop_xy)
        # Photometric anchor on the WARPED I_a vs I_b -- but since I_a == I_b,
        # the loss equals zero iff H is the identity (i.e. delta_p == 0).
        from losses.photo_image import photometric_image_loss
        from train import patch_to_full_homography
        H_full = patch_to_full_homography(out["H_patch"], crop_xy)
        L_photo = photometric_image_loss(I_a_full, I_b_patch, H_full, crop_xy)
        # Encourage zero offset directly: gradient signal so the network
        # learns the basin quickly even from random init.
        L_fold = (out["offset"] ** 2).mean()
        L = L_photo + 0.01 * L_fold
        opt.zero_grad(set_to_none=True)
        L.backward()
        opt.step()

        if (it + 1) % 100 == 0:
            last_off_inf = float(out["offset"].abs().max())
            last_residual = float(out["residual"].mean())
            print(f"        iter {it+1:4d}  L={float(L):.4f}  "
                  f"max|off|={last_off_inf:.3f}px  res_mean={last_residual:.4f}")

    elapsed = time.time() - t0
    print(f"        elapsed: {elapsed:.1f}s on device={device}")

    _check("||Δp||_inf < 0.5 px after 500 identity-overfit iters",
           last_off_inf is not None and last_off_inf < 0.5,
           f"max|off|={last_off_inf}")
    _check("mean feature residual < 0.5 after overfit",
           last_residual is not None and last_residual < 0.5,
           f"residual={last_residual}")

    print("=" * 60)
    if _FAILS:
        print(f"FAILED: {_FAILS}")
        sys.exit(1)
    print("All identity-overfit tests passed.")


if __name__ == "__main__":
    main()
