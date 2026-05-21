"""Diagnose where NaN first appears under AMP with realistic feature scales.

Mimics the training data range: normalized grayscale ~ N(0, 1)-ish."""

import os, sys, torch
_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS)

from configs.default import Config
from model.cdpc_net import CDPCNet


def _check(name, t):
    bad = bool(t.isnan().any() | t.isinf().any())
    print(f"  {name:18s} shape={tuple(t.shape)} dtype={t.dtype} "
          f"min={float(t.min()):.4f} max={float(t.max()):.4f} bad={bad}")
    return bad


def main():
    torch.manual_seed(0)
    cfg = Config()
    cfg.patch_h, cfg.patch_w = 315, 560
    device = "cuda"
    net = CDPCNet(
        patch_h=cfg.patch_h, patch_w=cfg.patch_w,
        backbone_pretrained=True,
        corr_radius=cfg.corr_radius, corr_out_channels=cfg.corr_out_channels,
        bb_quarter_channels=cfg.bb_quarter_channels,
        bb_eighth_channels=cfg.bb_eighth_channels,
    ).to(device)
    net.eval()
    B = 2
    I_a = torch.randn(B, 1, cfg.patch_h, cfg.patch_w, device=device) * 1.0
    I_b = I_a + 0.1 * torch.randn_like(I_a)

    print("=== AMP autocast on ===")
    with torch.amp.autocast("cuda", enabled=True):
        out = net(I_a, I_b)
    for k, v in out.items():
        if isinstance(v, torch.Tensor) and v.dim() > 0:
            _check(k, v)

    r = out["residual"]
    ls = out["log_sigma"]
    sig = ls.exp()
    x = r / (sig + 1e-8)
    print(f"\nx = r/sigma:  min={float(x.min()):.3f}  max={float(x.max()):.3f}")
    print(f"x^2:          min={float((x*x).min()):.3f}  max={float((x*x).max()):.3f}")
    rho = torch.sqrt(x * x + 1e-3 * 1e-3)
    print(f"rho:          min={float(rho.min()):.3f}  max={float(rho.max()):.3f}  "
          f"bad={bool(rho.isnan().any() | rho.isinf().any())}")


if __name__ == "__main__":
    main()
