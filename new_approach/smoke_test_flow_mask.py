import argparse
from types import SimpleNamespace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from Data.synthetic_flow_mask_dataset import SyntheticFlowMaskConfig, build_synthetic_flow_mask_loader
from new_approach.device_utils import resolve_device
from new_approach.losses_homo import flow_matching_loss, mask_area_loss, mask_total_variation_loss
from new_approach.modules.maskFlowHomo import DominantMaskFlow, build_mask_condition
from new_approach.modules.transformerHomo import Ms_Transformer


def _tiny_params():
    return SimpleNamespace(
        net_type="HomoGAN",
        crop_size=[32, 32],
        patch_size=4,
        in_channels=2,
        in_chans=2,
        num_basis=8,
        embed_dim=12,
        depths=[1, 1, 1],
        layer_depth=[3, 2, 1],
        num_heads=[1, 2, 4],
        num_decoder_layers=3,
        window_size=4,
        mlp_ratio=2.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        drop_path_rate=0.0,
        attn_drop_rate=0.0,
        ape=False,
        patch_norm=True,
        pretrain_phase=False,
        mask_use_fea=True,
        mask_method="flow_matching",
        mask_flow_cond_channels=7,
        mask_flow_base_channels=8,
        mask_flow_channel_mults=(1, 2),
        mask_flow_time_dim=32,
        mask_flow_steps=1,
        mask_flow_solver="euler",
        mask_flow_init="zero",
        mask_flow_detach_condition=True,
        mask_flow_sample_grad=False,
        return_h_matrix=True,
    )


def test_mask_flow_module(device):
    batch_size, height, width = 2, 32, 32
    reference = torch.rand(batch_size, 1, height, width, device=device)
    warped = torch.rand(batch_size, 1, height, width, device=device)
    flow = torch.randn(batch_size, 2, height, width, device=device) * 0.1
    target = torch.rand(batch_size, 1, height, width, device=device).round()

    cond = build_mask_condition(reference, warped, reference, warped, flow, detach=False)
    model = DominantMaskFlow(cond_channels=cond.shape[1], base_channels=8, channel_mults=(1, 2), time_dim=32).to(device)
    loss = flow_matching_loss(model, cond, target)
    loss = loss + 0.05 * mask_area_loss(model.sample(cond, requires_grad=True))
    loss = loss + 0.01 * mask_total_variation_loss(model.sample(cond, requires_grad=True))
    loss.backward()
    mask = model.sample(cond, steps=2, solver="heun")

    assert mask.shape == target.shape
    assert torch.isfinite(loss)
    assert mask.min() >= 0 and mask.max() <= 1
    print(f"mask_flow ok: cond={tuple(cond.shape)} mask_mean={mask.mean().item():.4f}")


def test_synthetic_loader(device):
    config = SyntheticFlowMaskConfig(height=32, width=32, length=4, seed=7, max_translation=3)
    loader = build_synthetic_flow_mask_loader(config, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    assert batch["imgs_gray_patch"].shape == (2, 2, 32, 32)
    assert batch["img1_dominant_mask"].shape == (2, 1, 32, 32)
    assert batch["img2_dominant_mask"].shape == (2, 1, 32, 32)
    print(
        "synthetic_loader ok: "
        f"img_mean={batch['imgs_gray_patch'].mean().item():.4f} "
        f"mask_mean={batch['img2_dominant_mask'].mean().item():.4f}"
    )


def test_homonet_forward(device):
    params = _tiny_params()
    model = Ms_Transformer(params=params).to(device)
    model.eval()
    batch = {
        "imgs_gray_patch": torch.rand(1, 2, 32, 32, device=device),
        "imgs_gray_full": torch.rand(1, 2, 32, 32, device=device),
        "start": torch.zeros(1, 2, 1, 1, device=device),
        "pts": torch.tensor([[[0, 0], [31, 0], [0, 31], [31, 31]]], dtype=torch.float32, device=device),
    }
    with torch.no_grad():
        output = model(batch)
    assert output["flow_f"].shape == (1, 32, 32, 2)
    assert output["img1_patch_mask"].shape == (1, 1, 32, 32)
    assert output["img2_patch_mask"].shape == (1, 1, 32, 32)
    assert output["H_f"].shape == (1, 3, 3)
    assert output["mask_method"] == "flow_matching"
    print(
        "homonet ok: "
        f"flow_abs={output['flow_f'].abs().mean().item():.4f} "
        f"mask_mean={output['img1_patch_mask'].mean().item():.4f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Smoke test flow-matching mask and HomoNet wiring.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    args = parser.parse_args()
    device = resolve_device(args.device)
    print(f"using device: {device}")
    torch.manual_seed(11)
    test_mask_flow_module(device)
    test_synthetic_loader(device)
    test_homonet_forward(device)


if __name__ == "__main__":
    main()
