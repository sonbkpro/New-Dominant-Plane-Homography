import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from Data.synthetic_flow_mask_dataset import SyntheticFlowMaskConfig, build_synthetic_flow_mask_loader
from new_approach.device_utils import resolve_device
from new_approach.losses_homo import (
    flow_matching_loss,
    mask_area_loss,
    mask_total_variation_loss,
)
from new_approach.modules.maskFlowHomo import DominantMaskFlow, build_mask_condition


def warp_with_flow(image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    if flow.shape[1] != 2:
        flow = flow.permute(0, 3, 1, 2).contiguous()
    batch_size, _, height, width = image.shape
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, height, device=image.device, dtype=image.dtype),
        torch.linspace(-1, 1, width, device=image.device, dtype=image.dtype),
        indexing="ij",
    )
    base_grid = torch.stack([xx, yy], dim=-1).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    flow_x = 2.0 * flow[:, 0] / max(width - 1, 1)
    flow_y = 2.0 * flow[:, 1] / max(height - 1, 1)
    sample_grid = base_grid - torch.stack([flow_x, flow_y], dim=-1)
    return F.grid_sample(image, sample_grid, mode="bilinear", padding_mode="border", align_corners=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthetic warmup for DominantMaskFlow.")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--save", default="models/dominant_mask_flow_synth.pth")
    parser.add_argument("--seed", type=int, default=230)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    print(f"using device: {device}")

    config = SyntheticFlowMaskConfig(height=args.height, width=args.width, seed=args.seed)
    loader = build_synthetic_flow_mask_loader(config, batch_size=args.batch_size, shuffle=True)

    model = DominantMaskFlow(cond_channels=7, base_channels=32).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    iterator = iter(loader)
    model.train()
    for step in range(1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)

        imgs = batch["imgs_gray_patch"].to(device)
        flow = batch["dominant_flow"].to(device)
        target_mask = batch["img2_dominant_mask"].to(device)
        img1 = imgs[:, :1]
        img2 = imgs[:, 1:]
        warped_img1 = warp_with_flow(img1, flow)

        cond = build_mask_condition(
            reference_feature=img2,
            warped_feature=warped_img1,
            reference_image=img2,
            warped_image=warped_img1,
            flow=flow,
            detach=False,
        )
        loss_fm = flow_matching_loss(model, cond, target_mask)
        sampled_mask = model.sample(cond, steps=1, requires_grad=True)
        loss = loss_fm + 0.05 * mask_area_loss(sampled_mask) + 0.01 * mask_total_variation_loss(sampled_mask)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step % 25 == 0:
            print(
                f"step={step:05d} loss={loss.item():.5f} "
                f"fm={loss_fm.item():.5f} mask_mean={sampled_mask.mean().item():.4f}"
            )

    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "args": vars(args)}, save_path)
    print(f"saved {save_path}")


if __name__ == "__main__":
    main()
