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


def soft_dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = pred.flatten(1)
    target = target.flatten(1)
    intersection = (pred * target).sum(dim=1)
    denom = pred.sum(dim=1) + target.sum(dim=1)
    return (1.0 - (2.0 * intersection + eps) / (denom + eps)).mean()


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
    parser.add_argument("--synthetic-max-translation", type=int, default=12)
    parser.add_argument("--synthetic-min-outliers", type=int, default=2)
    parser.add_argument("--synthetic-max-outliers", type=int, default=6)
    parser.add_argument("--synthetic-min-outlier-size", type=int, default=16)
    parser.add_argument("--synthetic-max-outlier-size", type=int, default=80)
    parser.add_argument("--synthetic-blur-kernel", type=int, default=7)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda-fm", type=float, default=0.5)
    parser.add_argument("--lambda-mask-bce", type=float, default=2.0)
    parser.add_argument("--lambda-mask-dice", type=float, default=1.0)
    parser.add_argument("--lambda-area", type=float, default=0.05)
    parser.add_argument("--lambda-tv", type=float, default=0.01)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--save", default="models/dominant_mask_flow_synth.pth")
    parser.add_argument("--seed", type=int, default=230)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    print(f"using device: {device}")

    config = SyntheticFlowMaskConfig(
        height=args.height,
        width=args.width,
        max_translation=args.synthetic_max_translation,
        min_outliers=args.synthetic_min_outliers,
        max_outliers=args.synthetic_max_outliers,
        min_outlier_size=args.synthetic_min_outlier_size,
        max_outlier_size=args.synthetic_max_outlier_size,
        blur_kernel=args.synthetic_blur_kernel,
        seed=args.seed,
    )
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
        loss_bce = F.binary_cross_entropy(sampled_mask.clamp(1e-6, 1.0 - 1e-6), target_mask)
        loss_dice = soft_dice_loss(sampled_mask, target_mask)
        loss_area = mask_area_loss(sampled_mask)
        loss_tv = mask_total_variation_loss(sampled_mask)
        loss = args.lambda_fm * loss_fm
        loss = loss + args.lambda_mask_bce * loss_bce + args.lambda_mask_dice * loss_dice
        loss = loss + args.lambda_area * loss_area + args.lambda_tv * loss_tv

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step % 25 == 0:
            print(
                f"step={step:05d} loss={loss.item():.5f} "
                f"fm={loss_fm.item():.5f} bce={loss_bce.item():.5f} dice={loss_dice.item():.5f} "
                f"mask_mean={sampled_mask.mean().item():.4f} target_mean={target_mask.mean().item():.4f}"
            )

    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "args": vars(args)}, save_path)
    print(f"saved {save_path}")


if __name__ == "__main__":
    main()
