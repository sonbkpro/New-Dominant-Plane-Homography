import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, Optional

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Data.homo_flow_dataset import build_homo_flow_loader
from new_approach.device_utils import resolve_device
from Data.synthetic_flow_mask_dataset import SyntheticFlowMaskConfig, build_synthetic_flow_mask_loader
from new_approach.losses_homo import (
    bidirectional_mask_intersection,
    feature_identity_loss,
    flow_matching_loss,
    mask_area_loss,
    mask_entropy_loss,
    mask_total_variation_loss,
    masked_triplet_alignment_loss,
    pseudo_mask_from_residuals,
)
from new_approach.modules.maskFlowHomo import DominantMaskFlow, build_mask_condition
from new_approach.modules.transformerHomo import Ms_Transformer


def default_homo_params(args, mode: str) -> SimpleNamespace:
    pretrain_phase = mode == "pretrain_homo"
    mask_method = "none" if pretrain_phase else args.mask_method
    return SimpleNamespace(
        net_type="HomoGAN",
        crop_size=[args.crop_h, args.crop_w],
        in_channels=2,
        patch_size=args.patch_size,
        rho=args.rho,
        in_chans=2,
        num_basis=8,
        embed_dim=args.embed_dim,
        depths=tuple(args.depths),
        layer_depth=tuple(args.layer_depth),
        num_heads=tuple(args.num_heads),
        num_decoder_layers=4,
        window_size=args.window_size,
        mlp_ratio=args.mlp_ratio,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        drop_path_rate=0.0,
        attn_drop_rate=0.0,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        mask_use_fea=True,
        use_open=args.use_open,
        pretrain_phase=pretrain_phase,
        mask_method=mask_method,
        mask_flow_cond_channels=7,
        mask_flow_base_channels=args.mask_flow_base_channels,
        mask_flow_channel_mults=tuple(args.mask_flow_channel_mults),
        mask_flow_time_dim=args.mask_flow_time_dim,
        mask_flow_steps=args.mask_flow_steps,
        mask_flow_solver=args.mask_flow_solver,
        mask_flow_init=args.mask_flow_init,
        mask_flow_temperature=args.mask_flow_temperature,
        mask_flow_noise_sigma=args.mask_flow_noise_sigma,
        mask_flow_detach_condition=args.mask_flow_detach_condition,
        mask_flow_sample_grad=args.mask_flow_sample_grad,
        return_h_matrix=args.return_h_matrix,
    )


def batch_to_device(batch: Dict, device: torch.device) -> Dict:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return moved


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    epoch: int,
    step: int,
    params: SimpleNamespace,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "epoch": epoch,
            "step": step,
            "params": vars(params),
            "args": vars(args),
        },
        path,
    )


def torch_load_checkpoint(path: str | Path) -> Dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_checkpoint(path: str, model, optimizer=None, strict: bool = False) -> Dict:
    ckpt = torch_load_checkpoint(path)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=strict)
    print(f"loaded checkpoint {path}")
    if missing:
        print(f"  missing keys: {len(missing)}")
    if unexpected:
        print(f"  unexpected keys: {len(unexpected)}")
    if optimizer is not None and isinstance(ckpt, dict) and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt if isinstance(ckpt, dict) else {}


def load_mask_flow_checkpoint(path: str, model) -> None:
    ckpt = torch_load_checkpoint(path)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.mask_flow.load_state_dict(state, strict=False)
    print(f"loaded mask_flow checkpoint {path}")


def build_real_loader(args, training: bool = True):
    image_dir = args.train_image_dir if training else args.val_image_dir
    list_path = args.train_list if training else args.val_list
    return build_homo_flow_loader(
        repo_root=ROOT,
        list_path=list_path,
        image_dir=image_dir,
        crop_size=(args.crop_h, args.crop_w),
        full_size=(args.full_h, args.full_w),
        rho=args.rho,
        shift=args.shift,
        batch_size=args.batch_size,
        shuffle=training,
        num_workers=args.num_workers,
        training=training,
        horizontal_flip_aug=args.horizontal_flip_aug if training else False,
        coordinate_dir=args.coordinate_dir if not training else None,
        max_items=args.max_train_items if training else args.max_val_items,
        seed=None if training else args.seed,
    )


def compute_homography_loss(output: Dict, use_masks: bool, use_open: bool) -> Dict[str, torch.Tensor]:
    img1_fea = output["img1_patch_fea"]
    img2_fea = output["img2_patch_fea"]
    warp_img1_fea = output["warp_img1_patch_fea"]
    warp_img2_fea = output["warp_img2_patch_fea"]

    if use_masks and output["img1_patch_mask"] is not None:
        trusted_f, trusted_b = bidirectional_mask_intersection(
            output["img1_patch_mask"],
            output["img2_patch_mask"],
            output["warp_img1_patch_mask"],
            output["warp_img2_patch_mask"],
            use_close=use_open,
        )
    else:
        trusted_f = torch.ones_like(img2_fea[:, :1])
        trusted_b = torch.ones_like(img1_fea[:, :1])

    forward = masked_triplet_alignment_loss(
        anchor=img2_fea,
        positive=warp_img1_fea,
        negative=img1_fea,
        mask=trusted_f,
    )
    backward = masked_triplet_alignment_loss(
        anchor=img1_fea,
        positive=warp_img2_fea,
        negative=img2_fea,
        mask=trusted_b,
    )
    fil = feature_identity_loss(
        output["img1_patch_warp_fea"],
        output["img2_patch_warp_fea"],
        output["warp_img1_patch_fea"],
        output["warp_img2_patch_fea"],
    )
    return {
        "align_forward": forward,
        "align_backward": backward,
        "fil": fil,
        "align_total": forward + backward + fil,
        "trusted_f_mean": trusted_f.mean().detach(),
        "trusted_b_mean": trusted_b.mean().detach(),
    }


def compute_pseudo_mask_flow_loss(model, output: Dict, data_batch: Dict, args) -> Dict[str, torch.Tensor]:
    if getattr(model, "mask_flow", None) is None:
        zero = output["img1_patch_fea"].new_tensor(0.0)
        return {"fm": zero, "pseudo_bce": zero, "pseudo_dice": zero, "pseudo_conf": zero}

    img1_patch = data_batch["imgs_gray_patch"][:, :1]
    img2_patch = data_batch["imgs_gray_patch"][:, 1:]
    target_1, conf_1 = pseudo_mask_from_residuals(
        output["img1_patch_fea"].detach(),
        output["warp_img2_patch_fea"].detach(),
        img1_patch.detach(),
        output["warp_img2_patch"].detach(),
        tau=args.pseudo_tau,
        temperature=args.pseudo_temperature,
        close_kernel=args.pseudo_close_kernel,
    )
    target_2, conf_2 = pseudo_mask_from_residuals(
        output["img2_patch_fea"].detach(),
        output["warp_img1_patch_fea"].detach(),
        img2_patch.detach(),
        output["warp_img1_patch"].detach(),
        tau=args.pseudo_tau,
        temperature=args.pseudo_temperature,
        close_kernel=args.pseudo_close_kernel,
    )
    loss_1 = flow_matching_loss(
        model.mask_flow,
        output["img1_mask_cond"],
        target_1,
        noise_sigma=args.mask_flow_noise_sigma,
        confidence=conf_1,
    )
    loss_2 = flow_matching_loss(
        model.mask_flow,
        output["img2_mask_cond"],
        target_2,
        noise_sigma=args.mask_flow_noise_sigma,
        confidence=conf_2,
    )
    sample_1 = model.mask_flow.sample(
        output["img1_mask_cond"],
        steps=args.mask_flow_steps,
        solver=args.mask_flow_solver,
        init=args.mask_flow_init,
        requires_grad=True,
    )
    sample_2 = model.mask_flow.sample(
        output["img2_mask_cond"],
        steps=args.mask_flow_steps,
        solver=args.mask_flow_solver,
        init=args.mask_flow_init,
        requires_grad=True,
    )
    bce = 0.5 * (
        weighted_bce_loss(sample_1, target_1, conf_1)
        + weighted_bce_loss(sample_2, target_2, conf_2)
    )
    dice = 0.5 * (
        soft_dice_loss(sample_1, target_1, conf_1)
        + soft_dice_loss(sample_2, target_2, conf_2)
    )
    return {
        "fm": 0.5 * (loss_1 + loss_2),
        "pseudo_bce": bce,
        "pseudo_dice": dice,
        "pseudo_conf": 0.5 * (conf_1.mean().detach() + conf_2.mean().detach()),
        "pseudo_area": 0.5 * (target_1.mean().detach() + target_2.mean().detach()),
        "pseudo_sample_mean": 0.5 * (sample_1.mean().detach() + sample_2.mean().detach()),
    }


def compute_mask_regularizers(output: Dict, args) -> Dict[str, torch.Tensor]:
    if output["img1_patch_mask"] is None:
        zero = output["img1_patch_fea"].new_tensor(0.0)
        return {"mask_area": zero, "mask_tv": zero, "mask_entropy": zero, "mask_mean": zero}
    mask1, mask2 = output["img1_patch_mask"], output["img2_patch_mask"]
    area = 0.5 * (
        mask_area_loss(mask1, args.mask_min_area, args.mask_max_area)
        + mask_area_loss(mask2, args.mask_min_area, args.mask_max_area)
    )
    tv = 0.5 * (mask_total_variation_loss(mask1) + mask_total_variation_loss(mask2))
    entropy = 0.5 * (mask_entropy_loss(mask1) + mask_entropy_loss(mask2))
    return {
        "mask_area": area,
        "mask_tv": tv,
        "mask_entropy": entropy,
        "mask_mean": 0.5 * (mask1.mean().detach() + mask2.mean().detach()),
    }


def weighted_bce_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    confidence: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    loss = F.binary_cross_entropy(pred.clamp(1e-6, 1.0 - 1e-6), target, reduction="none")
    if confidence is not None:
        if confidence.shape[-2:] != loss.shape[-2:]:
            confidence = F.interpolate(confidence, size=loss.shape[-2:], mode="bilinear", align_corners=False)
        loss = loss * confidence.clamp_min(0.0)
    return loss.mean()


def soft_dice_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    confidence: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    if confidence is not None:
        if confidence.shape[-2:] != pred.shape[-2:]:
            confidence = F.interpolate(confidence, size=pred.shape[-2:], mode="bilinear", align_corners=False)
        confidence = confidence.clamp_min(0.0)
        pred = pred * confidence
        target = target * confidence
    pred = pred.flatten(1)
    target = target.flatten(1)
    intersection = (pred * target).sum(dim=1)
    denom = pred.sum(dim=1) + target.sum(dim=1)
    return (1.0 - (2.0 * intersection + eps) / (denom + eps)).mean()


def log_losses(prefix: str, epoch: int, step: int, losses: Dict[str, torch.Tensor]) -> None:
    pieces = [f"{prefix} epoch={epoch:03d} step={step:06d}"]
    for key, value in losses.items():
        if torch.is_tensor(value):
            value = value.detach().float().mean().item()
        pieces.append(f"{key}={value:.5f}")
    print(" ".join(pieces))


@torch.no_grad()
def validate_epoch(model, loader, device, args, epoch: int, joint: bool) -> Dict[str, float]:
    model.eval()
    totals = {}
    count = 0
    for count, data_batch in enumerate(loader, start=1):
        data_batch = batch_to_device(data_batch, device)
        output = model(data_batch)
        h_losses = compute_homography_loss(output, use_masks=joint, use_open=args.use_open)
        mask_losses = compute_mask_regularizers(output, args) if joint else {}
        total = args.lambda_align * h_losses["align_total"]
        if joint:
            total = total + args.lambda_area * mask_losses["mask_area"] + args.lambda_tv * mask_losses["mask_tv"]
        log_dict = {"total": total, **h_losses, **mask_losses}
        for key, value in log_dict.items():
            if torch.is_tensor(value):
                totals[key] = totals.get(key, 0.0) + value.detach().float().mean().item()
        if args.max_val_steps and count >= args.max_val_steps:
            break
    if count == 0:
        return {}
    averaged = {key: value / count for key, value in totals.items()}
    log_losses("val_joint" if joint else "val_pretrain", epoch, count, averaged)
    return averaged


def train_homography_epoch(model, loader, optimizer, device, args, epoch: int, joint: bool) -> int:
    model.train()
    step_count = 0
    for step_count, data_batch in enumerate(loader, start=1):
        data_batch = batch_to_device(data_batch, device)
        output = model(data_batch)

        h_losses = compute_homography_loss(output, use_masks=joint, use_open=args.use_open)
        mask_losses = compute_mask_regularizers(output, args) if joint else {}
        pseudo_losses = compute_pseudo_mask_flow_loss(model, output, data_batch, args) if joint else {}

        total = args.lambda_align * h_losses["align_total"]
        if joint:
            total = total + args.lambda_fm * pseudo_losses["fm"]
            total = total + args.lambda_mask_bce * pseudo_losses["pseudo_bce"]
            total = total + args.lambda_mask_dice * pseudo_losses["pseudo_dice"]
            total = total + args.lambda_area * mask_losses["mask_area"]
            total = total + args.lambda_tv * mask_losses["mask_tv"]
            if epoch >= args.entropy_warmup_epochs:
                total = total + args.lambda_entropy * mask_losses["mask_entropy"]

        optimizer.zero_grad(set_to_none=True)
        total.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step_count == 1 or step_count % args.print_freq == 0:
            log_dict = {"total": total, **h_losses, **mask_losses, **pseudo_losses}
            log_losses("joint" if joint else "pretrain", epoch, step_count, log_dict)

        if args.max_steps_per_epoch and step_count >= args.max_steps_per_epoch:
            break
    return step_count


def train_synthetic_mask_epoch(mask_flow, optimizer, loader, device, args, epoch: int) -> int:
    mask_flow.train()
    step_count = 0
    for step_count, batch in enumerate(loader, start=1):
        batch = batch_to_device(batch, device)
        imgs = batch["imgs_gray_patch"]
        img1, img2 = imgs[:, :1], imgs[:, 1:]
        flow = batch["dominant_flow"]
        target = batch["img2_dominant_mask"]

        warped_img1 = warp_with_flow(img1, flow)
        cond = build_mask_condition(img2, warped_img1, img2, warped_img1, flow, detach=False)
        fm = flow_matching_loss(mask_flow, cond, target, noise_sigma=args.mask_flow_noise_sigma)
        sampled = mask_flow.sample(cond, steps=args.mask_flow_steps, solver=args.mask_flow_solver, requires_grad=True)
        bce = weighted_bce_loss(sampled, target)
        dice = soft_dice_loss(sampled, target)
        area = mask_area_loss(sampled, args.mask_min_area, args.mask_max_area)
        tv = mask_total_variation_loss(sampled)
        total = args.lambda_fm * fm
        total = total + args.lambda_mask_bce * bce + args.lambda_mask_dice * dice
        total = total + args.lambda_area * area + args.lambda_tv * tv

        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()

        if step_count == 1 or step_count % args.print_freq == 0:
            log_losses(
                "synthetic_mask",
                epoch,
                step_count,
                {
                    "total": total,
                    "fm": fm,
                    "bce": bce,
                    "dice": dice,
                    "area": area,
                    "tv": tv,
                    "mask_mean": sampled.mean(),
                    "target_mean": target.mean(),
                },
            )
        if args.max_steps_per_epoch and step_count >= args.max_steps_per_epoch:
            break
    return step_count


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


def run_synthetic_mask(args, device) -> Path:
    params = default_homo_params(args, mode="joint")
    mask_flow = DominantMaskFlow(
        cond_channels=7,
        base_channels=args.mask_flow_base_channels,
        channel_mults=tuple(args.mask_flow_channel_mults),
        time_dim=args.mask_flow_time_dim,
        temperature=args.mask_flow_temperature,
        noise_sigma=args.mask_flow_noise_sigma,
    ).to(device)
    if args.mask_flow_checkpoint and Path(args.mask_flow_checkpoint).exists():
        ckpt = torch_load_checkpoint(args.mask_flow_checkpoint)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        mask_flow.load_state_dict(state, strict=False)
        print(f"loaded mask_flow checkpoint {args.mask_flow_checkpoint}")

    config = SyntheticFlowMaskConfig(
        height=args.crop_h,
        width=args.crop_w,
        length=max(args.synthetic_length, args.batch_size * max(args.max_steps_per_epoch, 1)),
        max_translation=args.synthetic_max_translation,
        min_outliers=args.synthetic_min_outliers,
        max_outliers=args.synthetic_max_outliers,
        min_outlier_size=args.synthetic_min_outlier_size,
        max_outlier_size=args.synthetic_max_outlier_size,
        blur_kernel=args.synthetic_blur_kernel,
        seed=args.seed,
    )
    loader = build_synthetic_flow_mask_loader(
        config,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    optimizer = torch.optim.AdamW(mask_flow.parameters(), lr=args.lr_mask, weight_decay=args.weight_decay)
    global_steps = 0
    for epoch in range(1, args.epochs + 1):
        global_steps += train_synthetic_mask_epoch(mask_flow, optimizer, loader, device, args, epoch)
        save_checkpoint(args.output_dir / f"mask_flow_epoch_{epoch:03d}.pth", mask_flow, optimizer, epoch, global_steps, params, args)
    final_path = args.output_dir / "mask_flow_final.pth"
    save_checkpoint(final_path, mask_flow, optimizer, args.epochs, global_steps, params, args)
    return final_path


def run_real_training(args, device, mode: str) -> Path:
    params = default_homo_params(args, mode=mode)
    model = Ms_Transformer(params=params).to(device)
    if args.resume:
        load_checkpoint(args.resume, model, strict=False)
    if mode == "joint" and args.mask_flow_checkpoint:
        load_mask_flow_checkpoint(args.mask_flow_checkpoint, model)

    lr = args.lr_joint if mode == "joint" else args.lr_homo
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    loader = build_real_loader(args, training=True)
    val_loader = None
    if not args.skip_val and args.val_list and args.val_image_dir:
        try:
            val_loader = build_real_loader(args, training=False)
        except (FileNotFoundError, ValueError) as exc:
            print(f"validation disabled: {exc}")
    global_steps = 0
    for epoch in range(1, args.epochs + 1):
        global_steps += train_homography_epoch(model, loader, optimizer, device, args, epoch, joint=(mode == "joint"))
        if val_loader is not None and args.val_freq > 0 and epoch % args.val_freq == 0:
            validate_epoch(model, val_loader, device, args, epoch, joint=(mode == "joint"))
        save_checkpoint(args.output_dir / f"{mode}_epoch_{epoch:03d}.pth", model, optimizer, epoch, global_steps, params, args)
    final_path = args.output_dir / f"{mode}_final.pth"
    save_checkpoint(final_path, model, optimizer, args.epochs, global_steps, params, args)
    return final_path


def parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(v.strip()) for v in value.split(",") if v.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full training for transformer homography with flow-matching masks.")
    parser.add_argument("--mode", choices=["synthetic_mask", "pretrain_homo", "joint", "all"], default="joint")
    parser.add_argument("--output-dir", type=Path, default=Path("new_approach/checkpoints"))
    parser.add_argument("--resume", default=None)
    parser.add_argument("--mask-flow-checkpoint", default=None)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--seed", type=int, default=230)

    parser.add_argument("--train-list", default="Data/Train_List.txt")
    parser.add_argument("--val-list", default="Data/Test_List.txt")
    parser.add_argument("--train-image-dir", default="Data/Train")
    parser.add_argument("--val-image-dir", default="Data/Test")
    parser.add_argument("--coordinate-dir", default="Data/Coordinate-v2/Coordinate-v2")
    parser.add_argument("--full-h", type=int, default=360)
    parser.add_argument("--full-w", type=int, default=640)
    parser.add_argument("--crop-h", type=int, default=320)
    parser.add_argument("--crop-w", type=int, default=512)
    parser.add_argument("--rho", type=int, default=16)
    parser.add_argument("--shift", type=int, default=8)
    parser.add_argument("--horizontal-flip-aug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-items", type=int, default=None)
    parser.add_argument("--max-val-items", type=int, default=None)
    parser.add_argument("--max-steps-per-epoch", type=int, default=0)
    parser.add_argument("--max-val-steps", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--synthetic-length", type=int, default=10000)
    parser.add_argument("--synthetic-max-translation", type=int, default=12)
    parser.add_argument("--synthetic-min-outliers", type=int, default=2)
    parser.add_argument("--synthetic-max-outliers", type=int, default=5)
    parser.add_argument("--synthetic-min-outlier-size", type=int, default=64)
    parser.add_argument("--synthetic-max-outlier-size", type=int, default=180)
    parser.add_argument("--synthetic-blur-kernel", type=int, default=7)
    parser.add_argument("--val-freq", type=int, default=1)
    parser.add_argument("--skip-val", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--embed-dim", type=int, default=24)
    parser.add_argument("--depths", type=parse_ints, default=(2, 4, 6))
    parser.add_argument("--layer-depth", type=parse_ints, default=(3, 2, 1))
    parser.add_argument("--num-heads", type=parse_ints, default=(3, 12, 24))
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=float, default=3.0)

    parser.add_argument("--mask-method", choices=["flow_matching", "homogan_cnn", "none"], default="flow_matching")
    parser.add_argument("--mask-flow-base-channels", type=int, default=32)
    parser.add_argument("--mask-flow-channel-mults", type=parse_ints, default=(1, 2, 4, 4))
    parser.add_argument("--mask-flow-time-dim", type=int, default=128)
    parser.add_argument("--mask-flow-steps", type=int, default=1)
    parser.add_argument("--mask-flow-solver", choices=["euler", "heun"], default="euler")
    parser.add_argument("--mask-flow-init", choices=["zero", "noise"], default="zero")
    parser.add_argument("--mask-flow-temperature", type=float, default=1.0)
    parser.add_argument("--mask-flow-noise-sigma", type=float, default=1.0)
    parser.add_argument("--mask-flow-detach-condition", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mask-flow-sample-grad", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--return-h-matrix", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--lr-homo", type=float, default=1e-4)
    parser.add_argument("--lr-mask", type=float, default=1e-4)
    parser.add_argument("--lr-joint", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lambda-align", type=float, default=1.0)
    parser.add_argument("--lambda-fm", type=float, default=0.5)
    parser.add_argument("--lambda-mask-bce", type=float, default=2.0)
    parser.add_argument("--lambda-mask-dice", type=float, default=1.0)
    parser.add_argument("--lambda-area", type=float, default=0.05)
    parser.add_argument("--lambda-tv", type=float, default=0.01)
    parser.add_argument("--lambda-entropy", type=float, default=0.001)
    parser.add_argument("--entropy-warmup-epochs", type=int, default=2)
    parser.add_argument("--mask-min-area", type=float, default=0.15)
    parser.add_argument("--mask-max-area", type=float, default=0.90)
    parser.add_argument("--pseudo-tau", type=float, default=1.0)
    parser.add_argument("--pseudo-temperature", type=float, default=0.5)
    parser.add_argument("--pseudo-close-kernel", type=int, default=5)
    parser.add_argument("--use-open", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--print-freq", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "args.json").write_text(json.dumps(vars(args), indent=2, default=str))
    device = resolve_device(args.device)
    print(f"using device: {device}")

    if args.mode == "synthetic_mask":
        run_synthetic_mask(args, device)
    elif args.mode == "pretrain_homo":
        run_real_training(args, device, mode="pretrain_homo")
    elif args.mode == "joint":
        run_real_training(args, device, mode="joint")
    else:
        mask_path = run_synthetic_mask(args, device)
        args.mask_flow_checkpoint = str(mask_path)
        pretrain_path = run_real_training(args, device, mode="pretrain_homo")
        args.resume = str(pretrain_path)
        run_real_training(args, device, mode="joint")


if __name__ == "__main__":
    main()
