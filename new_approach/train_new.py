import argparse
import json
import math
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
from new_approach.modules.transformerHomo import Ms_Transformer, get_warp_flow


def default_homo_params(args, mode: str) -> SimpleNamespace:
    pretrain_phase = mode == "pretrain_homo"
    mask_method = "none" if pretrain_phase else args.mask_method
    feature_channels = int(args.feature_channels)
    corr_channels = (2 * int(args.corr_radius) + 1) ** 2 if args.use_correlation else 0
    in_chans = 2 * feature_channels + corr_channels
    return SimpleNamespace(
        net_type="HomoGAN",
        crop_size=[args.crop_h, args.crop_w],
        in_channels=2,
        patch_size=args.patch_size,
        rho=args.rho,
        in_chans=in_chans,
        est_ref_channels=feature_channels,
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
        refine_iters=args.refine_iters,
        refine_detach_between=args.refine_detach_between,
        refine_supervision=args.refine_supervision,
        refine_gamma=args.refine_gamma,
        feature_channels=feature_channels,
        use_correlation=args.use_correlation,
        corr_radius=args.corr_radius,
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


def apply_foreground_paste(batch: Dict, args) -> Dict:
    prob = float(getattr(args, "fg_paste_prob", 0.0))
    if prob <= 0.0:
        return batch
    imgs_full = batch["imgs_gray_full"].clone()
    imgs_patch = batch["imgs_gray_patch"].clone()
    starts = batch["start"]
    batch_size, _, full_h, full_w = imgs_full.shape
    _, _, crop_h, crop_w = imgs_patch.shape
    paste_mask = torch.zeros(batch_size, 1, crop_h, crop_w, device=imgs_patch.device, dtype=imgs_patch.dtype)
    min_count = int(getattr(args, "fg_paste_min", 1))
    max_count = max(int(getattr(args, "fg_paste_max", 4)), min_count)
    size_min = int(getattr(args, "fg_size_min", 16))
    size_max = max(int(getattr(args, "fg_size_max", 64)), size_min)

    for batch_idx in range(batch_size):
        if torch.rand((), device=imgs_full.device).item() > prob:
            continue
        count = int(torch.randint(min_count, max_count + 1, (1,), device=imgs_full.device).item())
        start_x = int(starts[batch_idx, 0, 0, 0].round().item())
        start_y = int(starts[batch_idx, 1, 0, 0].round().item())
        for _ in range(count):
            patch_h = int(torch.randint(size_min, size_max + 1, (1,), device=imgs_full.device).item())
            patch_w = int(torch.randint(size_min, size_max + 1, (1,), device=imgs_full.device).item())
            patch_h = min(patch_h, full_h)
            patch_w = min(patch_w, full_w)
            dst_y = int(torch.randint(0, full_h - patch_h + 1, (1,), device=imgs_full.device).item())
            dst_x = int(torch.randint(0, full_w - patch_w + 1, (1,), device=imgs_full.device).item())
            src_idx = (batch_idx + int(torch.randint(1, max(batch_size, 2), (1,), device=imgs_full.device).item())) % batch_size
            src_y = int(torch.randint(0, full_h - patch_h + 1, (1,), device=imgs_full.device).item())
            src_x = int(torch.randint(0, full_w - patch_w + 1, (1,), device=imgs_full.device).item())
            imgs_full[batch_idx, 1:2, dst_y:dst_y + patch_h, dst_x:dst_x + patch_w] = imgs_full[
                src_idx, 0:1, src_y:src_y + patch_h, src_x:src_x + patch_w
            ]

            y0 = max(dst_y, start_y)
            x0 = max(dst_x, start_x)
            y1 = min(dst_y + patch_h, start_y + crop_h)
            x1 = min(dst_x + patch_w, start_x + crop_w)
            if y1 > y0 and x1 > x0:
                paste_mask[batch_idx, :, y0 - start_y:y1 - start_y, x0 - start_x:x1 - start_x] = 1.0

        imgs_patch[batch_idx, 1:2] = imgs_full[batch_idx, 1:2, start_y:start_y + crop_h, start_x:start_x + crop_w]

    batch = dict(batch)
    batch["imgs_gray_full"] = imgs_full
    batch["imgs_gray_patch"] = imgs_patch
    batch["fg_paste_mask_patch"] = paste_mask
    return batch


def foreground_mask_loss(output: Dict, data_batch: Dict) -> torch.Tensor:
    mask = data_batch.get("fg_paste_mask_patch")
    pred = output.get("img2_patch_mask")
    if mask is None or pred is None:
        return output["img2_patch_fea"].new_tensor(0.0)
    if mask.shape[-2:] != pred.shape[-2:]:
        mask = F.interpolate(mask, size=pred.shape[-2:], mode="nearest")
    denom = mask.flatten(1).sum(dim=1).clamp_min(1.0)
    loss = F.binary_cross_entropy(pred.clamp(1e-6, 1.0 - 1e-6), torch.zeros_like(pred), reduction="none")
    return ((loss * mask).flatten(1).sum(dim=1) / denom).mean()


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


def compute_homography_loss(output: Dict, use_masks: bool, use_open: bool, args=None) -> Dict[str, torch.Tensor]:
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

    robust = getattr(args, "align_robust", "none") if args is not None else "none"
    trunc_frac = getattr(args, "align_trunc_frac", 0.15) if args is not None else 0.15
    tukey_c = getattr(args, "align_tukey_c", 4.685) if args is not None else 4.685
    use_intermediate = bool(getattr(args, "refine_supervision", False)) if args is not None else False
    iters_b = output.get("warp_img1_patch_fea_iters")
    iters_f = output.get("warp_img2_patch_fea_iters")
    if use_intermediate and iters_f is not None and iters_b is not None:
        gamma = float(getattr(args, "refine_gamma", 0.8))
        total_w = 0.0
        forward = img2_fea.new_tensor(0.0)
        backward = img1_fea.new_tensor(0.0)
        total_iters = min(len(iters_f), len(iters_b))
        for idx in range(total_iters):
            weight = gamma ** (total_iters - 1 - idx)
            total_w += weight
            forward = forward + weight * masked_triplet_alignment_loss(
                anchor=img2_fea,
                positive=iters_b[idx],
                negative=img1_fea,
                mask=trusted_f,
                robust=robust,
                trunc_frac=trunc_frac,
                tukey_c=tukey_c,
            )
            backward = backward + weight * masked_triplet_alignment_loss(
                anchor=img1_fea,
                positive=iters_f[idx],
                negative=img2_fea,
                mask=trusted_b,
                robust=robust,
                trunc_frac=trunc_frac,
                tukey_c=tukey_c,
            )
        forward = forward / max(total_w, 1e-6)
        backward = backward / max(total_w, 1e-6)
    else:
        forward = masked_triplet_alignment_loss(
            anchor=img2_fea,
            positive=warp_img1_fea,
            negative=img1_fea,
            mask=trusted_f,
            robust=robust,
            trunc_frac=trunc_frac,
            tukey_c=tukey_c,
        )
        backward = masked_triplet_alignment_loss(
            anchor=img1_fea,
            positive=warp_img2_fea,
            negative=img2_fea,
            mask=trusted_b,
            robust=robust,
            trunc_frac=trunc_frac,
            tukey_c=tukey_c,
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
    if getattr(args, "pseudo_agreement", True):
        flow_f = output["flow_f"].detach()
        flow_b = output["flow_b"].detach()
        if flow_f.shape[-1] == 2:
            flow_f = flow_f.permute(0, 3, 1, 2).contiguous()
        if flow_b.shape[-1] == 2:
            flow_b = flow_b.permute(0, 3, 1, 2).contiguous()
        start = data_batch["start"]
        target_2_to_1 = get_warp_flow(target_2, flow_f, start)
        target_1_to_2 = get_warp_flow(target_1, flow_b, start)
        conf_2_to_1 = get_warp_flow(conf_2, flow_f, start)
        conf_1_to_2 = get_warp_flow(conf_1, flow_b, start)
        target_1 = (target_1 * target_2_to_1).clamp(0.0, 1.0)
        target_2 = (target_2 * target_1_to_2).clamp(0.0, 1.0)
        conf_1 = (conf_1 * conf_2_to_1).clamp(0.0, 1.0)
        conf_2 = (conf_2 * conf_1_to_2).clamp(0.0, 1.0)
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
        h_losses = compute_homography_loss(output, use_masks=joint, use_open=args.use_open, args=args)
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


def train_homography_epoch(
    model,
    loader,
    optimizer,
    device,
    args,
    epoch: int,
    joint: bool,
    scheduler=None,
    scaler=None,
    amp_enabled: bool = False,
) -> int:
    model.train()
    step_count = 0
    for step_count, data_batch in enumerate(loader, start=1):
        data_batch = batch_to_device(data_batch, device)
        if joint:
            data_batch = apply_foreground_paste(data_batch, args)

        with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
            output = model(data_batch)

            h_losses = compute_homography_loss(output, use_masks=joint, use_open=args.use_open, args=args)
            mask_losses = compute_mask_regularizers(output, args) if joint else {}
            pseudo_losses = compute_pseudo_mask_flow_loss(model, output, data_batch, args) if joint else {}
            fg_loss = foreground_mask_loss(output, data_batch) if joint else h_losses["align_total"].new_tensor(0.0)

            total = args.lambda_align * h_losses["align_total"]
            if joint:
                total = total + args.lambda_fm * pseudo_losses["fm"]
                total = total + args.lambda_mask_bce * pseudo_losses["pseudo_bce"]
                total = total + args.lambda_mask_dice * pseudo_losses["pseudo_dice"]
                total = total + args.lambda_area * mask_losses["mask_area"]
                total = total + args.lambda_tv * mask_losses["mask_tv"]
                total = total + args.lambda_fg * fg_loss
                if epoch >= args.entropy_warmup_epochs:
                    total = total + args.lambda_entropy * mask_losses["mask_entropy"]

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None and amp_enabled:
            scaler.scale(total).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            total.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if step_count == 1 or step_count % args.print_freq == 0:
            log_dict = {"total": total, **h_losses, **mask_losses, **pseudo_losses, "fg": fg_loss}
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


def build_real_optimizer(model, args, mode: str):
    if mode != "joint":
        return torch.optim.AdamW(model.parameters(), lr=args.lr_homo, weight_decay=args.weight_decay)

    mask_params = []
    homo_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("mask_flow") or name.startswith("mask_pred"):
            mask_params.append(param)
        else:
            homo_params.append(param)
    groups = []
    if homo_params:
        groups.append({"params": homo_params, "lr": args.lr_homo})
    if mask_params:
        groups.append({"params": mask_params, "lr": args.lr_mask})
    return torch.optim.AdamW(groups, lr=args.lr_joint, weight_decay=args.weight_decay)


def build_lr_scheduler(optimizer, args, total_steps: int):
    scheduler_name = str(getattr(args, "lr_scheduler", "none")).lower()
    if scheduler_name == "none":
        return None
    warmup_steps = max(int(getattr(args, "warmup_steps", 0)), 0)
    total_steps = max(int(total_steps), 1)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
        progress = min(max(progress, 0.0), 1.0)
        if scheduler_name == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        if scheduler_name == "linear":
            return 1.0 - progress
        raise ValueError(f"unsupported lr scheduler: {scheduler_name}")

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def run_real_training(args, device, mode: str) -> Path:
    params = default_homo_params(args, mode=mode)
    model = Ms_Transformer(params=params).to(device)
    if args.resume:
        load_checkpoint(args.resume, model, strict=False)
    if mode == "joint" and args.mask_flow_checkpoint:
        load_mask_flow_checkpoint(args.mask_flow_checkpoint, model)

    optimizer = build_real_optimizer(model, args, mode)
    loader = build_real_loader(args, training=True)
    steps_per_epoch = args.max_steps_per_epoch if args.max_steps_per_epoch else len(loader)
    scheduler = build_lr_scheduler(optimizer, args, total_steps=steps_per_epoch * args.epochs)
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    val_loader = None
    if not args.skip_val and args.val_list and args.val_image_dir:
        try:
            val_loader = build_real_loader(args, training=False)
        except (FileNotFoundError, ValueError) as exc:
            print(f"validation disabled: {exc}")
    global_steps = 0
    for epoch in range(1, args.epochs + 1):
        global_steps += train_homography_epoch(
            model,
            loader,
            optimizer,
            device,
            args,
            epoch,
            joint=(mode == "joint"),
            scheduler=scheduler,
            scaler=scaler,
            amp_enabled=amp_enabled,
        )
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
    parser.add_argument("--synthetic-max-outliers", type=int, default=6)
    parser.add_argument("--synthetic-min-outlier-size", type=int, default=16)
    parser.add_argument("--synthetic-max-outlier-size", type=int, default=80)
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
    parser.add_argument("--refine-iters", type=int, default=1)
    parser.add_argument("--refine-detach-between", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--refine-supervision", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--refine-gamma", type=float, default=0.8)
    parser.add_argument("--feature-channels", type=int, default=1)
    parser.add_argument("--use-correlation", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--corr-radius", type=int, default=4)

    parser.add_argument("--mask-method", choices=["flow_matching", "homogan_cnn", "none"], default="homogan_cnn")
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
    parser.add_argument("--lr-scheduler", choices=["none", "cosine", "linear"], default="none")
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lambda-align", type=float, default=1.0)
    parser.add_argument("--lambda-fm", type=float, default=0.5)
    parser.add_argument("--lambda-mask-bce", type=float, default=2.0)
    parser.add_argument("--lambda-mask-dice", type=float, default=1.0)
    parser.add_argument("--lambda-area", type=float, default=0.05)
    parser.add_argument("--lambda-tv", type=float, default=0.01)
    parser.add_argument("--lambda-entropy", type=float, default=0.001)
    parser.add_argument("--lambda-fg", type=float, default=0.1)
    parser.add_argument("--entropy-warmup-epochs", type=int, default=2)
    parser.add_argument("--mask-min-area", type=float, default=0.15)
    parser.add_argument("--mask-max-area", type=float, default=0.90)
    parser.add_argument("--pseudo-tau", type=float, default=1.0)
    parser.add_argument("--pseudo-temperature", type=float, default=0.5)
    parser.add_argument("--pseudo-close-kernel", type=int, default=5)
    parser.add_argument("--pseudo-agreement", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--align-robust", choices=["none", "charbonnier", "truncated", "tukey"], default="none")
    parser.add_argument("--align-trunc-frac", type=float, default=0.15)
    parser.add_argument("--align-tukey-c", type=float, default=4.685)
    parser.add_argument("--fg-paste-prob", type=float, default=0.0)
    parser.add_argument("--fg-paste-min", type=int, default=1)
    parser.add_argument("--fg-paste-max", type=int, default=4)
    parser.add_argument("--fg-size-min", type=int, default=16)
    parser.add_argument("--fg-size-max", type=int, default=64)
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
