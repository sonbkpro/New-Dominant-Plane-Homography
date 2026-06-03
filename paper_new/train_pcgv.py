"""Training entry point for PCGV.

Recommended first PCGV stage:

    python -m paper_new.train_pcgv \
        --coarse_ckpt new_approach/checkpoints/baseline/baseline_final.pth \
        --freeze_coarse \
        --epochs 2 --batch_size 16 --lr_pcgv 1e-4 --lr_features 2e-5 \
        --lambda_align 1.0 --lambda_fil 0.5 --lambda_coarse_flow 0.0 \
        --refine_blend_init 0.05 --lambda_reproj 0 --lambda_cycle 0 --lambda_vote 0

This starts from the loaded coarse baseline, initializes the PCGV feature stack
from that baseline, then trains the PCGV voting/refinement path with the
alignment+FIL objective. PCGV-specific losses are present but disabled until
their lambdas are set above zero.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import Dict

import torch
from torch.utils.data import DataLoader

from new_approach.dataset_baseline import TrainDataset, move_batch

from paper_new.evaluate_pcgv import evaluate
from paper_new.losses_pcgv import dominant_plane_loss, pcgv_loss
from paper_new.model_pcgv import build_pcgv, make_pcgv_params

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))


def _str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("1", "true", "yes", "y"):
        return True
    if value in ("0", "false", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("expected boolean")


def _add_bool_arg(parser, *names, default=False, **kwargs):
    parser.add_argument(*names, type=_str2bool, nargs="?", const=True, default=default, **kwargs)


def _flatten_config(data: Dict) -> Dict:
    if not isinstance(data, dict):
        raise ValueError("--config must contain a JSON object")
    defaults = {}
    for key, value in data.items():
        if key in ("pcgv", "loss", "train", "data", "eval"):
            if not isinstance(value, dict):
                raise ValueError(f"config section '{key}' must be a JSON object")
            defaults.update(value)
        elif key in ("model", "crop_h", "crop_w"):
            continue
        else:
            defaults[key] = value
    return defaults


def _apply_config_defaults(parser, path: str):
    with open(path) as f:
        data = json.load(f)
    defaults = _flatten_config(data)
    valid = {action.dest for action in parser._actions}
    unknown = sorted(key for key in defaults if key not in valid)
    if unknown:
        raise ValueError(f"unknown --config keys: {', '.join(unknown)}")
    parser.set_defaults(**defaults)


def parse_args():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=None)
    config_args, _ = config_parser.parse_known_args()

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None,
                    help="Strict JSON config whose defaults are loaded before explicit CLI overrides.")
    ap.add_argument("--train_list", default=os.path.join(ROOT, "Data/Train_List.txt"))
    ap.add_argument("--train_img_dir", default=os.path.join(ROOT, "Data/Train"))
    ap.add_argument("--test_list", default=os.path.join(ROOT, "Data/Test_List.txt"))
    ap.add_argument("--test_img_dir", default=os.path.join(ROOT, "Data/Test"))
    ap.add_argument("--coord_dir", default=os.path.join(ROOT, "Data/Coordinate-v2/Coordinate-v2"))
    ap.add_argument("--out_dir", default=os.path.join(ROOT, "paper_new/checkpoints/pcgv_v1"))
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--rho", type=int, default=16)
    ap.add_argument("--seed", type=int, default=230)
    _add_bool_arg(ap, "--amp", default=False)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--print_freq", type=int, default=20)
    ap.add_argument("--max_train_items", type=int, default=None)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=1)
    ap.add_argument("--eval_max_items", type=int, default=None)
    ap.add_argument("--stage", choices=("custom", "vote_pretrain"), default="custom",
                    help="Optional curriculum preset. Explicit command-line values override preset defaults.")
    ap.add_argument("--save_step_every", type=int, default=0,
                    help="also save pcgv_latest.pth every N optimizer steps; 0 disables mid-epoch saves")
    ap.add_argument("--nonfinite_patience", type=int, default=20,
                    help="abort after this many consecutive non-finite losses")
    _add_bool_arg(ap, "--fail_on_nonfinite", default=False,
                  help="abort immediately on NaN/Inf loss instead of skipping that batch")

    ap.add_argument("--coarse_ckpt", default=None,
                    help="Optional baseline checkpoint to load into the coarse HomoGAN initializer.")
    ap.add_argument("--resume", default=None,
                    help="Optional full PCGV checkpoint to resume for staged loss activation.")
    ap.add_argument("--resume_optimizer", type=_str2bool, default=True,
                    help="Restore optimizer state from --resume checkpoints when present.")
    ap.add_argument("--override_resume_lrs", type=_str2bool, default=True,
                    help="After restoring optimizer state, apply the LR values from this command line.")
    _add_bool_arg(ap, "--freeze_coarse", default=False,
                  help="Freeze the coarse baseline while training PCGV.")
    ap.add_argument("--init_pcgv_features_from_coarse", type=_str2bool, default=True,
                    help="Initialize PCGV shallow/pyramid features from the loaded coarse baseline.")
    _add_bool_arg(ap, "--freeze_pcgv_shallow", default=False,
                  help="Freeze the copied shallow feature extractor during PCGV training.")
    _add_bool_arg(ap, "--freeze_pcgv_pyramid", default=False,
                  help="Freeze the copied PCGV feature pyramid while leaving projection/voting layers trainable.")
    ap.add_argument("--init_mode", choices=("identity", "coarse_flow_corners"),
                    default="coarse_flow_corners")
    ap.add_argument("--override_baseline_keys", type=_str2bool, default=True,
                    help="Apply baseline alignment/FIL loss to PCGV final warps.")
    ap.add_argument("--pcgv_enabled", type=_str2bool, default=True)

    ap.add_argument("--lr_pcgv", type=float, default=1e-4)
    ap.add_argument("--lr_features", type=float, default=2e-5)
    ap.add_argument("--lr_coarse", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=1e-4)

    ap.add_argument("--num_iters", type=int, default=4)
    ap.add_argument("--radius", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--feat_dim", type=int, default=64)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--pyramid_embed_dim", type=int, default=24,
                    help="PCGV feature-pyramid base channels; 24 matches the HomoGAN coarse checkpoint.")
    ap.add_argument("--pyramid_layers", type=int, default=3)
    ap.add_argument("--pcgv_level", type=int, default=0)
    ap.add_argument("--use_transformer", type=_str2bool, default=False)
    ap.add_argument("--use_plane_token", type=_str2bool, default=True)
    ap.add_argument("--use_leverage", type=_str2bool, default=True)
    ap.add_argument("--use_uncertainty", type=_str2bool, default=True)
    ap.add_argument("--update_alpha", type=float, default=0.7)
    ap.add_argument("--detach_dlt", type=_str2bool, default=True,
                    help="Stop gradients through the weighted DLT/SVD solve; keeps training numerically stable.")
    ap.add_argument("--refine_blend_init", type=float, default=0.05,
                    help="Initial final blend from coarse H to PCGV-refined H; small values preserve the baseline early.")
    ap.add_argument("--learn_refine_blend", type=_str2bool, default=True,
                    help="Learn the final PCGV refinement blend during training.")
    ap.add_argument("--set_refine_blend", type=float, default=None,
                    help="Overwrite the loaded PCGV final blend, useful for staged refinement after --resume.")
    _add_bool_arg(ap, "--mask_refine", default=False,
                  help="Use the learned mask upsampler instead of pure bilinear vote upsampling.")
    ap.add_argument("--blend_start", type=float, default=0.05)
    ap.add_argument("--blend_final", type=float, default=0.05)
    ap.add_argument("--blend_warmup_steps", type=int, default=0)

    ap.add_argument("--objective", choices=("legacy", "dominant_plane"), default="legacy")
    ap.add_argument("--lambda_align", type=float, default=1.0)
    ap.add_argument("--lambda_fil", type=float, default=0.5)
    ap.add_argument("--lambda_cov", type=float, default=0.1)
    ap.add_argument("--lambda_coarse_flow", type=float, default=0.0,
                    help="Anchor PCGV final patch flow to the coarse baseline patch flow.")
    ap.add_argument("--lambda_reproj", type=float, default=0.0)
    ap.add_argument("--lambda_cycle", type=float, default=0.0)
    ap.add_argument("--lambda_vote", type=float, default=0.0)
    ap.add_argument("--lambda_tv", type=float, default=0.0)
    ap.add_argument("--lambda_area", type=float, default=0.0)
    ap.add_argument("--lambda_entropy", type=float, default=0.0)
    ap.add_argument("--lambda_cond", type=float, default=0.0)
    ap.add_argument("--lambda_uncertainty", type=float, default=0.0)
    ap.add_argument("--target_area", type=float, default=0.35)
    ap.add_argument("--coverage_floor", type=float, default=0.35)
    ap.add_argument("--coverage_mode", choices=("hinge", "log"), default="hinge")
    ap.add_argument("--vote_tau", type=float, default=2.0)
    ap.add_argument("--vote_target_mode", choices=("robust", "exp"), default="robust")
    ap.add_argument("--vote_residual_low_q", type=float, default=0.25)
    ap.add_argument("--vote_residual_high_q", type=float, default=0.75)
    ap.add_argument("--vote_corr_weight", type=float, default=0.35)
    ap.add_argument("--vote_gamma", type=float, default=1.5)
    ap.add_argument("--vote_min_confidence_weight", type=float, default=0.25)
    ap.add_argument("--vote_low_thresh", type=float, default=0.30)
    ap.add_argument("--vote_high_thresh", type=float, default=0.70)
    _add_bool_arg(ap, "--use_masked_align", default=False)
    if config_args.config:
        _apply_config_defaults(ap, config_args.config)
    return ap.parse_args()


def _cli_provided(name: str) -> bool:
    return f"--{name}" in sys.argv[1:]


def _apply_stage_defaults(args):
    if args.stage != "vote_pretrain":
        return
    defaults = {
        "freeze_coarse": True,
        "freeze_pcgv_shallow": True,
        "freeze_pcgv_pyramid": True,
        "set_refine_blend": 0.0,
        "lambda_align": 0.0,
        "lambda_fil": 0.0,
        "lambda_coarse_flow": 0.0,
        "lambda_reproj": 0.2,
        "lambda_vote": 0.5,
        "lambda_tv": 0.002,
        "lambda_area": 0.05,
        "lambda_entropy": 0.01,
        "lambda_cycle": 0.0,
        "lambda_cond": 0.0,
        "lambda_uncertainty": 0.0,
        "target_area": 0.35,
        "vote_target_mode": "robust",
        "vote_residual_low_q": 0.25,
        "vote_residual_high_q": 0.75,
        "vote_corr_weight": 0.35,
        "vote_gamma": 1.5,
        "vote_min_confidence_weight": 0.25,
        "vote_low_thresh": 0.30,
        "vote_high_thresh": 0.70,
    }
    applied = []
    for name, value in defaults.items():
        if _cli_provided(name):
            continue
        setattr(args, name, value)
        applied.append(f"{name}={value}")
    if applied:
        print("Applied vote_pretrain defaults: " + ", ".join(applied))


def _infer_epoch_from_path(path: str) -> int:
    match = re.search(r"epoch_(\d+)\.pth$", os.path.basename(path))
    return int(match.group(1)) if match else 0


def _clean_state_dict(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state.items()
    }


def _extract_model_state(checkpoint) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, torch.nn.Module):
        checkpoint = checkpoint.state_dict()
    elif isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in checkpoint:
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint format: {type(checkpoint)}")
    return _clean_state_dict(checkpoint)


def _load_compatible_state_dict(model, state: Dict[str, torch.Tensor], label: str):
    """Load matching tensors and report missing/unexpected/shape-mismatched keys."""
    model_state = model.state_dict()
    compatible = {}
    unexpected = []
    mismatched = []
    for key, value in state.items():
        target = model_state.get(key)
        if target is None:
            unexpected.append(key)
            continue
        if tuple(target.shape) != tuple(value.shape):
            mismatched.append(key)
            continue
        compatible[key] = value.to(device=target.device, dtype=target.dtype)
    model_state.update(compatible)
    model.load_state_dict(model_state, strict=True)
    missing = [key for key in model_state.keys() if key not in compatible]
    if missing:
        print(f"[warn] {label} missing {len(missing)} keys")
    if unexpected:
        print(f"[warn] {label} has {len(unexpected)} unexpected keys")
    if mismatched:
        print(f"[warn] {label} skipped {len(mismatched)} shape-mismatched keys")
        print("[warn] If this is an older PCGV checkpoint, try matching its --pyramid_embed_dim.")
    return missing, unexpected, mismatched


def _load_coarse(model, path: str, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = _extract_model_state(checkpoint)
    missing, unexpected = model.coarse.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] coarse checkpoint missing {len(missing)} keys")
    if unexpected:
        print(f"[warn] coarse checkpoint has {len(unexpected)} unexpected keys")


def _load_resume_model(model, path: str, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = _extract_model_state(checkpoint)
    _load_compatible_state_dict(model, state, "resume checkpoint")

    start_epoch = _infer_epoch_from_path(path) + 1
    start_step = 0
    start_global_step = None
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        if checkpoint.get("step") is None:
            start_epoch = int(checkpoint.get("epoch", start_epoch - 1)) + 1
        else:
            start_epoch = int(checkpoint["epoch"])
            start_step = int(checkpoint["step"]) + 1
        if checkpoint.get("global_step") is not None:
            start_global_step = int(checkpoint["global_step"])
    return checkpoint, start_epoch, start_step, start_global_step


def _optimizer_lr_map(args):
    return {
        "pcgv": args.lr_pcgv,
        "features": args.lr_features,
        "coarse": args.lr_coarse,
    }


def _apply_optimizer_lrs(opt, group_names, args):
    lr_map = _optimizer_lr_map(args)
    applied = []
    for group, name in zip(opt.param_groups, group_names):
        if name not in lr_map:
            continue
        group["lr"] = lr_map[name]
        applied.append(f"{name}={lr_map[name]:.3g}")
    if applied:
        print("Applied command-line learning rates after resume: " + ", ".join(applied))


def _restore_training_state(checkpoint, opt, scaler, args, group_names):
    if not isinstance(checkpoint, dict):
        return False
    restored = False
    if args.resume_optimizer and "optimizer" in checkpoint:
        try:
            opt.load_state_dict(checkpoint["optimizer"])
            restored = True
        except ValueError as exc:
            print(f"[warn] optimizer state was not restored: {exc}")
    elif not args.resume_optimizer:
        print("Skipping optimizer state restore by request; using fresh optimizer.")
    if restored and args.override_resume_lrs:
        _apply_optimizer_lrs(opt, group_names, args)
    if "scaler" in checkpoint and checkpoint["scaler"] is not None:
        try:
            scaler.load_state_dict(checkpoint["scaler"])
        except ValueError as exc:
            print(f"[warn] AMP scaler state was not restored: {exc}")
    return restored


def save_checkpoint(path, model, opt, scaler, epoch, args, step=None, global_step=None):
    torch.save({
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scaler": scaler.state_dict() if scaler.is_enabled() else None,
        "epoch": epoch,
        "step": step,
        "global_step": global_step,
        "args": vars(args),
    }, path)


def _build_optimizer(model, args):
    groups = []
    pcgv_params = [p for p in model.pcgv.parameters() if p.requires_grad]
    feature_params = [p for p in model.features.parameters() if p.requires_grad]
    mask_params = [p for p in model.mask_upsampler.parameters() if p.requires_grad]
    coarse_params = [p for p in model.coarse.parameters() if p.requires_grad]
    if pcgv_params or mask_params:
        groups.append({"name": "pcgv", "params": pcgv_params + mask_params, "lr": args.lr_pcgv})
    if feature_params:
        groups.append({"name": "features", "params": feature_params, "lr": args.lr_features})
    if coarse_params:
        groups.append({"name": "coarse", "params": coarse_params, "lr": args.lr_coarse})
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay)


def _scheduled_blend(args, global_step: int) -> float:
    g = min(1.0, float(global_step) / float(max(args.blend_warmup_steps, 1)))
    return float(args.blend_start + (args.blend_final - args.blend_start) * g)


def _gradients_are_finite(model):
    for name, param in model.named_parameters():
        if param.grad is not None and not torch.isfinite(param.grad).all():
            return False, name
    return True, None


def main():
    args = parse_args()
    _apply_stage_defaults(args)
    use_blend_schedule = args.blend_warmup_steps > 0
    if use_blend_schedule and args.learn_refine_blend:
        args.learn_refine_blend = False
        print("Disabled learn_refine_blend because a deterministic blend schedule is active.")
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device(args.device)
    params = make_pcgv_params(
        pcgv_enabled=args.pcgv_enabled,
        pcgv_feat_dim=args.feat_dim,
        pcgv_hidden_dim=args.hidden_dim,
        pcgv_num_iters=args.num_iters,
        pcgv_radius=args.radius,
        pcgv_temperature=args.temperature,
        pcgv_use_transformer=args.use_transformer,
        pcgv_use_plane_token=args.use_plane_token,
        pcgv_use_uncertainty=args.use_uncertainty,
        pcgv_use_leverage=args.use_leverage,
        pcgv_update_alpha=args.update_alpha,
        pcgv_detach_dlt=args.detach_dlt,
        pcgv_refine_blend_init=args.refine_blend_init,
        pcgv_learn_refine_blend=args.learn_refine_blend,
        pcgv_mask_refine=args.mask_refine,
        pcgv_freeze_coarse=args.freeze_coarse,
        pcgv_init_mode=args.init_mode,
        pcgv_override_baseline_keys=args.override_baseline_keys,
        pcgv_pyramid_embed_dim=args.pyramid_embed_dim,
        pcgv_pyramid_layers=args.pyramid_layers,
        pcgv_level=args.pcgv_level,
    )
    net = build_pcgv(params).to(device)
    resume_checkpoint = None
    start_epoch = 1
    start_step = 0
    resume_global_step = None
    if args.resume:
        resume_checkpoint, start_epoch, start_step, resume_global_step = _load_resume_model(net, args.resume, device)
        print(f"Resumed PCGV weights from {args.resume} at epoch {start_epoch}, step {start_step}")
    elif args.coarse_ckpt:
        _load_coarse(net, args.coarse_ckpt, device)
        if args.init_pcgv_features_from_coarse:
            report = net.init_pcgv_features_from_coarse()
            print(
                "Initialized PCGV features from coarse: "
                f"shallow copied={report['shallow_copied']} skipped={report['shallow_skipped']}, "
                f"pyramid copied={report['pyramid_copied']} skipped={report['pyramid_skipped']}"
            )
    if args.set_refine_blend is not None:
        applied_blend = net.set_refine_blend(args.set_refine_blend)
        print(f"Set PCGV refine blend to {applied_blend:.4f}")
    else:
        print(f"PCGV refine blend: {net.get_refine_blend():.4f}")
    if args.freeze_coarse:
        net.freeze_coarse()
        net.coarse.eval()
    if args.freeze_pcgv_shallow:
        net.freeze_pcgv_shallow()
    if args.freeze_pcgv_pyramid:
        net.freeze_pcgv_pyramid()

    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"PCGV built: {n_params/1e6:.2f}M trainable params, device={device}")
    if args.freeze_coarse and not (args.coarse_ckpt or args.resume):
        print("[warn] coarse model is frozen without --coarse_ckpt; this is only useful for smoke tests.")

    ds = TrainDataset(args.train_list, args.train_img_dir, rho=args.rho,
                      max_items=args.max_train_items)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, drop_last=True,
                        pin_memory=(device.type == "cuda"))
    print(f"Train pairs: {len(ds)}  steps/epoch: {len(loader)}")
    if resume_global_step is None:
        global_step = max(0, (start_epoch - 1) * len(loader) + start_step)
    else:
        global_step = resume_global_step

    opt = _build_optimizer(net, args)
    optimizer_group_names = [group.get("name", str(i)) for i, group in enumerate(opt.param_groups)]
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if resume_checkpoint is not None:
        restored = _restore_training_state(resume_checkpoint, opt, scaler, args, optimizer_group_names)
        if restored:
            print("Resumed optimizer/scaler state.")
        else:
            print("Optimizer/scaler state was not present or could not be restored; continuing with fresh optimizer.")
    if start_epoch > args.epochs:
        print(
            f"Checkpoint already reached epoch {start_epoch - 1}; nothing to train for --epochs {args.epochs}. "
            f"Use --epochs {start_epoch} or larger to continue."
        )
        return

    consecutive_nonfinite = 0
    skipped_nonfinite = 0
    for epoch in range(start_epoch, args.epochs + 1):
        net.train()
        if args.freeze_coarse:
            net.coarse.eval()
        if args.freeze_pcgv_shallow:
            net.features.shallow.eval()
        if args.freeze_pcgv_pyramid:
            net.features.pyramid.eval()
        t0 = time.time()
        epoch_skipped_nonfinite = 0
        for step, batch in enumerate(loader):
            if epoch == start_epoch and step < start_step:
                continue
            batch = move_batch(batch, device)
            opt.zero_grad(set_to_none=True)
            active_blend = net.get_refine_blend()
            if use_blend_schedule:
                active_blend = net.set_refine_blend(_scheduled_blend(args, global_step))
            amp_device = "cuda" if device.type == "cuda" else "cpu"
            with torch.amp.autocast(amp_device, enabled=use_amp):
                out = net(batch)
                if args.objective == "dominant_plane":
                    loss, logs = dominant_plane_loss(
                        out,
                        lambda_align=args.lambda_align,
                        lambda_cov=args.lambda_cov,
                        lambda_fil=args.lambda_fil,
                        lambda_tv=args.lambda_tv,
                        lambda_cycle=args.lambda_cycle,
                        lambda_entropy=args.lambda_entropy,
                        coverage_floor=args.coverage_floor,
                        coverage_mode=args.coverage_mode,
                    )
                else:
                    loss, logs = pcgv_loss(
                        out,
                        lambda_align=args.lambda_align,
                        lambda_fil=args.lambda_fil,
                        lambda_coarse_flow=args.lambda_coarse_flow,
                        lambda_reproj=args.lambda_reproj,
                        lambda_cycle=args.lambda_cycle,
                        lambda_vote=args.lambda_vote,
                        lambda_tv=args.lambda_tv,
                        lambda_area=args.lambda_area,
                        lambda_entropy=args.lambda_entropy,
                        lambda_cond=args.lambda_cond,
                        lambda_uncertainty=args.lambda_uncertainty,
                        target_area=args.target_area,
                        vote_tau=args.vote_tau,
                        vote_target_mode=args.vote_target_mode,
                        vote_residual_low_q=args.vote_residual_low_q,
                        vote_residual_high_q=args.vote_residual_high_q,
                        vote_corr_weight=args.vote_corr_weight,
                        vote_gamma=args.vote_gamma,
                        vote_min_confidence_weight=args.vote_min_confidence_weight,
                        vote_low_thresh=args.vote_low_thresh,
                        vote_high_thresh=args.vote_high_thresh,
                        use_masked_align=args.use_masked_align,
                    )
            logs["blend"] = active_blend
            if not torch.isfinite(loss):
                consecutive_nonfinite += 1
                skipped_nonfinite += 1
                epoch_skipped_nonfinite += 1
                msg = (f"ep{epoch} [{step}/{len(loader)}] non-finite loss; "
                       f"skipping batch (consecutive={consecutive_nonfinite}, total_skipped={skipped_nonfinite})")
                print(msg, flush=True)
                if args.fail_on_nonfinite or consecutive_nonfinite >= args.nonfinite_patience:
                    raise FloatingPointError(msg)
                continue
            consecutive_nonfinite = 0

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            grads_finite, bad_grad_name = _gradients_are_finite(net)
            if not grads_finite:
                consecutive_nonfinite += 1
                skipped_nonfinite += 1
                epoch_skipped_nonfinite += 1
                opt.zero_grad(set_to_none=True)
                scaler.update()
                msg = (f"ep{epoch} [{step}/{len(loader)}] non-finite gradient"
                       f" in {bad_grad_name}; skipping optimizer step "
                       f"(consecutive={consecutive_nonfinite}, total_skipped={skipped_nonfinite})")
                print(msg, flush=True)
                if args.fail_on_nonfinite or consecutive_nonfinite >= args.nonfinite_patience:
                    raise FloatingPointError(msg)
                continue
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
            global_step += 1

            if args.save_step_every and (step + 1) % args.save_step_every == 0:
                save_checkpoint(
                    os.path.join(args.out_dir, "pcgv_latest.pth"),
                    net, opt, scaler, epoch, args, step=step, global_step=global_step,
                )

            if step % args.print_freq == 0:
                msg = (
                    f"ep{epoch} [{step}/{len(loader)}] "
                    f"total={logs['total']:.4f} align={logs['align']:.4f} "
                    f"fil={logs['fil']:.4f}"
                )
                if "coverage" in logs:
                    msg += f" cov={logs['coverage']:.4f}"
                if "coarse_flow" in logs:
                    msg += f" coarse={logs['coarse_flow']:.4f}"
                if "reproj" in logs:
                    msg += f" reproj={logs['reproj']:.4f}"
                if "vote" in logs:
                    msg += f" vote={logs['vote']:.4f}"
                if "mean_vote_f" in logs:
                    msg += f" mean_vote_f={logs['mean_vote_f']:.3f}"
                elif "mean_votes_f" in logs:
                    msg += f" mean_vote_f={logs['mean_votes_f']:.3f}"
                if "mean_residuals_f" in logs:
                    msg += f" residual_f={logs['mean_residuals_f']:.3f}"
                if "blend" in logs:
                    msg += f" blend={logs['blend']:.3f}"
                elif "refine_blend_f" in logs:
                    msg += f" blend_f={logs['refine_blend_f']:.3f}"
                if "mask_area_f" in logs:
                    msg += f" area_f={logs['mask_area_f']:.3f}"
                if "vote_std_f" in logs:
                    msg += f" vstd_f={logs['vote_std_f']:.3f}"
                if "dlt_fallback_rate_f" in logs:
                    msg += f" dltfb_f={logs['dlt_fallback_rate_f']:.4f}"
                if "pseudo_hi_f" in logs and "pseudo_lo_f" in logs:
                    msg += f" phi_f={logs['pseudo_hi_f']:.3f} plo_f={logs['pseudo_lo_f']:.3f}"
                if "pseudo_inlier_residual_f" in logs and "pseudo_outlier_residual_f" in logs:
                    msg += (
                        f" rin_f={logs['pseudo_inlier_residual_f']:.3f}"
                        f" rout_f={logs['pseudo_outlier_residual_f']:.3f}"
                    )
                print(msg)
            if args.max_steps and step + 1 >= args.max_steps:
                break
        print(
            f"epoch {epoch} done in {time.time() - t0:.1f}s "
            f"(skipped_nonfinite={epoch_skipped_nonfinite}, total_skipped={skipped_nonfinite})"
        )

        ckpt = os.path.join(args.out_dir, f"pcgv_epoch_{epoch:03d}.pth")
        save_checkpoint(ckpt, net, opt, scaler, epoch, args, global_step=global_step)
        save_checkpoint(os.path.join(args.out_dir, "pcgv_latest.pth"), net, opt, scaler, epoch, args,
                        global_step=global_step)

        if args.eval_every and epoch % args.eval_every == 0:
            report = evaluate(net, device, args.test_list, args.test_img_dir,
                              args.coord_dir, max_items=args.eval_max_items)
            net.train()
            if args.freeze_coarse:
                net.coarse.eval()
            if args.freeze_pcgv_shallow:
                net.features.shallow.eval()
            if args.freeze_pcgv_pyramid:
                net.features.pyramid.eval()
            print("  PME " + "  ".join(
                f"{k}={report[k]:.4f}" for k in ("RE", "LT", "LL", "SF", "LF", "AVG")))

    save_checkpoint(os.path.join(args.out_dir, "pcgv_final.pth"), net, opt, scaler, args.epochs, args,
                    global_step=global_step)
    print("saved final checkpoint")


if __name__ == "__main__":
    main()
