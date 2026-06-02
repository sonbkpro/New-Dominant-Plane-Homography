"""Training entry point for PCGV.

Recommended first PCGV stage:

    python -m paper_new.train_pcgv \
        --coarse_ckpt new_approach/checkpoints/baseline/baseline_final.pth \
        --freeze_coarse \
        --epochs 5 --batch_size 4 --lr_pcgv 1e-4 --lr_coarse 2e-5 \
        --lambda_reproj 0 --lambda_cycle 0 --lambda_vote 0

This starts with the existing alignment+FIL baseline loss.  PCGV-specific
losses are present but disabled until their lambdas are set above zero.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict

import torch
from torch.utils.data import DataLoader

from new_approach.dataset_baseline import TrainDataset, move_batch

from paper_new.evaluate_pcgv import evaluate
from paper_new.losses_pcgv import pcgv_loss
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


def parse_args():
    ap = argparse.ArgumentParser()
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
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--print_freq", type=int, default=20)
    ap.add_argument("--max_train_items", type=int, default=None)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=1)
    ap.add_argument("--eval_max_items", type=int, default=None)

    ap.add_argument("--coarse_ckpt", default=None,
                    help="Optional baseline checkpoint to load into the coarse HomoGAN initializer.")
    ap.add_argument("--freeze_coarse", action="store_true",
                    help="Freeze the coarse baseline while training PCGV.")
    ap.add_argument("--init_mode", choices=("identity", "coarse_flow_corners"),
                    default="coarse_flow_corners")
    ap.add_argument("--override_baseline_keys", type=_str2bool, default=True,
                    help="Apply baseline alignment/FIL loss to PCGV final warps.")
    ap.add_argument("--pcgv_enabled", type=_str2bool, default=True)

    ap.add_argument("--lr_pcgv", type=float, default=1e-4)
    ap.add_argument("--lr_features", type=float, default=1e-4)
    ap.add_argument("--lr_coarse", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=1e-4)

    ap.add_argument("--num_iters", type=int, default=4)
    ap.add_argument("--radius", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--feat_dim", type=int, default=64)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--pcgv_level", type=int, default=0)
    ap.add_argument("--use_transformer", type=_str2bool, default=False)
    ap.add_argument("--use_plane_token", type=_str2bool, default=True)
    ap.add_argument("--use_leverage", type=_str2bool, default=True)
    ap.add_argument("--use_uncertainty", type=_str2bool, default=True)
    ap.add_argument("--update_alpha", type=float, default=0.7)

    ap.add_argument("--lambda_align", type=float, default=1.0)
    ap.add_argument("--lambda_fil", type=float, default=0.5)
    ap.add_argument("--lambda_reproj", type=float, default=0.0)
    ap.add_argument("--lambda_cycle", type=float, default=0.0)
    ap.add_argument("--lambda_vote", type=float, default=0.0)
    ap.add_argument("--lambda_tv", type=float, default=0.0)
    ap.add_argument("--lambda_area", type=float, default=0.0)
    ap.add_argument("--lambda_entropy", type=float, default=0.0)
    ap.add_argument("--lambda_cond", type=float, default=0.0)
    ap.add_argument("--lambda_uncertainty", type=float, default=0.0)
    ap.add_argument("--target_area", type=float, default=0.35)
    ap.add_argument("--vote_tau", type=float, default=2.0)
    ap.add_argument("--use_masked_align", action="store_true")
    return ap.parse_args()


def _load_state(path: str, device) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if isinstance(checkpoint, torch.nn.Module):
        checkpoint = checkpoint.state_dict()
    elif isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in checkpoint:
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint format: {type(checkpoint)}")
    return {(k[7:] if k.startswith("module.") else k): v for k, v in checkpoint.items()}


def _load_coarse(model, path: str, device):
    state = _load_state(path, device)
    missing, unexpected = model.coarse.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] coarse checkpoint missing {len(missing)} keys")
    if unexpected:
        print(f"[warn] coarse checkpoint has {len(unexpected)} unexpected keys")


def _build_optimizer(model, args):
    groups = []
    pcgv_params = [p for p in model.pcgv.parameters() if p.requires_grad]
    feature_params = [p for p in model.features.parameters() if p.requires_grad]
    mask_params = [p for p in model.mask_upsampler.parameters() if p.requires_grad]
    coarse_params = [p for p in model.coarse.parameters() if p.requires_grad]
    if pcgv_params or mask_params:
        groups.append({"params": pcgv_params + mask_params, "lr": args.lr_pcgv})
    if feature_params:
        groups.append({"params": feature_params, "lr": args.lr_features})
    if coarse_params:
        groups.append({"params": coarse_params, "lr": args.lr_coarse})
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay)


def main():
    args = parse_args()
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
        pcgv_freeze_coarse=args.freeze_coarse,
        pcgv_init_mode=args.init_mode,
        pcgv_override_baseline_keys=args.override_baseline_keys,
        pcgv_level=args.pcgv_level,
    )
    net = build_pcgv(params).to(device)
    if args.coarse_ckpt:
        _load_coarse(net, args.coarse_ckpt, device)
    if args.freeze_coarse:
        net.freeze_coarse()
        net.coarse.eval()

    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"PCGV built: {n_params/1e6:.2f}M trainable params, device={device}")
    if args.freeze_coarse and not args.coarse_ckpt:
        print("[warn] coarse model is frozen without --coarse_ckpt; this is only useful for smoke tests.")

    ds = TrainDataset(args.train_list, args.train_img_dir, rho=args.rho,
                      max_items=args.max_train_items)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, drop_last=True,
                        pin_memory=(device.type == "cuda"))
    print(f"Train pairs: {len(ds)}  steps/epoch: {len(loader)}")

    opt = _build_optimizer(net, args)
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for epoch in range(1, args.epochs + 1):
        net.train()
        if args.freeze_coarse:
            net.coarse.eval()
        t0 = time.time()
        for step, batch in enumerate(loader):
            batch = move_batch(batch, device)
            opt.zero_grad(set_to_none=True)
            amp_device = "cuda" if device.type == "cuda" else "cpu"
            with torch.amp.autocast(amp_device, enabled=use_amp):
                out = net(batch)
                loss, logs = pcgv_loss(
                    out,
                    lambda_align=args.lambda_align,
                    lambda_fil=args.lambda_fil,
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
                    use_masked_align=args.use_masked_align,
                )
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()

            if step % args.print_freq == 0:
                msg = (
                    f"ep{epoch} [{step}/{len(loader)}] "
                    f"total={logs['total']:.4f} align={logs['align']:.4f} "
                    f"fil={logs['fil']:.4f} reproj={logs['reproj']:.4f} "
                    f"vote={logs['vote']:.4f}"
                )
                if "mean_votes_f" in logs:
                    msg += f" mean_vote_f={logs['mean_votes_f']:.3f}"
                if "mean_residuals_f" in logs:
                    msg += f" residual_f={logs['mean_residuals_f']:.3f}"
                print(msg)
            if args.max_steps and step + 1 >= args.max_steps:
                break
        print(f"epoch {epoch} done in {time.time() - t0:.1f}s")

        ckpt = os.path.join(args.out_dir, f"pcgv_epoch_{epoch:03d}.pth")
        torch.save({"model": net.state_dict(), "args": vars(args)}, ckpt)

        if args.eval_every and epoch % args.eval_every == 0:
            report = evaluate(net, device, args.test_list, args.test_img_dir,
                              args.coord_dir, max_items=args.eval_max_items)
            net.train()
            if args.freeze_coarse:
                net.coarse.eval()
            print("  PME " + "  ".join(
                f"{k}={report[k]:.4f}" for k in ("RE", "LT", "LL", "SF", "LF", "AVG")))

    torch.save({"model": net.state_dict(), "args": vars(args)},
               os.path.join(args.out_dir, "pcgv_final.pth"))
    print("saved final checkpoint")


if __name__ == "__main__":
    main()
