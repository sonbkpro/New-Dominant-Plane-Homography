"""Unsupervised training for the no-mask HomoGAN baseline.

Run from the project root:

    python -m new_approach.train_baseline \
        --epochs 6 --batch_size 16 --lr 4e-4 --num_workers 8

Smoke test (a handful of items / steps, fast end-to-end check):

    python -m new_approach.train_baseline \
        --max_train_items 16 --batch_size 2 --epochs 1 --max_steps 4 \
        --out_dir new_approach/checkpoints/baseline_smoke --eval_max_items 8
"""
import argparse
import json
import os
import time

import torch
from torch.utils.data import DataLoader

from .dataset_baseline import TrainDataset, move_batch
from .evaluate import evaluate
from .losses import baseline_loss
from .model_baseline import build_baseline, make_params

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_list", default=os.path.join(ROOT, "Data/Train_List.txt"))
    ap.add_argument("--train_img_dir", default=os.path.join(ROOT, "Data/Train"))
    ap.add_argument("--test_list", default=os.path.join(ROOT, "Data/Test_List.txt"))
    ap.add_argument("--test_img_dir", default=os.path.join(ROOT, "Data/Test"))
    ap.add_argument("--coord_dir", default=os.path.join(ROOT, "Data/Coordinate-v2/Coordinate-v2"))
    ap.add_argument("--out_dir", default=os.path.join(ROOT, "new_approach/checkpoints/baseline"))
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--rho", type=int, default=16)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--lambda_align", type=float, default=1.0)
    ap.add_argument("--lambda_fil", type=float, default=0.5)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--max_train_items", type=int, default=None, help="cap dataset size (smoke)")
    ap.add_argument("--max_steps", type=int, default=0, help="stop epoch early after N steps (smoke); 0 = full")
    ap.add_argument("--print_freq", type=int, default=20)
    ap.add_argument("--seed", type=int, default=230)
    ap.add_argument("--eval_every", type=int, default=1, help="eval every N epochs; 0 = never")
    ap.add_argument("--eval_max_items", type=int, default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device(args.device)
    params = make_params(pretrain_phase=True)
    net = build_baseline(params).to(device)
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Model built: {n_params/1e6:.2f}M trainable params, device={device}")

    ds = TrainDataset(args.train_list, args.train_img_dir, rho=args.rho,
                      max_items=args.max_train_items)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, drop_last=True,
                        pin_memory=(device.type == "cuda"))
    print(f"Train pairs: {len(ds)}  steps/epoch: {len(loader)}")

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for epoch in range(1, args.epochs + 1):
        net.train()
        t0 = time.time()
        for step, batch in enumerate(loader):
            batch = move_batch(batch, device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = net(batch)
                loss, logs = baseline_loss(out, args.lambda_align, args.lambda_fil)
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()

            if step % args.print_freq == 0:
                print(f"ep{epoch} [{step}/{len(loader)}] "
                      f"total={logs['total']:.4f} align={logs['align']:.4f} "
                      f"fil={logs['fil']:.4f}")
            if args.max_steps and step + 1 >= args.max_steps:
                break
        print(f"epoch {epoch} done in {time.time()-t0:.1f}s")

        ckpt = os.path.join(args.out_dir, f"baseline_epoch_{epoch:03d}.pth")
        torch.save(net.state_dict(), ckpt)

        if args.eval_every and epoch % args.eval_every == 0:
            report = evaluate(net, device, args.test_list, args.test_img_dir,
                              args.coord_dir, max_items=args.eval_max_items)
            net.train()
            print("  PME " + "  ".join(
                f"{k}={report[k]:.4f}" for k in ("RE", "LT", "LL", "SF", "LF", "AVG")))

    torch.save(net.state_dict(), os.path.join(args.out_dir, "baseline_final.pth"))
    print("saved final checkpoint")


if __name__ == "__main__":
    main()
