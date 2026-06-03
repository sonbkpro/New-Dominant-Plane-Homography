"""Visualize PCGV dominant-plane vote masks."""

from __future__ import annotations

import argparse
import os
import re

import cv2
import numpy as np
import torch

from new_approach.dataset_baseline import TestDataset, move_batch
from new_approach.evaluate import CATEGORIES, _category

from paper_new.evaluate_pcgv import _load_checkpoint, _load_compatible_state_dict
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


def _ckpt_arg(ckpt_args, *names, default):
    for name in names:
        if name in ckpt_args:
            return ckpt_args[name]
    return default


def _build_model(args, device):
    checkpoint, state = _load_checkpoint(args.ckpt, device)
    ckpt_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    params = make_pcgv_params(
        pcgv_num_iters=args.num_iters if args.num_iters is not None else _ckpt_arg(
            ckpt_args, "num_iters", "pcgv_num_iters", default=4),
        pcgv_radius=args.radius if args.radius is not None else _ckpt_arg(
            ckpt_args, "radius", "pcgv_radius", default=4),
        pcgv_feat_dim=args.feat_dim if args.feat_dim is not None else _ckpt_arg(
            ckpt_args, "feat_dim", "pcgv_feat_dim", default=64),
        pcgv_hidden_dim=args.hidden_dim if args.hidden_dim is not None else _ckpt_arg(
            ckpt_args, "hidden_dim", "pcgv_hidden_dim", default=128),
        pcgv_pyramid_embed_dim=args.pyramid_embed_dim if args.pyramid_embed_dim is not None else _ckpt_arg(
            ckpt_args, "pyramid_embed_dim", "pcgv_pyramid_embed_dim", default=24),
        pcgv_pyramid_layers=args.pyramid_layers if args.pyramid_layers is not None else _ckpt_arg(
            ckpt_args, "pyramid_layers", "pcgv_pyramid_layers", default=3),
        pcgv_mask_refine=args.mask_refine if args.mask_refine is not None else _ckpt_arg(
            ckpt_args, "mask_refine", "pcgv_mask_refine", default=False),
        pcgv_init_mode=args.init_mode,
    )
    net = build_pcgv(params).to(device)
    missing, unexpected, mismatched = _load_compatible_state_dict(net, state)
    if missing:
        print(f"[warn] checkpoint missing {len(missing)} keys")
    if unexpected:
        print(f"[warn] checkpoint has {len(unexpected)} unexpected keys")
    if mismatched:
        print(f"[warn] checkpoint skipped {len(mismatched)} shape-mismatched keys")
    if args.set_refine_blend is not None:
        applied = net.set_refine_blend(args.set_refine_blend)
        print(f"Set PCGV refine blend to {applied:.4f}")
    net.eval()
    return net


def _to_u8(value):
    arr = np.asarray(value, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.percentile(arr, [1.0, 99.0])
    if hi <= lo:
        lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return (arr * 255.0).astype(np.uint8)


def _rgb(gray_u8):
    return cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)


def _label(panel, text):
    cv2.putText(panel, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(panel, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1, cv2.LINE_AA)
    return panel


def _mask_overlay(img_u8, mask):
    heat = cv2.applyColorMap(np.clip(mask * 255.0, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(_rgb(img_u8), 0.60, heat, 0.40, 0)


def _contour_overlay(img_u8, mask):
    panel = _rgb(img_u8)
    binary = (mask >= 0.5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(panel, contours, -1, (0, 255, 255), 2)
    return panel


def _input_panel(img1_u8, img2_u8):
    h, w = img1_u8.shape
    left_w = w // 2
    left = cv2.resize(_rgb(img1_u8), (left_w, h), interpolation=cv2.INTER_AREA)
    right = cv2.resize(_rgb(img2_u8), (w - left_w, h), interpolation=cv2.INTER_AREA)
    panel = np.concatenate([left, right], axis=1)
    _label(panel[:, :left_w], "input A")
    _label(panel[:, left_w:], "input B")
    return panel


def _error_panel(error, mask):
    err_u8 = _to_u8(error)
    panel = cv2.applyColorMap(err_u8, cv2.COLORMAP_INFERNO)
    inside = mask >= 0.5
    outside = ~inside
    inside_mean = float(error[inside].mean()) if inside.any() else float("nan")
    outside_mean = float(error[outside].mean()) if outside.any() else float("nan")
    text = f"abs error  in={inside_mean:.3f}  out={outside_mean:.3f}"
    _label(panel, text)
    return panel


def _make_canvas(img1, img2, warped, mask):
    img1_u8 = _to_u8(img1)
    img2_u8 = _to_u8(img2)
    warped_u8 = _to_u8(warped)
    error = np.abs(img1.astype(np.float32) - warped.astype(np.float32))
    panels = [
        _input_panel(img1_u8, img2_u8),
        _label(_mask_overlay(img1_u8, mask), "vote mask overlay"),
        _label(_contour_overlay(warped_u8, mask), "warped B + mask contour"),
        _error_panel(error, mask),
    ]
    return np.concatenate([
        np.concatenate(panels[:2], axis=1),
        np.concatenate(panels[2:], axis=1),
    ], axis=0)


def _safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _selected_samples(ds, n_per_cat):
    counts = {k: 0 for k in CATEGORIES}
    selected = []
    for i in range(len(ds)):
        if all(count >= n_per_cat for count in counts.values()):
            break
        sample = ds[i]
        cat = _category(sample["video_id"])
        if cat is None or counts[cat] >= n_per_cat:
            continue
        if not os.path.exists(sample["npy_path"]):
            continue
        selected.append((i, cat, sample))
        counts[cat] += 1
    return selected


@torch.no_grad()
def visualize(args):
    device = torch.device(args.device)
    net = _build_model(args, device)
    ds = TestDataset(args.list, args.img_dir, args.coord_dir, max_items=args.max_items)
    os.makedirs(args.out_dir, exist_ok=True)
    selected = _selected_samples(ds, args.n_per_cat)
    written = 0
    for idx, cat, sample in selected:
        batch = {k: sample[k].unsqueeze(0) for k in ("imgs_gray_full", "imgs_gray_patch", "start", "pts")}
        batch = move_batch(batch, device)
        out = net(batch)
        patch = batch["imgs_gray_patch"][0].detach().cpu()
        img1 = patch[0].numpy()
        img2 = patch[1].numpy()
        warped = out["warp_img2_patch"][0, 0].detach().cpu().numpy()
        mask = out["pcgv_mask_f_patch"][0, 0].detach().float().cpu().clamp(0.0, 1.0).numpy()
        canvas = _make_canvas(img1, img2, warped, mask)
        stem = _safe_name(os.path.splitext(os.path.basename(sample["npy_path"]))[0])
        path = os.path.join(args.out_dir, f"{cat}_{idx:04d}_{stem}.png")
        cv2.imwrite(path, canvas)
        written += 1
    print(f"Wrote {written} mask visualizations to {args.out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", "--checkpoint", dest="ckpt", required=True)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--list", default=os.path.join(ROOT, "Data/Test_List.txt"))
    ap.add_argument("--img_dir", default=os.path.join(ROOT, "Data/Test"))
    ap.add_argument("--coord_dir", default=os.path.join(ROOT, "Data/Coordinate-v2/Coordinate-v2"))
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_per_cat", type=int, default=4)
    ap.add_argument("--max_items", type=int, default=None)
    ap.add_argument("--num_iters", type=int, default=None)
    ap.add_argument("--radius", type=int, default=None)
    ap.add_argument("--feat_dim", type=int, default=None)
    ap.add_argument("--hidden_dim", type=int, default=None)
    ap.add_argument("--pyramid_embed_dim", type=int, default=None)
    ap.add_argument("--pyramid_layers", type=int, default=None)
    ap.add_argument("--init_mode", choices=("identity", "coarse_flow_corners"), default="coarse_flow_corners")
    ap.add_argument("--mask_refine", type=_str2bool, default=None)
    ap.add_argument("--set_refine_blend", type=float, default=None)
    args = ap.parse_args()
    visualize(args)


if __name__ == "__main__":
    main()
