"""PME evaluation for PCGV checkpoints."""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

from new_approach.dataset_baseline import TestDataset, move_batch
from new_approach.evaluate import CATEGORIES, _category, _pair_error, _parse_num_points
from new_approach.geometry import flow_to_homography

from paper_new.geometry import torch_convert_homography_scale
from paper_new.model_pcgv import build_pcgv, make_pcgv_params

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))


def _start_xy(sample):
    start = sample["start"]
    return float(start[0, 0, 0]), float(start[1, 0, 0])


def _patch_h_to_full_h(H_patch: np.ndarray, start_xy):
    start_x, start_y = map(float, start_xy)
    T = np.array([[1.0, 0.0, start_x], [0.0, 1.0, start_y], [0.0, 0.0, 1.0]], dtype=np.float64)
    H = T @ H_patch.astype(np.float64) @ np.linalg.inv(T)
    return H / H[2, 2]


def _load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    raw_checkpoint = checkpoint
    if isinstance(checkpoint, torch.nn.Module):
        state = checkpoint.state_dict()
    elif isinstance(checkpoint, dict):
        state = checkpoint
        for key in ("model", "state_dict", "model_state_dict"):
            if key in state:
                state = state[key]
                break
    else:
        state = checkpoint
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint format: {type(state)}")
    state = {(key[7:] if key.startswith("module.") else key): value
             for key, value in state.items()}
    return raw_checkpoint, state


def _load_compatible_state_dict(model, state):
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
    return missing, unexpected, mismatched


def _write_report(report, output_dir):
    if output_dir is None:
        return
    os.makedirs(output_dir, exist_ok=True)

    def _json_safe(value):
        if isinstance(value, dict):
            return {k: _json_safe(v) for k, v in value.items()}
        if isinstance(value, float) and np.isnan(value):
            return None
        return value

    json_report = {key: _json_safe(value) for key, value in report.items()}
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(json_report, f, indent=2)


def _progress_indices(total, enabled=True, desc="Evaluating", fallback_every=100):
    indices = range(total)
    if not enabled:
        return indices
    if tqdm is not None:
        return tqdm(indices, total=total, desc=desc, unit="pair")

    def _fallback():
        for i in indices:
            yield i
            done = i + 1
            if done == 1 or done % fallback_every == 0 or done == total:
                print(f"{desc}: {done}/{total} pairs", flush=True)

    return _fallback()


@torch.no_grad()
def evaluate(net, device, list_path, img_dir, coord_dir, max_items=None,
             verbose=False, num_points=6, progress=True, h_source="raw"):
    net.eval()
    ds = TestDataset(list_path, img_dir, coord_dir, max_items=max_items)
    buckets = {k: [] for k in CATEGORIES}
    all_errs = []
    mask_areas = []
    mask_area_buckets = {k: [] for k in CATEGORIES}
    mask_entropies = []
    residuals = []
    cycle_errors = []
    skipped = 0

    for i in _progress_indices(len(ds), enabled=progress, desc="Evaluating"):
        sample = ds[i]
        if not os.path.exists(sample["npy_path"]):
            skipped += 1
            continue
        batch = {k: sample[k].unsqueeze(0) for k in
                 ("imgs_gray_full", "imgs_gray_patch", "start", "pts")}
        batch = move_batch(batch, device)
        out = net(batch)
        if h_source == "start" and "H0_f" in out:
            H_patch = out["H0_f"][0].detach().cpu().numpy()
            H = _patch_h_to_full_h(H_patch, _start_xy(sample))
        elif h_source == "raw" and "pcgv_H_raw_feat_f" in out:
            H_feat = out["pcgv_H_raw_feat_f"]
            _, _, h_feat, w_feat = out["pcgv_mask_f"].shape
            h_patch, w_patch = batch["imgs_gray_patch"].shape[-2:]
            H_patch_t = torch_convert_homography_scale(
                H_feat,
                (h_feat, w_feat),
                (h_patch, w_patch),
            )
            H_patch = H_patch_t[0].detach().cpu().numpy()
            H = _patch_h_to_full_h(H_patch, _start_xy(sample))
        elif "H_f" in out:
            H_patch = out["H_f"][0].detach().cpu().numpy()
            H = _patch_h_to_full_h(H_patch, _start_xy(sample))
        else:
            flow = out["flow_f_patch"][0].detach().cpu().numpy()
            H = flow_to_homography(flow, start_xy=_start_xy(sample))
        err = _pair_error(sample["npy_path"], H, num_points=num_points)
        if err is None:
            skipped += 1
            continue
        all_errs.append(err)
        cat = _category(sample["video_id"])
        if cat is not None:
            buckets[cat].append(err)

        mask = out.get("pcgv_mask_f_patch")
        if torch.is_tensor(mask):
            m = mask.detach().float().clamp(1e-6, 1.0 - 1e-6)
            area = float(m.mean().cpu())
            mask_areas.append(area)
            if cat is not None:
                mask_area_buckets[cat].append(area)
            ent = -(m * m.log() + (1.0 - m) * (1.0 - m).log()).mean()
            mask_entropies.append(float(ent.cpu()))
        res = out.get("pcgv_residuals_f")
        if torch.is_tensor(res):
            residuals.append(float(res.detach().mean().cpu()))
        if "H_f" in out and "H_b" in out:
            Hf = out["H_f"]
            Hb = out["H_b"]
            eye = torch.eye(3, device=Hf.device, dtype=Hf.dtype).unsqueeze(0)
            cyc = ((Hb @ Hf - eye).abs().mean() + (Hf @ Hb - eye).abs().mean())
            cycle_errors.append(float(cyc.detach().cpu()))
        if verbose:
            print(f"[{i}] {sample['video_id']}  PME={err:.4f}")

    report = {k: (float(np.mean(v)) if v else float("nan")) for k, v in buckets.items()}
    report["AVG"] = float(np.mean(all_errs)) if all_errs else float("nan")
    report["_n"] = len(all_errs)
    report["_skipped"] = skipped
    report["mask_area"] = float(np.mean(mask_areas)) if mask_areas else float("nan")
    report["mask_area_by_cat"] = {
        k: (float(np.mean(v)) if v else float("nan"))
        for k, v in mask_area_buckets.items()
    }
    report["mask_entropy"] = float(np.mean(mask_entropies)) if mask_entropies else float("nan")
    report["weighted_residual"] = float(np.mean(residuals)) if residuals else float("nan")
    report["cycle_error"] = float(np.mean(cycle_errors)) if cycle_errors else float("nan")
    return report


def _str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("1", "true", "yes", "y"):
        return True
    if value in ("0", "false", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("expected boolean")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", "--checkpoint", dest="ckpt", required=True)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--list", default=os.path.join(ROOT, "Data/Test_List.txt"))
    ap.add_argument("--img_dir", default=os.path.join(ROOT, "Data/Test"))
    ap.add_argument("--coord_dir", default=os.path.join(ROOT, "Data/Coordinate-v2/Coordinate-v2"))
    ap.add_argument("--max_items", type=int, default=None)
    ap.add_argument("--output_dir", "--output-dir", dest="output_dir", default=None)
    ap.add_argument("--num_points", type=_parse_num_points, default=6)
    ap.add_argument("--no_progress", "--no-progress", dest="progress", action="store_false")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--num_iters", type=int, default=None)
    ap.add_argument("--radius", type=int, default=None)
    ap.add_argument("--feat_dim", type=int, default=None)
    ap.add_argument("--hidden_dim", type=int, default=None)
    ap.add_argument("--pyramid_embed_dim", type=int, default=None,
                    help="PCGV pyramid base channels. Defaults to checkpoint args, else 24.")
    ap.add_argument("--pyramid_layers", type=int, default=None,
                    help="PCGV pyramid layers. Defaults to checkpoint args, else 3.")
    ap.add_argument("--init_mode", choices=("identity", "coarse_flow_corners"),
                    default="coarse_flow_corners")
    ap.add_argument("--h_source", choices=("final", "start", "raw"), default="raw",
                    help="Which PCGV homography to evaluate: final blended H, start/coarse H, or raw refined H.")
    ap.add_argument("--mask_refine", type=_str2bool, default=None,
                    help="Override checkpoint mask refinement mode for mask statistics.")
    ap.add_argument("--set_refine_blend", type=float, default=None,
                    help="Override the checkpoint's final coarse-to-refined blend before evaluation.")
    args = ap.parse_args()

    device = torch.device(args.device)
    checkpoint, state = _load_checkpoint(args.ckpt, device)
    ckpt_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    pyramid_embed_dim = args.pyramid_embed_dim
    if pyramid_embed_dim is None:
        pyramid_embed_dim = ckpt_args.get("pyramid_embed_dim", ckpt_args.get("pcgv_pyramid_embed_dim", 24))
    pyramid_layers = args.pyramid_layers
    if pyramid_layers is None:
        pyramid_layers = ckpt_args.get("pyramid_layers", ckpt_args.get("pcgv_pyramid_layers", 3))
    num_iters = args.num_iters
    if num_iters is None:
        num_iters = ckpt_args.get("num_iters", ckpt_args.get("pcgv_num_iters", 4))
    radius = args.radius
    if radius is None:
        radius = ckpt_args.get("radius", ckpt_args.get("pcgv_radius", 4))
    feat_dim = args.feat_dim
    if feat_dim is None:
        feat_dim = ckpt_args.get("feat_dim", ckpt_args.get("pcgv_feat_dim", 64))
    hidden_dim = args.hidden_dim
    if hidden_dim is None:
        hidden_dim = ckpt_args.get("hidden_dim", ckpt_args.get("pcgv_hidden_dim", 128))
    params = make_pcgv_params(
        pcgv_num_iters=num_iters,
        pcgv_radius=radius,
        pcgv_feat_dim=feat_dim,
        pcgv_hidden_dim=hidden_dim,
        pcgv_pyramid_embed_dim=pyramid_embed_dim,
        pcgv_pyramid_layers=pyramid_layers,
        pcgv_mask_refine=(
            args.mask_refine
            if args.mask_refine is not None
            else ckpt_args.get("mask_refine", ckpt_args.get("pcgv_mask_refine", False))
        ),
        pcgv_init_mode=args.init_mode,
    )
    net = build_pcgv(params).to(device)
    missing, unexpected, mismatched = _load_compatible_state_dict(net, state)
    if args.set_refine_blend is not None:
        applied_blend = net.set_refine_blend(args.set_refine_blend)
        print(f"Set PCGV refine blend to {applied_blend:.4f}")
    if missing:
        print(f"[warn] checkpoint missing {len(missing)} keys")
    if unexpected:
        print(f"[warn] checkpoint has {len(unexpected)} unexpected keys")
    if mismatched:
        print(f"[warn] checkpoint skipped {len(mismatched)} shape-mismatched keys")
        print("[warn] If this is an older PCGV checkpoint, retry with --pyramid_embed_dim 32.")

    report = evaluate(net, device, args.list, args.img_dir, args.coord_dir,
                      max_items=args.max_items, verbose=args.verbose,
                      num_points=args.num_points, progress=args.progress,
                      h_source=args.h_source)
    _write_report(report, args.output_dir)
    print("\n=== PME by category ===")
    for k in ("RE", "LT", "LL", "SF", "LF", "AVG"):
        print(f"  {k}: {report[k]:.4f}")
    print(f"  mask_area: {report['mask_area']:.4f}")
    print("  mask_area_by_cat: " + "  ".join(
        f"{k}={report['mask_area_by_cat'][k]:.4f}" for k in ("RE", "LT", "LL", "SF", "LF")))
    print(f"  mask_entropy: {report['mask_entropy']:.4f}")
    print(f"  weighted_residual: {report['weighted_residual']:.4f}")
    print(f"  cycle_error: {report['cycle_error']:.4f}")
    print(f"  (n={report['_n']}, skipped={report['_skipped']})")


if __name__ == "__main__":
    main()
