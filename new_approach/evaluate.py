"""Reprojection-error (PME) evaluation for the no-mask baseline.

Mirrors Oneline-DLTv1/test.py: mean L2 over the first six labelled
correspondence points of each Test_List pair, bucketed into 5 scene categories.
Because the human annotator did not fix a left/right point ordering, the
per-point error takes min(forward, swapped) using the same predicted H.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

if __package__ in (None, ""):
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    from new_approach.dataset_baseline import TestDataset, move_batch
    from new_approach.geometry import flow_to_homography, geometric_distance
    from new_approach.model_baseline import build_baseline
else:
    from .dataset_baseline import TestDataset, move_batch
    from .geometry import flow_to_homography, geometric_distance
    from .model_baseline import build_baseline

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))

# Scene-category video-id buckets (ported from Oneline-DLTv1/test.py).
CATEGORIES = {
    "RE": ['0000011', '0000016', '00000147', '00000155', '00000158', '00000107', '00000239', '0000030'],
    "LT": ['0000038', '0000044', '0000046', '0000047', '00000238', '00000177', '00000188', '00000181'],
    "LL": ['0000085', '00000100', '0000091', '0000092', '00000216', '00000226'],
    "SF": ['00000244', '00000251', '0000026', '0000034', '00000115'],
    "LF": ['00000104', '0000031', '0000035', '00000129', '00000141', '00000200'],
}


def _category(video_id):
    for name, ids in CATEGORIES.items():
        if video_id in ids:
            return name
    return None


def _parse_num_points(value):
    if value == "all":
        return None
    num_points = int(value)
    if num_points <= 0:
        raise argparse.ArgumentTypeError("--num_points must be positive or 'all'")
    return num_points


def _start_xy(sample):
    start = sample["start"]
    return float(start[0, 0, 0]), float(start[1, 0, 0])


def _pair_error(npy_path, H, num_points=6):
    data = np.load(npy_path, allow_pickle=True).item()
    matches = data["matche_pts"]
    if num_points is not None:
        matches = matches[:num_points]
    errs = []
    for p_lr in matches:
        p1, p2 = p_lr[0], p_lr[1]
        err_lr = geometric_distance(p1, p2, H)
        err_rl = geometric_distance(p2, p1, H)  # annotator order is not fixed
        errs.append(min(err_lr, err_rl))
    return float(np.mean(errs)) if errs else None


def _load_checkpoint_state(path, device):
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
    return {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in checkpoint.items()
    }


def _write_report(report, output_dir):
    if output_dir is None:
        return
    os.makedirs(output_dir, exist_ok=True)
    json_report = {
        key: (None if isinstance(value, float) and np.isnan(value) else value)
        for key, value in report.items()
    }
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
             verbose=False, num_points=6, progress=True):
    net.eval()
    ds = TestDataset(list_path, img_dir, coord_dir, max_items=max_items)
    buckets = {k: [] for k in CATEGORIES}
    all_errs = []
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
        flow = out["flow_f_patch"][0].cpu().numpy()    # (CROP_H, CROP_W, 2)
        H = flow_to_homography(flow, start_xy=_start_xy(sample))
        err = _pair_error(sample["npy_path"], H, num_points=num_points)
        if err is None:
            skipped += 1
            continue
        all_errs.append(err)
        cat = _category(sample["video_id"])
        if cat is not None:
            buckets[cat].append(err)
        if verbose:
            print(f"[{i}] {sample['video_id']}  PME={err:.4f}")

    report = {k: (float(np.mean(v)) if v else float("nan")) for k, v in buckets.items()}
    report["AVG"] = float(np.mean(all_errs)) if all_errs else float("nan")
    report["_n"] = len(all_errs)
    report["_skipped"] = skipped
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", "--checkpoint", dest="ckpt", required=True)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--list", default=os.path.join(ROOT, "Data/Test_List.txt"))
    ap.add_argument("--img_dir", default=os.path.join(ROOT, "Data/Test"))
    ap.add_argument("--coord_dir", default=os.path.join(ROOT, "Data/Coordinate-v2/Coordinate-v2"))
    ap.add_argument("--max_items", type=int, default=None)
    ap.add_argument("--batch_size", "--batch-size", dest="batch_size", type=int, default=1,
                    help="Accepted for CLI compatibility; this evaluator currently processes one pair at a time.")
    ap.add_argument("--output_dir", "--output-dir", dest="output_dir", default=None)
    ap.add_argument("--num_points", type=_parse_num_points, default=6,
                    help="PME points per pair: 6 matches Oneline-DLTv1/test.py; use 'all' for every labelled point.")
    ap.add_argument("--no_progress", "--no-progress", dest="progress", action="store_false",
                    help="Disable the evaluation progress bar.")
    ap.add_argument("--verbose", action="store_true")
    ap.set_defaults(progress=True)
    args = ap.parse_args()

    device = torch.device(args.device)
    net = build_baseline().to(device)
    state = _load_checkpoint_state(args.ckpt, device)
    try:
        net.load_state_dict(state)
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint is not compatible with new_approach.model_baseline.build_baseline(). "
            "Use a checkpoint produced by new_approach.train_baseline, for example "
            "baseline_epoch_001.pth or baseline_final.pth."
        ) from exc

    report = evaluate(net, device, args.list, args.img_dir, args.coord_dir,
                      max_items=args.max_items, verbose=args.verbose,
                      num_points=args.num_points, progress=args.progress)
    _write_report(report, args.output_dir)
    print("\n=== PME by category ===")
    for k in ("RE", "LT", "LL", "SF", "LF", "AVG"):
        print(f"  {k}: {report[k]:.4f}")
    print(f"  (n={report['_n']}, skipped={report['_skipped']})")


if __name__ == "__main__":
    main()
