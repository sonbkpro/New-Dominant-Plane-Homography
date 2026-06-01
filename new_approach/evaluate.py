"""Reprojection-error (PME) evaluation for the no-mask baseline.

Mirrors Oneline-DLTv1/test.py: mean L2 over the first six labelled
correspondence points of each Test_List pair, bucketed into 5 scene categories.
Because the human annotator did not fix a left/right point ordering, the
per-point error takes min(forward, swapped) using the same predicted H.
"""
import argparse
import os

import numpy as np
import torch

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


@torch.no_grad()
def evaluate(net, device, list_path, img_dir, coord_dir, max_items=None,
             verbose=False, num_points=6):
    net.eval()
    ds = TestDataset(list_path, img_dir, coord_dir, max_items=max_items)
    buckets = {k: [] for k in CATEGORIES}
    all_errs = []
    skipped = 0

    for i in range(len(ds)):
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
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--list", default=os.path.join(ROOT, "Data/Test_List.txt"))
    ap.add_argument("--img_dir", default=os.path.join(ROOT, "Data/Test"))
    ap.add_argument("--coord_dir", default=os.path.join(ROOT, "Data/Coordinate-v2/Coordinate-v2"))
    ap.add_argument("--max_items", type=int, default=None)
    ap.add_argument("--num_points", type=_parse_num_points, default=6,
                    help="PME points per pair: 6 matches Oneline-DLTv1/test.py; use 'all' for every labelled point.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device)
    net = build_baseline().to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    net.load_state_dict(state)

    report = evaluate(net, device, args.list, args.img_dir, args.coord_dir,
                      max_items=args.max_items, verbose=args.verbose,
                      num_points=args.num_points)
    print("\n=== PME by category ===")
    for k in ("RE", "LT", "LL", "SF", "LF", "AVG"):
        print(f"  {k}: {report[k]:.4f}")
    print(f"  (n={report['_n']}, skipped={report['_skipped']})")


if __name__ == "__main__":
    main()
