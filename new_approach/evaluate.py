import argparse
import ast
import csv
import json
import sys
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Data.homo_flow_dataset import build_homo_flow_loader
from new_approach.device_utils import resolve_device
from new_approach.geometry_homo import flow_mask_to_homography
from new_approach.modules.transformerHomo import Ms_Transformer

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


VIDEO_GROUPS = OrderedDict(
    [
        ("RE", ["0000011", "0000016", "00000147", "00000155", "00000158", "00000107", "00000239", "0000030"]),
        ("LT", ["0000038", "0000044", "0000046", "0000047", "00000238", "00000177", "00000188", "00000181"]),
        ("LL", ["0000085", "00000100", "0000091", "0000092", "00000216", "00000226"]),
        ("SF", ["00000244", "00000251", "0000026", "0000030", "0000034", "00000115"]),
        ("LF", ["00000104", "0000031", "0000035", "00000129", "00000141", "00000200"]),
    ]
)


def parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(v.strip()) for v in value.split(",") if v.strip())


def batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


def collated_get(value: Any, index: int) -> Any:
    if isinstance(value, (list, tuple)):
        return value[index]
    return value


def default_params(args: argparse.Namespace) -> SimpleNamespace:
    crop_h = 320 if args.crop_h is None else args.crop_h
    crop_w = 512 if args.crop_w is None else args.crop_w
    return SimpleNamespace(
        net_type="HomoGAN",
        crop_size=[crop_h, crop_w],
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
        use_open=True,
        pretrain_phase=args.pretrain_phase,
        mask_method=args.mask_method,
        mask_flow_cond_channels=7,
        mask_flow_base_channels=args.mask_flow_base_channels,
        mask_flow_channel_mults=tuple(args.mask_flow_channel_mults),
        mask_flow_time_dim=args.mask_flow_time_dim,
        mask_flow_steps=args.mask_flow_steps,
        mask_flow_solver=args.mask_flow_solver,
        mask_flow_init=args.mask_flow_init,
        mask_flow_temperature=args.mask_flow_temperature,
        mask_flow_noise_sigma=args.mask_flow_noise_sigma,
        mask_flow_detach_condition=True,
        mask_flow_sample_grad=False,
        return_h_matrix=False,
    )


def torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_checkpoint_state(ckpt: dict[str, Any] | Any, path: Path, model: torch.nn.Module) -> None:
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    try:
        missing, unexpected = model.load_state_dict(state, strict=False)
    except RuntimeError:
        stripped = {key.removeprefix("module."): value for key, value in state.items()}
        missing, unexpected = model.load_state_dict(stripped, strict=False)
    print(f"loaded checkpoint: {path}")
    if missing:
        print(f"  missing keys: {len(missing)}")
    if unexpected:
        print(f"  unexpected keys: {len(unexpected)}")


def params_from_checkpoint(args: argparse.Namespace, ckpt: dict[str, Any] | None = None) -> SimpleNamespace:
    if ckpt and isinstance(ckpt.get("params"), dict):
        params = SimpleNamespace(**ckpt["params"])
    else:
        params = default_params(args)

    params.mask_flow_steps = args.mask_flow_steps
    params.mask_flow_solver = args.mask_flow_solver
    params.mask_flow_init = args.mask_flow_init
    params.mask_flow_sample_grad = False
    params.return_h_matrix = False
    return params


def load_point_set(pair_name: str, coordinate_dir: Path) -> list:
    img1_rel, img2_rel = pair_name.split()[:2]
    pt_name = f"{Path(img1_rel).name}_{Path(img2_rel).name}.npy"
    pt_path = coordinate_dir / pt_name
    if not pt_path.exists():
        raise FileNotFoundError(f"coordinate file not found for pair {pair_name}: {pt_path}")
    obj = np.load(pt_path, allow_pickle=True).item()
    return obj["matche_pts"]


def parse_point_set_from_batch(batch: dict[str, Any], index: int, coordinate_dir: Path) -> list:
    if "pt_set" in batch:
        text = collated_get(batch["pt_set"], index)
        if isinstance(text, str) and text:
            try:
                parsed = ast.literal_eval(text)
            except (SyntaxError, ValueError):
                parsed = eval(text, {"np": np})
            return parsed["matche_pts"] if isinstance(parsed, dict) else parsed
    pair_name = collated_get(batch["pair_name"], index)
    return load_point_set(pair_name, coordinate_dir)


def transform_points(points: torch.Tensor, homography: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    ones = torch.ones(points.shape[0], 1, device=points.device, dtype=points.dtype)
    points_h = torch.cat([points, ones], dim=1)
    warped = points_h @ homography.transpose(0, 1)
    denom = warped[:, 2:]
    denom = torch.where(denom.abs() < eps, torch.full_like(denom, eps), denom)
    return warped[:, :2] / denom


def compute_h_from_output(output: dict[str, Any], use_mask_weights: bool) -> tuple[torch.Tensor, torch.Tensor]:
    flow_f = output["flow_f"]
    flow_b = output["flow_b"]
    mask_f = None
    mask_b = None
    if use_mask_weights and output.get("img1_patch_mask") is not None:
        height, width = flow_f.shape[1:3]
        mask_f = F.interpolate(output["img1_patch_mask"], size=(height, width), mode="bilinear", align_corners=False)
        mask_b = F.interpolate(output["img2_patch_mask"], size=(height, width), mode="bilinear", align_corners=False)
    h_f = flow_mask_to_homography(flow_f, mask_f)
    h_b = flow_mask_to_homography(flow_b, mask_b)
    return h_f, h_b


def sample_error(
    point_set: list,
    h_f: torch.Tensor,
    h_b: torch.Tensor,
    num_matches: int,
    point_scale: tuple[float, float],
) -> torch.Tensor:
    errors = []
    scale = torch.tensor(point_scale, device=h_f.device, dtype=h_f.dtype)
    for pair in point_set[:num_matches]:
        src = torch.as_tensor(pair[0], device=h_f.device, dtype=h_f.dtype).view(1, 2) * scale
        dst = torch.as_tensor(pair[1], device=h_f.device, dtype=h_f.dtype).view(1, 2) * scale
        err_forward = torch.linalg.norm(transform_points(src, h_f) - dst, dim=1).mean()
        err_backward = torch.linalg.norm(transform_points(dst, h_b) - src, dim=1).mean()
        errors.append(torch.minimum(err_forward, err_backward))
    if not errors:
        return torch.tensor(float("nan"), device=h_f.device)
    return torch.stack(errors).mean()


def video_group(video_name: str) -> str:
    for name, videos in VIDEO_GROUPS.items():
        if video_name in videos:
            return name
    return "OTHER"


def evaluate(model: torch.nn.Module, loader, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    buckets: dict[str, list[float]] = {name: [] for name in list(VIDEO_GROUPS.keys()) + ["OTHER"]}
    rows = []
    coordinate_dir = Path(args.coordinate_dir)
    point_scale = (float(args.full_w) / float(args.point_w), float(args.full_h) / float(args.point_h))

    model.eval()
    iterator = loader
    if tqdm is not None:
        iterator = tqdm(loader, total=len(loader), ncols=100, desc="evaluate")

    with torch.no_grad():
        for step, batch_cpu in enumerate(iterator, start=1):
            batch = batch_to_device(batch_cpu, device)
            output = model(batch)
            h_f, h_b = compute_h_from_output(output, use_mask_weights=not args.no_mask_weighted_h)
            batch_size = h_f.shape[0]

            for idx in range(batch_size):
                pair_name = collated_get(batch_cpu["pair_name"], idx)
                video_name = collated_get(batch_cpu.get("video_names", [pair_name.split("/")[0]]), idx)
                point_set = parse_point_set_from_batch(batch_cpu, idx, coordinate_dir)
                err = sample_error(point_set, h_f[idx], h_b[idx], args.num_matches, point_scale)
                err_value = float(err.detach().cpu())
                group = video_group(video_name)
                buckets[group].append(err_value)
                rows.append({"pair_name": pair_name, "video_name": video_name, "group": group, "error": err_value})

            if args.max_steps and step >= args.max_steps:
                break

    group_metrics = {}
    for name, values in buckets.items():
        if values:
            group_metrics[name] = float(np.mean(values))
    realsh_groups = [name for name in VIDEO_GROUPS.keys() if name in group_metrics]
    macro_avg = float(np.mean([group_metrics[name] for name in realsh_groups])) if realsh_groups else float("nan")
    all_avg = float(np.mean([row["error"] for row in rows])) if rows else float("nan")
    return {"groups": group_metrics, "AVG": macro_avg, "ALL": all_avg, "count": len(rows), "rows": rows}


def write_outputs(metrics: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {key: value for key, value in metrics.items() if key != "rows"}
    (output_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2))
    with (output_dir / "eval_pairs.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pair_name", "video_name", "group", "error"])
        writer.writeheader()
        writer.writerows(metrics["rows"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate new_approach homography checkpoints with RealSH metrics.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("new_approach/eval_results"))
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=0)

    parser.add_argument("--test-list", default="Data/Test_List.txt")
    parser.add_argument("--test-image-dir", default="Data/Test")
    parser.add_argument("--coordinate-dir", default="Data/Coordinate-v2/Coordinate-v2")
    parser.add_argument("--full-h", type=int, default=360)
    parser.add_argument("--full-w", type=int, default=640)
    parser.add_argument("--crop-h", type=int, default=None)
    parser.add_argument("--crop-w", type=int, default=None)
    parser.add_argument("--point-h", type=int, default=360)
    parser.add_argument("--point-w", type=int, default=640)
    parser.add_argument("--rho", type=int, default=16)
    parser.add_argument("--shift", type=int, default=8)
    parser.add_argument("--num-matches", type=int, default=6)
    parser.add_argument("--no-mask-weighted-h", action="store_true")

    parser.add_argument("--pretrain-phase", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--mask-method", choices=["flow_matching", "homogan_cnn", "none"], default="flow_matching")
    parser.add_argument("--embed-dim", type=int, default=24)
    parser.add_argument("--depths", type=parse_ints, default=(2, 4, 6))
    parser.add_argument("--layer-depth", type=parse_ints, default=(3, 2, 1))
    parser.add_argument("--num-heads", type=parse_ints, default=(3, 12, 24))
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=float, default=3.0)
    parser.add_argument("--mask-flow-base-channels", type=int, default=32)
    parser.add_argument("--mask-flow-channel-mults", type=parse_ints, default=(1, 2, 4, 4))
    parser.add_argument("--mask-flow-time-dim", type=int, default=128)
    parser.add_argument("--mask-flow-steps", type=int, default=1)
    parser.add_argument("--mask-flow-solver", choices=["euler", "heun"], default="euler")
    parser.add_argument("--mask-flow-init", choices=["zero", "noise"], default="zero")
    parser.add_argument("--mask-flow-temperature", type=float, default=1.0)
    parser.add_argument("--mask-flow-noise-sigma", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt = torch_load(args.checkpoint)
    params = params_from_checkpoint(args, ckpt if isinstance(ckpt, dict) else None)

    if args.crop_h is None or args.crop_w is None:
        crop_h, crop_w = params.crop_size
        args.crop_h = int(crop_h)
        args.crop_w = int(crop_w)

    device = resolve_device(args.device)
    print(f"using device: {device}")

    model = Ms_Transformer(params=params).to(device)
    load_checkpoint_state(ckpt, args.checkpoint, model)

    loader = build_homo_flow_loader(
        repo_root=ROOT,
        list_path=args.test_list,
        image_dir=args.test_image_dir,
        crop_size=(args.crop_h, args.crop_w),
        full_size=(args.full_h, args.full_w),
        rho=args.rho,
        shift=args.shift,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        training=False,
        horizontal_flip_aug=False,
        coordinate_dir=args.coordinate_dir,
        max_items=args.max_items,
        seed=230,
    )

    metrics = evaluate(model, loader, args, device)
    write_outputs(metrics, args.output_dir)

    groups = metrics["groups"]
    print(
        "RealSH-style MSE: "
        f"AVG={metrics['AVG']:.4f} "
        f"RE={groups.get('RE', float('nan')):.4f} "
        f"LT={groups.get('LT', float('nan')):.4f} "
        f"LL={groups.get('LL', float('nan')):.4f} "
        f"SF={groups.get('SF', float('nan')):.4f} "
        f"LF={groups.get('LF', float('nan')):.4f} "
        f"ALL={metrics['ALL']:.4f} "
        f"count={metrics['count']}"
    )
    print(f"wrote {args.output_dir / 'eval_summary.json'}")
    print(f"wrote {args.output_dir / 'eval_pairs.csv'}")


if __name__ == "__main__":
    main()
