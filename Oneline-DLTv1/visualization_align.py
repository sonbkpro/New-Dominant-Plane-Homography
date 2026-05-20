#!/usr/bin/env python3
"""
Visualize model-predicted image alignment from a correspondence .npy file.

The homography is computed by the trained network (same pipeline as test.py),
NOT from the ground-truth correspondence points.

Outputs two side-by-side images:
  1. <stem>_alignment.png  –  Original | Warp Overlay | Second image
  2. <stem>_points.png     –  6 pts on Original | 6 pts on Warp | 6 pts on Second image
  3. <stem>_mask.png       –  Learned patch mask heatmap and overlay
  4. <stem>_mask_raw.png   –  Learned patch mask as grayscale

Usage:
    python visualization_align.py \\
        --npy ../Data/Coordinate-v2/00000100_10001.jpg_00000100_10005.jpg.npy \\
        --model_path ../train_log_Oneline-FastDLT/real_models/resnet34_iter_44000.pth
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn

from dataset import TestDataset
from torch_homography_model import build_model
from utils import transformer as spatial_transform


# ── Visual style ──────────────────────────────────────────────────────────────
POINT_COLORS = [
    (  0,   0, 230),   # red
    (  0, 140, 255),   # orange
    ( 30, 180,  30),   # green
    (230,   0,   0),   # blue
    (200,   0, 200),   # magenta
    (  0, 200, 200),   # cyan
]
FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.65
LINE_TYPE  = cv2.LINE_AA
PT_RADIUS  = 9
SEP_W      = 5
SEP_COLOR  = (70, 70, 70)
BANNER_H   = 38
BANNER_BG  = (30, 30, 30)
BANNER_FG  = (240, 240, 240)


# ── Utilities ─────────────────────────────────────────────────────────────────

def die(msg):
    sys.exit('[ERROR] ' + msg)


def project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── Model helpers (identical logic to test.py) ────────────────────────────────

def load_model_weights(net, model_path):
    if not os.path.isfile(model_path):
        die('Checkpoint not found: ' + model_path)

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    if isinstance(checkpoint, nn.DataParallel):
        state_dict = checkpoint.module.state_dict()
    elif isinstance(checkpoint, nn.Module):
        state_dict = checkpoint.state_dict()
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint.get('state_dict',
                     checkpoint.get('model_state_dict', checkpoint))
    else:
        die('Unsupported checkpoint type: {}'.format(type(checkpoint)))

    cleaned = {(k[7:] if k.startswith('module.') else k): v
               for k, v in state_dict.items()}

    model_dict = net.state_dict()
    matched    = {k: v for k, v in cleaned.items() if k in model_dict}
    if not matched:
        die('No matching parameters found in checkpoint: ' + model_path)

    model_dict.update(matched)
    net.load_state_dict(model_dict)
    print('[INFO] Loaded {}/{} tensors from {}'.format(
        len(matched), len(model_dict), model_path))
    return net


def geometric_distance(correspondence, H):
    """Reprojection error: map pt1 through H and compare to pt2 (same as test.py)."""
    p1       = np.array([[correspondence[0][0]], [correspondence[0][1]], [1.0]])
    est_p2   = H @ p1
    est_p2   = est_p2 / est_p2[2, 0]
    p2       = np.array([[correspondence[1][0]], [correspondence[1][1]], [1.0]])
    return float(np.linalg.norm(p2 - est_p2))


# ── Pair resolution ───────────────────────────────────────────────────────────

def find_test_index(npy_path, test_list_path, npy_data):
    """Return the 0-based index in Test_List.txt for the given npy file."""
    if not os.path.isfile(test_list_path):
        die('Test_List.txt not found: ' + test_list_path)

    p1_base = os.path.basename(npy_data.get('path1', ''))
    p2_base = os.path.basename(npy_data.get('path2', ''))

    # Primary: match via path1/path2 stored in the npy
    with open(test_list_path) as f:
        for idx, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            if (os.path.basename(parts[0]) == p1_base and
                    os.path.basename(parts[1]) == p2_base):
                return idx

    # Fallback: infer from the npy filename itself
    # Format: "<path1>_<path2>.npy"
    stem = os.path.splitext(os.path.basename(npy_path))[0]
    if '.jpg_' in stem:
        left, right = stem.split('.jpg_', 1)
        p1_inf, p2_inf = left + '.jpg', right
        with open(test_list_path) as f:
            for idx, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) != 2:
                    continue
                if (os.path.basename(parts[0]) == p1_inf and
                        os.path.basename(parts[1]) == p2_inf):
                    return idx

    die('Cannot find a matching test pair for: ' + os.path.basename(npy_path))


# ── Drawing helpers ───────────────────────────────────────────────────────────

def separator(h):
    return np.full((h, SEP_W, 3), SEP_COLOR, dtype=np.uint8)


def with_banner(img, text):
    out = img.copy()
    h, w = out.shape[:2]
    cv2.rectangle(out, (0, 0), (w, BANNER_H), BANNER_BG, -1)
    (tw, th), _ = cv2.getTextSize(text, FONT, FONT_SCALE + 0.05, 2)
    tx = max(8, (w - tw) // 2)
    ty = BANNER_H - (BANNER_H - th) // 2 - 2
    cv2.putText(out, text, (tx, ty), FONT, FONT_SCALE + 0.05, BANNER_FG, 2, LINE_TYPE)
    return out


def draw_points(img, pts):
    out = img.copy()
    h, w = out.shape[:2]
    for i, (px, py) in enumerate(pts):
        px, py = int(round(float(px))), int(round(float(py)))
        if not (0 <= px < w and 0 <= py < h):
            continue
        color = POINT_COLORS[i % len(POINT_COLORS)]
        cv2.circle(out, (px, py), PT_RADIUS + 3, (255, 255, 255), -1, LINE_TYPE)
        cv2.circle(out, (px, py), PT_RADIUS,     color,           -1, LINE_TYPE)
        lx, ly = px + PT_RADIUS + 4, py + 5
        cv2.putText(out, str(i + 1), (lx + 1, ly + 1), FONT, FONT_SCALE,
                    (0, 0, 0),       2, LINE_TYPE)
        cv2.putText(out, str(i + 1), (lx,     ly),     FONT, FONT_SCALE,
                    (255, 255, 255), 1, LINE_TYPE)
    return out


def hstack(panels):
    h   = panels[0].shape[0]
    sep = separator(h)
    out = []
    for i, p in enumerate(panels):
        if i:
            out.append(sep)
        out.append(p)
    return np.concatenate(out, axis=1)


# ── Strip builders ────────────────────────────────────────────────────────────

def build_alignment_strip(img1, warped, img2):
    overlay = cv2.addWeighted(warped, 0.5, img2, 0.5, 0)
    return hstack([
        with_banner(img1,    'Original image'),
        with_banner(overlay, 'Warped original over second image'),
        with_banner(img2,    'Second image'),
    ])


def build_points_strip(img1, warped, img2, pts1, pts1_proj, pts2):
    return hstack([
        with_banner(draw_points(img1,   pts1),      '6 points on original image'),
        with_banner(draw_points(warped, pts1_proj), '6 points on warped image'),
        with_banner(draw_points(img2,   pts2),      '6 points on second image'),
    ])


def tensor_mask_to_uint8(mask_tensor):
    """Convert model mask output to a single-channel uint8 image."""
    mask = mask_tensor.detach().cpu().float().numpy()
    mask = np.squeeze(mask)
    if mask.ndim != 2:
        die('Expected 2D mask after squeeze, got shape {}'.format(mask.shape))
    mask = np.nan_to_num(mask, nan=0.0, posinf=1.0, neginf=0.0)
    mask = np.clip(mask, 0.0, 1.0)
    return (mask * 255.0).round().astype(np.uint8)


def crop_patch_from_h4p(img, h4p):
    corners = np.asarray(h4p, dtype=np.float32).reshape(4, 2)
    x0 = int(round(np.min(corners[:, 0])))
    y0 = int(round(np.min(corners[:, 1])))
    x1 = int(round(np.max(corners[:, 0])))
    y1 = int(round(np.max(corners[:, 1])))

    h, w = img.shape[:2]
    x0 = max(0, min(w, x0))
    x1 = max(0, min(w, x1))
    y0 = max(0, min(h, y0))
    y1 = max(0, min(h, y1))
    if x1 <= x0 or y1 <= y0:
        die('Invalid patch corners from h4p: {}'.format(corners.tolist()))
    return img[y0:y1, x0:x1]


def build_mask_strip(mask_u8, target_patch):
    """Build a visualization of the learned combined mask M_b * warped(M_a)."""
    mask_bgr = cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR)
    heatmap = cv2.applyColorMap(mask_u8, cv2.COLORMAP_JET)

    target_patch = cv2.resize(target_patch, (mask_u8.shape[1], mask_u8.shape[0]))
    overlay = cv2.addWeighted(target_patch, 0.55, heatmap, 0.45, 0)

    return hstack([
        with_banner(mask_bgr, 'Learned mask'),
        with_banner(heatmap,  'Mask heatmap'),
        with_banner(overlay,  'Mask over second-image patch'),
    ])


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Visualize model-predicted alignment for one test pair.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    p.add_argument('--npy', required=True,
                   help='Path to .npy correspondence file (Coordinate or Coordinate-v2).')
    p.add_argument('--model_path', default=None,
                   help='Checkpoint .pth file. '
                        'Defaults to ../train_log_Oneline-FastDLT/real_models/'
                        'resnet34_iter_44000.pth')
    p.add_argument('--model_name', default='resnet34',
                   choices=['resnet34', 'resnet50', 'resnet101', 'resnet152'])
    p.add_argument('--data_root',  default=None,
                   help='Path to Data/. Auto-detected from script location.')
    p.add_argument('--output_dir', default='viz_output')
    p.add_argument('--img_w',        type=int, default=640)
    p.add_argument('--img_h',        type=int, default=360)
    p.add_argument('--patch_size_w', type=int, default=560)
    p.add_argument('--patch_size_h', type=int, default=315)
    p.add_argument('--n_pts', type=int, default=6,
                   help='Number of correspondence points to draw (default: 6).')
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    root      = project_root()
    data_root = args.data_root or os.path.join(root, 'Data')
    test_dir  = os.path.join(data_root, 'Test')
    test_list = os.path.join(data_root, 'Test_List.txt')

    if not os.path.isfile(args.npy):
        die('npy file not found: ' + args.npy)
    if not os.path.isdir(test_dir):
        die('Test directory not found: ' + test_dir)

    # ── Load npy ──────────────────────────────────────────────────────────────
    npy_data   = np.load(args.npy, allow_pickle=True).item()
    matche_pts = npy_data['matche_pts'][:args.n_pts]
    pts1 = np.array([pt[0] for pt in matche_pts], dtype=np.float32)  # (N,2) img1 pixels
    pts2 = np.array([pt[1] for pt in matche_pts], dtype=np.float32)  # (N,2) img2 pixels

    print('[INFO] Pair : {}  ↔  {}'.format(npy_data.get('path1'), npy_data.get('path2')))

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('[INFO] Device:', device)

    # ── Build and load model (same as test.py) ────────────────────────────────
    model_path = args.model_path or os.path.join(
        root, 'train_log_Oneline-FastDLT', 'real_models', 'resnet34_iter_44000.pth')

    net = build_model(args.model_name, pretrained=False)
    net = load_model_weights(net, model_path)
    net = net.to(device).eval()

    # ── Normalization matrices M / M_inv (same as test.py) ───────────────────
    M = torch.tensor([[args.img_w / 2.0, 0.,             args.img_w / 2.0],
                      [0.,               args.img_h / 2.0, args.img_h / 2.0],
                      [0.,               0.,             1.            ]],
                     device=device)
    M_tile     = M.unsqueeze(0)                    # (1, 3, 3)
    M_tile_inv = torch.inverse(M).unsqueeze(0)     # (1, 3, 3)

    # ── Find matching index in Test_List.txt, then load via TestDataset ───────
    idx = find_test_index(args.npy, test_list, npy_data)
    print('[INFO] Matched test index: {}'.format(idx))

    test_data = TestDataset(
        data_path=root,
        patch_w=args.patch_size_w,
        patch_h=args.patch_size_h,
        rho=16,
        WIDTH=args.img_w,
        HEIGHT=args.img_h,
    )
    (org_imges, input_tesnors, patch_indices,
     h4p, print_img_1, print_img_2, _, _) = test_data[idx]

    # Add batch dim and send to device
    to_t = lambda arr: torch.tensor(arr).float().unsqueeze(0).to(device)
    org_imges_t     = to_t(org_imges)
    input_tesnors_t = to_t(input_tesnors)
    patch_indices_t = to_t(patch_indices)
    h4p_t           = to_t(h4p)

    # ── Model inference ───────────────────────────────────────────────────────
    with torch.no_grad():
        batch_out = net(org_imges_t, input_tesnors_t, h4p_t, patch_indices_t)
    H_mat = batch_out['H_mat']   # (1, 3, 3)  raw DLT homography (pixel space)
    mask_ap = batch_out.get('mask_ap_d')
    if mask_ap is None:
        die('Model output does not contain mask_ap_d. Check resnet.py out_dict.')

    # ── H for warping – denormalized, same formula as test.py ─────────────────
    H_warp = torch.matmul(torch.matmul(M_tile_inv, H_mat), M_tile)   # (1, 3, 3)

    # ── H for point evaluation – inv + normalize, identical to test.py ────────
    H_np   = H_mat.squeeze(0).cpu().numpy()
    H_eval = np.linalg.inv(H_np)
    H_eval = H_eval / H_eval[2, 2]          # normalize by H[2,2]  (= .item(8))
    # H_eval: maps img1 pixels → img2 pixels

    # ── Warp img1 using the spatial transformer (same as test.py) ─────────────
    print_img_1_t = torch.tensor(print_img_1, dtype=torch.float32).unsqueeze(0).to(device)
    pred_full, _  = spatial_transform(print_img_1_t, H_warp, (args.img_h, args.img_w))
    warped_bgr    = np.clip(pred_full.cpu().numpy()[0], 0, 255).astype(np.uint8)  # (H,W,C) BGR

    # ── Color images for the visualization panels ──────────────────────────────
    img1_bgr = np.transpose(print_img_1, [1, 2, 0]).astype(np.uint8)
    img2_bgr = np.transpose(print_img_2, [1, 2, 0]).astype(np.uint8)
    img1_bgr = cv2.resize(img1_bgr, (args.img_w, args.img_h))
    img2_bgr = cv2.resize(img2_bgr, (args.img_w, args.img_h))

    # ── Project pts1 through H_eval → where they land in warp / img2 space ───
    pts1_proj = cv2.perspectiveTransform(
        pts1.reshape(-1, 1, 2), H_eval).reshape(-1, 2)

    # ── Geometric error (exact same logic as test.py) ─────────────────────────
    errors = []
    for pt in matche_pts:
        err_LR = geometric_distance(pt,             H_eval)   # pt[0]→img1, pt[1]→img2
        err_RL = geometric_distance([pt[1], pt[0]], H_eval)   # reversed labelling
        errors.append(min(err_LR, err_RL))
    avg_err = sum(errors) / len(errors)
    print('[INFO] Geometric error  avg={:.4f} px   max={:.4f} px'.format(
        avg_err, max(errors)))

    # ── Build output strips ───────────────────────────────────────────────────
    strip_align  = build_alignment_strip(img1_bgr, warped_bgr, img2_bgr)
    strip_points = build_points_strip(
        img1_bgr, warped_bgr, img2_bgr, pts1, pts1_proj, pts2)
    mask_u8 = tensor_mask_to_uint8(mask_ap)
    img2_patch = crop_patch_from_h4p(img2_bgr, h4p)
    strip_mask = build_mask_strip(mask_u8, img2_patch)

    os.makedirs(args.output_dir, exist_ok=True)
    stem        = os.path.splitext(os.path.basename(args.npy))[0]
    out_align   = os.path.join(args.output_dir, stem + '_alignment.png')
    out_points  = os.path.join(args.output_dir, stem + '_points.png')
    out_mask    = os.path.join(args.output_dir, stem + '_mask.png')
    out_mask_raw = os.path.join(args.output_dir, stem + '_mask_raw.png')

    cv2.imwrite(out_align,  strip_align)
    cv2.imwrite(out_points, strip_points)
    cv2.imwrite(out_mask,   strip_mask)
    cv2.imwrite(out_mask_raw, mask_u8)
    print('[OK]  Alignment  →', out_align)
    print('[OK]  Points     →', out_points)
    print('[OK]  Mask       →', out_mask)
    print('[OK]  Mask raw   →', out_mask_raw)


if __name__ == '__main__':
    main()
