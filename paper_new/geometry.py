"""Differentiable geometry utilities for PCGV.

The original ``new_approach.geometry`` module intentionally stays NumPy/OpenCV
based for baseline evaluation.  This module contains the PyTorch geometry used
inside the new PCGV model: point grids, homography projection, weighted DLT,
flow conversion, and grid-sample warping.
"""

from __future__ import annotations

from contextlib import nullcontext
import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def _autocast_disabled_for(tensor: torch.Tensor):
    if tensor.device.type in ("cuda", "cpu"):
        return torch.amp.autocast(tensor.device.type, enabled=False)
    return nullcontext()


def torch_make_pixel_grid(batch: int, height: int, width: int,
                          device=None, dtype=None) -> torch.Tensor:
    """Return an ``[B, H*W, 2]`` grid in ``(x, y)`` pixel coordinates."""
    ys, xs = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    grid = torch.stack((xs, ys), dim=-1).reshape(1, height * width, 2)
    return grid.repeat(batch, 1, 1)


def torch_pixel_to_norm(points: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Convert pixel coordinates to grid_sample coordinates with align_corners=True."""
    x = points[..., 0]
    y = points[..., 1]
    if width > 1:
        x = 2.0 * x / float(width - 1) - 1.0
    else:
        x = torch.zeros_like(x)
    if height > 1:
        y = 2.0 * y / float(height - 1) - 1.0
    else:
        y = torch.zeros_like(y)
    return torch.stack((x, y), dim=-1)


def torch_norm_to_pixel(points: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Convert normalized grid_sample coordinates to pixel coordinates."""
    x = points[..., 0]
    y = points[..., 1]
    if width > 1:
        x = (x + 1.0) * float(width - 1) / 2.0
    else:
        x = torch.zeros_like(x)
    if height > 1:
        y = (y + 1.0) * float(height - 1) / 2.0
    else:
        y = torch.zeros_like(y)
    return torch.stack((x, y), dim=-1)


def torch_normalize_homography(H: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Scale homographies so ``H[..., 2, 2] == 1`` whenever possible."""
    denom = H[..., 2:3, 2:3]
    safe = torch.where(denom.abs() < eps, denom.sign().clamp(min=0.0) + eps, denom)
    return H / safe


def torch_warp_points_h(points: torch.Tensor, H: torch.Tensor,
                        eps: float = 1e-8) -> torch.Tensor:
    """Project ``[B, N, 2]`` points through ``[B, 3, 3]`` homographies."""
    ones = torch.ones_like(points[..., :1])
    homog = torch.cat((points, ones), dim=-1)
    warped = torch.bmm(homog, H.transpose(1, 2))
    z = warped[..., 2:3].clamp_min(eps)
    z = torch.where(warped[..., 2:3] < 0, warped[..., 2:3].clamp_max(-eps), z)
    return warped[..., :2] / z


def _as_batched_start(start_xy: Optional[torch.Tensor], batch: int,
                      device, dtype) -> torch.Tensor:
    if start_xy is None:
        return torch.zeros(batch, 2, device=device, dtype=dtype)
    start = start_xy.to(device=device, dtype=dtype)
    if start.ndim == 4:
        start = start[:, :, 0, 0]
    elif start.ndim == 3:
        start = start.squeeze(-1)
    if start.ndim == 1:
        start = start.view(1, 2).repeat(batch, 1)
    return start


def _hartley_normalize(points: torch.Tensor, weights: torch.Tensor,
                       eps: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Weighted Hartley point normalization."""
    batch = points.shape[0]
    wsum = weights.sum(dim=1, keepdim=True).clamp_min(eps)
    centroid = (points * weights.unsqueeze(-1)).sum(dim=1, keepdim=True) / wsum.unsqueeze(-1)
    centered = points - centroid
    dist = centered.square().sum(dim=-1).sqrt()
    mean_dist = (dist * weights).sum(dim=1) / wsum.squeeze(1)
    scale = math.sqrt(2.0) / mean_dist.clamp_min(eps)

    T = torch.zeros(batch, 3, 3, device=points.device, dtype=points.dtype)
    T[:, 0, 0] = scale
    T[:, 1, 1] = scale
    T[:, 2, 2] = 1.0
    T[:, 0, 2] = -scale * centroid[:, 0, 0]
    T[:, 1, 2] = -scale * centroid[:, 0, 1]

    ones = torch.ones_like(points[..., :1])
    homog = torch.cat((points, ones), dim=-1)
    norm = torch.bmm(homog, T.transpose(1, 2))[..., :2]
    return norm, T


def _build_dlt_matrix(src_pts: torch.Tensor, dst_pts: torch.Tensor) -> torch.Tensor:
    x, y = src_pts[..., 0], src_pts[..., 1]
    xp, yp = dst_pts[..., 0], dst_pts[..., 1]
    zeros = torch.zeros_like(x)
    ones = torch.ones_like(x)

    row1 = torch.stack((-x, -y, -ones, zeros, zeros, zeros, xp * x, xp * y, xp), dim=-1)
    row2 = torch.stack((zeros, zeros, zeros, -x, -y, -ones, yp * x, yp * y, yp), dim=-1)
    return torch.stack((row1, row2), dim=2).reshape(src_pts.shape[0], -1, 9)


def torch_weighted_dlt(src_pts: torch.Tensor, dst_pts: torch.Tensor,
                       weights: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Solve a weighted homography with differentiable DLT.

    Args:
        src_pts: ``[B, N, 2]`` source pixel coordinates.
        dst_pts: ``[B, N, 2]`` destination pixel coordinates.
        weights: ``[B, N]`` or ``[B, N, 1]`` non-negative correspondence weights.

    Returns:
        ``[B, 3, 3]`` homographies mapping ``src_pts`` to ``dst_pts``.
    """
    if src_pts.ndim != 3 or dst_pts.ndim != 3:
        raise ValueError("src_pts and dst_pts must have shape [B, N, 2]")
    if src_pts.shape != dst_pts.shape or src_pts.shape[-1] != 2:
        raise ValueError("src_pts and dst_pts must have matching [B, N, 2] shapes")
    if src_pts.shape[1] < 4:
        raise ValueError("weighted DLT needs at least four correspondences")

    orig_dtype = src_pts.dtype
    with _autocast_disabled_for(src_pts):
        src = src_pts.float()
        dst = dst_pts.float()
        w = weights.float().squeeze(-1).clamp_min(eps)

        src_n, T_src = _hartley_normalize(src, w, eps)
        dst_n, T_dst = _hartley_normalize(dst, w, eps)
        A = _build_dlt_matrix(src_n, dst_n)
        row_weights = w.sqrt().repeat_interleave(2, dim=1).unsqueeze(-1)
        A = A * row_weights

        _, _, Vh = torch.linalg.svd(A, full_matrices=True)
        Hn = Vh[:, -1].reshape(-1, 3, 3)
        H = torch.linalg.inv(T_dst) @ Hn @ T_src
        H = torch_normalize_homography(H, eps=eps)
    return H.to(dtype=orig_dtype)


def torch_dlt_condition_number(src_pts: torch.Tensor, dst_pts: torch.Tensor,
                               weights: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Return the SVD condition number of the weighted DLT system."""
    with _autocast_disabled_for(src_pts):
        src = src_pts.float()
        dst = dst_pts.float()
        w = weights.float().squeeze(-1).clamp_min(eps)
        src_n, _ = _hartley_normalize(src, w, eps)
        dst_n, _ = _hartley_normalize(dst, w, eps)
        A = _build_dlt_matrix(src_n, dst_n)
        A = A * w.sqrt().repeat_interleave(2, dim=1).unsqueeze(-1)
        s = torch.linalg.svdvals(A)
    return s[:, 0] / s[:, -1].clamp_min(eps)


def torch_dlt_leverage(src_pts: torch.Tensor, dst_pts: torch.Tensor,
                       weights: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Return a detached regularized DLT row/point leverage cue.

    This is a regularized DLT row/point leverage cue, not exact homography
    influence on the final null-space solution.
    """
    if src_pts.ndim != 3 or dst_pts.ndim != 3:
        raise ValueError("src_pts and dst_pts must have shape [B, N, 2]")
    if src_pts.shape != dst_pts.shape or src_pts.shape[-1] != 2:
        raise ValueError("src_pts and dst_pts must have matching [B, N, 2] shapes")
    if src_pts.shape[1] < 4:
        raise ValueError("DLT leverage needs at least four correspondences")

    orig_dtype = src_pts.dtype
    with torch.no_grad(), _autocast_disabled_for(src_pts):
        src = src_pts.float()
        dst = dst_pts.float()
        w = weights.float().squeeze(-1).clamp_min(eps)
        src_n, _ = _hartley_normalize(src, w, eps)
        dst_n, _ = _hartley_normalize(dst, w, eps)
        A = _build_dlt_matrix(src_n, dst_n)
        A = A * w.sqrt().repeat_interleave(2, dim=1).unsqueeze(-1)

        normal = A.transpose(1, 2) @ A
        eye = torch.eye(9, device=A.device, dtype=A.dtype).unsqueeze(0)
        diag_mean = normal.diagonal(dim1=-2, dim2=-1).mean(dim=1).clamp_min(1.0)
        normal = normal + (eps * diag_mean).view(-1, 1, 1) * eye
        inv_normal = torch.linalg.pinv(normal)
        row_leverage = (A @ inv_normal * A).sum(dim=-1).clamp_min(0.0)
        point_leverage = row_leverage.reshape(src.shape[0], src.shape[1], 2).mean(dim=-1, keepdim=True)
        mean = point_leverage.mean(dim=1, keepdim=True).clamp_min(eps)
        leverage = point_leverage / mean
        leverage = torch.nan_to_num(leverage, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    return leverage.to(dtype=orig_dtype)


def torch_homography_to_flow(H: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Convert patch-space homographies to dense ``[B, H, W, 2]`` flow."""
    grid = torch_make_pixel_grid(H.shape[0], height, width, H.device, H.dtype)
    warped = torch_warp_points_h(grid, H)
    flow = warped - grid
    return flow.reshape(H.shape[0], height, width, 2)


def torch_flow_to_homography_from_corners(flow: torch.Tensor,
                                          start_xy: Optional[torch.Tensor] = None,
                                          eps: float = 1e-6) -> torch.Tensor:
    """Recover a homography from the four corner displacements of a dense flow."""
    if flow.ndim != 4:
        raise ValueError("flow must be [B, H, W, 2] or [B, 2, H, W]")
    if flow.shape[1] == 2 and flow.shape[-1] != 2:
        flow = flow.permute(0, 2, 3, 1)
    batch, height, width, _ = flow.shape
    device, dtype = flow.device, flow.dtype
    local = torch.tensor(
        [[0.0, 0.0], [width - 1.0, 0.0], [width - 1.0, height - 1.0], [0.0, height - 1.0]],
        device=device,
        dtype=dtype,
    ).view(1, 4, 2).repeat(batch, 1, 1)
    corner_flow = torch.stack(
        (flow[:, 0, 0], flow[:, 0, width - 1], flow[:, height - 1, width - 1], flow[:, height - 1, 0]),
        dim=1,
    )
    start = _as_batched_start(start_xy, batch, device, dtype).unsqueeze(1)
    src = local + start
    dst = src + corner_flow
    weights = torch.ones(batch, 4, 1, device=device, dtype=dtype)
    return torch_weighted_dlt(src, dst, weights, eps=eps)


def torch_symmetric_transfer_error(H: torch.Tensor, src_pts: torch.Tensor,
                                   dst_pts: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Symmetric transfer error for correspondences."""
    pred_dst = torch_warp_points_h(src_pts, H, eps=eps)
    H_inv = torch.linalg.inv(H.float()).to(dtype=H.dtype)
    pred_src = torch_warp_points_h(dst_pts, H_inv, eps=eps)
    return (pred_dst - dst_pts).square().sum(dim=-1).sqrt() + (
        pred_src - src_pts).square().sum(dim=-1).sqrt()


def torch_convert_homography_scale(H: torch.Tensor,
                                   from_hw: Tuple[int, int],
                                   to_hw: Tuple[int, int]) -> torch.Tensor:
    """Convert a homography between two coordinate grids of different sizes.

    ``H`` maps points in the ``from_hw`` coordinate system to points in the same
    coordinate system.  The returned matrix maps points in ``to_hw`` coordinates
    to ``to_hw`` coordinates.
    """
    from_h, from_w = from_hw
    to_h, to_w = to_hw
    sx = float(to_w - 1) / float(max(from_w - 1, 1))
    sy = float(to_h - 1) / float(max(from_h - 1, 1))
    S = H.new_tensor([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]])
    S = S.unsqueeze(0).expand(H.shape[0], -1, -1)
    H_scaled = S @ H @ torch.linalg.inv(S)
    return torch_normalize_homography(H_scaled)


def torch_warp_tensor_with_flow(src: torch.Tensor, flow: torch.Tensor,
                                padding_mode: str = "border") -> torch.Tensor:
    """Sample ``src`` at ``target_grid + flow`` using bilinear grid_sample."""
    if flow.ndim != 4:
        raise ValueError("flow must be [B, H, W, 2] or [B, 2, H, W]")
    if flow.shape[1] == 2 and flow.shape[-1] != 2:
        flow = flow.permute(0, 2, 3, 1)
    batch, _, height, width = src.shape
    if flow.shape[:3] != (batch, height, width):
        raise ValueError("flow spatial size must match src")
    grid = torch_make_pixel_grid(batch, height, width, src.device, src.dtype).reshape(batch, height, width, 2)
    sample = grid + flow.to(dtype=src.dtype)
    norm = torch_pixel_to_norm(sample, height, width)
    return F.grid_sample(src, norm, mode="bilinear", padding_mode=padding_mode, align_corners=True)


def torch_warp_full_with_patch_flow(src: torch.Tensor, flow: torch.Tensor,
                                    start_xy: Optional[torch.Tensor] = None,
                                    padding_mode: str = "border") -> torch.Tensor:
    """Warp a full-frame tensor into a crop using patch-local flow.

    This mirrors HomoGAN's ``get_warp_flow(full_tensor, patch_flow, start)``:
    the output has the patch-flow spatial size, and each target crop pixel
    samples ``src`` at ``start_xy + local_xy + flow`` in full-frame coordinates.
    """
    if flow.ndim != 4:
        raise ValueError("flow must be [B, H, W, 2] or [B, 2, H, W]")
    if flow.shape[1] == 2 and flow.shape[-1] != 2:
        flow = flow.permute(0, 2, 3, 1)
    batch, _, full_h, full_w = src.shape
    if flow.shape[0] != batch:
        raise ValueError("flow batch size must match src")
    patch_h, patch_w = flow.shape[1:3]
    local = torch_make_pixel_grid(batch, patch_h, patch_w, src.device, src.dtype)
    local = local.reshape(batch, patch_h, patch_w, 2)
    start = _as_batched_start(start_xy, batch, src.device, src.dtype).view(batch, 1, 1, 2)
    sample = local + start + flow.to(device=src.device, dtype=src.dtype)
    norm = torch_pixel_to_norm(sample, full_h, full_w)
    return F.grid_sample(src, norm, mode="bilinear", padding_mode=padding_mode, align_corners=True)


def torch_warp_tensor_with_homography(src: torch.Tensor, H: torch.Tensor,
                                      out_hw: Optional[Tuple[int, int]] = None,
                                      padding_mode: str = "border") -> torch.Tensor:
    """Warp ``src`` into a target frame using a target-to-source homography."""
    batch, _, src_h, src_w = src.shape
    out_h, out_w = out_hw if out_hw is not None else (src_h, src_w)
    grid = torch_make_pixel_grid(batch, out_h, out_w, src.device, src.dtype)
    sample = torch_warp_points_h(grid, H.to(dtype=src.dtype)).reshape(batch, out_h, out_w, 2)
    norm = torch_pixel_to_norm(sample, src_h, src_w)
    return F.grid_sample(src, norm, mode="bilinear", padding_mode=padding_mode, align_corners=True)
