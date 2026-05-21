"""Warping wrappers built on F.grid_sample.

Convention: `H` is the source -> destination homography (i.e. for a source
pixel at (x, y), H * [x, y, 1]^T is its homogeneous destination location).
To synthesize the warped image at destination coordinates, we apply H^{-1}
to the destination grid and sample the source image."""

from typing import Tuple

import torch
import torch.nn.functional as F


def _make_pixel_grid(H_out: int, W_out: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Returns a (3, H_out * W_out) tensor of homogeneous pixel coordinates."""
    y, x = torch.meshgrid(
        torch.arange(H_out, device=device, dtype=dtype),
        torch.arange(W_out, device=device, dtype=dtype),
        indexing="ij",
    )
    ones = torch.ones_like(x)
    grid = torch.stack([x, y, ones], dim=0).reshape(3, -1)
    return grid


def warp_by_homography(
    img: torch.Tensor,
    H_src_to_dst: torch.Tensor,
    out_size: Tuple[int, int],
    out_origin_xy: Tuple[int, int] = (0, 0),
    padding_mode: str = "zeros",
    align_corners: bool = True,
) -> torch.Tensor:
    """Warp an image so that the result lies in destination coordinates.

    Args:
        img: (B, C, H_in, W_in) source image.
        H_src_to_dst: (B, 3, 3) source -> destination homography.
        out_size: (H_out, W_out) of the warped image.
        out_origin_xy: top-left corner of the output patch in destination
            coordinates. Use this when the destination region you want is a
            patch inside a larger destination image (default (0, 0) assumes
            the destination origin is the image origin).
    Returns:
        (B, C, H_out, W_out) warped image, sampled in destination coordinates.
    """
    B, C, H_in, W_in = img.shape
    H_out, W_out = out_size
    device, dtype = img.device, img.dtype

    grid = _make_pixel_grid(H_out, W_out, device, dtype)        # (3, N)
    grid = grid.unsqueeze(0).expand(B, -1, -1).clone()          # (B, 3, N)
    grid[:, 0, :] += out_origin_xy[0]
    grid[:, 1, :] += out_origin_xy[1]

    # linalg.inv requires fp32; force it inside any autocast scope.
    with torch.amp.autocast(device_type=device.type, enabled=False):
        H_inv = torch.linalg.inv(H_src_to_dst.float())            # (B, 3, 3)
    H_inv = H_inv.to(dtype)
    src_coords = torch.bmm(H_inv, grid)                          # (B, 3, N)
    src_coords = src_coords / (src_coords[:, 2:3, :] + 1e-8)
    sx = src_coords[:, 0, :]
    sy = src_coords[:, 1, :]

    # Normalize to [-1, 1] for grid_sample (align_corners=True).
    nx = 2.0 * sx / max(W_in - 1, 1) - 1.0
    ny = 2.0 * sy / max(H_in - 1, 1) - 1.0
    sample_grid = torch.stack([nx, ny], dim=-1).reshape(B, H_out, W_out, 2)

    warped = F.grid_sample(
        img, sample_grid, mode="bilinear",
        padding_mode=padding_mode, align_corners=align_corners,
    )
    return warped


def warp_patch_by_homography(
    feat: torch.Tensor,
    H_src_to_dst: torch.Tensor,
    padding_mode: str = "zeros",
) -> torch.Tensor:
    """Convenience wrapper that warps a patch-aligned feature map of size
    (H, W) onto itself, returning the same spatial extent. The homography is
    assumed to be expressed in the same pixel coordinates as the feature map.
    """
    B, C, H_in, W_in = feat.shape
    return warp_by_homography(feat, H_src_to_dst, (H_in, W_in), (0, 0), padding_mode)


def make_validity_mask(
    H_src_to_dst: torch.Tensor,
    out_size: Tuple[int, int],
    in_size: Tuple[int, int],
    out_origin_xy: Tuple[int, int] = (0, 0),
) -> torch.Tensor:
    """Returns (B, 1, H_out, W_out) float mask indicating where the warp
    drew from inside the source image (1) vs outside (0). Used to suppress
    border artifacts in alignment / cycle losses."""
    B = H_src_to_dst.shape[0]
    device = H_src_to_dst.device
    dtype = H_src_to_dst.dtype
    H_in, W_in = in_size
    H_out, W_out = out_size

    grid = _make_pixel_grid(H_out, W_out, device, dtype)
    grid = grid.unsqueeze(0).expand(B, -1, -1).clone()
    grid[:, 0, :] += out_origin_xy[0]
    grid[:, 1, :] += out_origin_xy[1]

    with torch.amp.autocast(device_type=device.type, enabled=False):
        H_inv = torch.linalg.inv(H_src_to_dst.float())
    H_inv = H_inv.to(dtype)
    src_coords = torch.bmm(H_inv, grid)
    src_coords = src_coords / (src_coords[:, 2:3, :] + 1e-8)
    sx = src_coords[:, 0, :]
    sy = src_coords[:, 1, :]

    inside = (
        (sx >= 0) & (sx <= W_in - 1) & (sy >= 0) & (sy <= H_in - 1)
    ).float()
    return inside.reshape(B, 1, H_out, W_out)
