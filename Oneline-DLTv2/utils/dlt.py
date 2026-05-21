"""Differentiable 4-point DLT, simplified port of v1 utils.DLT_solve for the
single-mesh case (one homography per pair). Solves H such that
dst_corners = H * src_corners, where corners are (x, y) image coordinates."""

import torch


def DLT_solve(src_p_flat: torch.Tensor, off_set_flat: torch.Tensor) -> torch.Tensor:
    """
    Args:
        src_p_flat:   (B, 8) tensor, four source corners [x1,y1,x2,y2,x3,y3,x4,y4]
        off_set_flat: (B, 8) tensor, predicted corner offsets in the same order
    Returns:
        H: (B, 3, 3) homography mapping src corners to dst = src + offset.
    """
    bs = src_p_flat.shape[0]
    device = src_p_flat.device
    dtype = src_p_flat.dtype

    src_ps = src_p_flat.reshape(bs, 4, 2)
    off_sets = off_set_flat.reshape(bs, 4, 2)
    dst_p = src_ps + off_sets

    ones = torch.ones(bs, 4, 1, device=device, dtype=dtype)
    xy1 = torch.cat((src_ps, ones), dim=2)            # (B, 4, 3)
    zeros = torch.zeros_like(xy1)
    xyu = torch.cat((xy1, zeros), dim=2)              # (B, 4, 6)
    xyd = torch.cat((zeros, xy1), dim=2)              # (B, 4, 6)
    M1 = torch.cat((xyu, xyd), dim=2).reshape(bs, -1, 6)  # (B, 8, 6)

    M2 = torch.matmul(
        dst_p.reshape(bs * 4, 2, 1),
        src_ps.reshape(bs * 4, 1, 2),
    ).reshape(bs, -1, 2)                              # (B, 8, 2)

    A = torch.cat((M1, -M2), dim=2)                   # (B, 8, 8)
    b = dst_p.reshape(bs, 8, 1)                       # (B, 8, 1)

    # Solve via lstsq for numerical robustness; falls back to torch.linalg.solve.
    try:
        h8 = torch.linalg.solve(A, b)                 # (B, 8, 1)
    except RuntimeError:
        h8 = torch.linalg.lstsq(A, b).solution

    h9 = torch.cat([h8.squeeze(-1), torch.ones(bs, 1, device=device, dtype=dtype)], dim=1)
    H = h9.reshape(bs, 3, 3)
    return H
