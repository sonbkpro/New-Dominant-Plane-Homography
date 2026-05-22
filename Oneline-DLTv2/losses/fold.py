"""L_fold: penalize fold-over of the warped quadrilateral.

The four corner offsets define a destination quadrilateral. For a non-degenerate
homography the quad must remain simple (non-self-intersecting) and convex; in
practice, that means all four signed cross products of consecutive edges share
the same sign and stay bounded away from zero.

We normalize the signed cross product by the source-quad area so the threshold
`eps_frac` is a fraction of the original quad area (0.1 = penalize when any
corner's signed area drops below 10% of the source area).

Corner order is the canonical TL, BL, BR, TR convention used by
cdpc_net.py:h4p_patch and DLT_solve. A previous version of this file used
TL, TR, BR, BL, which silently mis-attributed per-corner penalties because
the offset vector emitted by the head is in TL, BL, BR, TR order.
"""

import torch


def fold_loss(
    offset: torch.Tensor,
    patch_h: int,
    patch_w: int,
    eps_frac: float = 0.1,
) -> torch.Tensor:
    """
    Args:
        offset: (B, 8) corner offsets in pixel coords, ordered TL, BL, BR, TR
                as [x0,y0, x1,y1, x2,y2, x3,y3] -- same as DLT_solve.
        patch_h, patch_w: source patch dimensions in pixels.
        eps_frac: minimum signed area at each corner, as a fraction of the
                  source quad area.
    Returns: scalar mean penalty over the batch (zero when all quads convex).
    """
    B = offset.shape[0]
    device, dtype = offset.device, offset.dtype

    src = torch.tensor(
        [[0.0,     0.0],
         [0.0,     patch_h],
         [patch_w, patch_h],
         [patch_w, 0.0]],
        device=device, dtype=dtype,
    ).view(1, 4, 2).expand(B, -1, -1)
    dst = src + offset.view(B, 4, 2)

    e = torch.roll(dst, shifts=-1, dims=1) - dst          # (B, 4, 2)
    e_next = torch.roll(e, shifts=-1, dims=1)
    cross = e[..., 0] * e_next[..., 1] - e[..., 1] * e_next[..., 0]   # (B, 4)

    # In image coords (y down), the canonical source quad has all crosses
    # positive; we orient by the batch-wise mean sign so this works under
    # either convention.
    sign = torch.sign(cross.mean(dim=1, keepdim=True)).clamp(min=-1.0, max=1.0)
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    signed = cross * sign

    norm = float(patch_h * patch_w)
    signed_norm = signed / norm
    penalty = torch.clamp(eps_frac - signed_norm, min=0.0)            # (B, 4)
    return penalty.mean()
