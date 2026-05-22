"""Hartley-normalized 4-point DLT.

Same input/output shape as utils.dlt.DLT_solve, but each point set is centered
at its centroid and rescaled so the average distance from the centroid is
sqrt(2). The DLT system is solved in normalized coordinates, then the result
is de-normalized via the similarity transforms. Defensive numerical stability
that matters most when the predicted homography is far from identity or when
patches use large pixel coordinates.

Reference: Hartley & Zisserman, "Multiple View Geometry", section 4.4.4.
"""

import torch


def _normalize_points(p: torch.Tensor):
    """p: (B, 4, 2). Returns (p_norm, T) where p_norm = T @ [p; 1] and T is the
    (B, 3, 3) similarity transform."""
    B = p.shape[0]
    device, dtype = p.device, p.dtype

    centroid = p.mean(dim=1, keepdim=True)                          # (B, 1, 2)
    centered = p - centroid                                          # (B, 4, 2)
    avg_dist = centered.norm(dim=-1).mean(dim=1, keepdim=True)       # (B, 1)
    scale = (2.0 ** 0.5) / avg_dist.clamp(min=1e-6)                  # (B, 1)

    T = torch.zeros(B, 3, 3, device=device, dtype=dtype)
    s = scale.squeeze(-1)
    T[:, 0, 0] = s
    T[:, 1, 1] = s
    T[:, 0, 2] = -s * centroid[:, 0, 0]
    T[:, 1, 2] = -s * centroid[:, 0, 1]
    T[:, 2, 2] = 1.0

    p_norm = centered * scale.unsqueeze(-1)
    return p_norm, T


def DLT_solve_normalized(src_p_flat: torch.Tensor, off_set_flat: torch.Tensor) -> torch.Tensor:
    """Hartley-normalized 4-point DLT.

    Args:
        src_p_flat:   (B, 8) source corners, [x0,y0, x1,y1, x2,y2, x3,y3].
        off_set_flat: (B, 8) corner offsets in pixels, same order.
    Returns:
        H: (B, 3, 3) homography mapping src -> src + offset.
    """
    bs = src_p_flat.shape[0]
    device = src_p_flat.device
    out_dtype = src_p_flat.dtype

    with torch.amp.autocast(device_type=device.type, enabled=False):
        src = src_p_flat.float().reshape(bs, 4, 2)
        off = off_set_flat.float().reshape(bs, 4, 2)
        dst = src + off

        src_n, T_src = _normalize_points(src)
        dst_n, T_dst = _normalize_points(dst)

        ones = torch.ones(bs, 4, 1, device=device, dtype=torch.float32)
        xy1 = torch.cat((src_n, ones), dim=2)                # (B, 4, 3)
        zeros = torch.zeros_like(xy1)
        xyu = torch.cat((xy1, zeros), dim=2)                 # (B, 4, 6)
        xyd = torch.cat((zeros, xy1), dim=2)                 # (B, 4, 6)
        M1 = torch.cat((xyu, xyd), dim=2).reshape(bs, -1, 6) # (B, 8, 6)

        M2 = torch.matmul(
            dst_n.reshape(bs * 4, 2, 1),
            src_n.reshape(bs * 4, 1, 2),
        ).reshape(bs, -1, 2)                                 # (B, 8, 2)

        A = torch.cat((M1, -M2), dim=2)                      # (B, 8, 8)
        b = dst_n.reshape(bs, 8, 1)

        try:
            h8 = torch.linalg.solve(A, b)
        except RuntimeError:
            h8 = torch.linalg.lstsq(A, b).solution

        h9 = torch.cat([h8.squeeze(-1),
                        torch.ones(bs, 1, device=device, dtype=torch.float32)],
                       dim=1)
        H_n = h9.reshape(bs, 3, 3)

        # De-normalize: H = T_dst^{-1} @ H_n @ T_src
        T_dst_inv = torch.linalg.inv(T_dst)
        H = torch.bmm(torch.bmm(T_dst_inv, H_n), T_src)

    return H.to(out_dtype)
