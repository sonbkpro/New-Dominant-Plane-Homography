import torch

from new_approach.geometry_homo import _to_bhw2

__all__ = ["weighted_normalized_dlt"]


def _normalize_points(points: torch.Tensor, weights: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    weights = weights.clamp_min(eps)
    weights = weights / weights.sum().clamp_min(eps)
    centroid = (points * weights[:, None]).sum(dim=0)
    centered = points - centroid
    mean_dist = (centered.pow(2).sum(dim=1).sqrt() * weights).sum().clamp_min(eps)
    scale = torch.sqrt(points.new_tensor(2.0)) / mean_dist
    transform = torch.eye(3, device=points.device, dtype=points.dtype)
    transform[0, 0] = scale
    transform[1, 1] = scale
    transform[0, 2] = -scale * centroid[0]
    transform[1, 2] = -scale * centroid[1]
    ones = torch.ones(points.shape[0], 1, device=points.device, dtype=points.dtype)
    points_h = torch.cat([points, ones], dim=1)
    normalized = points_h @ transform.transpose(0, 1)
    return normalized[:, :2], transform


def weighted_normalized_dlt(
    flow: torch.Tensor,
    mask: torch.Tensor | None = None,
    max_points: int = 2048,
    min_points: int = 16,
    eps: float = 1e-6,
) -> torch.Tensor:
    flow = _to_bhw2(flow)
    batch_size, height, width, _ = flow.shape
    device, dtype = flow.device, flow.dtype

    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    src = torch.stack([xx, yy], dim=-1).view(-1, 2)
    flow_flat = flow.view(batch_size, -1, 2)

    if mask is None:
        weights = torch.ones(batch_size, height * width, device=device, dtype=dtype)
    else:
        if mask.ndim != 4:
            raise ValueError("mask must be [B, 1, H, W]")
        if mask.shape[-2:] != (height, width):
            raise ValueError("mask and flow spatial sizes must match")
        weights = mask.mean(dim=1).flatten(1).to(dtype=dtype)

    point_count = height * width
    take_count = min(int(max_points), point_count)
    identity = torch.eye(3, device=device, dtype=dtype)
    homographies = []

    for batch_idx in range(batch_size):
        w = weights[batch_idx].clamp_min(0.0)
        if torch.count_nonzero(w > eps).item() < min_points:
            w = torch.ones_like(w)

        if take_count < point_count:
            _, indices = torch.topk(w, k=take_count, largest=True, sorted=False)
        else:
            indices = torch.arange(point_count, device=device)

        src_i = src[indices]
        dst_i = src_i + flow_flat[batch_idx, indices]
        w_i = w[indices].clamp_min(eps)
        src_n, t_src = _normalize_points(src_i, w_i, eps)
        dst_n, t_dst = _normalize_points(dst_i, w_i, eps)

        x, y = src_n[:, 0:1], src_n[:, 1:2]
        u, v = dst_n[:, 0:1], dst_n[:, 1:2]
        ones = torch.ones_like(x)
        zeros = torch.zeros_like(x)
        row1 = torch.cat([-x, -y, -ones, zeros, zeros, zeros, u * x, u * y, u], dim=1)
        row2 = torch.cat([zeros, zeros, zeros, -x, -y, -ones, v * x, v * y, v], dim=1)
        A = torch.stack([row1, row2], dim=1).reshape(-1, 9)
        A = A * w_i.sqrt().view(-1, 1).repeat_interleave(2, dim=0)

        try:
            _, _, vh = torch.linalg.svd(A, full_matrices=False)
            h_norm = vh[-1].view(3, 3)
            H = torch.linalg.inv(t_dst) @ h_norm @ t_src
            if torch.isfinite(H).all() and torch.abs(H[-1, -1]) > eps:
                H = H / H[-1, -1]
            else:
                H = identity
        except RuntimeError:
            H = identity
        homographies.append(H)

    return torch.stack(homographies, dim=0)
