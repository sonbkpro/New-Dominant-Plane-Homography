import torch

__all__ = ["flow_mask_to_homography"]


def _identity(batch_size: int, device, dtype) -> torch.Tensor:
    return torch.eye(3, device=device, dtype=dtype).view(1, 3, 3).repeat(batch_size, 1, 1)


def _to_bhw2(flow: torch.Tensor) -> torch.Tensor:
    if flow.ndim != 4:
        raise ValueError("flow must be [B, 2, H, W] or [B, H, W, 2]")
    if flow.shape[-1] == 2:
        return flow
    if flow.shape[1] == 2:
        return flow.permute(0, 2, 3, 1).contiguous()
    raise ValueError("flow must have a 2-channel flow dimension")


def flow_mask_to_homography(
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
        weights = mask
        if weights.shape[-2:] != (height, width):
            raise ValueError("mask and flow spatial sizes must match")
        weights = weights.mean(dim=1).flatten(1).to(dtype=dtype)

    homographies = []
    identity = torch.eye(3, device=device, dtype=dtype)
    point_count = height * width
    take_count = min(max_points, point_count)

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
        w_i = w[indices].clamp_min(eps).sqrt().view(-1, 1)

        x, y = src_i[:, 0:1], src_i[:, 1:2]
        u, v = dst_i[:, 0:1], dst_i[:, 1:2]
        ones = torch.ones_like(x)
        zeros = torch.zeros_like(x)

        row1 = torch.cat([-x, -y, -ones, zeros, zeros, zeros, u * x, u * y, u], dim=1)
        row2 = torch.cat([zeros, zeros, zeros, -x, -y, -ones, v * x, v * y, v], dim=1)
        A = torch.stack([row1, row2], dim=1).reshape(-1, 9)
        A = A * w_i.repeat_interleave(2, dim=0)

        try:
            _, _, vh = torch.linalg.svd(A, full_matrices=False)
            H = vh[-1].view(3, 3)
            if torch.abs(H[-1, -1]) > eps:
                H = H / H[-1, -1]
            else:
                H = identity
        except RuntimeError:
            H = identity
        homographies.append(H)

    return torch.stack(homographies, dim=0)
