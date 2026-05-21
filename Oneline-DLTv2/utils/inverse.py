"""Safe 3x3 matrix inverse with a condition-number guard. Used by L_cycle to
skip ill-conditioned predictions rather than blow up training."""

import torch


def condition_number_3x3(H: torch.Tensor) -> torch.Tensor:
    """Batched 2-norm condition number for (B, 3, 3) matrices.
    Runs in fp32 because svdvals does not support fp16."""
    device = H.device
    with torch.amp.autocast(device_type=device.type, enabled=False):
        s = torch.linalg.svdvals(H.float())               # (B, 3)
    return s[:, 0] / (s[:, -1] + 1e-12)


def safe_inverse_3x3(H: torch.Tensor, cond_max: float = 1.0e4):
    """Returns (H_inv, valid_mask) where valid_mask is (B,) float in {0, 1}
    marking samples whose condition number is below `cond_max`. For invalid
    samples, H_inv is filled with identity to avoid NaNs but the mask is 0
    so any consumer can zero out the loss.

    All linear-algebra ops run in fp32 (AMP-safe) and the result is cast
    back to H's dtype."""
    B = H.shape[0]
    device, dtype = H.device, H.dtype
    cond = condition_number_3x3(H)
    valid = (cond < cond_max).float()

    H_safe = torch.where(
        valid.view(B, 1, 1) > 0.5,
        H,
        torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(B, 3, 3),
    )
    with torch.amp.autocast(device_type=device.type, enabled=False):
        H_inv = torch.linalg.inv(H_safe.float())
    return H_inv.to(dtype), valid
