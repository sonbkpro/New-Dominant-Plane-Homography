"""Compact tensorboard visualization helpers."""

import cv2
import numpy as np
import torch


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    return cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)


def _pick_hw(tensor: torch.Tensor) -> np.ndarray:
    """Extract a single-channel HxW numpy image from a (B, C, H, W) tensor."""
    arr = tensor.detach().cpu().float().numpy()
    if arr.ndim == 4:
        arr = arr[0, 0]
    elif arr.ndim == 3:
        arr = arr[0]
    return arr


def log_training_panel(
    writer,
    step: int,
    *,
    I_a: torch.Tensor,
    I_b: torch.Tensor,
    pred_I_b: torch.Tensor,
    q_map: torch.Tensor,
    log_sigma_map: torch.Tensor,
    residual_map: torch.Tensor,
):
    writer.add_image("input/I_a",    _to_uint8(_pick_hw(I_a)),    step, dataformats="HW")
    writer.add_image("input/I_b",    _to_uint8(_pick_hw(I_b)),    step, dataformats="HW")
    writer.add_image("warp/pred_I_b", _to_uint8(_pick_hw(pred_I_b)), step, dataformats="HW")
    writer.add_image("q/posterior",  _to_uint8(_pick_hw(q_map)),  step, dataformats="HW")
    writer.add_image("sigma/log",    _to_uint8(_pick_hw(log_sigma_map)), step, dataformats="HW")
    writer.add_image("residual/r",   _to_uint8(_pick_hw(residual_map)),  step, dataformats="HW")
