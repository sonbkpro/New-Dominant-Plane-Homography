"""Synthetic supervised-H dataset (planv2 Phase 3 / Stage 2).

For every training image we:
  1. Load and normalize the full image I_a (single channel).
  2. Sample random 4-corner perturbations Delta_p* ~ U(-rho_s, rho_s)^8.
  3. Build H* by 4-point DLT from (corners, corners + Delta_p*).
  4. Warp the full I_a by H* to produce a synthetic I_b.
  5. Crop both I_a and I_b at a fixed random (x, y) to get patches.

The H-only branch is trained with L = Huber(Delta_p_pred, Delta_p*). No real
pairs and no posterior involved -- pure geometric bootstrap.

`H*` is constructed in FULL-image pixel coords so the synthetic I_b is what v1
would have produced (warp-then-crop)."""

import os
from typing import Dict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


_MEAN = np.array([118.93, 113.97, 102.60], dtype=np.float32).reshape(1, 1, 3)
_STD  = np.array([69.85,  68.81,  72.45 ], dtype=np.float32).reshape(1, 1, 3)


def _load_normalized_gray(path: str, W: int, H: int) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    if img.shape[1] != W or img.shape[0] != H:
        img = cv2.resize(img, (W, H))
    img = (img.astype(np.float32) - _MEAN) / _STD
    img = img.mean(axis=2, keepdims=True)
    return np.transpose(img, [2, 0, 1])              # (1, H, W)


def _solve_homography_4pt(src_xy: np.ndarray, dst_xy: np.ndarray) -> np.ndarray:
    """4-point DLT for a single (4, 2) src/dst correspondence pair.
    Identical algebra to utils.dlt.DLT_solve but in numpy and unbatched."""
    A = np.zeros((8, 8), dtype=np.float64)
    b = np.zeros((8,), dtype=np.float64)
    for i in range(4):
        x, y = src_xy[i]
        xp, yp = dst_xy[i]
        A[2 * i]     = [x, y, 1, 0, 0, 0, -xp * x, -xp * y]
        A[2 * i + 1] = [0, 0, 0, x, y, 1, -yp * x, -yp * y]
        b[2 * i]     = xp
        b[2 * i + 1] = yp
    h = np.linalg.solve(A, b)
    H = np.array([[h[0], h[1], h[2]],
                  [h[3], h[4], h[5]],
                  [h[6], h[7], 1.0]], dtype=np.float64)
    return H


class SynthPairDataset(Dataset):
    """Synthetic supervised pairs from a list of single training images.

    Reuses Train_List.txt's first column as the source-image source; the
    second column is ignored. This means each training pair contributes one
    synthetic example per epoch, but with random Delta_p* per call."""

    def __init__(
        self,
        data_list_path: str,
        train_root: str,
        patch_h: int = 315,
        patch_w: int = 560,
        img_h: int = 360,
        img_w: int = 640,
        rho: int = 16,
        rho_s: int = 32,
    ):
        super().__init__()
        with open(data_list_path, "r") as f:
            self.lines = [ln.strip() for ln in f if ln.strip()]
        self.train_root = train_root
        self.patch_h, self.patch_w = patch_h, patch_w
        self.img_h, self.img_w = img_h, img_w
        self.rho = rho
        self.rho_s = rho_s

    def __len__(self) -> int:
        return len(self.lines)

    def _random_crop_xy(self) -> np.ndarray:
        x = np.random.randint(self.rho, self.img_w - self.rho - self.patch_w)
        y = np.random.randint(self.rho, self.img_h - self.rho - self.patch_h)
        return np.array([x, y], dtype=np.float32)

    def _random_offset(self) -> np.ndarray:
        """Per-corner displacement in pixels, uniform in [-rho_s, +rho_s]^8."""
        return np.random.uniform(-self.rho_s, self.rho_s, size=(4, 2)).astype(np.float32)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        names = self.lines[idx].split()
        img_a_full = _load_normalized_gray(
            os.path.join(self.train_root, names[0]),
            self.img_w, self.img_h,
        )  # (1, H_full, W_full)

        # Patch crop chosen first, then we define H* in patch coords for the
        # supervised label, and convert to full-image coords for warping.
        crop_xy = self._random_crop_xy()                              # (2,)
        x, y = int(crop_xy[0]), int(crop_xy[1])

        # 4 patch corners in patch-local coords. Canonical order across the
        # whole codebase is TL, BL, BR, TR (v1-compatible). Must match
        # `h4p_patch` in cdpc_net.py and `src` in losses/fold.py so that
        # offset[2i:2i+2] perturbs the SAME corner everywhere -- otherwise
        # the supervised Huber pairs pred-BL with GT-TR (and vice versa)
        # and the supervised stage learns a label-permuted H.
        ph, pw = self.patch_h, self.patch_w
        src_patch = np.array(
            [[0,  0],
             [0,  ph],
             [pw, ph],
             [pw, 0]],
            dtype=np.float32,
        )
        offset = self._random_offset()                                # (4, 2)
        dst_patch = src_patch + offset
        H_patch = _solve_homography_4pt(src_patch, dst_patch)         # (3, 3)

        # Convert to full-image pixel coords by conjugation with translation.
        T = np.array([[1.0, 0.0, x], [0.0, 1.0, y], [0.0, 0.0, 1.0]], dtype=np.float64)
        T_inv = np.array([[1.0, 0.0, -x], [0.0, 1.0, -y], [0.0, 0.0, 1.0]], dtype=np.float64)
        H_full = T @ H_patch @ T_inv

        # Warp the full I_a by H_full to produce I_b_full (v1's pipeline).
        # cv2.warpPerspective with INTER_LINEAR and BORDER_REPLICATE matches
        # grid_sample's bilinear + replicate behavior closely enough for
        # supervised training; we don't compare pixels, only corners.
        img_a_hw = img_a_full[0]  # (H_full, W_full)
        img_b_hw = cv2.warpPerspective(
            img_a_hw, H_full,
            dsize=(self.img_w, self.img_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        img_b_full = img_b_hw[None, ...]                              # (1, H, W)

        I_a_patch = img_a_full[:, y:y + ph, x:x + pw]
        I_b_patch = img_b_full[:, y:y + ph, x:x + pw]

        # Targets: corner offset in patch-local coords (8,).
        offset_flat = offset.reshape(-1).astype(np.float32)

        # planv3 §5: also expose H_full_gt for the Frobenius-supervision term.
        H_full_t = (H_full / H_full[2, 2]).astype(np.float32)

        return {
            "I_a_full":  torch.from_numpy(img_a_full.copy()).float(),
            "I_b_full":  torch.from_numpy(img_b_full.copy()).float(),
            "I_a_patch": torch.from_numpy(I_a_patch.copy()).float(),
            "I_b_patch": torch.from_numpy(I_b_patch.copy()).float(),
            "crop_xy":   torch.from_numpy(crop_xy).float(),
            "offset_gt": torch.from_numpy(offset_flat).float(),
            "H_full_gt": torch.from_numpy(H_full_t).float(),
        }
