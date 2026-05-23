"""planv3 B6 / §6: structured-affine + projective synthetic pairs.

The v2 synth dataset (`data/synth_pairs.py`) samples 4 corner offsets
uniformly in [-rho, rho]^8. That distribution under-represents structured
warps (small rotations, scale changes, near-affine projective tilts) which
dominate real CA-Homography pairs. A network supervised only on uniform
corner perturbations learns to handle them but extrapolates poorly to the
real-pair distribution.

This module samples H* as a composition of four random simple transforms:

    H* = H_persp · H_rot(theta) · H_scale(s) · H_trans(t)

with the ranges below. The planv3 §B6 spec originally allowed
|h31|, |h32| ~ U(0, 5e-4), but on a 360×640 image that produces patch-
corner displacements of ~100 px purely from the projective division --
which exceeds the pyramid's bounded-tanh budget Σ ρ_t = 56 px and pins
L_sup_corner at a floor (~10 px). The persp range is therefore tightened
to 5e-5; theta and translation are unchanged.

    theta ~ U(-4 deg,  4 deg)
    s     ~ U( 0.95,    1.05)
    t     ~ U(-15,       15)        (per-axis, in pixels)
    |h31|, |h32| ~ U( 0,  5e-5)     (sign uniform)

After H* is constructed, the patch-corner offset is computed analytically
and rejected (resampled) if any |Δp_i| exceeds `max_offset_px`. If 10
consecutive resamples all exceed the cap (rare), the sample falls back
to uniform-corner sampling within ±rho_s, which is always reachable by
the pyramid as long as rho_s ≤ Σ ρ_t.

The structured / uniform mix is controlled by `structured_frac`; 30% of
samples are uniform by default so the model sees both distributions.

Canonical corner order is TL, BL, BR, TR throughout (planv3 B1).
"""

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
    return np.transpose(img, [2, 0, 1])


# ---------------------------------------------------------------------------
# Random structured H* builders
# ---------------------------------------------------------------------------

def _rand_translation(t_range: float, rng: np.random.Generator) -> np.ndarray:
    tx = rng.uniform(-t_range, t_range)
    ty = rng.uniform(-t_range, t_range)
    H = np.eye(3, dtype=np.float64)
    H[0, 2] = tx
    H[1, 2] = ty
    return H


def _rand_rotation(theta_deg_range: float, center_xy: np.ndarray,
                   rng: np.random.Generator) -> np.ndarray:
    theta = np.deg2rad(rng.uniform(-theta_deg_range, theta_deg_range))
    c, s = np.cos(theta), np.sin(theta)
    cx, cy = float(center_xy[0]), float(center_xy[1])
    R = np.eye(3, dtype=np.float64)
    R[0, 0] = c
    R[0, 1] = -s
    R[0, 2] = cx - c * cx + s * cy
    R[1, 0] = s
    R[1, 1] = c
    R[1, 2] = cy - s * cx - c * cy
    return R


def _rand_scale(s_min: float, s_max: float, center_xy: np.ndarray,
                rng: np.random.Generator) -> np.ndarray:
    s = rng.uniform(s_min, s_max)
    cx, cy = float(center_xy[0]), float(center_xy[1])
    S = np.eye(3, dtype=np.float64)
    S[0, 0] = s
    S[1, 1] = s
    S[0, 2] = (1 - s) * cx
    S[1, 2] = (1 - s) * cy
    return S


def _rand_perspective(p_range: float, rng: np.random.Generator) -> np.ndarray:
    h31 = rng.uniform(-p_range, p_range)
    h32 = rng.uniform(-p_range, p_range)
    H = np.eye(3, dtype=np.float64)
    H[2, 0] = h31
    H[2, 1] = h32
    return H


def _structured_H(image_w: int, image_h: int,
                  theta_deg: float = 4.0,
                  scale_min: float = 0.95, scale_max: float = 1.05,
                  trans_px: float = 15.0,
                  persp: float = 5e-5,
                  rng: np.random.Generator = None) -> np.ndarray:
    """Compose persp @ rot @ scale @ trans around the image center."""
    if rng is None:
        rng = np.random.default_rng()
    center = np.array([image_w / 2.0, image_h / 2.0], dtype=np.float64)
    H_t = _rand_translation(trans_px, rng)
    H_s = _rand_scale(scale_min, scale_max, center, rng)
    H_r = _rand_rotation(theta_deg, center, rng)
    H_p = _rand_perspective(persp, rng)
    return H_p @ H_r @ H_s @ H_t


# ---------------------------------------------------------------------------
# Algebraic patch-corner offset from a full-image H
# ---------------------------------------------------------------------------

def _apply_H_to_pts(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """pts: (N, 2) in pixel coords. Returns (N, 2) after H."""
    ones = np.ones((pts.shape[0], 1), dtype=pts.dtype)
    p_h = np.concatenate([pts, ones], axis=1).T          # (3, N)
    q_h = H @ p_h
    q   = q_h[:2] / (q_h[2:3] + 1e-12)
    return q.T


def _solve_homography_4pt_np(src_xy: np.ndarray, dst_xy: np.ndarray) -> np.ndarray:
    """Unbatched 4-point DLT in numpy. Identical algebra to utils/dlt.py."""
    A = np.zeros((8, 8), dtype=np.float64)
    b = np.zeros((8,),   dtype=np.float64)
    for i in range(4):
        x, y = src_xy[i]
        xp, yp = dst_xy[i]
        A[2 * i]     = [x, y, 1, 0, 0, 0, -xp * x, -xp * y]
        A[2 * i + 1] = [0, 0, 0, x, y, 1, -yp * x, -yp * y]
        b[2 * i]     = xp
        b[2 * i + 1] = yp
    h = np.linalg.solve(A, b)
    return np.array([[h[0], h[1], h[2]],
                     [h[3], h[4], h[5]],
                     [h[6], h[7], 1.0]], dtype=np.float64)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SynthPairDatasetV3(Dataset):
    """Synthetic supervised pairs with structured-affine + projective warps.

    Sample blend:
        70% structured H*  (rotation + scale + translation + small projective)
        30% uniform-corner H* (legacy uniform delta_p ~ U(-rho_s, rho_s)^8)
    """

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
        structured_frac: float = 0.7,
        # Structured-H ranges (planv3 §B6, tightened so corner offsets stay
        # within the pyramid's Σ ρ_t = 56 px budget).
        theta_deg: float = 4.0,
        scale_min: float = 0.95, scale_max: float = 1.05,
        trans_px: float = 15.0,
        persp: float = 5e-5,
        # Reject-and-resample cap: a structured sample whose max |Δp_i|
        # exceeds this is regenerated. Falls back to uniform-corner sampling
        # after `max_resamples` consecutive failures.
        max_offset_px: float = 48.0,
        max_resamples: int = 10,
    ):
        super().__init__()
        with open(data_list_path, "r") as f:
            self.lines = [ln.strip() for ln in f if ln.strip()]
        self.train_root = train_root
        self.patch_h, self.patch_w = patch_h, patch_w
        self.img_h, self.img_w = img_h, img_w
        self.rho = rho
        self.rho_s = rho_s
        self.structured_frac = structured_frac
        self.theta_deg = theta_deg
        self.scale_min, self.scale_max = scale_min, scale_max
        self.trans_px = trans_px
        self.persp = persp
        self.max_offset_px = float(max_offset_px)
        self.max_resamples = int(max_resamples)

    def __len__(self) -> int:
        return len(self.lines)

    def _random_crop_xy(self, rng: np.random.Generator) -> np.ndarray:
        x = rng.integers(self.rho, self.img_w - self.rho - self.patch_w)
        y = rng.integers(self.rho, self.img_h - self.rho - self.patch_h)
        return np.array([x, y], dtype=np.float32)

    def _patch_offset_from_H_full(self, H_full: np.ndarray, x: int, y: int) -> np.ndarray:
        """Given a full-image H, compute the corresponding 4 patch-corner
        offsets in canonical TL,BL,BR,TR order in patch-local coords."""
        ph, pw = self.patch_h, self.patch_w
        # Source corners in FULL-image coords:
        src_full = np.array(
            [[x,         y],
             [x,         y + ph],
             [x + pw,    y + ph],
             [x + pw,    y]],
            dtype=np.float64,
        )
        dst_full = _apply_H_to_pts(H_full, src_full)
        # Convert both into patch-local coords (subtract crop origin).
        src_patch = src_full - np.array([x, y], dtype=np.float64)
        dst_patch = dst_full - np.array([x, y], dtype=np.float64)
        offset = (dst_patch - src_patch).astype(np.float32)        # (4, 2)
        return offset

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rng = np.random.default_rng(seed=None)
        names = self.lines[idx].split()
        img_a_full = _load_normalized_gray(
            os.path.join(self.train_root, names[0]),
            self.img_w, self.img_h,
        )                                                          # (1, H, W)
        crop_xy = self._random_crop_xy(rng)
        x, y = int(crop_xy[0]), int(crop_xy[1])
        ph, pw = self.patch_h, self.patch_w

        use_structured = (rng.random() < self.structured_frac)

        if use_structured:
            # Reject-and-resample: structured H combos can produce corner
            # offsets that exceed the pyramid's bounded budget; the network
            # then cannot fit the GT and L_sup_corner stalls at the clip
            # residual. Cap with `max_offset_px` and retry up to
            # `max_resamples` times before falling back to uniform.
            H_full = None
            offset = None
            for _ in range(self.max_resamples):
                H_full_try = _structured_H(
                    self.img_w, self.img_h,
                    theta_deg=self.theta_deg,
                    scale_min=self.scale_min, scale_max=self.scale_max,
                    trans_px=self.trans_px,
                    persp=self.persp,
                    rng=rng,
                )
                offset_try = self._patch_offset_from_H_full(H_full_try, x, y)
                if float(np.abs(offset_try).max()) <= self.max_offset_px:
                    H_full = H_full_try
                    offset = offset_try
                    break
            if H_full is None:
                # All resamples exceeded the cap. Force a uniform-corner
                # fallback so the sample is still usable.
                use_structured = False

        if not use_structured:
            # Legacy uniform-corner sampling (canonical TL,BL,BR,TR order).
            offset = rng.uniform(-self.rho_s, self.rho_s, size=(4, 2)).astype(np.float32)
            src_patch = np.array(
                [[0,  0], [0, ph], [pw, ph], [pw, 0]],
                dtype=np.float64,
            )
            dst_patch = src_patch + offset
            H_patch = _solve_homography_4pt_np(src_patch, dst_patch)
            T = np.array(
                [[1.0, 0.0, x],
                 [0.0, 1.0, y],
                 [0.0, 0.0, 1.0]], dtype=np.float64)
            T_inv = np.array(
                [[1.0, 0.0, -x],
                 [0.0, 1.0, -y],
                 [0.0, 0.0, 1.0]], dtype=np.float64)
            H_full = T @ H_patch @ T_inv

        # Warp the full I_a by H_full to produce I_b_full.
        img_a_hw = img_a_full[0]
        img_b_hw = cv2.warpPerspective(
            img_a_hw, H_full,
            dsize=(self.img_w, self.img_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        img_b_full = img_b_hw[None, ...]

        I_a_patch = img_a_full[:, y:y + ph, x:x + pw]
        I_b_patch = img_b_full[:, y:y + ph, x:x + pw]

        # Final supervised label: 8-vector of patch-local corner offsets,
        # canonical TL,BL,BR,TR.
        offset_flat = offset.reshape(-1).astype(np.float32)

        # Also stash H_full as label for the optional Frobenius supervision
        # (planv3 §5 L_sup_H). Normalize so it can be compared in float.
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
