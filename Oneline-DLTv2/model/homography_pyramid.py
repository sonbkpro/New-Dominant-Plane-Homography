"""planv3 §3.3: coarse-to-fine homography pyramid.

Three levels at strides 1/16, 1/8, 1/4. Each level:

  1. Computes H_t = DLT(canonical_corners, canonical + delta_p_accum), where
     delta_p_accum is the 8-vector running sum of all previous-level corner
     updates.
  2. Pre-warps F_b^(t) by H_t^{-1} into F_a's coordinate frame so the
     regressor only has to resolve the residual misalignment.
  3. Builds a local cosine correlation cost volume between F_a^(t) and the
     pre-warped F_b^(t). Reduces it to a compact per-pixel feature.
  4. Concatenates [F_a, F_b_warped, correlation, coord channels] and feeds
     a conv tower + attentive pooling + MLP that outputs Delta_p^(t) in R^8.
  5. Bounds the update: Delta_p^(t) = rho_t * tanh(out_t).
  6. Accumulates: delta_p_accum <- delta_p_accum + Delta_p^(t).

Final H is DLT(canonical, canonical + delta_p_final).

The canonical corner order is TL, BL, BR, TR (matching DLT_solve and
cdpc_net.h4p_patch). The per-level corner offsets are in this same order.
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.dlt import DLT_solve
from utils.dlt_normalized import DLT_solve_normalized
from utils.warping import warp_by_homography
from utils.inverse import safe_inverse_3x3


# ---------------------------------------------------------------------------
# Local cosine correlation -- inlined here so each level can configure its
# own radius without sharing reduction conv weights across scales.
# ---------------------------------------------------------------------------

class _LevelCorrelation(nn.Module):
    """Local cosine correlation with a small 1x1 / 3x3 reducer."""

    def __init__(self, radius: int = 4, out_channels: int = 32):
        super().__init__()
        self.radius = radius
        self.cost_dim = (2 * radius + 1) ** 2
        self.reduce = nn.Sequential(
            nn.Conv2d(self.cost_dim, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, F_a: torch.Tensor, F_b: torch.Tensor) -> torch.Tensor:
        B, C, H, W = F_a.shape
        R = self.radius
        Fa_n = F.normalize(F_a, dim=1, eps=1e-6)
        Fb_n = F.normalize(F_b, dim=1, eps=1e-6)
        Fb_pad = F.pad(Fb_n, [R, R, R, R], mode="replicate")

        costs = []
        for dy in range(-R, R + 1):
            for dx in range(-R, R + 1):
                Fb_shift = Fb_pad[:, :, R + dy: R + dy + H, R + dx: R + dx + W]
                costs.append((Fa_n * Fb_shift).sum(dim=1, keepdim=True))
        cv = torch.cat(costs, dim=1)
        return self.reduce(cv)


# ---------------------------------------------------------------------------
# Attentive pooling -- learned spatial softmax replaces GAP. The pooled
# embedding is a single 256-d vector per sample.
# ---------------------------------------------------------------------------

class AttentivePool(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.score = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) -> (B, C)"""
        B, C, H, W = x.shape
        logits = self.score(x)                        # (B, 1, H, W)
        attn = torch.softmax(logits.flatten(2), dim=2).view(B, 1, H, W)
        return (x * attn).sum(dim=(2, 3))             # (B, C)


# ---------------------------------------------------------------------------
# Per-level corner-offset regressor.
# ---------------------------------------------------------------------------

def _conv_bn_relu(in_c: int, out_c: int, k: int = 3) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=k, padding=k // 2, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class _LevelRegressor(nn.Module):
    """Conv tower + attentive pool + MLP -> 8-dim corner offset."""

    def __init__(self, in_channels: int, hidden: int = 256):
        super().__init__()
        self.tower = nn.Sequential(
            _conv_bn_relu(in_channels, hidden),
            _conv_bn_relu(hidden, hidden),
            _conv_bn_relu(hidden, hidden),
            _conv_bn_relu(hidden, hidden),
        )
        self.pool = AttentivePool(hidden)
        self.fc = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 8),
        )
        # Init scale chosen so initial raw is O(0.5) and tanh stays in
        # linear regime; bias 0 -> initial mean offset 0.
        nn.init.normal_(self.fc[-1].weight, std=0.05)
        nn.init.zeros_(self.fc[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.tower(x)
        h = self.pool(h)               # (B, hidden)
        return self.fc(h)              # (B, 8)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _add_coord_channels(x: torch.Tensor) -> torch.Tensor:
    """Append (x/W in [-1,1], y/H in [-1,1]) to a (B,C,H,W) tensor."""
    B, C, H, W = x.shape
    device, dtype = x.device, x.dtype
    ys = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype).view(1, 1, H, 1).expand(B, 1, H, W)
    xs = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype).view(1, 1, 1, W).expand(B, 1, H, W)
    return torch.cat([x, xs, ys], dim=1)


def _rescale_homography(H: torch.Tensor, scale: float) -> torch.Tensor:
    """H operates on patch-pixel coords (scale 1). Return H' that operates on
    coords at `scale` (e.g. 0.25 for 1/4-resolution feature map)."""
    B = H.shape[0]
    device, dtype = H.device, H.dtype
    S = torch.tensor([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], device=device, dtype=dtype)
    Si = torch.tensor([[1.0 / scale, 0, 0], [0, 1.0 / scale, 0], [0, 0, 1]], device=device, dtype=dtype)
    Sb = S.unsqueeze(0).expand(B, -1, -1)
    Sib = Si.unsqueeze(0).expand(B, -1, -1)
    return torch.bmm(torch.bmm(Sb, H), Sib)


def _solve_h(use_normalized: bool, h4p: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
    if use_normalized:
        return DLT_solve_normalized(h4p, offset)
    return DLT_solve(h4p, offset)


# ---------------------------------------------------------------------------
# Top-level pyramid.
# ---------------------------------------------------------------------------

class HomographyPyramid(nn.Module):
    """Three-level coarse-to-fine corner-offset regressor.

    Forward signature (called from CDPCNet):
        features_a: dict with keys "stride16", "stride8", "stride4"
        features_b: same
        patch_h, patch_w: physical patch size in pixels (needed for DLT solve)
    Returns: dict with H_patch (final), per-level offsets, per-level Hs, etc.
    """

    LEVEL_STRIDES = (1.0 / 16.0, 1.0 / 8.0, 1.0 / 4.0)
    LEVEL_KEYS    = ("stride16", "stride8", "stride4")

    def __init__(
        self,
        in_channels_per_level: Tuple[int, int, int] = (256, 128, 64),
        corr_radius: int = 4,
        corr_out_channels: int = 32,
        rho_per_level: Tuple[float, float, float] = (32.0, 16.0, 8.0),
        n_levels: int = 3,
        use_normalized_dlt: bool = True,
    ):
        super().__init__()
        assert n_levels in (1, 2, 3), "n_levels must be 1, 2, or 3"
        self.n_levels = n_levels
        self.rho_per_level = rho_per_level
        self.use_normalized_dlt = use_normalized_dlt

        # Last `n_levels` levels (coarse-to-fine within the slice).
        self._active_indices = list(range(3 - n_levels, 3))

        # Per-level correlation and regressor. Input to each regressor is
        # [F_a, F_b_warped, correlation, x_coord, y_coord].
        self.correlations = nn.ModuleList()
        self.regressors = nn.ModuleList()
        for i in range(3):
            c_in_feat = in_channels_per_level[i]
            corr = _LevelCorrelation(radius=corr_radius, out_channels=corr_out_channels)
            self.correlations.append(corr)
            in_ch = 2 * c_in_feat + corr_out_channels + 2  # F_a + F_b_warped + corr + 2 coord
            self.regressors.append(_LevelRegressor(in_channels=in_ch))

    @staticmethod
    def _canonical_corners(B: int, ph: int, pw: int,
                           device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Canonical (TL, BL, BR, TR) flat 8-vector, repeated across B."""
        h4p = torch.tensor(
            [0.0, 0.0,
             0.0, float(ph),
             float(pw), float(ph),
             float(pw), 0.0],
            device=device, dtype=dtype,
        )
        return h4p.unsqueeze(0).expand(B, -1).contiguous()

    def _solve(self, h4p: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        return _solve_h(self.use_normalized_dlt, h4p, offset)

    def forward(
        self,
        features_a: Dict[str, torch.Tensor],
        features_b: Dict[str, torch.Tensor],
        patch_h: int,
        patch_w: int,
    ) -> Dict[str, torch.Tensor]:
        # Take any reference tensor for device/dtype/B.
        any_a = features_a["stride8"]
        B = any_a.shape[0]
        device, dtype = any_a.device, any_a.dtype

        h4p = self._canonical_corners(B, patch_h, patch_w, device, dtype)
        delta_accum = torch.zeros(B, 8, device=device, dtype=dtype)

        per_level_offsets: List[torch.Tensor] = []   # per-level *update* Δp^(t)
        per_level_H: List[torch.Tensor] = []
        per_level_cumulative: List[torch.Tensor] = []  # δp at end of level

        for t in self._active_indices:
            level_key = self.LEVEL_KEYS[t]
            level_scale = self.LEVEL_STRIDES[t]
            F_a_t = features_a[level_key]
            F_b_t = features_b[level_key]
            ph_t  = F_a_t.shape[-2]
            pw_t  = F_a_t.shape[-1]

            # Build the running H from delta_accum so far (patch coords).
            H_t_patch = self._solve(h4p, delta_accum)
            # Rescale H to operate on this level's feature-map pixel coords.
            H_t_lvl = _rescale_homography(H_t_patch, level_scale)
            # Pre-warp F_b: source = F_b, dest = F_a-frame. We want
            # F_b_warped(p) = F_b(H_t @ p), so use H_b_to_a = inv(H_t_lvl) as
            # the "source->destination" argument to warp_by_homography (which
            # internally inverts and samples).
            H_b_to_a, _ = safe_inverse_3x3(H_t_lvl)
            F_b_warped = warp_by_homography(
                F_b_t, H_b_to_a,
                out_size=(ph_t, pw_t),
                out_origin_xy=(0, 0),
                padding_mode="zeros",
            )

            # Local correlation between F_a and the pre-warped F_b.
            corr_t = self.correlations[t](F_a_t, F_b_warped)

            # Stack input and regress residual corner offset.
            x_in = torch.cat([F_a_t, F_b_warped, corr_t], dim=1)
            x_in = _add_coord_channels(x_in)
            raw  = self.regressors[t](x_in)              # (B, 8)
            rho_t = self.rho_per_level[t]
            delta_t = rho_t * torch.tanh(raw)            # (B, 8) bounded

            delta_accum = delta_accum + delta_t
            per_level_offsets.append(delta_t)
            per_level_cumulative.append(delta_accum.clone())
            per_level_H.append(self._solve(h4p, delta_accum))

        H_final_patch = self._solve(h4p, delta_accum)

        return {
            "H_patch":          H_final_patch,
            "offset":           delta_accum,
            "per_level_offset": per_level_offsets,
            "per_level_delta":  per_level_cumulative,
            "per_level_H":      per_level_H,
        }
