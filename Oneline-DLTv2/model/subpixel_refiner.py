"""CDPC-v4: sub-pixel correlation refinement of H_init.

This module is the *sub-pixel localization mechanism* that v3 was missing.
The joint trunk produces an initial H_init in the right basin of attraction
(within ~32 px corner offset), and this refiner improves it to sub-pixel
accuracy via a local cost volume with a soft-argmax displacement field.

Forward:

  1.  Per-image features at 1/4 scale from a small Siamese refinement trunk
      (ResNet-18 stem + layer1, 64-ch). Trained jointly with the joint trunk
      from random init.
  2.  Pre-warp F_b by H_init (at 1/4 scale) into F_a's coordinate frame ->
      F_b_warped. Now |residual displacement| at any pixel is at most a few
      feature cells.
  3.  Local cost volume: cos(F_a(p), F_b_warped(p + (dx,dy))) for
      (dx,dy) in [-R, R]^2 -> (B, (2R+1)^2, H/4, W/4) cost tensor.
  4.  Softmax across the (2R+1)^2 displacement candidates and take the
      expected value over the (dx, dy) grid -> dense sub-pixel displacement
      field d(x, y) at 1/4 scale. Multiply by 4 to map back to patch pixels.
  5.  Confidence c(x, y) = max softmax probability (used to weight
      corner-offset aggregation).
  6.  For each of the 4 patch corners, average d within a window around the
      corner weighted by c -> 4 corner displacements. Add them to the
      pre-warped corner positions to obtain destination corners; DLT to ΔH.
  7.  H_refined = H_init composed with ΔH (so the refinement is residual).

Sub-pixel argmax is the *mechanism* that lets us localize below feature-cell
resolution. Without it, no FC over pooled features can hit < 0.3 px accuracy.

Caveat: the refiner runs in patch coordinates. The trunk emits H_init in
patch coords too, so the composition is straightforward. The image-coords
H_full is then `T_crop @ H_refined @ T_crop^{-1}` as usual.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

from utils.dlt_normalized import DLT_solve_normalized
from utils.dlt import DLT_solve
from utils.warping import warp_by_homography


# ---------------------------------------------------------------------------
# Small Siamese trunk producing F^(1/4) per image.
# ---------------------------------------------------------------------------

class _Quarter1chTrunk(nn.Module):
    """ResNet-18 stem + layer1, 1-channel input, output at 1/4 scale, 64-ch.

    Random init by default (matching v1's no-pretrained convention). Pretrained
    is supported for ablation but typically hurts at this task.
    """

    def __init__(self, pretrained: bool = False, in_channels: int = 1,
                 out_channels: int = 64):
        super().__init__()
        try:
            weights = tv_models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            net = tv_models.resnet18(weights=weights)
        except (AttributeError, TypeError):
            net = tv_models.resnet18(pretrained=pretrained)

        orig_conv1 = net.conv1
        new_conv1 = nn.Conv2d(
            in_channels, orig_conv1.out_channels,
            kernel_size=orig_conv1.kernel_size, stride=orig_conv1.stride,
            padding=orig_conv1.padding, bias=False,
        )
        if pretrained:
            with torch.no_grad():
                mean_w = orig_conv1.weight.mean(dim=1, keepdim=True)
                new_conv1.weight.copy_(mean_w.expand(-1, in_channels, -1, -1))
        net.conv1 = new_conv1

        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1                        # 1/4, 64-ch

        self.proj = (
            nn.Conv2d(64, out_channels, kernel_size=1, bias=False)
            if out_channels != 64 else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        return self.proj(x)


# ---------------------------------------------------------------------------
# Sub-pixel correlation refiner
# ---------------------------------------------------------------------------

def _rescale_homography(H: torch.Tensor, scale: float) -> torch.Tensor:
    """Conjugate H by the scaling matrix so it operates on coords at `scale`."""
    B = H.shape[0]
    device, dtype = H.device, H.dtype
    S = torch.tensor([[scale, 0, 0], [0, scale, 0], [0, 0, 1]],
                     device=device, dtype=dtype)
    Si = torch.tensor([[1.0 / scale, 0, 0], [0, 1.0 / scale, 0], [0, 0, 1]],
                      device=device, dtype=dtype)
    Sb  = S.unsqueeze(0).expand(B, -1, -1)
    Sib = Si.unsqueeze(0).expand(B, -1, -1)
    return torch.bmm(torch.bmm(Sb, H), Sib)


class SubpixelRefiner(nn.Module):
    """Sub-pixel correlation refinement of H_init.

    The refiner has its own small Siamese trunk so the geometry trunk can
    stay focused on coarse 8-DoF regression. Both are trained jointly from
    random init.
    """

    def __init__(
        self,
        patch_h: int = 315,
        patch_w: int = 560,
        feat_channels: int = 64,
        radius: int = 6,
        feat_pretrained: bool = False,
        corner_window: int = 7,
        confidence_temperature: float = 1.0,
        rho_max: float = 16.0,
        use_normalized_dlt: bool = True,
    ):
        super().__init__()
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.radius = int(radius)
        self.corner_window = int(corner_window)
        self.tau = float(confidence_temperature)
        self.rho_max = float(rho_max)
        self.use_normalized_dlt = use_normalized_dlt

        self.feat = _Quarter1chTrunk(
            pretrained=feat_pretrained, in_channels=1, out_channels=feat_channels,
        )

        # Precompute displacement grid (dx, dy) for the (2R+1)^2 candidates.
        R = self.radius
        coords = torch.arange(-R, R + 1, dtype=torch.float32)
        dy, dx = torch.meshgrid(coords, coords, indexing="ij")
        # Flatten to (2R+1)^2 in the same order as the cost-volume channel axis
        # (dy outer, dx inner) -- matches the nested loop below.
        self.register_buffer("disp_dx", dx.flatten().view(1, -1, 1, 1))  # (1, K, 1, 1)
        self.register_buffer("disp_dy", dy.flatten().view(1, -1, 1, 1))

    # ----- cost volume helper -------------------------------------------------

    def _cost_volume(self, F_a: torch.Tensor, F_b_warped: torch.Tensor) -> torch.Tensor:
        """Returns (B, K, H, W) of cosine similarities for K=(2R+1)^2 disps."""
        B, C, H, W = F_a.shape
        R = self.radius
        Fa_n = F.normalize(F_a, dim=1, eps=1e-6)
        Fb_n = F.normalize(F_b_warped, dim=1, eps=1e-6)
        Fb_pad = F.pad(Fb_n, [R, R, R, R], mode="replicate")
        costs = []
        for dy in range(-R, R + 1):
            for dx in range(-R, R + 1):
                shifted = Fb_pad[:, :, R + dy: R + dy + H, R + dx: R + dx + W]
                costs.append((Fa_n * shifted).sum(dim=1, keepdim=True))   # (B,1,H,W)
        return torch.cat(costs, dim=1)                                     # (B, K, H, W)

    # ----- displacement and confidence ---------------------------------------

    def _displacement_and_confidence(self, cost: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Softmax over candidates; return (expected (dx,dy), max prob) per pixel.

        Output dx/dy are in *feature-pixel* units (i.e. 1/4 of patch pixels).
        """
        probs = F.softmax(cost / self.tau, dim=1)                          # (B, K, H, W)
        ex = (probs * self.disp_dx).sum(dim=1, keepdim=True)               # (B, 1, H, W)
        ey = (probs * self.disp_dy).sum(dim=1, keepdim=True)
        conf, _ = probs.max(dim=1, keepdim=True)                           # (B, 1, H, W)
        return ex, ey, conf

    # ----- corner aggregation -------------------------------------------------

    def _aggregate_corner_offsets(
        self,
        ex_patch: torch.Tensor,    # (B, 1, H_p, W_p) -- expected dx in PATCH px
        ey_patch: torch.Tensor,    # (B, 1, H_p, W_p)
        conf: torch.Tensor,        # (B, 1, H_p, W_p)
    ) -> torch.Tensor:
        """For each of the 4 corners (TL, BL, BR, TR), compute the confidence-
        weighted average displacement in a window around the corner.

        The window has side `corner_window` *feature-pixels* (so ~corner_window*4
        patch pixels). We use feature-pixel windowing because the inputs are
        already at patch resolution after upsampling.

        Returns (B, 8) flat corner-offset vector in patch pixels, canonical
        TL, BL, BR, TR order.
        """
        B, _, H_p, W_p = ex_patch.shape
        w = self.corner_window
        # Convert corner_window from feature-pixel units to patch-pixel units.
        # ex_patch / ey_patch / conf are already at PATCH resolution (we
        # upsampled them by 4 before this call), so the window is 4*w patch
        # pixels on a side, but anchored at the four corners.
        wp = w * 4
        # Clamp window to half the smaller patch dim, just in case.
        wp = max(1, min(wp, min(H_p, W_p) // 2))

        # Build per-corner anchor (top-left of the window in the patch).
        # Patch corner positions (in patch pixel coords, TL/BL/BR/TR):
        corners_xy = [(0, 0), (0, H_p - 1), (W_p - 1, H_p - 1), (W_p - 1, 0)]
        offsets = []
        for (cx, cy) in corners_xy:
            x0 = max(0, cx - wp // 2)
            y0 = max(0, cy - wp // 2)
            x1 = min(W_p, x0 + wp)
            y1 = min(H_p, y0 + wp)
            ex_c   = ex_patch[:, :, y0:y1, x0:x1]
            ey_c   = ey_patch[:, :, y0:y1, x0:x1]
            conf_c = conf[:,    :, y0:y1, x0:x1]
            denom = conf_c.sum(dim=(2, 3), keepdim=True).clamp(min=1e-6)
            dx_mean = (ex_c * conf_c).sum(dim=(2, 3), keepdim=True) / denom
            dy_mean = (ey_c * conf_c).sum(dim=(2, 3), keepdim=True) / denom
            offsets.append(dx_mean.view(B, 1))
            offsets.append(dy_mean.view(B, 1))
        return torch.cat(offsets, dim=1)                                   # (B, 8)

    # ----- forward ------------------------------------------------------------

    def _solve(self, h4p: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        if self.use_normalized_dlt:
            return DLT_solve_normalized(h4p, offset)
        return DLT_solve(h4p, offset)

    @staticmethod
    def _canonical_corners(B, ph, pw, device, dtype):
        h4p = torch.tensor(
            [0.0, 0.0,
             0.0, float(ph),
             float(pw), float(ph),
             float(pw), 0.0],
            device=device, dtype=dtype,
        )
        return h4p.unsqueeze(0).expand(B, -1).contiguous()

    def forward(
        self,
        I_a_patch: torch.Tensor,
        I_b_patch: torch.Tensor,
        H_init_patch: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            I_a_patch, I_b_patch: (B, 1, ph, pw)
            H_init_patch:         (B, 3, 3) in patch coords from the joint trunk.
        Returns dict with:
            H_refined_patch:  (B, 3, 3)  final refined homography in patch coords
            offset_residual:  (B, 8)     ΔH's corner offsets (bounded by rho_max)
            F_a_quarter:      (B, C, ph/4, pw/4)
            F_b_quarter:      (B, C, ph/4, pw/4)
            F_b_warped:       (B, C, ph/4, pw/4)
            confidence_map:   (B, 1, ph/4, pw/4)
            displacement_xy:  (B, 2, ph/4, pw/4) in PATCH pixels
        """
        B = I_a_patch.shape[0]
        device, dtype = I_a_patch.device, I_a_patch.dtype
        ph, pw = self.patch_h, self.patch_w

        # 1. Per-image features at 1/4.
        F_a = self.feat(I_a_patch)                                          # (B, C, ph4, pw4)
        F_b = self.feat(I_b_patch)
        h4 = F_a.shape[-2]
        w4 = F_a.shape[-1]

        # 2. Pre-warp F_b into F_a's frame using H_init (rescaled to 1/4 coords).
        H_init_q = _rescale_homography(H_init_patch, 0.25)
        # We want F_b_warped(p) = F_b(H_init @ p), so the "source->destination"
        # arg to warp_by_homography (which inverts internally) is inv(H_init_q).
        from utils.inverse import safe_inverse_3x3
        H_b_to_a_q, _ = safe_inverse_3x3(H_init_q)
        F_b_warped = warp_by_homography(
            F_b, H_b_to_a_q,
            out_size=(h4, w4),
            out_origin_xy=(0, 0),
            padding_mode="zeros",
        )

        # 3. Local cost volume + soft-argmax displacement field.
        cost = self._cost_volume(F_a, F_b_warped)                           # (B, K, h4, w4)
        # Run softmax/expected value in fp32 to avoid AMP precision losses.
        ex_q, ey_q, conf = self._displacement_and_confidence(cost.float())
        ex_q = ex_q.to(dtype)
        ey_q = ey_q.to(dtype)
        conf = conf.to(dtype)

        # 4. Upsample displacement + confidence to PATCH resolution and convert
        #    units from feature pixels to patch pixels (multiply by 4).
        ex_p = F.interpolate(ex_q, size=(ph, pw), mode="bilinear", align_corners=True) * 4.0
        ey_p = F.interpolate(ey_q, size=(ph, pw), mode="bilinear", align_corners=True) * 4.0
        conf_p = F.interpolate(conf, size=(ph, pw), mode="bilinear", align_corners=True)

        # 5. Confidence-weighted aggregation per corner -> 8-vector.
        corner_off = self._aggregate_corner_offsets(ex_p, ey_p, conf_p)     # (B, 8)
        # Cap the residual so a noisy refiner cannot blow up H_init.
        corner_off = self.rho_max * torch.tanh(corner_off / self.rho_max)

        # 6. ΔH from residual corner offsets; H_refined = H_init @ ΔH (so the
        #    refinement is applied in source coordinates).
        h4p = self._canonical_corners(B, ph, pw, device, dtype)
        dH_patch = self._solve(h4p, corner_off)
        H_refined_patch = torch.bmm(H_init_patch, dH_patch)

        disp_xy = torch.cat([ex_p, ey_p], dim=1)                            # (B, 2, ph, pw)

        return {
            "H_refined_patch":  H_refined_patch,
            "offset_residual":  corner_off,
            "F_a_quarter":      F_a,
            "F_b_quarter":      F_b,
            "F_b_warped":       F_b_warped,
            "confidence_map":   conf,            # at 1/4 resolution
            "displacement_xy":  disp_xy,         # at PATCH resolution, units = patch px
        }
