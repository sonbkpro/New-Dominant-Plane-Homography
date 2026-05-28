import math
from typing import Iterable, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "DominantMaskFlow",
    "MaskFlowUNet",
    "build_mask_condition",
    "logits_from_mask",
    "mask_from_logits",
    "make_flow_matching_batch",
]


def _num_groups(channels: int, max_groups: int = 8) -> int:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def _as_tuple(values: Iterable[int]) -> Tuple[int, ...]:
    return tuple(int(v) for v in values)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        if dim < 2:
            raise ValueError("time embedding dimension must be >= 2")
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 0:
            t = t[None]
        t = t.float().view(-1, 1)
        half_dim = self.dim // 2
        device = t.device
        dtype = t.dtype
        exponent = -math.log(10000.0) * torch.arange(half_dim, device=device, dtype=dtype)
        exponent = exponent / max(half_dim - 1, 1)
        freqs = torch.exp(exponent).view(1, -1)
        args = t * freqs
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.norm = nn.GroupNorm(_num_groups(out_channels), out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(_num_groups(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.norm2 = nn.GroupNorm(_num_groups(out_channels), out_channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.time_proj(t_emb).view(t_emb.shape[0], -1, 1, 1)
        h = self.conv2(self.dropout(self.act(self.norm2(h))))
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class MaskFlowUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        base_channels: int = 32,
        channel_mults: Sequence[int] = (1, 2, 4, 4),
        time_dim: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        if len(channel_mults) == 0:
            raise ValueError("channel_mults must contain at least one level")

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(inplace=True),
            nn.Linear(time_dim, time_dim),
        )
        self.input_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        channel_mults = _as_tuple(channel_mults)
        self.downs = nn.ModuleList()
        self.skip_channels = []
        ch = base_channels
        for idx, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            blocks = nn.ModuleList(
                [
                    ResBlock(ch, out_ch, time_dim, dropout=dropout),
                    ResBlock(out_ch, out_ch, time_dim, dropout=dropout),
                ]
            )
            down = Downsample(out_ch) if idx < len(channel_mults) - 1 else nn.Identity()
            self.downs.append(nn.ModuleDict({"blocks": blocks, "down": down}))
            self.skip_channels.append(out_ch)
            ch = out_ch

        self.mid1 = ResBlock(ch, ch, time_dim, dropout=dropout)
        self.mid2 = ResBlock(ch, ch, time_dim, dropout=dropout)

        self.ups = nn.ModuleList()
        for skip_ch in reversed(self.skip_channels):
            out_ch = skip_ch
            blocks = nn.ModuleList(
                [
                    ResBlock(ch + skip_ch, out_ch, time_dim, dropout=dropout),
                    ResBlock(out_ch, out_ch, time_dim, dropout=dropout),
                ]
            )
            self.ups.append(blocks)
            ch = out_ch

        self.output_norm = nn.GroupNorm(_num_groups(ch), ch)
        self.output_conv = nn.Conv2d(ch, out_channels, 3, padding=1)
        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 0:
            t = t[None].repeat(x.shape[0])
        elif t.ndim == 2 and t.shape[1] == 1:
            t = t[:, 0]
        t_emb = self.time_embed(t.to(device=x.device, dtype=x.dtype))

        h = self.input_conv(x)
        skips = []
        for down in self.downs:
            for block in down["blocks"]:
                h = block(h, t_emb)
            skips.append(h)
            h = down["down"](h)

        h = self.mid1(h, t_emb)
        h = self.mid2(h, t_emb)

        for blocks, skip in zip(self.ups, reversed(skips)):
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = torch.cat([h, skip], dim=1)
            for block in blocks:
                h = block(h, t_emb)

        h = F.silu(self.output_norm(h))
        return self.output_conv(h)


class DominantMaskFlow(nn.Module):
    def __init__(
        self,
        cond_channels: int,
        base_channels: int = 32,
        channel_mults: Sequence[int] = (1, 2, 4, 4),
        time_dim: int = 128,
        temperature: float = 1.0,
        noise_sigma: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.cond_channels = int(cond_channels)
        self.temperature = float(temperature)
        self.noise_sigma = float(noise_sigma)
        self.net = MaskFlowUNet(
            in_channels=1 + self.cond_channels,
            out_channels=1,
            base_channels=base_channels,
            channel_mults=channel_mults,
            time_dim=time_dim,
            dropout=dropout,
        )

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if cond.shape[1] != self.cond_channels:
            raise ValueError(
                f"expected {self.cond_channels} condition channels, got {cond.shape[1]}"
            )
        if z_t.shape[-2:] != cond.shape[-2:]:
            z_t = F.interpolate(z_t, size=cond.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([z_t, cond], dim=1)
        return self.net(x, t)

    def sample(
        self,
        cond: torch.Tensor,
        steps: int = 1,
        solver: str = "euler",
        init: str = "zero",
        requires_grad: bool = False,
    ) -> torch.Tensor:
        if steps < 1:
            raise ValueError("steps must be >= 1")
        solver = solver.lower()
        if solver not in {"euler", "heun"}:
            raise ValueError(f"unsupported solver: {solver}")

        batch_size, _, height, width = cond.shape
        if init == "noise":
            z = self.noise_sigma * torch.randn(
                batch_size, 1, height, width, device=cond.device, dtype=cond.dtype
            )
        elif init == "zero":
            z = torch.zeros(batch_size, 1, height, width, device=cond.device, dtype=cond.dtype)
        else:
            raise ValueError(f"unsupported init: {init}")

        dt = 1.0 / float(steps)
        with torch.set_grad_enabled(requires_grad):
            for idx in range(steps):
                t = torch.full((batch_size,), idx / float(steps), device=cond.device, dtype=cond.dtype)
                if solver == "heun" and steps > 1:
                    v1 = self.forward(z, t, cond)
                    z_pred = z + dt * v1
                    t_next = torch.full(
                        (batch_size,),
                        min((idx + 1) / float(steps), 1.0),
                        device=cond.device,
                        dtype=cond.dtype,
                    )
                    v2 = self.forward(z_pred, t_next, cond)
                    z = z + 0.5 * dt * (v1 + v2)
                else:
                    z = z + dt * self.forward(z, t, cond)
        return mask_from_logits(z, temperature=self.temperature)


def logits_from_mask(mask: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    mask = mask.clamp(eps, 1.0 - eps)
    return torch.log(mask) - torch.log1p(-mask)


def mask_from_logits(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    return torch.sigmoid(logits / max(float(temperature), 1e-6))


def _ensure_nchw_flow(flow: torch.Tensor) -> torch.Tensor:
    if flow.ndim != 4:
        raise ValueError("flow must have shape [B, 2, H, W] or [B, H, W, 2]")
    if flow.shape[1] == 2:
        return flow
    if flow.shape[-1] == 2:
        return flow.permute(0, 3, 1, 2).contiguous()
    raise ValueError("flow must have a channel dimension of size 2")


def build_mask_condition(
    reference_feature: torch.Tensor,
    warped_feature: torch.Tensor,
    reference_image: Optional[torch.Tensor] = None,
    warped_image: Optional[torch.Tensor] = None,
    flow: Optional[torch.Tensor] = None,
    detach: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    if detach:
        reference_feature = reference_feature.detach()
        warped_feature = warped_feature.detach()
        reference_image = reference_image.detach() if reference_image is not None else None
        warped_image = warped_image.detach() if warped_image is not None else None
        flow = flow.detach() if flow is not None else None

    if warped_feature.shape[-2:] != reference_feature.shape[-2:]:
        warped_feature = F.interpolate(
            warped_feature,
            size=reference_feature.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    feature_residual = torch.abs(reference_feature - warped_feature)
    cond = [reference_feature, warped_feature, feature_residual]

    if reference_image is not None and warped_image is not None:
        if warped_image.shape[-2:] != reference_feature.shape[-2:]:
            warped_image = F.interpolate(
                warped_image,
                size=reference_feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        if reference_image.shape[-2:] != reference_feature.shape[-2:]:
            reference_image = F.interpolate(
                reference_image,
                size=reference_feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        cond.append(torch.abs(reference_image - warped_image))

    if flow is not None:
        flow = _ensure_nchw_flow(flow)
        if flow.shape[-2:] != reference_feature.shape[-2:]:
            flow = F.interpolate(flow, size=reference_feature.shape[-2:], mode="bilinear", align_corners=True)
        _, _, height, width = flow.shape
        flow_x = flow[:, 0:1] / max(width, 1)
        flow_y = flow[:, 1:2] / max(height, 1)
        flow_mag = torch.sqrt(flow_x * flow_x + flow_y * flow_y + eps)
        cond.extend([flow_x, flow_y, flow_mag])

    return torch.cat(cond, dim=1)


def make_flow_matching_batch(
    target_mask: torch.Tensor,
    noise_sigma: float = 1.0,
    t_min: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x1 = logits_from_mask(target_mask)
    x0 = float(noise_sigma) * torch.randn_like(x1)
    batch_size = target_mask.shape[0]
    t = torch.rand(batch_size, device=target_mask.device, dtype=target_mask.dtype)
    t = t.clamp_min(float(t_min))
    t_view = t.view(batch_size, 1, 1, 1)
    z_t = (1.0 - t_view) * x0 + t_view * x1
    target_velocity = x1 - x0
    return z_t, t, target_velocity
