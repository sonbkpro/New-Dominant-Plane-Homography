from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

__all__ = [
    "SyntheticFlowMaskConfig",
    "SyntheticFlowMaskDataset",
    "build_synthetic_flow_mask_loader",
]


@dataclass
class SyntheticFlowMaskConfig:
    height: int = 128
    width: int = 128
    length: int = 10000
    max_translation: int = 12
    min_outliers: int = 1
    max_outliers: int = 4
    min_outlier_size: int = 16
    max_outlier_size: int = 48
    blur_kernel: int = 7
    seed: Optional[int] = None


def _randint(generator: torch.Generator, low: int, high: int) -> int:
    if high <= low:
        return int(low)
    return int(torch.randint(low, high, (1,), generator=generator).item())


def _smooth_image(image: torch.Tensor, rounds: int = 2) -> torch.Tensor:
    image = image.unsqueeze(0)
    for _ in range(rounds):
        image = F.avg_pool2d(image, kernel_size=5, stride=1, padding=2)
    return image.squeeze(0)


def _make_base_image(height: int, width: int, generator: torch.Generator) -> torch.Tensor:
    image = torch.rand(1, height, width, generator=generator)
    image = _smooth_image(image, rounds=3)
    yy = torch.linspace(0, 1, height).view(1, height, 1)
    xx = torch.linspace(0, 1, width).view(1, 1, width)
    image = 0.65 * image + 0.2 * xx + 0.15 * yy
    return image.clamp(0.0, 1.0)


def _translate(image: torch.Tensor, dx: float, dy: float, padding_mode: str = "border") -> torch.Tensor:
    _, height, width = image.shape
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, height, device=image.device),
        torch.linspace(-1, 1, width, device=image.device),
        indexing="ij",
    )
    norm_dx = 2.0 * dx / max(width - 1, 1)
    norm_dy = 2.0 * dy / max(height - 1, 1)
    grid = torch.stack([xx - norm_dx, yy - norm_dy], dim=-1).unsqueeze(0)
    return F.grid_sample(
        image.unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=True,
    ).squeeze(0)


def _soften_mask(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
    if kernel_size <= 1:
        return mask
    pad = kernel_size // 2
    mask = F.avg_pool2d(mask.unsqueeze(0), kernel_size=kernel_size, stride=1, padding=pad).squeeze(0)
    return mask.clamp(0.0, 1.0)


class SyntheticFlowMaskDataset(Dataset):
    def __init__(self, config: SyntheticFlowMaskConfig | None = None):
        self.config = config or SyntheticFlowMaskConfig()
        self.base_generator = torch.Generator()
        if self.config.seed is not None:
            self.base_generator.manual_seed(int(self.config.seed))

    def __len__(self) -> int:
        return int(self.config.length)

    def _generator_for_index(self, idx: int) -> torch.Generator:
        generator = torch.Generator()
        if self.config.seed is None:
            generator.seed()
        else:
            generator.manual_seed(int(self.config.seed) + int(idx) * 9973)
        return generator

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        cfg = self.config
        generator = self._generator_for_index(idx)
        height, width = int(cfg.height), int(cfg.width)

        img1 = _make_base_image(height, width, generator)
        dx = _randint(generator, -cfg.max_translation, cfg.max_translation + 1)
        dy = _randint(generator, -cfg.max_translation, cfg.max_translation + 1)
        img2 = _translate(img1, dx=dx, dy=dy)

        mask2 = torch.ones(1, height, width)
        outlier_count = _randint(generator, cfg.min_outliers, cfg.max_outliers + 1)
        for _ in range(outlier_count):
            max_rh = min(cfg.max_outlier_size, height)
            max_rw = min(cfg.max_outlier_size, width)
            min_rh = min(cfg.min_outlier_size, max_rh)
            min_rw = min(cfg.min_outlier_size, max_rw)
            rh = _randint(generator, min_rh, max_rh + 1)
            rw = _randint(generator, min_rw, max_rw + 1)
            top = _randint(generator, 0, max(height - rh + 1, 1))
            left = _randint(generator, 0, max(width - rw + 1, 1))
            patch = _make_base_image(rh, rw, generator)
            patch_shift = _translate(
                patch,
                dx=_randint(generator, -cfg.max_translation, cfg.max_translation + 1),
                dy=_randint(generator, -cfg.max_translation, cfg.max_translation + 1),
            )
            img2[:, top:top + rh, left:left + rw] = patch_shift
            mask2[:, top:top + rh, left:left + rw] = 0.0

        mask1 = _translate(mask2, dx=-dx, dy=-dy, padding_mode="zeros")
        mask1 = _soften_mask(mask1, cfg.blur_kernel)
        mask2 = _soften_mask(mask2, cfg.blur_kernel)

        pts = torch.tensor(
            [[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]],
            dtype=torch.float32,
        )
        flow = torch.zeros(2, height, width, dtype=torch.float32)
        flow[0].fill_(float(dx))
        flow[1].fill_(float(dy))

        imgs = torch.cat([img1, img2], dim=0).float()
        return {
            "imgs_gray_patch": imgs,
            "imgs_gray_full": imgs.clone(),
            "img1_dominant_mask": mask1.float(),
            "img2_dominant_mask": mask2.float(),
            "dominant_flow": flow,
            "start": torch.zeros(2, 1, 1, dtype=torch.float32),
            "pts": pts,
            "ori_size": torch.tensor([width, height], dtype=torch.float32),
        }


def build_synthetic_flow_mask_loader(
    config: SyntheticFlowMaskConfig | None = None,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        SyntheticFlowMaskDataset(config),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
    )
