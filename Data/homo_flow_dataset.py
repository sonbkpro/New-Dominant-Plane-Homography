import os
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    import cv2
except ImportError as exc:
    cv2 = None
    _CV2_IMPORT_ERROR = exc
else:
    _CV2_IMPORT_ERROR = None

try:
    from PIL import Image
except ImportError as exc:
    Image = None
    _PIL_IMPORT_ERROR = exc
else:
    _PIL_IMPORT_ERROR = None

__all__ = ["HomoFlowDataset", "build_homo_flow_loader"]


def _as_hw(size: Sequence[int]) -> Tuple[int, int]:
    if len(size) != 2:
        raise ValueError("size must contain height and width")
    return int(size[0]), int(size[1])


def _resize_array(image: np.ndarray, size_hw: Tuple[int, int]) -> np.ndarray:
    height, width = size_hw
    if image.shape[0] == height and image.shape[1] == width:
        return image
    if cv2 is not None:
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

    if image.ndim == 2:
        tensor = torch.from_numpy(image).float()[None, None]
        resized = F.interpolate(tensor, size=(height, width), mode="bilinear", align_corners=True)
        return resized[0, 0].numpy()
    tensor = torch.from_numpy(np.ascontiguousarray(image)).float().permute(2, 0, 1)[None]
    resized = F.interpolate(tensor, size=(height, width), mode="bilinear", align_corners=True)
    return resized[0].permute(1, 2, 0).numpy()


def _find_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    if cv2 is not None:
        homography, _ = cv2.findHomography(src.astype(np.float32), dst.astype(np.float32))
        if homography is not None:
            return homography.astype(np.float32)

    rows = []
    for (x, y), (u, v) in zip(src.astype(np.float64), dst.astype(np.float64)):
        rows.append([-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u])
        rows.append([0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v])
    _, _, vh = np.linalg.svd(np.asarray(rows, dtype=np.float64))
    homography = vh[-1].reshape(3, 3)
    if abs(homography[2, 2]) > 1e-8:
        homography = homography / homography[2, 2]
    return homography.astype(np.float32)


def worker_init_fn(worker_id: int) -> None:
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed + worker_id)


class HomoFlowDataset(Dataset):
    """HomoGAN-style image-pair dataset for the local Data/Train and Data/Test folders.

    The official HomoGAN training loader reads explicit image pairs, applies optional
    horizontal flip, BGR mean/std normalization, grayscale conversion, random crop
    with a rho margin, and returns the crop start plus a random perturbation
    homography. Validation/test resize the full frame pair to the model crop size.

    Expected list file format, matching Data/Train_List.txt and Data/Test_List.txt:
        video/frame_a.jpg video/frame_b.jpg

    Expected image directory layout by default:
        Data/Train/<video>/<frame>.jpg
        Data/Test/<video>/<frame>.jpg
    """

    def __init__(
        self,
        repo_root: str | os.PathLike,
        list_path: str | os.PathLike,
        image_dir: str | os.PathLike,
        crop_size: Sequence[int] = (320, 512),
        full_size: Sequence[int] = (360, 640),
        rho: int = 16,
        shift: int = 8,
        training: bool = True,
        horizontal_flip_aug: bool = True,
        fixed_start: Optional[Tuple[int, int]] = None,
        normalize: bool = True,
        coordinate_dir: Optional[str | os.PathLike] = None,
        max_items: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        if cv2 is None and Image is None:
            raise ImportError("HomoFlowDataset requires either cv2 or PIL") from _PIL_IMPORT_ERROR

        self.repo_root = Path(repo_root)
        self.list_path = Path(list_path)
        self.image_dir = Path(image_dir)
        self.crop_h, self.crop_w = _as_hw(crop_size)
        self.full_h, self.full_w = _as_hw(full_size)
        self.rho = int(rho)
        self.shift = int(shift)
        self.training = bool(training)
        self.horizontal_flip_aug = bool(horizontal_flip_aug)
        self.fixed_start = fixed_start
        self.normalize = bool(normalize)
        self.seed = seed
        self.coordinate_dir = Path(coordinate_dir) if coordinate_dir is not None else None

        if not self.list_path.is_absolute():
            self.list_path = self.repo_root / self.list_path
        if not self.image_dir.is_absolute():
            self.image_dir = self.repo_root / self.image_dir
        if self.coordinate_dir is not None and not self.coordinate_dir.is_absolute():
            self.coordinate_dir = self.repo_root / self.coordinate_dir

        if not self.list_path.exists():
            raise FileNotFoundError(f"list file not found: {self.list_path}")
        if not self.image_dir.exists():
            raise FileNotFoundError(f"image directory not found: {self.image_dir}")

        pairs = []
        for line in self.list_path.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) >= 2:
                pairs.append((parts[0], parts[1]))
        if max_items is not None:
            pairs = pairs[: int(max_items)]
        if not pairs:
            raise ValueError(f"no image pairs found in {self.list_path}")
        self.pairs = pairs

        self.mean_bgr = np.reshape(np.array([118.93, 113.97, 102.60], dtype=np.float32), (1, 1, 3))
        self.std_bgr = np.reshape(np.array([69.85, 68.81, 72.45], dtype=np.float32), (1, 1, 3))

    def __len__(self) -> int:
        return len(self.pairs)

    def _rng(self, idx: int) -> np.random.Generator:
        if self.seed is None:
            return np.random.default_rng()
        return np.random.default_rng(int(self.seed) + int(idx) * 7919)

    def _read_bgr(self, relative_path: str) -> np.ndarray:
        path = self.image_dir / relative_path
        if cv2 is not None:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"failed to read image: {path}")
            if image.shape[0] != self.full_h or image.shape[1] != self.full_w:
                image = cv2.resize(image, (self.full_w, self.full_h), interpolation=cv2.INTER_LINEAR)
        else:
            if not path.exists():
                raise FileNotFoundError(f"image not found: {path}")
            pil_image = Image.open(path).convert("RGB")
            image = np.asarray(pil_image, dtype=np.float32)[..., ::-1]
            image = np.ascontiguousarray(image)
            if image.shape[0] != self.full_h or image.shape[1] != self.full_w:
                image = _resize_array(image, (self.full_h, self.full_w))
        return image.astype(np.float32)

    def _to_gray(self, image: np.ndarray) -> np.ndarray:
        if self.normalize:
            image = (image - self.mean_bgr) / self.std_bgr
            return image.mean(axis=2, keepdims=True).astype(np.float32)
        else:
            return ((0.114 * image[..., 0] + 0.587 * image[..., 1] + 0.299 * image[..., 2])[..., None] / 255.0).astype(
                np.float32
            )

    @staticmethod
    def _to_chw(image: np.ndarray) -> torch.Tensor:
        if image.ndim == 2:
            image = image[..., None]
        return torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float()

    def _sample_start(self, idx: int) -> Tuple[int, int]:
        if self.fixed_start is not None:
            return int(self.fixed_start[0]), int(self.fixed_start[1])

        if self.crop_w == self.full_w and self.crop_h == self.full_h:
            return 0, 0

        high_x = self.full_w - self.rho - self.crop_w
        high_y = self.full_h - self.rho - self.crop_h
        if high_x <= self.rho or high_y <= self.rho:
            raise ValueError(
                "crop_size must fit inside full_size with rho margin: "
                f"crop=({self.crop_h},{self.crop_w}) full=({self.full_h},{self.full_w}) rho={self.rho}"
            )

        if self.training:
            rng = self._rng(idx)
            x = int(rng.integers(self.rho, high_x))
            y = int(rng.integers(self.rho, high_y))
        else:
            x = 0
            y = 0
        return x, y

    def _random_perturb(self, start: Tuple[int, int], rng: np.random.Generator) -> np.ndarray:
        x, y = start
        src = np.array(
            [[0, 0], [self.crop_w - 1, 0], [0, self.crop_h - 1], [self.crop_w - 1, self.crop_h - 1]],
            dtype=np.float32,
        )
        src[:, 0] += x
        src[:, 1] += y
        if self.shift > 0:
            shift = rng.integers(-self.shift, self.shift, size=(4, 2)).astype(np.float32)
        else:
            shift = np.zeros((4, 2), dtype=np.float32)
        dst = src + shift
        return _find_homography(src, dst)

    def _coordinate_payload(self, img1_rel: str, img2_rel: str) -> dict[str, str]:
        if self.coordinate_dir is None:
            return {}
        img1_name = Path(img1_rel).name
        img2_name = Path(img2_rel).name
        pt_name = f"{img1_name}_{img2_name}.npy"
        pt_path = self.coordinate_dir / pt_name
        payload = {"pt_names": pt_name, "video_names": img1_rel.split("/")[0]}
        if pt_path.exists():
            pt_set = np.load(pt_path, allow_pickle=True)
            try:
                payload["pt_set"] = str(pt_set.item())
            except ValueError:
                payload["pt_set"] = str(pt_set)
        return payload

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        img1_rel, img2_rel = self.pairs[idx]
        rng = self._rng(idx)
        img1_bgr = self._read_bgr(img1_rel)
        img2_bgr = self._read_bgr(img2_rel)

        if self.training and self.horizontal_flip_aug and rng.random() <= 0.5:
            img1_bgr = np.ascontiguousarray(np.flip(img1_bgr, axis=1))
            img2_bgr = np.ascontiguousarray(np.flip(img2_bgr, axis=1))

        img1_gray = self._to_gray(img1_bgr)
        img2_gray = self._to_gray(img2_bgr)
        imgs_gray_full = torch.cat([self._to_chw(img1_gray), self._to_chw(img2_gray)], dim=0)
        imgs_full = torch.cat([self._to_chw(img1_bgr), self._to_chw(img2_bgr)], dim=0)

        if self.training:
            x, y = self._sample_start(idx)
            img1_patch_gray = img1_gray[y:y + self.crop_h, x:x + self.crop_w]
            img2_patch_gray = img2_gray[y:y + self.crop_h, x:x + self.crop_w]
            img1_patch_bgr = img1_bgr[y:y + self.crop_h, x:x + self.crop_w]
            img2_patch_bgr = img2_bgr[y:y + self.crop_h, x:x + self.crop_w]
            ori_size = torch.tensor([self.crop_w, self.crop_h], dtype=torch.float32)
            h_random = torch.from_numpy(self._random_perturb((x, y), rng)).float()
        else:
            x, y = 0, 0
            img1_patch_gray = _resize_array(img1_gray, (self.crop_h, self.crop_w))
            img2_patch_gray = _resize_array(img2_gray, (self.crop_h, self.crop_w))
            img1_patch_bgr = _resize_array(img1_bgr, (self.crop_h, self.crop_w))
            img2_patch_bgr = _resize_array(img2_bgr, (self.crop_h, self.crop_w))
            ori_size = torch.tensor([self.full_w, self.full_h], dtype=torch.float32)
            h_random = torch.eye(3, dtype=torch.float32)

        imgs_patch = torch.cat([self._to_chw(img1_patch_gray), self._to_chw(img2_patch_gray)], dim=0)
        imgs_patch_rgb = torch.cat([self._to_chw(img1_patch_bgr), self._to_chw(img2_patch_bgr)], dim=0)

        pts = torch.tensor(
            [[0, 0], [self.crop_w - 1, 0], [0, self.crop_h - 1], [self.crop_w - 1, self.crop_h - 1]],
            dtype=torch.float32,
        )
        start = torch.tensor([x, y], dtype=torch.float32).view(2, 1, 1)

        data_dict = {
            "imgs_patch_rgb": imgs_patch_rgb,
            "imgs_gray_full": imgs_gray_full,
            "imgs_full": imgs_full,
            "imgs_gray_patch": imgs_patch,
            "start": start,
            "pts": pts,
            "ori_size": ori_size,
            "H_random": h_random,
            "img_name": img1_rel,
            "pair_name": f"{img1_rel} {img2_rel}",
        }
        if not self.training:
            data_dict.update(self._coordinate_payload(img1_rel, img2_rel))
        return data_dict


def build_homo_flow_loader(
    repo_root: str | os.PathLike,
    list_path: str | os.PathLike,
    image_dir: str | os.PathLike,
    crop_size: Sequence[int] = (320, 512),
    full_size: Sequence[int] = (360, 640),
    rho: int = 16,
    shift: int = 8,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
    training: bool = True,
    horizontal_flip_aug: bool = True,
    coordinate_dir: Optional[str | os.PathLike] = None,
    max_items: Optional[int] = None,
    seed: Optional[int] = None,
) -> DataLoader:
    dataset = HomoFlowDataset(
        repo_root=repo_root,
        list_path=list_path,
        image_dir=image_dir,
        crop_size=crop_size,
        full_size=full_size,
        rho=rho,
        shift=shift,
        training=training,
        horizontal_flip_aug=horizontal_flip_aug,
        coordinate_dir=coordinate_dir,
        max_items=max_items,
        seed=seed,
    )
    loader_kwargs = {}
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 3
        loader_kwargs["worker_init_fn"] = worker_init_fn
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=training,
        pin_memory=torch.cuda.is_available(),
        **loader_kwargs,
    )
