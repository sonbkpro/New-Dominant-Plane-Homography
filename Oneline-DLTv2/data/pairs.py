"""Natural-pair training dataset matching v1's TrainDataset format.

Reads Data/Train_List.txt with lines of the form:
    <video_id>/<frame_a>.jpg <video_id>/<frame_b>.jpg

Returns aligned grayscale patch pairs (I_a_patch, I_b_patch) of size
(patch_h, patch_w), cropped at the same random (x, y) within the image."""

import os
from typing import Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


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


class TrainPairDataset(Dataset):
    def __init__(
        self,
        data_list_path: str,
        train_root: str,
        patch_h: int = 315,
        patch_w: int = 560,
        img_h: int = 360,
        img_w: int = 640,
        rho: int = 16,
    ):
        super().__init__()
        with open(data_list_path, "r") as f:
            self.lines = [ln.strip() for ln in f if ln.strip()]
        self.train_root = train_root
        self.patch_h, self.patch_w = patch_h, patch_w
        self.img_h, self.img_w = img_h, img_w
        self.rho = rho

    def __len__(self) -> int:
        return len(self.lines)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        names = self.lines[idx].split()
        img_a = _load_normalized_gray(os.path.join(self.train_root, names[0]),
                                      self.img_w, self.img_h)
        img_b = _load_normalized_gray(os.path.join(self.train_root, names[1]),
                                      self.img_w, self.img_h)

        x = np.random.randint(self.rho, self.img_w - self.rho - self.patch_w)
        y = np.random.randint(self.rho, self.img_h - self.rho - self.patch_h)
        I_a_patch = img_a[:, y:y + self.patch_h, x:x + self.patch_w]
        I_b_patch = img_b[:, y:y + self.patch_h, x:x + self.patch_w]

        I_a_patch = torch.from_numpy(I_a_patch.copy()).float()
        I_b_patch = torch.from_numpy(I_b_patch.copy()).float()
        crop_xy = torch.tensor([x, y], dtype=torch.float32)
        return I_a_patch, I_b_patch, crop_xy


def make_patch_loader(
    data_list_path: str,
    train_root: str,
    batch_size: int,
    num_workers: int = 4,
    shuffle: bool = True,
    drop_last: bool = True,
    **dataset_kwargs,
) -> DataLoader:
    ds = TrainPairDataset(data_list_path, train_root, **dataset_kwargs)
    return DataLoader(
        ds, batch_size=batch_size, num_workers=num_workers,
        shuffle=shuffle, drop_last=drop_last, pin_memory=True,
    )
