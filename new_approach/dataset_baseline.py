"""Dataloaders that emit the `data_batch` dict expected by HomoNet.forward.

HomoNet.forward reads:
    data_batch["imgs_gray_full"]  : (2, FULL_H, FULL_W)   img1, img2 stacked on dim 0
    data_batch["imgs_gray_patch"] : (2, CROP_H, CROP_W)   the cropped patch pair
    data_batch["start"]           : (2, 1, 1)             [start_x, start_y] crop offset
    data_batch["pts"]             : (4, 2)                patch corners (legacy, unused)

Image preprocessing follows Oneline-DLTv1/dataset.py exactly: resize to
640x360, per-channel mean/std normalize, then average to a single grayscale
channel. The crop is random within an `rho` margin for training and fixed at
the frame centre for evaluation.
"""
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .model_baseline import FULL_H, FULL_W, CROP_H, CROP_W

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))

MEAN_I = np.reshape(np.array([118.93, 113.97, 102.60]), (1, 1, 3))
STD_I = np.reshape(np.array([69.85, 68.81, 72.45]), (1, 1, 3))


def _load_gray(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    if img.shape[0] != FULL_H or img.shape[1] != FULL_W:
        img = cv2.resize(img, (FULL_W, FULL_H))
    img = (img - MEAN_I) / STD_I
    img = np.mean(img, axis=2, keepdims=True)          # -> (H, W, 1)
    return np.transpose(img, [2, 0, 1]).astype(np.float32)  # (1, H, W)


def _clean_token(tok):
    """Strip whitespace and any trailing 'M' marker, keep up to '.jpg'."""
    tok = tok.strip()
    if ".jpg" in tok:
        tok = tok[: tok.index(".jpg") + 4]
    return tok


def _pack(full, patch, start_x, start_y):
    pts = np.array(
        [[start_x, start_y],
         [start_x, start_y + CROP_H],
         [start_x + CROP_W, start_y + CROP_H],
         [start_x + CROP_W, start_y]],
        dtype=np.float32,
    )
    return {
        "imgs_gray_full": torch.from_numpy(full),
        "imgs_gray_patch": torch.from_numpy(patch),
        "start": torch.tensor([[[start_x]], [[start_y]]], dtype=torch.float32),  # (2,1,1)
        "pts": torch.from_numpy(pts),
    }


class TrainDataset(Dataset):
    def __init__(self, list_path, img_dir, rho=16, max_items=None):
        with open(list_path) as f:
            self.pairs = [ln for ln in f.read().splitlines() if ln.strip()]
        if max_items is not None:
            self.pairs = self.pairs[:max_items]
        self.img_dir = img_dir
        self.rho = rho

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        a, b = self.pairs[idx].split(" ")[:2]
        img1 = _load_gray(os.path.join(self.img_dir, _clean_token(a)))
        img2 = _load_gray(os.path.join(self.img_dir, _clean_token(b)))
        full = np.concatenate([img1, img2], axis=0)        # (2, H, W)

        x = np.random.randint(self.rho, FULL_W - self.rho - CROP_W + 1)
        y = np.random.randint(self.rho, FULL_H - self.rho - CROP_H + 1)
        patch = full[:, y:y + CROP_H, x:x + CROP_W]
        return _pack(full, patch.copy(), x, y)


class TestDataset(Dataset):
    """Centre-crop dataset that also returns the labelled-point .npy path."""

    def __init__(self, list_path, img_dir, coord_dir, max_items=None):
        with open(list_path) as f:
            self.pairs = [ln for ln in f.read().splitlines() if ln.strip()]
        if max_items is not None:
            self.pairs = self.pairs[:max_items]
        self.img_dir = img_dir
        self.coord_dir = coord_dir
        self.start_x = (FULL_W - CROP_W) // 2
        self.start_y = (FULL_H - CROP_H) // 2

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        a, b = self.pairs[idx].split(" ")[:2]
        a, b = _clean_token(a), _clean_token(b)
        img1 = _load_gray(os.path.join(self.img_dir, a))
        img2 = _load_gray(os.path.join(self.img_dir, b))
        full = np.concatenate([img1, img2], axis=0)
        patch = full[:, self.start_y:self.start_y + CROP_H,
                     self.start_x:self.start_x + CROP_W]

        npy_name = a.split("/")[-1] + "_" + b.split("/")[-1] + ".npy"
        sample = _pack(full, patch.copy(), self.start_x, self.start_y)
        sample["npy_path"] = os.path.join(self.coord_dir, npy_name)
        sample["video_id"] = a.split("/")[0]
        return sample


def move_batch(batch, device):
    """Move the tensor entries of a collated batch dict to `device`."""
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
