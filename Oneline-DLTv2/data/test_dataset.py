"""Test dataset matching v1's protocol: same patch crop at fixed (40, 23),
and exposes the manual 6-point correspondences (from Data/Coordinate/*.npy)
for L2 reprojection evaluation."""

import os
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .pairs import _load_normalized_gray


SCENE_TAGS = {
    "RE": ["0000011", "0000016", "00000147", "00000155", "00000158",
           "00000107", "00000239", "0000030"],
    "LT": ["0000038", "0000044", "0000046", "0000047", "00000238",
           "00000177", "00000188", "00000181"],
    "LL": ["0000085", "00000100", "0000091", "0000092", "00000216", "00000226"],
    "SF": ["00000244", "00000251", "0000026", "0000034", "00000115"],
    "LF": ["00000104", "0000031", "0000035", "00000129", "00000141", "00000200"],
}


def scene_category(video_name: str) -> str:
    for tag, vids in SCENE_TAGS.items():
        if video_name in vids:
            return tag
    return "??"


class TestDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        patch_h: int = 315,
        patch_w: int = 560,
        img_h: int = 360,
        img_w: int = 640,
        patch_x: int = 40,
        patch_y: int = 23,
    ):
        super().__init__()
        self.work_dir = os.path.join(data_root, "Data")
        self.test_root = os.path.join(self.work_dir, "Test")
        self.npy_root = os.path.join(self.work_dir, "Coordinate")
        with open(os.path.join(self.work_dir, "Test_List.txt"), "r") as f:
            self.lines = [ln.strip() for ln in f if ln.strip()]

        self.patch_h, self.patch_w = patch_h, patch_w
        self.img_h, self.img_w = img_h, img_w
        self.patch_x, self.patch_y = patch_x, patch_y

    def __len__(self) -> int:
        return len(self.lines)

    def _resolve(self, token: str) -> Tuple[str, str]:
        """v1's Test_List has trailing 'M' / newline markers; strip them."""
        token = token.rstrip()
        if token.endswith("M"):
            token = token[:-1]
        return token

    def __getitem__(self, idx: int):
        ln = self.lines[idx]
        parts = ln.split(" ")
        a_token, b_token = self._resolve(parts[0]), self._resolve(parts[1])
        video_name = ln.split("/")[0]

        a_path = os.path.join(self.test_root, a_token)
        b_path = os.path.join(self.test_root, b_token)
        img_a = _load_normalized_gray(a_path, self.img_w, self.img_h)
        img_b = _load_normalized_gray(b_path, self.img_w, self.img_h)

        x, y = self.patch_x, self.patch_y
        I_a_patch = img_a[:, y:y + self.patch_h, x:x + self.patch_w]
        I_b_patch = img_b[:, y:y + self.patch_h, x:x + self.patch_w]

        # Manual point correspondences (in full-image coordinates).
        # Note: v1 wrote `parts[1].split("/")[1][:-1]` because v1 did NOT strip
        # the trailing newline from the line. v2 strips it in __init__, so
        # do NOT chop the last char or "00000100_10011.jpg" becomes ".jp".
        npy_name = (parts[0].split("/")[1] + "_" +
                    parts[1].split("/")[1] + ".npy")
        npy_path = os.path.join(self.npy_root, npy_name)
        point_dict = np.load(npy_path, allow_pickle=True).item()
        pts = np.array(point_dict["matche_pts"], dtype=np.float32)   # (K, 2, 2)

        return {
            "I_a_patch": torch.from_numpy(I_a_patch.copy()).float(),
            "I_b_patch": torch.from_numpy(I_b_patch.copy()).float(),
            "points": torch.from_numpy(pts),                          # (K, 2, 2)
            "crop_xy": torch.tensor([x, y], dtype=torch.float32),
            "video_name": video_name,
            "scene": scene_category(video_name),
        }
