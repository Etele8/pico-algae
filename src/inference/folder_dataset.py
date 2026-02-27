from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.inference.image_pairs import ImagePair


@dataclass(frozen=True)
class ResizeSpec:
    width: int
    height: int


class PairOgInferenceDataset(Dataset):
    def __init__(self, pairs: List[ImagePair], resize: Optional[ResizeSpec] = None):
        self.pairs = list(pairs)
        self.resize = resize

    def __len__(self) -> int:
        return len(self.pairs)

    def _load_og_rgb(self, idx: int) -> np.ndarray:
        p = self.pairs[idx].og_path
        img_bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Could not read image: {p}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        if self.resize is not None:
            img_rgb = cv2.resize(img_rgb, (self.resize.width, self.resize.height), interpolation=cv2.INTER_AREA)
        return img_rgb

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        img_rgb = self._load_og_rgb(idx)
        img_t = torch.from_numpy(img_rgb).permute(2, 0, 1).contiguous().float() / 255.0

        target = {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.int64),
            "image_id": torch.tensor([idx], dtype=torch.int64),
        }
        return img_t, target

    def image_for_vis(self, idx: int) -> np.ndarray:
        return self._load_og_rgb(idx)
