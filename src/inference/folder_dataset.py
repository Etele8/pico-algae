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

    def _load_rgb(self, path, *, kind: str) -> np.ndarray:
        img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Could not read {kind} image: {path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        if self.resize is not None:
            img_rgb = cv2.resize(img_rgb, (self.resize.width, self.resize.height), interpolation=cv2.INTER_AREA)
        return img_rgb

    def _load_pair_rgb(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        pair = self.pairs[idx]
        og_rgb = self._load_rgb(pair.og_path, kind='og')
        if pair.red_path is None:
            raise FileNotFoundError(f"Missing red pair for: {pair.og_path}")
        red_rgb = self._load_rgb(pair.red_path, kind='red')
        if red_rgb.shape[:2] != og_rgb.shape[:2]:
            raise ValueError(
                f"Size mismatch og vs red for idx={idx}: og={og_rgb.shape[:2]} red={red_rgb.shape[:2]} "
                f"({pair.og_path} vs {pair.red_path})"
            )
        return og_rgb, red_rgb

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        og_rgb, red_rgb = self._load_pair_rgb(idx)
        og_t = torch.from_numpy(og_rgb).permute(2, 0, 1).contiguous().float() / 255.0
        red_t = torch.from_numpy(red_rgb).permute(2, 0, 1).contiguous().float() / 255.0
        img_t = torch.cat([og_t, red_t], dim=0)

        target = {
            'boxes': torch.zeros((0, 4), dtype=torch.float32),
            'labels': torch.zeros((0,), dtype=torch.int64),
            'image_id': torch.tensor([idx], dtype=torch.int64),
        }
        return img_t, target

    def image_for_vis(self, idx: int) -> np.ndarray:
        og_rgb, _ = self._load_pair_rgb(idx)
        return og_rgb
