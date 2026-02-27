from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.utils.boxes import clip_boxes_xyxy_, remove_degenerate


@dataclass
class PicoClasses:
    # raw ids: EUK=0, FE=1, FC=2, colony=3
    @staticmethod
    def raw_to_tv(raw_cls: int) -> int:
        # torchvision uses 0 as background, so shift by +1
        return int(raw_cls) + 1

    @staticmethod
    def tv_to_name(tv_label: int) -> str:
        return {1: "EUK", 2: "FE", 3: "FC", 4: "colony"}.get(int(tv_label), f"class_{tv_label}")


def _imread_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def _parse_label_txt(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    # line: class_id x_min y_min x_max y_max (absolute px)
    if not path.exists():
        raise FileNotFoundError(f"Label file not found: {path}")

    boxes = []
    labels = []
    for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        parts = ln.split()
        if len(parts) < 5:
            raise ValueError(f"Bad label line in {path}: {ln!r}")

        raw = int(float(parts[0]))
        x1, y1, x2, y2 = map(float, parts[1:5])

        xa, xb = (x1, x2) if x1 <= x2 else (x2, x1)
        ya, yb = (y1, y2) if y1 <= y2 else (y2, y1)

        if xb - xa < 1.0 or yb - ya < 1.0:
            continue

        boxes.append([xa, ya, xb, yb])
        labels.append(PicoClasses.raw_to_tv(raw))

    if len(boxes) == 0:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.int64)
    return np.asarray(boxes, np.float32), np.asarray(labels, np.int64)


class PicoOgRedDetectionDataset(Dataset):
    """
    6-channel dataset: concat [og_rgb, red_rgb] -> (6,H,W), float32 in [0,1].
    Assumes df has columns: og_webp, red_webp, label_path.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        transform: Optional[Callable[[torch.Tensor, Dict], Tuple[torch.Tensor, Dict]]] = None,
        keep_empty: bool = True,
    ):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.keep_empty = keep_empty

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        r = self.df.iloc[idx]
        og_path = Path(r["og_webp"])
        red_path = Path(r["red_webp"])
        lbl_path = Path(r["label_path"])

        og_bgr = _imread_bgr(og_path)
        red_bgr = _imread_bgr(red_path)

        og_rgb = cv2.cvtColor(og_bgr, cv2.COLOR_BGR2RGB)
        red_rgb = cv2.cvtColor(red_bgr, cv2.COLOR_BGR2RGB)

        h, w = og_rgb.shape[:2]
        if red_rgb.shape[:2] != (h, w):
            raise ValueError(
                f"Size mismatch og vs red for idx={idx}: og={og_rgb.shape[:2]} red={red_rgb.shape[:2]} "
                f"({og_path} vs {red_path})"
            )

        boxes_np, labels_np = _parse_label_txt(lbl_path)

        og_t = torch.from_numpy(og_rgb).permute(2, 0, 1).contiguous().float() / 255.0
        red_t = torch.from_numpy(red_rgb).permute(2, 0, 1).contiguous().float() / 255.0
        img_t = torch.cat([og_t, red_t], dim=0)  # (6,H,W)

        boxes = torch.from_numpy(boxes_np).float()
        labels = torch.from_numpy(labels_np).long()

        if boxes.numel() > 0:
            clip_boxes_xyxy_(boxes, w=w, h=h)
            boxes, labels = remove_degenerate(boxes, labels)

        if (not self.keep_empty) and (boxes.shape[0] == 0):
            return self.__getitem__((idx + 1) % len(self.df))

        area = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": area,
            "iscrowd": iscrowd,
        }

        if self.transform is not None:
            img_t, target = self.transform(img_t, target)

        return img_t, target