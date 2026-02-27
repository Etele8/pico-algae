from __future__ import annotations
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.data.pico_dataset import PicoClasses
from src.utils.io import ensure_dir


def draw_xyxy(img_rgb_u8: np.ndarray, boxes: np.ndarray, labels: np.ndarray, scores: Optional[np.ndarray] = None):
    out = img_rgb_u8.copy()
    class_colors = {
        1: (0, 0, 255),    # blue
        2: (255, 255, 0),  # yellow
        3: (255, 0, 0),    # red
        4: (0, 255, 0),    # green
    }
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        lab = int(labels[i])
        color = class_colors.get(lab, (255, 255, 255))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        name = PicoClasses.tv_to_name(lab)
        if scores is not None:
            name = f"{name} {float(scores[i]):.2f}"
        cv2.putText(out, name, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(out, name, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    return out


def save_vis(path: Path, img_rgb_u8: np.ndarray):
    ensure_dir(path.parent)
    bgr = cv2.cvtColor(img_rgb_u8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)
