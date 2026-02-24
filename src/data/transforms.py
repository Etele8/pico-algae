from __future__ import annotations
from typing import Dict, Tuple, Optional

import torch


class IdentityTransform:
    def __call__(self, image: torch.Tensor, target: Dict) -> Tuple[torch.Tensor, Dict]:
        return image, target


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = float(p)

    def __call__(self, image: torch.Tensor, target: Dict):
        if torch.rand(()) > self.p:
            return image, target

        # image: (C,H,W)
        _, h, w = image.shape
        image = torch.flip(image, dims=[2])  # flip W

        boxes = target["boxes"].clone()
        x1 = boxes[:, 0].clone()
        x2 = boxes[:, 2].clone()
        boxes[:, 0] = (w - 1) - x2
        boxes[:, 2] = (w - 1) - x1
        target["boxes"] = boxes
        return image, target