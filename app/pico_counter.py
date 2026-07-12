"""
Offline pico-algae detection + counting core for the desktop UI.

Loads a 3-channel Faster R-CNN checkpoint and runs single-image inference.
Everything is self-contained: the ResNet50-FPN backbone is built WITHOUT
downloading COCO weights (weights=None), then the trained checkpoint is loaded
on top. This means the app runs fully offline on a colleague's PC.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import AnchorGenerator

# Foreground class ids -> human names (background is 0)
CLASS_NAMES: Dict[int, str] = {1: "EUK", 2: "FE", 3: "FC", 4: "colony"}

# BGR-ish colors reused from src/inference/visualize.py, expressed as RGB here.
CLASS_COLORS: Dict[int, Tuple[int, int, int]] = {
    1: (0, 0, 255),      # EUK  - blue
    2: (255, 200, 0),    # FE   - amber
    3: (255, 0, 0),      # FC   - red
    4: (0, 200, 0),      # colony - green
}

# Model training/inference resolution (dataset was 2048x1500).
TARGET_W = 2048
TARGET_H = 1500


def default_device() -> torch.device:
    """Pick the compute device.

    Uses the GPU automatically when a CUDA-enabled PyTorch build and an NVIDIA
    GPU are present, otherwise CPU. Set the PICO_DEVICE env var to "cpu" or
    "cuda" to force a specific device.
    """
    override = os.environ.get("PICO_DEVICE", "").strip().lower()
    if override == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("[pico] PICO_DEVICE=cuda requested but no CUDA GPU is available; using CPU.")
        return torch.device("cpu")
    if override == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Prediction:
    counts: Dict[str, int]                 # per-class kept counts (only counted classes)
    total: int                             # sum of counted classes
    image_rgb: np.ndarray                  # resized plain RGB the boxes refer to
    width: int                             # image_rgb width  (== TARGET_W)
    height: int                            # image_rgb height (== TARGET_H)
    boxes: List[List[float]]               # [x1,y1,x2,y2] per detection, image_rgb pixel coords
    labels: List[int]                      # class id per detection (1..4)
    scores: List[float]                    # confidence per detection


@dataclass
class PicoCounter:
    ckpt_path: Path
    device: torch.device = field(default_factory=default_device)

    @property
    def device_label(self) -> str:
        if self.device.type == "cuda":
            try:
                return f"GPU · {torch.cuda.get_device_name(self.device)}"
            except Exception:  # noqa: BLE001
                return "GPU"
        return "CPU"

    def __post_init__(self) -> None:
        self.ckpt_path = Path(self.ckpt_path)
        ck = torch.load(self.ckpt_path, map_location="cpu")
        state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck

        in_ch = int(state["backbone.body.conv1.weight"].shape[1])
        if in_ch != 3:
            raise ValueError(
                f"This UI supports 3-channel checkpoints; {self.ckpt_path.name} "
                f"expects {in_ch} input channels."
            )
        n_classes = int(state["roi_heads.box_predictor.cls_score.weight"].shape[0])

        params = ck.get("params", {}) if isinstance(ck, dict) else {}
        anchor_sizes = _tuples(params.get("anchor_sizes", [[16], [32], [64], [128], [256]]))
        aspect_ratios = _tuples(
            params.get("aspect_ratios", [[0.5, 1.0, 2.0]] * len(anchor_sizes))
        )
        det_per_img = int(params.get("detections_per_image", 300))
        nms = float(params.get("box_nms_thresh", 0.5))

        # Inference settings the checkpoint was validated with.
        self.score_thresh = float(ck.get("val_score_thresh", 0.5)) if isinstance(ck, dict) else 0.5
        self.classes_to_count = [
            int(c) for c in (ck.get("classes_to_count", [1, 2, 3]) if isinstance(ck, dict) else [1, 2, 3])
        ]

        model = _build_frcnn(n_classes, anchor_sizes, aspect_ratios, det_per_img, nms)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            # Should be empty for a matching checkpoint; surface if not.
            print(f"[pico] load_state_dict missing={missing} unexpected={unexpected}")
        model.to(self.device).eval()
        self.model = model

    @torch.no_grad()
    def predict(self, image_rgb: np.ndarray, score_thresh: float | None = None) -> Prediction:
        thr = self.score_thresh if score_thresh is None else float(score_thresh)
        resized = cv2.resize(image_rgb, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
        t = torch.from_numpy(resized).permute(2, 0, 1).contiguous().float() / 255.0
        out = self.model([t.to(self.device)])[0]

        scores = out["scores"].cpu().numpy()
        labels = out["labels"].cpu().numpy()
        boxes = out["boxes"].cpu().numpy()

        keep = scores >= thr
        boxes, labels, scores = boxes[keep], labels[keep], scores[keep]

        counts = {
            CLASS_NAMES[c]: int((labels == c).sum())
            for c in self.classes_to_count
            if c in CLASS_NAMES
        }
        total = int(sum(counts.values()))

        return Prediction(
            counts=counts,
            total=total,
            image_rgb=resized,
            width=TARGET_W,
            height=TARGET_H,
            boxes=[[round(float(v), 1) for v in b] for b in boxes.tolist()],
            labels=[int(x) for x in labels.tolist()],
            scores=[round(float(s), 4) for s in scores.tolist()],
        )


def _tuples(x) -> Tuple[Tuple, ...]:
    return tuple(tuple(row) for row in x)


def _build_frcnn(num_classes, anchor_sizes, aspect_ratios, det_per_img, nms) -> FasterRCNN:
    anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)
    backbone = resnet_fpn_backbone("resnet50", weights=None, trainable_layers=0)
    return FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_detections_per_img=det_per_img,
        box_nms_thresh=nms,
    )


def load_image_rgb(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode image (unsupported or corrupt file).")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
