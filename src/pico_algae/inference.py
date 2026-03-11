from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from src.inference.visualize import draw_xyxy, save_vis
from src.models.frcnn import build_frcnn_resnet50_fpn_coco
from src.models.frcnn_6ch import build_frcnn_resnet50_fpn_coco_6ch


def load_rgb(path: Path, *, resize: tuple[int, int] | None = None) -> np.ndarray:
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    if resize is not None:
        img_rgb = cv2.resize(img_rgb, resize, interpolation=cv2.INTER_AREA)
    return img_rgb


def find_red_pair_path(image_path: Path) -> Path | None:
    stem = image_path.stem
    if stem.endswith("_og"):
        red_candidate = image_path.with_name(f"{stem[:-3]}_red{image_path.suffix}")
        if red_candidate.exists():
            return red_candidate
    return None


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Run pico-algae detection on one image.")
    parser.add_argument("--image", required=True, type=Path, help="Path to input image (typically *_og.png).")
    parser.add_argument(
        "--red-image",
        type=Path,
        default=None,
        help="Optional paired red-channel image. If omitted, auto-detected from *_og -> *_red or falls back to --image.",
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/train_run01/checkpoints/best_mae.pt"))
    parser.add_argument("--output", type=Path, default=Path("examples/detection_results"))
    parser.add_argument("--score-thresh", type=float, default=0.22)
    parser.add_argument("--target-width", type=int, default=2048)
    parser.add_argument("--target-height", type=int, default=1500)
    args = parser.parse_args()

    if not args.image.exists():
        raise FileNotFoundError(f"Input image not found: {args.image}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    resize = (args.target_width, args.target_height)
    og_rgb = load_rgb(args.image, resize=resize)
    red_path = args.red_image if args.red_image is not None else find_red_pair_path(args.image)
    if red_path is not None and red_path.exists():
        red_rgb = load_rgb(red_path, resize=resize)
    else:
        red_rgb = og_rgb.copy()

    og_t = torch.from_numpy(og_rgb).permute(2, 0, 1).contiguous().float() / 255.0
    red_t = torch.from_numpy(red_rgb).permute(2, 0, 1).contiguous().float() / 255.0
    image_6ch = torch.cat([og_t, red_t], dim=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    conv1 = ckpt["model"]["backbone.body.conv1.weight"]
    in_ch = int(conv1.shape[1])

    if in_ch == 6:
        model = build_frcnn_resnet50_fpn_coco_6ch(num_classes=5, trainable_backbone_layers=2)
    elif in_ch == 3:
        model = build_frcnn_resnet50_fpn_coco(num_classes=5, trainable_backbone_layers=2)
    else:
        raise RuntimeError(f"Unsupported checkpoint input channels: {in_ch}")

    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    model_input = image_6ch if in_ch == 6 else image_6ch[:3]
    output = model([model_input.to(device)])[0]
    scores = output["scores"].detach().cpu()
    keep = scores >= float(args.score_thresh)
    boxes = output["boxes"].detach().cpu()[keep].numpy()
    labels = output["labels"].detach().cpu()[keep].numpy()
    kept_scores = scores[keep].numpy()

    vis = draw_xyxy(og_rgb, boxes, labels, kept_scores)
    args.output.mkdir(parents=True, exist_ok=True)
    out_path = args.output / f"{args.image.stem}_detection.png"
    save_vis(out_path, vis)

    print(f"Input: {args.image}")
    if red_path is not None and red_path.exists():
        print(f"Red pair: {red_path}")
    else:
        print("Red pair: not found, reused --image")
    print(f"Detections kept: {int(keep.sum().item())}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

