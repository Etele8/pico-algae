from __future__ import annotations
import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

# python scripts/predict_frcnn.py --images_dir data/raw/images_og --ckpt runs/train_run/checkpoints/best_mae.pt --out_dir runs/predict_run --predict_yaml src/configs/predict_frcnn.yaml

from src.data.collate import detection_collate
from src.inference.image_pairs import discover_og_red_pairs
from src.inference.folder_dataset import PairOgInferenceDataset, ResizeSpec
from src.inference.predict import predict_on_loader
from src.inference.visualize import draw_xyxy, save_vis
from src.models.frcnn import build_frcnn_resnet50_fpn_coco
from src.models.weights import load_checkpoint


def as_bool(x) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True, type=str)
    ap.add_argument("--ckpt", required=True, type=str)
    ap.add_argument("--out_dir", required=True, type=str)
    ap.add_argument("--predict_yaml", required=True, type=str)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.predict_yaml).read_text(encoding="utf-8")) or {}
    recursive = as_bool(cfg.get("recursive", False))
    require_red_pair = as_bool(cfg.get("require_red_pair", True))

    score_thresh = float(cfg.get("score_thresh", 0.5))
    n = int(cfg.get("n", 0))
    trainable_backbone_layers = int(cfg.get("trainable_backbone_layers", 2))
    detections_per_image = int(cfg.get("detections_per_image", 300))
    box_nms_thresh = float(cfg.get("box_nms_thresh", 0.5))
    resize_to_target = as_bool(cfg.get("resize_to_target", True))
    target_width = int(cfg.get("target_width", 2048))
    target_height = int(cfg.get("target_height", 1500))

    classes_to_keep = cfg.get("classes_to_keep", cfg.get("classes_to_count", None))
    per_class_score_thresh_raw = cfg.get("per_class_score_thresh", None)
    per_class_score_thresh = None
    if isinstance(per_class_score_thresh_raw, dict):
        per_class_score_thresh = {int(k): float(v) for k, v in per_class_score_thresh_raw.items()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pairs = discover_og_red_pairs(Path(args.images_dir), recursive=recursive, require_red_pair=require_red_pair)
    if not pairs:
        if require_red_pair:
            raise RuntimeError(f"No *_og.png + *_red.png pairs found under: {args.images_dir}")
        raise RuntimeError(f"No *_og.png images found under: {args.images_dir}")

    if n > 0:
        pairs = pairs[:n]

    resize = ResizeSpec(width=target_width, height=target_height) if resize_to_target else None
    ds = PairOgInferenceDataset(pairs, resize=resize)
    dl = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=detection_collate)

    model = build_frcnn_resnet50_fpn_coco(
        num_classes=5,
        trainable_backbone_layers=trainable_backbone_layers,
        detections_per_image=detections_per_image,
        box_nms_thresh=box_nms_thresh,
    )
    load_checkpoint(args.ckpt, model, map_location=device)
    model.to(device)

    results = predict_on_loader(
        model,
        dl,
        device,
        score_thresh=score_thresh,
        per_class_score_thresh=per_class_score_thresh,
        classes_to_keep=classes_to_keep,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, res in enumerate(results):
        raw_count = int(res.get("raw_count", 0))
        kept_count = int(len(res["pred_boxes"]))
        hit_cap = raw_count >= detections_per_image
        cap_note = " [HIT_CAP]" if hit_cap else ""
        print(f"{pairs[i].stem}: raw={raw_count}, kept={kept_count}, cap={detections_per_image}{cap_note}")

        keep_mask = res.get("keep_mask", None)
        raw_boxes = res.get("raw_boxes", None)
        deleted_boxes = None
        if keep_mask is not None and raw_boxes is not None:
            deleted_boxes = raw_boxes[~keep_mask]

        img_rgb = ds.image_for_vis(i)
        vis = draw_xyxy(
            img_rgb,
            res["pred_boxes"],
            res["pred_labels"],
            res["pred_scores"],
            deleted_boxes=deleted_boxes,
        )
        save_vis(out_dir / f"{pairs[i].stem}_pred.png", vis)

    print("Wrote predictions to:", out_dir)


if __name__ == "__main__":
    main()
