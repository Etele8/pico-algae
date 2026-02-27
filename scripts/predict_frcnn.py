from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset_index import load_index
from src.data.pico_dataset import PicoOgDetectionDataset
from src.data.collate import detection_collate
from src.inference.predict import predict_on_loader
from src.inference.visualize import draw_xyxy, save_vis
from src.models.frcnn import build_frcnn_resnet50_fpn_coco
from src.models.weights import load_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_csv", required=True, type=str)
    ap.add_argument("--ckpt", required=True, type=str)
    ap.add_argument("--out_dir", required=True, type=str)
    ap.add_argument("--predict_yaml", required=True, type=str)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.predict_yaml).read_text(encoding="utf-8")) or {}
    score_thresh = float(cfg.get("score_thresh", 0.5))
    n = int(cfg.get("n", 50))
    trainable_backbone_layers = int(cfg.get("trainable_backbone_layers", 2))
    detections_per_image = int(cfg.get("detections_per_image", 300))
    box_nms_thresh = float(cfg.get("box_nms_thresh", 0.5))

    classes_to_keep = cfg.get("classes_to_keep", cfg.get("classes_to_count", None))
    per_class_score_thresh_raw = cfg.get("per_class_score_thresh", None)
    per_class_score_thresh = None
    if isinstance(per_class_score_thresh_raw, dict):
        per_class_score_thresh = {int(k): float(v) for k, v in per_class_score_thresh_raw.items()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = load_index(Path(args.index_csv)).head(n)
    ds = PicoOgDetectionDataset(df)
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
        img_t, tgt = ds[i]
        img_rgb = (img_t.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)

        vis = draw_xyxy(img_rgb, res["pred_boxes"], res["pred_labels"], res["pred_scores"])
        save_vis(out_dir / f"{df.iloc[i]['stem']}_pred.png", vis)

    print("Wrote predictions to:", out_dir)


if __name__ == "__main__":
    main()
