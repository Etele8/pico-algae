from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset_index import load_index
from src.data.pico_dataset import PicoOgDetectionDataset
from src.data.collate import detection_collate
from src.inference.predict import predict_on_loader
from src.inference.visualize import draw_xyxy, save_vis
from src.models.frcnn import build_frcnn_resnet50_fpn
from src.models.weights import load_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_csv", required=True, type=str)
    ap.add_argument("--ckpt", required=True, type=str)
    ap.add_argument("--out_dir", required=True, type=str)
    ap.add_argument("--score_thresh", type=float, default=0.5)
    ap.add_argument("--n", type=int, default=50)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = load_index(Path(args.index_csv)).head(args.n)
    ds = PicoOgDetectionDataset(df)
    dl = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=detection_collate)

    model = build_frcnn_resnet50_fpn(num_classes=5, pretrained_backbone=False)
    load_checkpoint(args.ckpt, model, map_location=device)
    model.to(device)

    results = predict_on_loader(model, dl, device, score_thresh=args.score_thresh)

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