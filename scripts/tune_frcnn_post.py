from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.utils.seed import seed_everything
from src.utils.io import ensure_dir
from src.utils.logging import append_jsonl

from src.data.dataset_index_6ch import load_index_6ch
from src.data.pico_dataset_6ch import PicoOgRedDetectionDataset
from src.data.collate import detection_collate
from src.data.transforms import IdentityTransform

from src.models.frcnn_6ch import build_frcnn_resnet50_fpn_coco_6ch
from src.train.evaluate import evaluate_count_metrics

# python scripts/tune_frcnn_post.py --index_csv data/processed/dataset_2048x1500_webp/index.csv --post_yaml src/configs/tune_frcnn_post.yaml --checkpoint runs/tuning/train/best_train_model.pt --out_dir runs/tuning/post --device cuda

def to_tuple_of_tuples(x: Any):
    return tuple(tuple(v) for v in x)

def kfold_indices(n: int, k: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, k)
    out = []
    for i in range(k):
        val_idx = folds[i]
        tr_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        out.append((tr_idx, val_idx))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_csv", required=True, type=str)
    ap.add_argument("--post_yaml", required=True, type=str)
    ap.add_argument("--checkpoint", required=True, type=str, help="Path to best_train_model.pt")
    ap.add_argument("--out_dir", required=True, type=str)
    ap.add_argument("--device", default="", type=str)
    args = ap.parse_args()

    post_cfg = yaml.safe_load(Path(args.post_yaml).read_text(encoding="utf-8"))

    seed = int(post_cfg.get("seed", 42))
    seed_everything(seed)

    k = int(post_cfg.get("k_folds", 5))
    num_workers = int(post_cfg.get("num_workers", 8))
    classes_to_count = post_cfg.get("classes_to_count", [1, 2, 3])

    score_grid = [float(x) for x in post_cfg.get("score_thresh_grid", [0.2])]
    nms_grid = [float(x) for x in post_cfg.get("box_nms_thresh_grid", [0.5])]
    det_grid = [int(x) for x in post_cfg.get("detections_per_image_grid", [400])]

    # Optional per-class score thresholds grid (list of dicts)
    per_class_thresh_grid = post_cfg.get("per_class_score_thresh_grid", None)
    if per_class_thresh_grid is not None:
        # ensure keys are ints, vals floats
        cleaned = []
        for d in per_class_thresh_grid:
            cleaned.append({int(k): float(v) for k, v in d.items()})
        per_class_thresh_grid = cleaned

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    results_jsonl = out_dir / "tuning_post_results.jsonl"
    results_csv = out_dir / "tuning_post_results.csv"
    best_summary_path = out_dir / "best_post_summary.json"

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Load dataset and folds
    df = load_index_6ch(Path(args.index_csv))
    n = len(df)
    folds = kfold_indices(n, k=k, seed=seed)

    # Load checkpoint (trained weights + base params)
    ckpt = torch.load(Path(args.checkpoint), map_location="cpu")
    base_params: Dict[str, Any] = ckpt.get("params", {})
    print("Loaded checkpoint params:\n", json.dumps(base_params, indent=2))

    # Prepare anchor/aspect from params (if present)
    anchor_sizes = base_params.get("anchor_sizes", None)
    aspect_ratios = base_params.get("aspect_ratios", None)
    anchor_sizes_tt = to_tuple_of_tuples(anchor_sizes) if anchor_sizes is not None else None
    aspect_ratios_tt = to_tuple_of_tuples(aspect_ratios) if aspect_ratios is not None else None

    # Build val loaders once per fold (no training)
    fold_val_loaders = []
    for fold_i, (_tr_idx, va_idx) in enumerate(folds, start=1):
        df_va = df.iloc[va_idx].reset_index(drop=True)
        ds_va = PicoOgRedDetectionDataset(df_va, transform=IdentityTransform(), keep_empty=True)
        dl_va = DataLoader(
            ds_va,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=detection_collate,
        )
        fold_val_loaders.append(dl_va)

    all_rows = []
    best = None  # (mean_mae, row)

    # Build sweep list
    if per_class_thresh_grid is None:
        sweep_items = [(s, nms, det, None) for s in score_grid for nms in nms_grid for det in det_grid]
    else:
        sweep_items = [(s, nms, det, pc) for s in score_grid for nms in nms_grid for det in det_grid for pc in per_class_thresh_grid]

    for sweep_id, (score_t, nms_t, dets, per_cls_t) in enumerate(sweep_items, start=1):
        print("\n" + "-" * 80)
        print(f"POST SWEEP {sweep_id}/{len(sweep_items)}  score={score_t} nms={nms_t} det={dets} per_cls={per_cls_t}")

        fold_mae = []
        fold_bias = []

        # Rebuild model with current post params (NMS + dets) and load SAME weights
        model = build_frcnn_resnet50_fpn_coco_6ch(
            num_classes=5,
            trainable_backbone_layers=int(base_params.get("trainable_backbone_layers", 2)),
            anchor_sizes=anchor_sizes_tt,
            aspect_ratios=aspect_ratios_tt,
            detections_per_image=int(dets),
            box_nms_thresh=float(nms_t),
        )
        model.load_state_dict(ckpt["model"], strict=True)
        model.to(device)
        model.eval()

        for fold_i, dl_va in enumerate(fold_val_loaders, start=1):
            m = evaluate_count_metrics(
                model,
                dl_va,
                device,
                score_thresh=float(score_t),
                classes_to_count=classes_to_count,
                per_class_score_thresh=per_cls_t,
            )
            fold_mae.append(float(m["count_mae"]))
            fold_bias.append(float(m["count_bias"]))

        mean_mae = float(np.mean(fold_mae))
        std_mae = float(np.std(fold_mae))
        mean_bias = float(np.mean(fold_bias))

        row = {
            "sweep_id": sweep_id,
            "mean_count_mae": mean_mae,
            "std_count_mae": std_mae,
            "mean_count_bias": mean_bias,
            "fold_mae": fold_mae,
            "fold_bias": fold_bias,
            "score_thresh": float(score_t),
            "box_nms_thresh": float(nms_t),
            "detections_per_image": int(dets),
            "classes_to_count": classes_to_count,
            "per_class_score_thresh": per_cls_t,
            "checkpoint": str(args.checkpoint),
        }
        append_jsonl(results_jsonl, row)
        all_rows.append(row)

        print(f"POST summary: mean_mae={mean_mae:.4f} ± {std_mae:.4f} | mean_bias={mean_bias:.4f}")

        if best is None or mean_mae < best[0]:
            best = (mean_mae, row)
            best_summary_path.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
            print("  NEW BEST ✅")

        del model
        torch.cuda.empty_cache()

    df_out = pd.DataFrame(all_rows)
    df_out.to_csv(results_csv, index=False)
    print("\nWrote:", results_jsonl)
    print("Wrote:", results_csv)
    if best_summary_path.exists():
        print("Wrote:", best_summary_path)


if __name__ == "__main__":
    main()