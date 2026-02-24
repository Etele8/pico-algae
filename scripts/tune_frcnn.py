from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import itertools
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.utils.seed import seed_everything
from src.utils.io import ensure_dir
from src.utils.logging import append_jsonl

from src.data.dataset_index import load_index
from src.data.pico_dataset import PicoOgDetectionDataset
from src.data.collate import detection_collate
from src.data.transforms import IdentityTransform

from src.models.frcnn import build_frcnn_resnet50_fpn_coco
from src.train.optimizer import build_optimizer_two_groups
from src.train.scheduler import build_scheduler
from src.train.amp import get_scaler
from src.train.train_one_epoch import train_one_epoch
from src.train.evaluate import evaluate_count_metrics


# ---------- helpers ----------

def to_tuple_of_tuples(x: Any) -> Tuple[Tuple[Any, ...], ...]:
    # YAML gives lists; torchvision expects tuple-of-tuples
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

def sample_trials(space: Dict[str, List[Any]], trials: int, seed: int) -> List[Dict[str, Any]]:
    """
    Randomly sample combinations from a discrete search space.
    """
    rng = random.Random(seed)
    keys = list(space.keys())
    out = []
    for _ in range(trials):
        cfg = {k: rng.choice(space[k]) for k in keys}
        out.append(cfg)
    return out

def score_thresh_sweep(
    model,
    dl_val,
    device: torch.device,
    thresh_grid: List[float],
) -> Tuple[float, Dict[str, float]]:
    """
    Pick score_thresh minimizing count_mae on this fold.
    Returns: (best_thresh, best_metrics)
    """
    best_t = None
    best_mae = float("inf")
    best_metrics = None

    for t in thresh_grid:
        m = evaluate_count_metrics(model, dl_val, device, score_thresh=float(t))
        if m["count_mae"] < best_mae:
            best_mae = m["count_mae"]
            best_t = float(t)
            best_metrics = m

    return best_t, best_metrics if best_metrics is not None else {"count_mae": float("inf")}


@dataclass
class TrialResult:
    trial_id: int
    params: Dict[str, Any]
    fold_mae: List[float]
    fold_best_thresh: List[float]
    mean_mae: float
    std_mae: float


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_csv", required=True, type=str)
    ap.add_argument("--tune_yaml", required=True, type=str)
    ap.add_argument("--out_dir", required=True, type=str)
    ap.add_argument("--device", default="", type=str, help="'' auto, or 'cuda', or 'cpu'")
    args = ap.parse_args()

    tune_cfg = yaml.safe_load(Path(args.tune_yaml).read_text(encoding="utf-8"))

    seed = int(tune_cfg.get("seed", 42))
    seed_everything(seed)

    k = int(tune_cfg.get("k_folds", 5))
    trials = int(tune_cfg.get("trials", 10))
    epochs = int(tune_cfg.get("epochs", 10))
    amp = bool(tune_cfg.get("amp", True))
    num_workers = int(tune_cfg.get("num_workers", 8))
    thresh_grid = [float(x) for x in tune_cfg.get("val_score_thresh_grid", [0.5])]

    space = tune_cfg["space"]

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    ensure_dir(out_dir / "checkpoints_per_fold")  # optional
    results_jsonl = out_dir / "tuning_results.jsonl"
    results_csv = out_dir / "tuning_results.csv"

    # device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    df = load_index(Path(args.index_csv))
    n = len(df)
    folds = kfold_indices(n, k=k, seed=seed)

    # generate sampled trials
    trial_params_list = sample_trials(space, trials=trials, seed=seed)

    all_rows = []

    best_overall = None  # (mean_mae, trial_id, params)

    for trial_id, params in enumerate(trial_params_list, start=1):
        print("\n" + "=" * 80)
        print(f"TRIAL {trial_id}/{trials}")
        print(json.dumps(params, indent=2))

        # normalize anchors/aspect ratios if present
        anchor_sizes = params.get("anchor_sizes", None)
        aspect_ratios = params.get("aspect_ratios", None)

        if anchor_sizes is not None:
            anchor_sizes_tt = to_tuple_of_tuples(anchor_sizes)
        else:
            anchor_sizes_tt = None

        if aspect_ratios is not None:
            aspect_ratios_tt = to_tuple_of_tuples(aspect_ratios)
        else:
            aspect_ratios_tt = None

        fold_mae = []
        fold_best_thresh = []

        for fold_i, (tr_idx, va_idx) in enumerate(folds, start=1):
            print(f"\n--- Fold {fold_i}/{k} ---")

            df_tr = df.iloc[tr_idx].reset_index(drop=True)
            df_va = df.iloc[va_idx].reset_index(drop=True)

            ds_tr = PicoOgDetectionDataset(df_tr, transform=IdentityTransform(), keep_empty=True)
            ds_va = PicoOgDetectionDataset(df_va, transform=IdentityTransform(), keep_empty=True)

            batch_size = int(params.get("batch_size", 2))

            dl_tr = DataLoader(
                ds_tr,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                collate_fn=detection_collate,
            )
            dl_va = DataLoader(
                ds_va,
                batch_size=1,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                collate_fn=detection_collate,
            )

            model = build_frcnn_resnet50_fpn_coco(
                num_classes=5,
                trainable_backbone_layers=int(params.get("trainable_backbone_layers", 2)),
                anchor_sizes=anchor_sizes_tt,
                aspect_ratios=aspect_ratios_tt,
            )
            model.to(device)

            optimizer = build_optimizer_two_groups(
                model,
                lr_backbone=float(params.get("lr_backbone", 1e-5)),
                lr_heads=float(params.get("lr_heads", 1e-4)),
                weight_decay=float(params.get("weight_decay", 1e-4)),
            )
            scheduler = build_scheduler(optimizer, step_size=max(1, epochs // 3), gamma=0.5)
            scaler = get_scaler(use_amp=amp, device=device)

            # train for budgeted epochs
            for ep in range(1, epochs + 1):
                tr_metrics = train_one_epoch(model, optimizer, dl_tr, device, epoch=ep, scaler=scaler, log_every=0)
                scheduler.step()
                if ep == 1 or ep == epochs or ep % 5 == 0:
                    print(f"  ep {ep}/{epochs} loss={tr_metrics['loss']:.4f}")

            # pick best score threshold on this fold for count MAE
            best_t, best_m = score_thresh_sweep(model, dl_va, device, thresh_grid)
            fold_mae.append(float(best_m["count_mae"]))
            fold_best_thresh.append(float(best_t))

            print(f"  fold best thresh={best_t:.2f}  count_mae={best_m['count_mae']:.4f}")

            # optional: save fold checkpoint
            ckpt_path = out_dir / "checkpoints_per_fold" / f"trial{trial_id:03d}_fold{fold_i:02d}.pt"
            torch.save({"model": model.state_dict(), "params": params, "fold": fold_i}, ckpt_path)

            # free memory
            del model
            torch.cuda.empty_cache()

        mean_mae = float(np.mean(fold_mae))
        std_mae = float(np.std(fold_mae))
        row = {
            "trial_id": trial_id,
            "mean_count_mae": mean_mae,
            "std_count_mae": std_mae,
            "fold_mae": fold_mae,
            "fold_best_thresh": fold_best_thresh,
            **params,
        }

        append_jsonl(results_jsonl, row)
        all_rows.append(row)

        print(f"\nTRIAL {trial_id} summary: mean_mae={mean_mae:.4f} ± {std_mae:.4f}")

        if best_overall is None or mean_mae < best_overall[0]:
            best_overall = (mean_mae, trial_id, params)
            print("  NEW BEST ✅")

    # write CSV summary
    df_out = pd.DataFrame(all_rows)
    df_out.to_csv(results_csv, index=False)
    print("\nWrote:", results_jsonl)
    print("Wrote:", results_csv)
    if best_overall:
        print("\nBEST OVERALL:")
        print("  mean_mae:", best_overall[0])
        print("  trial_id:", best_overall[1])
        print("  params:", json.dumps(best_overall[2], indent=2))


if __name__ == "__main__":
    main()