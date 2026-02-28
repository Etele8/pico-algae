from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import json
import random
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
from src.train.optimizer import build_optimizer_two_groups
from src.train.scheduler import build_scheduler
from src.train.amp import get_scaler
from src.train.train_one_epoch import train_one_epoch
from src.train.evaluate import evaluate_count_metrics

# python scripts/tune_frcnn_train.py --index_csv data/processed/dataset_2048x1500_webp/index.csv --tune_yaml src/configs/tune_frcnn_train.yaml --out_dir runs/tuning

def to_tuple_of_tuples(x: Any):
    return tuple(tuple(v) for v in x)


def as_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(x)

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
    rng = random.Random(seed)
    keys = list(space.keys())
    out = []
    for _ in range(trials):
        cfg = {k: rng.choice(space[k]) for k in keys}
        out.append(cfg)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_csv", required=True, type=str)
    ap.add_argument("--tune_yaml", required=True, type=str)
    ap.add_argument("--out_dir", required=True, type=str)
    ap.add_argument("--device", default='cuda', type=str, help="'' auto, or 'cuda', or 'cpu'")
    args = ap.parse_args()

    tune_cfg = yaml.safe_load(Path(args.tune_yaml).read_text(encoding="utf-8"))

    seed = int(tune_cfg.get("seed", 42))
    seed_everything(seed)

    k = int(tune_cfg.get("k_folds", 5))
    trials = int(tune_cfg.get("trials", 10))
    epochs = int(tune_cfg.get("epochs", 10))
    amp = bool(tune_cfg.get("amp", True))
    num_workers = int(tune_cfg.get("num_workers", 8))
    prefetch_factor = int(tune_cfg.get("prefetch_factor", 2))
    use_torch_compile = as_bool(tune_cfg.get("torch_compile", False))
    torch_compile_mode = str(tune_cfg.get("torch_compile_mode", "default"))

    # This is a fixed validation score threshold during TRAIN tuning
    val_score_thresh = float(tune_cfg.get("val_score_thresh", 0.5))

    # Which classes count toward MAE objective
    classes_to_count = tune_cfg.get("classes_to_count", [1, 2, 3])

    space = tune_cfg["space"]

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    results_jsonl = out_dir / "tuning_train_results.jsonl"
    results_csv = out_dir / "tuning_train_results.csv"
    best_model_path = out_dir / "best_train_model.pt"
    best_summary_path = out_dir / "best_train_model_summary.json"

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    pin_memory = device.type == "cuda"
    persistent_workers = num_workers > 0

    df = load_index_6ch(Path(args.index_csv))
    n = len(df)
    folds = kfold_indices(n, k=k, seed=seed)

    trial_params_list = sample_trials(space, trials=trials, seed=seed)
    all_rows = []

    best_overall = None  # (mean_mae, trial_id, params, best_fold_idx, best_fold_state)

    for trial_id, params in enumerate(trial_params_list, start=1):
        print("\n" + "=" * 80)
        print(f"TRAIN TRIAL {trial_id}/{trials}")
        print(json.dumps(params, indent=2))

        anchor_sizes = params.get("anchor_sizes", None)
        aspect_ratios = params.get("aspect_ratios", None)

        anchor_sizes_tt = to_tuple_of_tuples(anchor_sizes) if anchor_sizes is not None else None
        aspect_ratios_tt = to_tuple_of_tuples(aspect_ratios) if aspect_ratios is not None else None

        fold_mae = []
        best_fold_idx = None
        best_fold_mae = None
        best_fold_state_cpu: Optional[Dict[str, torch.Tensor]] = None

        for fold_i, (tr_idx, va_idx) in enumerate(folds, start=1):
            print(f"\n--- Fold {fold_i}/{k} ---")

            df_tr = df.iloc[tr_idx].reset_index(drop=True)
            df_va = df.iloc[va_idx].reset_index(drop=True)

            ds_tr = PicoOgRedDetectionDataset(df_tr, transform=IdentityTransform(), keep_empty=True)
            ds_va = PicoOgRedDetectionDataset(df_va, transform=IdentityTransform(), keep_empty=True)

            batch_size = int(params.get("batch_size", 2))

            dl_kwargs = {
                "num_workers": num_workers,
                "pin_memory": pin_memory,
                "persistent_workers": persistent_workers,
                "collate_fn": detection_collate,
            }
            if num_workers > 0:
                dl_kwargs["prefetch_factor"] = prefetch_factor

            dl_tr = DataLoader(
                ds_tr,
                batch_size=batch_size,
                shuffle=True,
                **dl_kwargs,
            )
            dl_va = DataLoader(
                ds_va,
                batch_size=1,
                shuffle=False,
                **dl_kwargs,
            )

            model = build_frcnn_resnet50_fpn_coco_6ch(
                num_classes=5,
                trainable_backbone_layers=int(params.get("trainable_backbone_layers", 2)),
                anchor_sizes=anchor_sizes_tt,
                aspect_ratios=aspect_ratios_tt,
                detections_per_image=int(params.get("detections_per_image", 400)),
                box_nms_thresh=float(params.get("box_nms_thresh", 0.5)),
            )
            model.to(device)
            if use_torch_compile and device.type == "cuda" and hasattr(torch, "compile"):
                model = torch.compile(model, mode=torch_compile_mode)

            optimizer = build_optimizer_two_groups(
                model,
                lr_backbone=float(params.get("lr_backbone", 1e-5)),
                lr_heads=float(params.get("lr_heads", 1e-4)),
                weight_decay=float(params.get("weight_decay", 1e-4)),
            )
            scheduler = build_scheduler(optimizer, step_size=max(1, epochs // 3), gamma=0.5)
            scaler = get_scaler(use_amp=amp, device=device)

            for ep in range(1, epochs + 1):
                tr_metrics = train_one_epoch(model, optimizer, dl_tr, device, epoch=ep, scaler=scaler, log_every=0)
                scheduler.step()
                if ep == 1 or ep == epochs or ep % 5 == 0:
                    print(f"  ep {ep}/{epochs} loss={tr_metrics['loss']:.4f}")

            m = evaluate_count_metrics(
                model,
                dl_va,
                device,
                score_thresh=val_score_thresh,
                classes_to_count=classes_to_count,
            )
            current_fold_mae = float(m["count_mae"])
            fold_mae.append(current_fold_mae)
            print(f"  val(score={val_score_thresh:.2f}) count_mae={m['count_mae']:.4f} bias={m['count_bias']:.4f}")

            if best_fold_mae is None or current_fold_mae < best_fold_mae:
                state_model = model._orig_mod if hasattr(model, "_orig_mod") else model
                best_fold_mae = current_fold_mae
                best_fold_idx = fold_i - 1
                best_fold_state_cpu = {k: v.detach().cpu() for k, v in state_model.state_dict().items()}

            del model
            torch.cuda.empty_cache()

        mean_mae = float(np.mean(fold_mae))
        std_mae = float(np.std(fold_mae))

        row = {
            "trial_id": trial_id,
            "mean_count_mae": mean_mae,
            "std_count_mae": std_mae,
            "fold_mae": fold_mae,
            "val_score_thresh": val_score_thresh,
            "classes_to_count": classes_to_count,
            **params,
        }
        append_jsonl(results_jsonl, row)
        all_rows.append(row)

        print(f"\nTRAIN TRIAL {trial_id} summary: mean_mae={mean_mae:.4f} ± {std_mae:.4f}")

        if best_overall is None or mean_mae < best_overall[0]:
            if best_fold_idx is None or best_fold_state_cpu is None:
                raise RuntimeError(f"No best fold state captured for trial {trial_id}")

            best_fold_id = best_fold_idx + 1
            best_overall = (mean_mae, trial_id, params, best_fold_idx)

            best_ckpt = {
                "model": best_fold_state_cpu,
                "trial_id": trial_id,
                "fold": best_fold_id,
                "fold_count_mae": float(fold_mae[best_fold_idx]),
                "mean_count_mae": mean_mae,
                "std_count_mae": std_mae,
                "val_score_thresh": val_score_thresh,
                "classes_to_count": classes_to_count,
                "params": params,
            }
            torch.save(best_ckpt, best_model_path)
            best_summary_path.write_text(
                json.dumps(
                    {
                        "trial_id": trial_id,
                        "best_fold": best_fold_id,
                        "fold_count_mae": float(fold_mae[best_fold_idx]),
                        "mean_count_mae": mean_mae,
                        "std_count_mae": std_mae,
                        "val_score_thresh": val_score_thresh,
                        "classes_to_count": classes_to_count,
                        "params": params,
                        "checkpoint_path": str(best_model_path),
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            print("  NEW BEST ✅")

    df_out = pd.DataFrame(all_rows)
    df_out.to_csv(results_csv, index=False)
    print("\nWrote:", results_jsonl)
    print("Wrote:", results_csv)
    if best_model_path.exists():
        print("Wrote:", best_model_path)
        print("Wrote:", best_summary_path)


if __name__ == "__main__":
    main()
