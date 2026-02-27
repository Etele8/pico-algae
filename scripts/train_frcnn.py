from __future__ import annotations
import argparse
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from src.utils.seed import seed_everything
from src.utils.io import ensure_dir
from src.utils.logging import append_jsonl

from src.data.dataset_index import load_index, random_split_df
from src.data.pico_dataset import PicoOgDetectionDataset
from src.data.collate import detection_collate
from src.data.transforms import IdentityTransform, RandomHorizontalFlip

from src.models.frcnn import build_frcnn_resnet50_fpn_coco
from src.models.weights import save_checkpoint

from src.train.optimizer import build_optimizer_two_groups
from src.train.scheduler import build_scheduler
from src.train.amp import get_scaler
from src.train.train_one_epoch import train_one_epoch
from src.train.evaluate import evaluate_count_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_csv", required=True, type=str)
    ap.add_argument("--out_dir", required=True, type=str)
    ap.add_argument("--train_yaml", required=True, type=str)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.train_yaml).read_text(encoding="utf-8")) or {}

    seed = int(cfg.get("seed", 42))
    epochs = int(cfg.get("epochs", 20))
    batch_size = int(cfg.get("batch_size", 2))
    num_workers = int(cfg.get("num_workers", 4))
    weight_decay = float(cfg.get("weight_decay", 1e-4))
    trainable_backbone_layers = int(cfg.get("trainable_backbone_layers", 2))
    lr_backbone = float(cfg.get("lr_backbone", cfg.get("lr", 1e-5)))
    lr_heads = float(cfg.get("lr_heads", cfg.get("lr", 1e-4)))
    val_frac = float(cfg.get("val_frac", 0.2))
    use_amp = bool(cfg.get("amp", False))
    score_thresh = float(cfg.get("score_thresh", 0.5))
    flip_p = float(cfg.get("flip_p", 0.0))

    seed_everything(seed)

    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    ensure_dir(ckpt_dir)
    log_path = out_dir / "logs.jsonl"
    ensure_dir(out_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    df = load_index(Path(args.index_csv))
    split = random_split_df(df, val_frac=val_frac, seed=seed)

    # transforms (keep geometry simple at first)
    if flip_p > 0:
        tfm = RandomHorizontalFlip(p=flip_p)
    else:
        tfm = IdentityTransform()

    ds_tr = PicoOgDetectionDataset(split.train, transform=tfm, keep_empty=True)
    ds_va = PicoOgDetectionDataset(split.val, transform=IdentityTransform(), keep_empty=True)

    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,
                       num_workers=num_workers, pin_memory=True, collate_fn=detection_collate)
    dl_va = DataLoader(ds_va, batch_size=1, shuffle=False,
                       num_workers=num_workers, pin_memory=True, collate_fn=detection_collate)

    model = build_frcnn_resnet50_fpn_coco(
        num_classes=5,
        trainable_backbone_layers=trainable_backbone_layers,
    )
    model.to(device)

    optimizer = build_optimizer_two_groups(
        model,
        lr_backbone=lr_backbone,
        lr_heads=lr_heads,
        weight_decay=weight_decay,
    )
    scheduler = build_scheduler(optimizer, step_size=max(1, epochs // 3), gamma=0.5)

    scaler = get_scaler(use_amp, device)

    best = float("inf")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        tr = train_one_epoch(model, optimizer, dl_tr, device, epoch, scaler=scaler, log_every=20)
        va = evaluate_count_metrics(model, dl_va, device, score_thresh=score_thresh)
        scheduler.step()

        row = {"epoch": epoch, **tr, **va, "lr": float(optimizer.param_groups[0]["lr"]), "time_s": time.time() - t0}
        print(row)
        append_jsonl(log_path, row)

        save_checkpoint(ckpt_dir / "last.pt", model, optimizer=optimizer, epoch=epoch)

        if va["count_mae"] < best:
            best = va["count_mae"]
            save_checkpoint(ckpt_dir / "best_mae.pt", model, epoch=epoch, extra={"best_mae": best})
            print("  saved best_mae.pt (best_mae=", best, ")")

    print("Done. Best MAE:", best)


if __name__ == "__main__":
    main()
