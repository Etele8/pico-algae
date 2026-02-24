from __future__ import annotations
import argparse
import time
from pathlib import Path

import torch
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

    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=4)

    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    
    ap.add_argument("--trainable_backbone_layers", type=int, default=2)
    ap.add_argument("--lr_backbone", type=float, default=1e-5)
    ap.add_argument("--lr_heads", type=float, default=1e-4)

    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--score_thresh", type=float, default=0.5)

    ap.add_argument("--flip_p", type=float, default=0.0, help="Start with 0.0; set e.g. 0.5 after baseline works.")
    args = ap.parse_args()

    seed_everything(args.seed)

    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    ensure_dir(ckpt_dir)
    log_path = out_dir / "logs.jsonl"
    ensure_dir(out_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    df = load_index(Path(args.index_csv))
    split = random_split_df(df, val_frac=args.val_frac, seed=args.seed)

    # transforms (keep geometry simple at first)
    if args.flip_p > 0:
        tfm = RandomHorizontalFlip(p=args.flip_p)
    else:
        tfm = IdentityTransform()

    ds_tr = PicoOgDetectionDataset(split.train, transform=tfm, keep_empty=True)
    ds_va = PicoOgDetectionDataset(split.val, transform=IdentityTransform(), keep_empty=True)

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.num_workers, pin_memory=True, collate_fn=detection_collate)
    dl_va = DataLoader(ds_va, batch_size=1, shuffle=False,
                       num_workers=args.num_workers, pin_memory=True, collate_fn=detection_collate)

    model = build_frcnn_resnet50_fpn_coco(num_classes=5, pretrained_backbone=True)
    model.to(device)

    optimizer = build_optimizer_two_groups(model, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, step_size=max(1, args.epochs // 3), gamma=0.5)

    scaler = get_scaler(args.amp, device)

    best = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        tr = train_one_epoch(model, optimizer, dl_tr, device, epoch, scaler=scaler, log_every=20)
        va = evaluate_count_metrics(model, dl_va, device, score_thresh=args.score_thresh)
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