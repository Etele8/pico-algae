"""Wrap a trained 3-channel checkpoint into the format app/pico_counter.py
expects: the raw state_dict under "model", plus the tuned inference settings
(anchors/NMS under "params", and val_score_thresh / classes_to_count).

Without the "params" block the app silently falls back to DEFAULT anchors
([[16],[32],[64],[128],[256]]) instead of the trained ones, so detections
degrade. This script copies the settings from the training yaml.

Usage:
  python scripts/package_for_app.py \
      --ckpt runs/merged_3ch/checkpoints/best_mae.pt \
      --train_yaml src/configs/train_frcnn.yaml \
      --out best_train_model.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="best_mae.pt from training (3-channel).")
    ap.add_argument("--train_yaml", default="src/configs/train_frcnn.yaml")
    ap.add_argument("--out", default="best_train_model.pt")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.train_yaml).read_text(encoding="utf-8")) or {}
    space = cfg.get("space", cfg)

    ck = torch.load(args.ckpt, map_location="cpu")
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck

    in_ch = int(state["backbone.body.conv1.weight"].shape[1])
    if in_ch != 3:
        raise SystemExit(
            f"Refusing to package: {args.ckpt} is a {in_ch}-channel checkpoint, "
            "but the app only serves 3-channel models."
        )

    out = {
        "model": state,
        "params": {
            "anchor_sizes": space["anchor_sizes"],
            "aspect_ratios": space["aspect_ratios"],
            "detections_per_image": int(cfg.get("detections_per_image", 300)),
            "box_nms_thresh": float(cfg.get("box_nms_thresh", 0.5)),
        },
        "val_score_thresh": float(cfg.get("score_thresh", 0.5)),
        "classes_to_count": [int(c) for c in cfg.get("classes_to_count", [1, 2, 3])],
    }
    torch.save(out, args.out)
    print(f"Wrote {args.out} (3ch, app-ready). "
          f"anchors={out['params']['anchor_sizes']} "
          f"score_thresh={out['val_score_thresh']} "
          f"count={out['classes_to_count']}")


if __name__ == "__main__":
    main()
