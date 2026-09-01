"""Wrap a trained 3-channel checkpoint into the format app/pico_counter.py
expects: the raw state_dict under "model", plus the tuned inference settings
(anchors/NMS under "params", and val_score_thresh / classes_to_count).

Without the "params" block the app silently falls back to DEFAULT anchors
([[16],[32],[64],[128],[256]]) instead of the trained ones, so detections
degrade. This script copies the settings from the training yaml.

Ship the all-data model, but take the score/NMS thresholds from the HONEST
held-out CV sweep (post_3ch/…), NOT a sweep of the all-data model itself — that
would be leakage. Pass that sweep's best_post_summary.json via --post_summary,
or set --score_thresh / --box_nms explicitly.

Usage:
  python scripts/package_for_app.py \
      --ckpt runs/ship_3ch/checkpoints/best_mae.pt \
      --train_yaml src/configs/train_frcnn.yaml \
      --post_summary runs/post_3ch/best_post_summary.json \
      --out best_train_model.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="best_mae.pt from training (3-channel).")
    ap.add_argument("--train_yaml", default="src/configs/train_frcnn.yaml")
    ap.add_argument("--post_summary", default="",
                    help="best_post_summary.json from an HONEST (held-out CV) sweep to "
                         "pull box_nms + score threshold from. Do NOT use a sweep of the "
                         "all-data model (leakage).")
    ap.add_argument("--score_thresh", type=float, default=None, help="Override val_score_thresh.")
    ap.add_argument("--box_nms", type=float, default=None, help="Override box_nms_thresh.")
    ap.add_argument("--out", default="best_train_model.pt")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.train_yaml).read_text(encoding="utf-8")) or {}
    space = cfg.get("space", cfg)

    ck = torch.load(args.ckpt, map_location="cpu")
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    # Prefer anchors embedded in the checkpoint; fall back to the training yaml.
    ck_params = ck.get("params", {}) if isinstance(ck, dict) else {}

    in_ch = int(state["backbone.body.conv1.weight"].shape[1])
    if in_ch != 3:
        raise SystemExit(
            f"Refusing to package: {args.ckpt} is a {in_ch}-channel checkpoint, "
            "but the app only serves 3-channel models."
        )

    # Inference thresholds: yaml defaults, overridden by the honest post-sweep,
    # then by explicit flags.
    score_thresh = float(cfg.get("score_thresh", 0.5))
    box_nms = float(ck_params.get("box_nms_thresh", cfg.get("box_nms_thresh", 0.5)))
    if args.post_summary:
        ps = json.loads(Path(args.post_summary).read_text(encoding="utf-8"))
        box_nms = float(ps.get("box_nms_thresh", box_nms))
        pcs = ps.get("per_class_score_thresh") or {}
        if pcs:
            vals = sorted({float(v) for v in pcs.values()})
            if len(vals) > 1:
                print(f"[warn] per-class thresholds not uniform {vals}; the app uses a "
                      f"single threshold — using the max ({vals[-1]}).")
            score_thresh = vals[-1]
        elif "score_thresh" in ps:
            score_thresh = float(ps["score_thresh"])
    if args.score_thresh is not None:
        score_thresh = args.score_thresh
    if args.box_nms is not None:
        box_nms = args.box_nms

    out = {
        "model": state,
        "params": {
            "anchor_sizes": ck_params.get("anchor_sizes", space["anchor_sizes"]),
            "aspect_ratios": ck_params.get("aspect_ratios", space["aspect_ratios"]),
            "detections_per_image": int(ck_params.get("detections_per_image",
                                                      cfg.get("detections_per_image", 300))),
            "box_nms_thresh": box_nms,
        },
        "val_score_thresh": score_thresh,
        "classes_to_count": [int(c) for c in cfg.get("classes_to_count", [1, 2, 3])],
    }
    torch.save(out, args.out)
    print(f"Wrote {args.out} (3ch, app-ready). "
          f"anchors={out['params']['anchor_sizes']} "
          f"score_thresh={out['val_score_thresh']} "
          f"box_nms={out['params']['box_nms_thresh']} "
          f"count={out['classes_to_count']}")


if __name__ == "__main__":
    main()
