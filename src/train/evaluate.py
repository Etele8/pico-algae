from __future__ import annotations
from typing import Dict
import numpy as np
import torch


@torch.no_grad()
def evaluate_count_metrics(model, data_loader, device, score_thresh: float = 0.5) -> Dict[str, float]:
    model.eval()

    abs_errs = []
    sq_errs = []
    diffs = []

    per_cls_abs = {1: [], 2: [], 3: [], 4: []}

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        outputs = model(images)

        for out, tgt in zip(outputs, targets):
            gt_labels = tgt["labels"].numpy()
            gt_count = int(len(gt_labels))

            scores = out["scores"].detach().cpu().numpy()
            pred_labels = out["labels"].detach().cpu().numpy()
            keep = scores >= score_thresh

            pred_count = int(keep.sum())
            d = pred_count - gt_count

            diffs.append(d)
            abs_errs.append(abs(d))
            sq_errs.append(d * d)

            for c in per_cls_abs.keys():
                gt_c = int((gt_labels == c).sum())
                pred_c = int(((pred_labels == c) & keep).sum())
                per_cls_abs[c].append(abs(pred_c - gt_c))

    def safe_mean(x):
        return float(np.mean(x)) if len(x) else 0.0

    return {
        "count_mae": safe_mean(abs_errs),
        "count_rmse": float(np.sqrt(safe_mean(sq_errs))),
        "count_bias": safe_mean(diffs),
        "mae_EUK": safe_mean(per_cls_abs[1]),
        "mae_FE": safe_mean(per_cls_abs[2]),
        "mae_FC": safe_mean(per_cls_abs[3]),
        "mae_colony": safe_mean(per_cls_abs[4]),
    }