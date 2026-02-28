from __future__ import annotations
from typing import Dict, Iterable, Optional, Mapping
import numpy as np
import torch


@torch.no_grad()
def evaluate_count_metrics(
    model,
    data_loader,
    device,
    score_thresh: float = 0.5,
    classes_to_count: Optional[Iterable[int]] = None,
    per_class_score_thresh: Optional[Mapping[int, float]] = None,
) -> Dict[str, float]:
    """
    Count-based metrics for object detection.

    - Applies model's internal NMS as usual (torchvision FasterRCNN).
    - Applies score thresholding AFTER NMS.

    Args:
        score_thresh: global score threshold (used if per_class_score_thresh is None)
        classes_to_count: if provided, counts/MAE/bias computed ONLY on these class ids
        per_class_score_thresh: optional dict like {1:0.2, 2:0.15, 3:0.3, 4:0.5}
            If provided, per-detection threshold is looked up by predicted label;
            labels not in the dict fall back to global score_thresh.
    """
    model.eval()

    abs_errs = []
    sq_errs = []
    diffs = []

    # Keep your per-class abs error logging (classes assumed 1..4 here)
    per_cls_abs = {1: [], 2: [], 3: [], 4: []}

    if classes_to_count is not None:
        classes_to_count = set(int(c) for c in classes_to_count)

    def safe_mean(x):
        return float(np.mean(x)) if len(x) else 0.0

    for images, targets in data_loader:
        images = [img.to(device, non_blocking=True) for img in images]
        outputs = model(images)

        for out, tgt in zip(outputs, targets):
            gt_labels = tgt["labels"].numpy()

            scores = out["scores"].detach().cpu().numpy()
            pred_labels = out["labels"].detach().cpu().numpy()

            # ---- choose which classes to count in TOTAL MAE/Bias ----
            if classes_to_count is None:
                gt_mask_total = np.ones_like(gt_labels, dtype=bool)
                pred_mask_total = np.ones_like(pred_labels, dtype=bool)
            else:
                gt_mask_total = np.isin(gt_labels, list(classes_to_count))
                pred_mask_total = np.isin(pred_labels, list(classes_to_count))

            gt_count = int(gt_mask_total.sum())

            # ---- thresholding after NMS ----
            if per_class_score_thresh is None:
                keep_score = scores >= float(score_thresh)
            else:
                # threshold depends on predicted class label (fallback to global threshold)
                thr = np.array(
                    [float(per_class_score_thresh.get(int(lbl), score_thresh)) for lbl in pred_labels],
                    dtype=np.float32,
                )
                keep_score = scores >= thr

            keep_total = keep_score & pred_mask_total
            pred_count = int(keep_total.sum())

            d = pred_count - gt_count
            diffs.append(d)
            abs_errs.append(abs(d))
            sq_errs.append(d * d)

            # ---- per-class MAE logging (independent of classes_to_count) ----
            for c in per_cls_abs.keys():
                gt_c = int((gt_labels == c).sum())
                pred_c = int(((pred_labels == c) & keep_score).sum())
                per_cls_abs[c].append(abs(pred_c - gt_c))

    return {
        "count_mae": safe_mean(abs_errs),
        "count_rmse": float(np.sqrt(safe_mean(sq_errs))),
        "count_bias": safe_mean(diffs),
        "mae_EUK": safe_mean(per_cls_abs[1]),
        "mae_FE": safe_mean(per_cls_abs[2]),
        "mae_FC": safe_mean(per_cls_abs[3]),
        "mae_colony": safe_mean(per_cls_abs[4]),
    }
