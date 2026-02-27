from __future__ import annotations
from typing import List, Dict, Any, Optional, Mapping, Iterable

import torch


@torch.no_grad()
def predict_on_loader(
    model,
    data_loader,
    device,
    score_thresh: float = 0.5,
    per_class_score_thresh: Optional[Mapping[int, float]] = None,
    classes_to_keep: Optional[Iterable[int]] = None,
) -> List[Dict[str, Any]]:
    model.eval()
    results = []
    classes_to_keep_set = set(int(c) for c in classes_to_keep) if classes_to_keep is not None else None

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        outputs = model(images)

        for out, tgt in zip(outputs, targets):
            boxes = out["boxes"].detach().cpu()
            scores = out["scores"].detach().cpu()
            labels = out["labels"].detach().cpu()
            raw_count = int(scores.shape[0])

            if per_class_score_thresh is None:
                keep_score = scores >= float(score_thresh)
            else:
                thr = torch.tensor(
                    [float(per_class_score_thresh.get(int(lbl), score_thresh)) for lbl in labels.tolist()],
                    dtype=scores.dtype,
                )
                keep_score = scores >= thr

            if classes_to_keep_set is None:
                keep_cls = torch.ones_like(keep_score, dtype=torch.bool)
            else:
                keep_cls = torch.tensor([int(lbl) in classes_to_keep_set for lbl in labels.tolist()], dtype=torch.bool)

            keep = keep_score & keep_cls

            results.append({
                "image_id": int(tgt["image_id"][0]),
                "raw_count": raw_count,
                "raw_boxes": boxes.numpy(),
                "raw_labels": labels.numpy(),
                "raw_scores": scores.numpy(),
                "keep_mask": keep.numpy(),
                "pred_boxes": boxes[keep].numpy(),
                "pred_labels": labels[keep].numpy(),
                "pred_scores": scores[keep].numpy(),
                "gt_boxes": tgt["boxes"].numpy(),
                "gt_labels": tgt["labels"].numpy(),
            })
    return results
