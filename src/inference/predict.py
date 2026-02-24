from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any

import torch


@torch.no_grad()
def predict_on_loader(model, data_loader, device, score_thresh: float = 0.5) -> List[Dict[str, Any]]:
    model.eval()
    results = []

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        outputs = model(images)

        for out, tgt in zip(outputs, targets):
            scores = out["scores"].detach().cpu()
            keep = scores >= score_thresh

            results.append({
                "image_id": int(tgt["image_id"][0]),
                "pred_boxes": out["boxes"].detach().cpu()[keep].numpy(),
                "pred_labels": out["labels"].detach().cpu()[keep].numpy(),
                "pred_scores": scores[keep].numpy(),
                "gt_boxes": tgt["boxes"].numpy(),
                "gt_labels": tgt["labels"].numpy(),
            })
    return results