import torch


def clip_boxes_xyxy_(boxes: torch.Tensor, w: int, h: int) -> torch.Tensor:
    boxes[:, 0].clamp_(0, w - 1)
    boxes[:, 2].clamp_(0, w - 1)
    boxes[:, 1].clamp_(0, h - 1)
    boxes[:, 3].clamp_(0, h - 1)
    return boxes


def remove_degenerate(boxes: torch.Tensor, labels: torch.Tensor):
    keep = (boxes[:, 2] - boxes[:, 0] >= 1.0) & (boxes[:, 3] - boxes[:, 1] >= 1.0)
    return boxes[keep], labels[keep]