from __future__ import annotations
from typing import Optional, Sequence

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import AnchorGenerator


def build_frcnn_resnet50_fpn(
    num_classes: int,
    pretrained_backbone: bool = True,
    anchor_sizes: Optional[Sequence[Sequence[int]]] = None,
    aspect_ratios: Optional[Sequence[Sequence[float]]] = None,
) -> FasterRCNN:
    backbone = resnet_fpn_backbone("resnet50", weights="DEFAULT" if pretrained_backbone else None)

    if anchor_sizes is None:
        anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    if aspect_ratios is None:
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)

    anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,  # includes background
        rpn_anchor_generator=anchor_generator,
    )
    return model