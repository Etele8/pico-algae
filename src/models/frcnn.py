from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# COCO pretrained weights (detector weights, not just backbone)
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)


def build_frcnn_resnet50_fpn_coco(
    num_classes: int,
    trainable_backbone_layers: int = 2,
    anchor_sizes: Optional[Sequence[Sequence[int]]] = None,
    aspect_ratios: Optional[Sequence[Sequence[float]]] = None,
    detections_per_image: Optional[int] = 300,
) -> FasterRCNN:
    """
    Builds Faster R-CNN ResNet50-FPN with:
      - COCO-pretrained detector weights loaded
      - ResNet backbone partially trainable (trainable_backbone_layers)
      - Custom anchor generator (e.g., include 16)
      - New box predictor for your classes

    num_classes includes background.
    For your 4 classes => num_classes = 5
    """

    if anchor_sizes is None:
        anchor_sizes = ((16,), (32,), (64,), (128,), (256,))
    if aspect_ratios is None:
        # Keep default as a baseline
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)

    anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

    # Build a fresh model with custom anchors and desired trainable layers.
    # trainable_layers counts from the end: 2 => train layer4 and layer3, freeze earlier.
    backbone = resnet_fpn_backbone(
        backbone_name="resnet50",
        weights="DEFAULT",  # we will load COCO detector weights into the whole model
        trainable_layers=int(trainable_backbone_layers),
    )

    model = FasterRCNN(
        backbone=backbone,
        num_classes=91,  # temporary; we'll replace predictor after loading weights
        rpn_anchor_generator=anchor_generator,
        rpn_pre_nms_top_n_train=4000,
        rpn_post_nms_top_n_train=2000,
        rpn_pre_nms_top_n_test=2000,
        rpn_post_nms_top_n_test=2000,
        box_detections_per_img=detections_per_image,
    )

    # Load COCO pretrained detector weights
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    state = weights.get_state_dict(progress=True)

    # Load with strict=False because we'll swap the predictor (class head) anyway
    missing, unexpected = model.load_state_dict(state, strict=False)

    # Now replace the classifier head to match your num_classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model