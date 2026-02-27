from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights


def _expand_conv1_weight_3_to_6(w3: torch.Tensor, extra_scale: float = 0.5) -> torch.Tensor:
    """
    w3: (out, 3, k, k) -> returns (out, 6, k, k)
    Strategy A: copy pretrained RGB weights into first 3 channels,
    and repeat them into channels 4-6 (optionally scaled).
    """
    if w3.ndim != 4 or w3.shape[1] != 3:
        raise ValueError(f"Expected conv1 weight shape (out,3,k,k), got {tuple(w3.shape)}")

    w_extra = w3.clone() * float(extra_scale)
    w6 = torch.cat([w3, w_extra], dim=1)
    return w6


def build_frcnn_resnet50_fpn_coco_6ch(
    num_classes: int,
    trainable_backbone_layers: int = 2,
    anchor_sizes: Optional[Sequence[Sequence[int]]] = None,
    aspect_ratios: Optional[Sequence[Sequence[float]]] = None,
    detections_per_image: Optional[int] = 300,
    box_nms_thresh: float = 0.5,
    extra_scale: float = 0.5,
) -> FasterRCNN:
    if anchor_sizes is None:
        anchor_sizes = ((16,), (32,), (64,), (128,), (256,))
    if aspect_ratios is None:
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)

    anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

    backbone = resnet_fpn_backbone(
        backbone_name="resnet50",
        weights="DEFAULT",
        trainable_layers=int(trainable_backbone_layers),
    )

    # ---- Patch first conv: 3ch -> 6ch ----
    conv1 = backbone.body.conv1
    if not isinstance(conv1, nn.Conv2d):
        raise TypeError(f"Unexpected backbone.body.conv1 type: {type(conv1)}")

    if conv1.in_channels != 3:
        raise ValueError(f"Expected conv1.in_channels=3 before patch, got {conv1.in_channels}")

    new_conv1 = nn.Conv2d(
        in_channels=6,
        out_channels=conv1.out_channels,
        kernel_size=conv1.kernel_size,
        stride=conv1.stride,
        padding=conv1.padding,
        dilation=conv1.dilation,
        groups=conv1.groups,
        bias=(conv1.bias is not None),
        padding_mode=conv1.padding_mode,
    )
    backbone.body.conv1 = new_conv1

    model = FasterRCNN(
        backbone=backbone,
        num_classes=91,  # will be replaced below
        rpn_anchor_generator=anchor_generator,
        rpn_pre_nms_top_n_train=4000,
        rpn_post_nms_top_n_train=2000,
        rpn_pre_nms_top_n_test=2000,
        rpn_post_nms_top_n_test=2000,
        box_detections_per_img=detections_per_image,
        box_nms_thresh=float(box_nms_thresh),
    )

    # ---- Load COCO weights, but adapt conv1 weight to 6ch ----
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    state = weights.get_state_dict(progress=True)

    k = "backbone.body.conv1.weight"
    if k not in state:
        raise KeyError(f"Expected key {k!r} in COCO state dict, found {len(state)} keys")

    w3 = state[k]
    state[k] = _expand_conv1_weight_3_to_6(w3, extra_scale=extra_scale)

    missing, unexpected = model.load_state_dict(state, strict=False)

    # Replace ROI head for num_classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model