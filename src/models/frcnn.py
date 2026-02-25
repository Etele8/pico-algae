from __future__ import annotations

from typing import Optional, Sequence

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights


def build_frcnn_resnet50_fpn_coco(
    num_classes: int,
    trainable_backbone_layers: int = 2,
    anchor_sizes: Optional[Sequence[Sequence[int]]] = None,
    aspect_ratios: Optional[Sequence[Sequence[float]]] = None,
    detections_per_image: Optional[int] = 300,
    box_nms_thresh: float = 0.5,
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

    model = FasterRCNN(
        backbone=backbone,
        num_classes=91,
        rpn_anchor_generator=anchor_generator,
        rpn_pre_nms_top_n_train=4000,
        rpn_post_nms_top_n_train=2000,
        rpn_pre_nms_top_n_test=2000,
        rpn_post_nms_top_n_test=2000,
        box_detections_per_img=detections_per_image,
        box_nms_thresh=float(box_nms_thresh),
    )

    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    state = weights.get_state_dict(progress=True)
    model.load_state_dict(state, strict=False)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model