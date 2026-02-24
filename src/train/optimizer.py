from __future__ import annotations

import torch


def build_optimizer_two_groups(
    model,
    lr_backbone: float = 1e-5,
    lr_heads: float = 1e-4,
    weight_decay: float = 1e-4,
):
    """
    Two LR groups:
      - backbone (incl. FPN) gets smaller LR
      - everything else gets larger LR

    This is very helpful with small datasets (~250 images).
    """
    backbone_params = []
    head_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("backbone."):
            backbone_params.append(p)
        else:
            head_params.append(p)

    param_groups = [
        {"params": backbone_params, "lr": lr_backbone},
        {"params": head_params, "lr": lr_heads},
    ]

    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)