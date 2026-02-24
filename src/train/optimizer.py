import torch


def build_optimizer(model, lr: float = 1e-4, weight_decay: float = 1e-4):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)