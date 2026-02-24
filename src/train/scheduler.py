def build_scheduler(optimizer, step_size: int = 8, gamma: float = 0.5):
    import torch
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)