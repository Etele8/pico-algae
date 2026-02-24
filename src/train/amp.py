import torch


def get_scaler(use_amp: bool, device: torch.device):
    if use_amp and device.type == "cuda":
        return  torch.amp.GradScaler("cuda")
    return None