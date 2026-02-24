from __future__ import annotations
from typing import Dict

import torch
from torch.amp import autocast


def train_one_epoch(model, optimizer, data_loader, device, epoch: int, scaler=None, log_every: int = 20) -> Dict[str, float]:
    model.train()
    loss_sum = 0.0
    n = 0

    for it, (images, targets) in enumerate(data_loader, start=1):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad(set_to_none=True)

        use_amp = scaler is not None
        with autocast("cuda", enabled=use_amp):
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

        if torch.isnan(loss) or torch.isinf(loss):
            raise RuntimeError(f"NaN/Inf loss at epoch {epoch}, iter {it}: { {k: float(v) for k,v in loss_dict.items()} }")

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        loss_sum += float(loss.item())
        n += 1

        if log_every and it % log_every == 0:
            ld = {k: float(v.item()) for k, v in loss_dict.items()}
            print(f"[epoch {epoch} iter {it}] loss={loss.item():.4f} " + " ".join([f"{k}={v:.4f}" for k, v in ld.items()]))

    return {"loss": loss_sum / max(n, 1)}