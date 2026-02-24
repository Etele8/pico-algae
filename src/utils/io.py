import os
import random
import numpy as np
import torch
from pathlib import Path


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)