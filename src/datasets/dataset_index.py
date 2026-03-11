from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd
import torch


@dataclass
class IndexSplit:
    train: pd.DataFrame
    val: pd.DataFrame


def load_index(index_csv: Path) -> pd.DataFrame:
    index_csv = Path(index_csv)
    if not index_csv.exists():
        raise FileNotFoundError(f"index.csv not found: {index_csv}")
    df = pd.read_csv(index_csv)
    for col in ["stem", "og_webp", "label_path"]:
        if col not in df.columns:
            raise ValueError(f"index.csv missing column: {col}")
    df = df[df["og_webp"].notna() & df["label_path"].notna()].copy()
    df["og_webp"] = df["og_webp"].astype(str)
    df["label_path"] = df["label_path"].astype(str)
    df["stem"] = df["stem"].astype(str)
    return df.reset_index(drop=True)


def random_split_df(df: pd.DataFrame, val_frac: float, seed: int) -> IndexSplit:
    n = len(df)
    n_val = int(round(n * val_frac))
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    val_idx = set(perm[:n_val])
    tr = df.iloc[[i for i in range(n) if i not in val_idx]].reset_index(drop=True)
    va = df.iloc[[i for i in range(n) if i in val_idx]].reset_index(drop=True)
    return IndexSplit(train=tr, val=va)