from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class ImagePair:
    stem: str
    og_path: Path
    red_path: Optional[Path]


def discover_og_red_pairs(images_dir: Path, recursive: bool = False, require_red_pair: bool = True) -> List[ImagePair]:
    images_dir = Path(images_dir)
    if not images_dir.exists():
        raise FileNotFoundError(f"Image folder not found: {images_dir}")

    globber = images_dir.rglob if recursive else images_dir.glob
    og_files = sorted(p for p in globber("*_og.png") if p.is_file())

    pairs: List[ImagePair] = []
    for og_path in og_files:
        name = og_path.stem
        if not name.endswith("_og"):
            continue
        stem = name[:-3]
        red_path = og_path.with_name(f"{stem}_red{og_path.suffix}")
        if not red_path.exists():
            if require_red_pair:
                continue
            red_path = None
        pairs.append(ImagePair(stem=stem, og_path=og_path, red_path=red_path))

    return pairs
