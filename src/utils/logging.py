from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any

from .io import ensure_dir


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")