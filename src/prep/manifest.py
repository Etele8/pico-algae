# manifest.py
# Builds:
#   data/processed/manifest.csv
#   reports/sanity_check.txt
#
# Run:
#   python manifest.py --root "D:\intezet\Pico_algae"

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
from PIL import Image


@dataclass
class Paths:
    root: Path
    og_dir: Path
    red_dir: Path
    labels_dir: Path
    processed_dir: Path
    reports_dir: Path

    @staticmethod
    def from_root(root: Path) -> "Paths":
        return Paths(
            root=root,
            og_dir=root / "data" / "raw" / "images_og",
            red_dir=root / "data" / "raw" / "images_red",
            labels_dir=root / "data" / "processed" / "labels_abs",
            processed_dir=root / "data" / "processed",
            reports_dir=root / "reports",
        )


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def stem_from_og_filename(p: Path) -> Optional[str]:
    # Image_5444_og.png -> Image_5444
    name = p.stem
    if name.endswith("_og"):
        return name[:-3]
    return None


def image_size(path: Path) -> Tuple[int, int]:
    with Image.open(path) as im:
        return im.size  # (W, H)


def read_abs_label_file(txt_path: Path) -> List[Tuple[int, float, float, float, float]]:
    """
    Reads ABS labels:
      class_id x1 y1 x2 y2
    """
    boxes = []
    if not txt_path.exists():
        return boxes
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            continue
        cls = int(parts[0])
        x1, y1, x2, y2 = map(float, parts[1:])
        boxes.append((cls, x1, y1, x2, y2))
    return boxes


def build_manifest(paths: Paths, ext=(".png")) -> pd.DataFrame:
    og_files: List[Path] = []
    og_files.extend(paths.og_dir.glob(f"*{ext}"))
    og_files = sorted(og_files)

    rows = []
    sanity_lines = []
    missing_red = 0
    missing_label = 0
    size_mismatch = 0

    for og_path in og_files:
        stem = stem_from_og_filename(og_path)
        if stem is None:
            continue

        red_path = paths.red_dir / f"{stem}_red{og_path.suffix}"
        label_path = paths.labels_dir / f"{stem}_combined.txt"  # expected

        if not red_path.exists():
            missing_red += 1
            sanity_lines.append(f"[MISSING_RED] {stem}  expected: {red_path}")
            continue

        if not label_path.exists():
            missing_label += 1
            sanity_lines.append(f"[MISSING_LABEL] {stem}  expected: {label_path}")
            continue

        w_og, h_og = image_size(og_path)
        w_red, h_red = image_size(red_path)
        if (w_og, h_og) != (w_red, h_red):
            size_mismatch += 1
            sanity_lines.append(
                f"[SIZE_MISMATCH] {stem}  og=({w_og},{h_og}) red=({w_red},{h_red})"
            )
            continue

        boxes = read_abs_label_file(label_path)
        n_boxes = len(boxes)

        counts: Dict[int, int] = {}
        bad_boxes = 0
        for cls, x1, y1, x2, y2 in boxes:
            counts[cls] = counts.get(cls, 0) + 1
            if x2 <= x1 or y2 <= y1:
                bad_boxes += 1
            if x1 < 0 or y1 < 0 or x2 > w_og or y2 > h_og:
                bad_boxes += 1

        if bad_boxes > 0:
            sanity_lines.append(f"[BAD_BOXES] {stem}  bad_boxes={bad_boxes}")

        row = {
            "stem": stem,
            "og_path": str(og_path),
            "red_path": str(red_path),
            "label_path": str(label_path),
            "width": w_og,
            "height": h_og,
            "n_boxes": n_boxes,
            "n_bad_boxes": bad_boxes,
        }
        for cls_id, c in counts.items():
            row[f"count_cls_{cls_id}"] = c

        rows.append(row)

    df = pd.DataFrame(rows).sort_values("stem").reset_index(drop=True)

    ensure_dir(paths.reports_dir)
    sanity_path = paths.reports_dir / "sanity_check.txt"
    ensure_dir(paths.processed_dir)

    summary = [
        f"ROOT: {paths.root}",
        f"OG DIR: {paths.og_dir}",
        f"RED DIR: {paths.red_dir}",
        f"LABELS DIR: {paths.labels_dir}",
        "",
        f"Total OG images scanned: {len(og_files)}",
        f"Manifest rows created: {len(df)}",
        f"Missing RED pairs: {missing_red}",
        f"Missing labels: {missing_label}",
        f"OG/RED size mismatches: {size_mismatch}",
        "",
        "Issues (first 500):",
        *sanity_lines[:500],
    ]
    sanity_path.write_text("\n".join(summary) + "\n", encoding="utf-8")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-name", default="manifest.csv", help="Filename under data/processed/")
    args = ap.parse_args()

    paths = Paths.from_root(Path(__file__).resolve().parent.parent)
    

    for p in [paths.og_dir, paths.red_dir, paths.labels_dir]:
        if not p.exists():
            raise FileNotFoundError(f"Expected folder missing: {p}")

    df = build_manifest(paths)
    manifest_path = paths.processed_dir / args.out_name
    df.to_csv(manifest_path, index=False, encoding="utf-8")
    print(f"Wrote manifest: {manifest_path} (rows={len(df)})")
    print(f"Wrote sanity report: {paths.reports_dir / 'sanity_check.txt'}")


if __name__ == "__main__":
    main()