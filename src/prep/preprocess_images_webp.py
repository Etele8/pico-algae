# preprocess_images_webp.py
#
# What it does (combined + simplified):
# - Reads manifest.csv (default: data/processed/manifest.csv)
#   expecting columns at least: stem, og_path, red_path, label_path
# - Loads OG + RED
# - If an image is not 2048x1500, it is resized to 2048x1500
# - Rescales the bounding boxes in the label txt accordingly
#   Label format per line: class_id x_min y_min x_max y_max   (absolute pixels)
# - Saves resized images as lossless WEBP
# - Writes resized label txt files
# - Writes data/processed/index.csv pointing to the new webps and resized labels
#
# WEBP stores uint8, so the correct workflow is:
#   - store images losslessly as uint8 WEBP
#   - in training loader: img = img.astype(np.float32) / 255.0
# This achieves exact [0,1] normalization for training without destroying information on disk.

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd


TARGET_W = 2048
TARGET_H = 1500


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def imread_bgr_unicode(path: Path) -> np.ndarray:
    """Robust read for Windows paths (unicode-safe). Returns BGR uint8."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def imwrite_webp_lossless_unicode(path_no_ext: Path, img_bgr_u8: np.ndarray) -> Path:
    """
    Write lossless WEBP if supported. Falls back to quality=100.
    Uses imencode + tofile for unicode-safe writing.
    """
    ensure_dir(path_no_ext.parent)
    out_path = path_no_ext.with_suffix(".webp")

    params = []
    # Try to force true lossless if available in this OpenCV build.
    if hasattr(cv2, "IMWRITE_WEBP_LOSSLESS"):
        params += [cv2.IMWRITE_WEBP_LOSSLESS, 1]
    # Keep quality high as well (even though lossless ignores it).
    params += [cv2.IMWRITE_WEBP_QUALITY, 100]

    ok, buf = cv2.imencode(".webp", img_bgr_u8, params)
    if not ok:
        raise RuntimeError(f"Failed to encode WEBP: {out_path}")

    buf.tofile(str(out_path))
    return out_path


def resize_to_target(img: np.ndarray, target_w: int, target_h: int) -> Tuple[np.ndarray, float, float]:
    """Resize to exact target size. Returns resized image and scale factors (sx, sy)."""
    h, w = img.shape[:2]
    if (w, h) == (target_w, target_h):
        return img, 1.0, 1.0
    sx = target_w / float(w)
    sy = target_h / float(h)
    resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
    return resized, sx, sy


def parse_label_file(path: Path) -> List[Tuple[int, float, float, float, float]]:
    """
    Parse txt with lines:
      class_id x_min y_min x_max y_max
    Values are absolute pixels (floats allowed).
    Ignores empty/comment lines.
    """
    if not path.exists():
        raise FileNotFoundError(f"Label file not found: {path}")

    items: List[Tuple[int, float, float, float, float]] = []
    txt = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for ln in txt:
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        parts = ln.split()
        if len(parts) < 5:
            raise ValueError(f"Bad label line in {path}: {ln!r}")
        cls = int(float(parts[0]))
        x1 = float(parts[1]); y1 = float(parts[2]); x2 = float(parts[3]); y2 = float(parts[4])
        items.append((cls, x1, y1, x2, y2))
    return items


def clamp_box(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> Optional[Tuple[float, float, float, float]]:
    """
    Clamp to image bounds, ensure proper ordering.
    Drops boxes that become invalid or near-empty.
    """
    # order
    xa, xb = (x1, x2) if x1 <= x2 else (x2, x1)
    ya, yb = (y1, y2) if y1 <= y2 else (y2, y1)

    # clamp
    xa = max(0.0, min(float(w - 1), xa))
    xb = max(0.0, min(float(w - 1), xb))
    ya = max(0.0, min(float(h - 1), ya))
    yb = max(0.0, min(float(h - 1), yb))

    # must have area
    if xb - xa < 1.0 or yb - ya < 1.0:
        return None
    return xa, ya, xb, yb


def rescale_boxes(
    boxes: List[Tuple[int, float, float, float, float]],
    sx: float,
    sy: float,
    target_w: int,
    target_h: int
) -> List[Tuple[int, float, float, float, float]]:
    out: List[Tuple[int, float, float, float, float]] = []
    for cls, x1, y1, x2, y2 in boxes:
        x1s = x1 * sx
        x2s = x2 * sx
        y1s = y1 * sy
        y2s = y2 * sy
        clamped = clamp_box(x1s, y1s, x2s, y2s, target_w, target_h)
        if clamped is None:
            continue
        xa, ya, xb, yb = clamped
        out.append((cls, xa, ya, xb, yb))
    return out


def write_label_file(path: Path, boxes: List[Tuple[int, float, float, float, float]]) -> None:
    ensure_dir(path.parent)
    # Keep decimals (your labels are floats). Use a consistent precision.
    lines = [f"{cls} {x1:.3f} {y1:.3f} {x2:.3f} {y2:.3f}" for cls, x1, y1, x2, y2 in boxes]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifest.csv", help="Relative to data/processed/, e.g. manifest.csv")
    ap.add_argument("--out_dirname", default="dataset_2048x1500_webp", help="Folder name under data/processed/")
    ap.add_argument("--overwrite", action="store_true", help="Rewrite existing WEBP/label outputs.")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    manifest_path = root / "data" / "processed" / args.manifest
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)
    required = {"stem", "og_path", "red_path", "label_path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"manifest.csv missing columns: {sorted(missing)}")

    base_out = root / "data" / "processed" / args.out_dirname
    out_og = base_out / "images_og"
    out_red = base_out / "images_red"
    out_lbl = base_out / "labels"

    ensure_dir(out_og); ensure_dir(out_red); ensure_dir(out_lbl)

    rows = []
    dropped_boxes_total = 0

    for i, r in df.iterrows():
        stem = str(r["stem"])
        og_path = Path(r["og_path"])
        red_path = Path(r["red_path"])
        label_path = Path(r["label_path"])

        og_out = (out_og / stem).with_suffix(".webp")
        red_out = (out_red / stem).with_suffix(".webp")
        lbl_out = out_lbl / f"{stem}.txt"

        if (not args.overwrite) and og_out.exists() and red_out.exists() and lbl_out.exists():
            existing_boxes = parse_label_file(lbl_out)
            rows.append({
                "stem": stem,
                "og_webp": str(og_out),
                "red_webp": str(red_out),
                "label_path": str(lbl_out),
                "width": TARGET_W,
                "height": TARGET_H,
                "n_boxes": int(len(existing_boxes)),
                "normalize": "img_float = img_uint8/255.0",
            })
            continue

        og = imread_bgr_unicode(og_path)
        red = imread_bgr_unicode(red_path)

        # Resize images to target; use OG scale for labels (OG + RED should share geometry)
        og_rs, sx, sy = resize_to_target(og, TARGET_W, TARGET_H)
        red_rs, sx_r, sy_r = resize_to_target(red, TARGET_W, TARGET_H)

        # If OG and RED original sizes differ (shouldn't), it still proceeds, but labels follow OG.

        boxes = parse_label_file(label_path)
        boxes_rs = rescale_boxes(boxes, sx, sy, TARGET_W, TARGET_H)
        dropped_boxes_total += (len(boxes) - len(boxes_rs))

        og_out = imwrite_webp_lossless_unicode(out_og / stem, og_rs)
        red_out = imwrite_webp_lossless_unicode(out_red / stem, red_rs)
        write_label_file(lbl_out, boxes_rs)

        rows.append({
            "stem": stem,
            "og_webp": str(og_out),
            "red_webp": str(red_out),
            "label_path": str(lbl_out),
            "width": TARGET_W,
            "height": TARGET_H,
            "n_boxes": int(len(boxes_rs)),
            # training note:
            "normalize": "img_float = img_uint8/255.0",
        })

        if (i + 1) % 200 == 0:
            print(f"[{i+1}/{len(df)}] processed...")

    out_index = base_out / "index.csv"
    pd.DataFrame(rows).to_csv(out_index, index=False, encoding="utf-8")

    print("Wrote dataset to:", base_out)
    print("Index:", out_index)
    if dropped_boxes_total > 0:
        print("Dropped boxes (became invalid after clamp/resize):", dropped_boxes_total)
    print("Done.")


if __name__ == "__main__":
    main()
