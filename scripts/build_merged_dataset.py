"""Merge the original annotated dataset with a new training_export folder into
one self-contained dataset, and (re)generate its index.csv + frozen split.

Two phases, so the same tool works locally and on the training box (HPC):

  1. Assemble (default): copy og / red / label files from --old-dir and
     --new-dir into --out-dir/{images_og,images_red,labels}, dedupe by stem,
     drop blank/malformed label lines, then write index.csv + split.csv.

  2. Index-only (--index-only): do NOT copy anything; just re-scan an already
     assembled --out-dir and rewrite index.csv (+ split.csv) with paths rooted
     at --root. Run this on the HPC after uploading the merged folder, e.g.
         python scripts/build_merged_dataset.py --index-only \
             --out-dir /workspace/pico-algae/data/processed/dataset_merged \
             --root  /workspace/pico-algae/data/processed/dataset_merged

The index columns match what the 3ch and 6ch loaders need
(stem, og_webp, red_webp, label_path) plus width/height/n_boxes/split for
sanity. Paths use forward slashes and are absolute under --root.

Examples
--------
# Local assemble (default old/new/out paths):
python scripts/build_merged_dataset.py

# Explicit:
python scripts/build_merged_dataset.py \
    --old-dir "data/processed/dataset_2048x1500_webp" \
    --new-dir "Pico-Algae Captures/training_export" \
    --out-dir "data/processed/dataset_merged"
"""
from __future__ import annotations

import argparse
import csv
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from PIL import Image
except Exception:  # PIL is optional; width/height are left blank without it
    Image = None

IMG_EXTS = (".webp", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
CLASS_NAMES = {0: "EUK", 1: "FE", 2: "FC", 3: "colony"}


def find_image(dir_: Path, stem: str) -> Optional[Path]:
    """Return the first image file matching `stem` in `dir_`, or None."""
    for ext in IMG_EXTS:
        p = dir_ / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def clean_label_lines(src: Path) -> Tuple[List[str], int]:
    """Read a label .txt, keep only well-formed `cls x1 y1 x2 y2` lines.

    Returns (kept_lines, n_dropped).
    """
    kept: List[str] = []
    dropped = 0
    for ln in src.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = ln.split()
        if len(parts) < 5:
            if ln.strip():
                dropped += 1
            continue
        try:
            cls = int(float(parts[0]))
            x1, y1, x2, y2 = map(float, parts[1:5])
        except ValueError:
            dropped += 1
            continue
        kept.append(f"{cls} {x1:g} {y1:g} {x2:g} {y2:g}")
    return kept, dropped


def label_class_counts(lines: List[str]) -> Counter:
    c: Counter = Counter()
    for ln in lines:
        c[int(float(ln.split()[0]))] += 1
    return c


def collect_stems(old_dir: Path, new_dir: Path) -> List[Tuple[str, Path, Path]]:
    """List (source_name, images_root, labels_root) sub-roots to scan.

    Both datasets share the same {images_og,images_red,labels} layout.
    """
    return [("old", old_dir, old_dir), ("new", new_dir, new_dir)]


def image_size(p: Path) -> Tuple[Optional[int], Optional[int]]:
    if Image is None:
        return None, None
    try:
        with Image.open(p) as im:
            return im.size  # (w, h)
    except Exception:
        return None, None


def deterministic_split(stems: List[str], val_frac: float, seed: int) -> Dict[str, str]:
    """Assign each stem to 'train' or 'val' deterministically (stable across
    machines and runs — based on a hash of the stem, not list order)."""
    import hashlib

    split: Dict[str, str] = {}
    for s in stems:
        h = hashlib.sha1(f"{seed}:{s}".encode("utf-8")).hexdigest()
        frac = int(h[:8], 16) / 0xFFFFFFFF
        split[s] = "val" if frac < val_frac else "train"
    return split


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old-dir", default="data/processed/dataset_2048x1500_webp")
    ap.add_argument("--new-dir", default="Pico-Algae Captures/training_export")
    ap.add_argument("--out-dir", default="data/processed/dataset_merged")
    ap.add_argument("--root", default=None,
                    help="Path prefix written into index.csv (default: absolute --out-dir). "
                         "Set this to the dataset's path ON THE TRAINING MACHINE.")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--index-only", action="store_true",
                    help="Skip copying; just rebuild index.csv/split.csv from --out-dir.")
    ap.add_argument("--symlink", action="store_true",
                    help="Symlink instead of copy (local only; do not use for a folder you will upload).")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    og_out = out_dir / "images_og"
    red_out = out_dir / "images_red"
    lbl_out = out_dir / "labels"
    for d in (og_out, red_out, lbl_out):
        d.mkdir(parents=True, exist_ok=True)

    root = args.root if args.root is not None else str(out_dir.resolve())
    root = root.replace("\\", "/").rstrip("/")

    # ---- Phase 1: assemble (copy) -------------------------------------------
    total_dropped = 0
    collisions: List[str] = []
    src_of: Dict[str, str] = {}

    if not args.index_only:
        seen: Dict[str, str] = {}
        for src_name, img_root, lbl_root in collect_stems(Path(args.old_dir), Path(args.new_dir)):
            labels_dir = Path(lbl_root) / "labels"
            og_dir = Path(img_root) / "images_og"
            red_dir = Path(img_root) / "images_red"
            if not labels_dir.is_dir():
                print(f"[warn] no labels/ under {lbl_root} — skipping this source")
                continue
            for lbl in sorted(labels_dir.glob("*.txt")):
                stem = lbl.stem
                og = find_image(og_dir, stem)
                if og is None:
                    print(f"[warn] {src_name}:{stem} has a label but no og image — skipped")
                    continue
                if stem in seen:
                    collisions.append(f"{stem} (in {seen[stem]} and {src_name})")
                    continue
                seen[stem] = src_name
                src_of[stem] = src_name

                lines, dropped = clean_label_lines(lbl)
                total_dropped += dropped
                (lbl_out / f"{stem}.txt").write_text(
                    "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

                _copy(og, og_out / og.name, args.symlink)
                red = find_image(red_dir, stem)
                if red is not None:
                    _copy(red, red_out / red.name, args.symlink)

        if collisions:
            print(f"[warn] {len(collisions)} stem collision(s) skipped:")
            for c in collisions[:20]:
                print("   ", c)
        print(f"[info] dropped {total_dropped} blank/malformed label line(s)")

    # ---- Phase 2: (re)build index.csv + split.csv ---------------------------
    stems = sorted(p.stem for p in lbl_out.glob("*.txt"))
    if not stems:
        raise SystemExit(f"No labels found under {lbl_out} — nothing to index.")

    split = deterministic_split(stems, args.val_frac, args.seed)

    class_hist: Counter = Counter()
    n_with_red = 0
    rows: List[dict] = []
    for stem in stems:
        og = find_image(og_out, stem)
        if og is None:
            print(f"[warn] {stem}: no og image in {og_out} — skipped from index")
            continue
        red = find_image(red_out, stem)
        if red is not None:
            n_with_red += 1
        lines, _ = clean_label_lines(lbl_out / f"{stem}.txt")
        class_hist.update(label_class_counts(lines))
        w, h = image_size(og)
        rows.append({
            "stem": stem,
            "og_webp": f"{root}/images_og/{og.name}",
            "red_webp": (f"{root}/images_red/{red.name}" if red is not None else ""),
            "label_path": f"{root}/labels/{stem}.txt",
            "width": w if w is not None else "",
            "height": h if h is not None else "",
            "n_boxes": len(lines),
            "split": split[stem],
        })

    index_csv = out_dir / "index.csv"
    with index_csv.open("w", newline="", encoding="utf-8") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(rows)

    with (out_dir / "split.csv").open("w", newline="", encoding="utf-8") as f:
        wtr = csv.writer(f)
        wtr.writerow(["stem", "split"])
        for stem in stems:
            wtr.writerow([stem, split[stem]])

    # ---- Report --------------------------------------------------------------
    n_train = sum(1 for s in stems if split[s] == "train")
    n_val = len(stems) - n_train
    print("\n=== merged dataset summary ===")
    print(f"out-dir      : {out_dir}")
    print(f"index root   : {root}")
    print(f"images       : {len(rows)}  (with red pair: {n_with_red})")
    print(f"split        : train {n_train} / val {n_val}  (val_frac={args.val_frac}, seed={args.seed})")
    print(f"total boxes  : {sum(class_hist.values())}")
    for cls in sorted(class_hist):
        print(f"   {cls} {CLASS_NAMES.get(cls, '?'):7s}: {class_hist[cls]}")
    if src_of:
        by_src = Counter(src_of.values())
        print(f"sources      : " + ", ".join(f"{k}={v}" for k, v in by_src.items()))
    print(f"wrote        : {index_csv.name}, split.csv")


def _copy(src: Path, dst: Path, symlink: bool) -> None:
    if dst.exists():
        return
    if symlink:
        try:
            dst.symlink_to(src.resolve())
            return
        except OSError:
            pass  # fall back to copy (e.g. no symlink privilege on Windows)
    shutil.copy2(src, dst)


if __name__ == "__main__":
    main()
