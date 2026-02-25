# eda.py
# Reads:
#   data/processed/manifest.csv
# Writes:
#   reports/eda/*.png
#   reports/eda/summary.md
#
# Run:
#   python eda.py

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class Paths:
    root: Path
    processed_dir: Path
    reports_dir: Path
    eda_dir: Path

    @staticmethod
    def from_root(root: Path) -> "Paths":
        return Paths(
            root=root,
            processed_dir=root / "data" / "processed",
            reports_dir=root / "reports",
            eda_dir=root / "reports" / "eda",
        )


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_abs_label_file(txt_path: Path) -> List[Tuple[int, float, float, float, float]]:
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


def explode_boxes(df_manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df_manifest.iterrows():
        stem = r["stem"]
        w_img = int(r["width"])
        h_img = int(r["height"])
        label_path = Path(r["label_path"])
        boxes = read_abs_label_file(label_path)
        for cls, x1, y1, x2, y2 in boxes:
            bw = max(0.0, x2 - x1)
            bh = max(0.0, y2 - y1)
            rows.append(
                {
                    "stem": stem,
                    "cls": int(cls),
                    "bw": bw,
                    "bh": bh,
                    "area": bw * bh,
                    "img_w": w_img,
                    "img_h": h_img,
                }
            )
    return pd.DataFrame(rows)


def save_hist(series, title: str, out_path: Path, bins: int = 50) -> None:
    plt.figure()
    plt.hist(series.dropna().values, bins=bins)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def make_eda(paths: Paths, df_manifest: pd.DataFrame) -> None:
    ensure_dir(paths.eda_dir)
    df_boxes = explode_boxes(df_manifest)

    if df_boxes.empty:
        (paths.eda_dir / "summary.md").write_text(
            "# EDA Summary\n\nNo boxes found. Check label conversion.\n", encoding="utf-8"
        )
        return

    save_hist(df_boxes["bw"], "BBox width (px) - all", paths.eda_dir / "bbox_width_all.png")
    save_hist(df_boxes["bh"], "BBox height (px) - all", paths.eda_dir / "bbox_height_all.png")
    save_hist(df_boxes["area"], "BBox area (px^2) - all", paths.eda_dir / "bbox_area_all.png")

    save_hist(df_manifest["n_boxes"], "Objects per image", paths.eda_dir / "objects_per_image.png", bins=30)

    cls_ids = sorted(df_boxes["cls"].unique().tolist())
    per_cls_lines = []

    def pct(s: pd.Series, p: float) -> float:
        return float(s.quantile(p))

    for cls in cls_ids:
        d = df_boxes[df_boxes["cls"] == cls]
        if d.empty:
            continue
        save_hist(d["bw"], f"BBox width (px) - cls {cls}", paths.eda_dir / f"bbox_width_cls_{cls}.png")
        save_hist(d["bh"], f"BBox height (px) - cls {cls}", paths.eda_dir / f"bbox_height_cls_{cls}.png")
        save_hist(d["area"], f"BBox area (px^2) - cls {cls}", paths.eda_dir / f"bbox_area_cls_{cls}.png")

        per_cls_lines.append(
            f"## Class {cls}\n"
            f"- n: {len(d)}\n"
            f"- bw px: p01={pct(d['bw'],0.01):.2f}, p05={pct(d['bw'],0.05):.2f}, "
            f"p50={pct(d['bw'],0.50):.2f}, p95={pct(d['bw'],0.95):.2f}, p99={pct(d['bw'],0.99):.2f}\n"
            f"- bh px: p01={pct(d['bh'],0.01):.2f}, p05={pct(d['bh'],0.05):.2f}, "
            f"p50={pct(d['bh'],0.50):.2f}, p95={pct(d['bh'],0.95):.2f}, p99={pct(d['bh'],0.99):.2f}\n"
            f"- area:  p01={pct(d['area'],0.01):.2f}, p50={pct(d['area'],0.50):.2f}, p99={pct(d['area'],0.99):.2f}\n"
        )

    densest = df_manifest.sort_values("n_boxes", ascending=False).head(15)[["stem", "n_boxes"]]

    summary_lines = [
        "# EDA Summary",
        "",
        f"- Images in manifest: **{len(df_manifest)}**",
        f"- Total boxes: **{len(df_boxes)}**",
        "",
        "## Objects per image",
        f"- min: {df_manifest['n_boxes'].min()}",
        f"- median: {df_manifest['n_boxes'].median()}",
        f"- mean: {df_manifest['n_boxes'].mean():.2f}",
        f"- max: {df_manifest['n_boxes'].max()}",
        "",
        "## BBox size (all classes)",
        f"- bw (px): p01={df_boxes['bw'].quantile(0.01):.2f}, p05={df_boxes['bw'].quantile(0.05):.2f}, "
        f"p50={df_boxes['bw'].quantile(0.50):.2f}, p95={df_boxes['bw'].quantile(0.95):.2f}, p99={df_boxes['bw'].quantile(0.99):.2f}",
        f"- bh (px): p01={df_boxes['bh'].quantile(0.01):.2f}, p05={df_boxes['bh'].quantile(0.05):.2f}, "
        f"p50={df_boxes['bh'].quantile(0.50):.2f}, p95={df_boxes['bh'].quantile(0.95):.2f}, p99={df_boxes['bh'].quantile(0.99):.2f}",
        "",
        "## Densest images (top 15)",
        "",
        densest.to_markdown(index=False),
        "",
        *per_cls_lines,
    ]

    (paths.eda_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifest.csv", help="Manifest filename under data/processed/")
    args = ap.parse_args()

    paths = Paths.from_root(Path(__file__).resolve().parent.parent.parent)
    manifest_path = paths.processed_dir / args.manifest

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}. Run make_manifest.py first.")

    df_manifest = pd.read_csv(manifest_path)
    if df_manifest.empty:
        print("Manifest is empty. Check sanity_check.txt and label/image pairing.")
        return

    make_eda(paths, df_manifest)
    print(f"Wrote EDA to: {paths.eda_dir}")


if __name__ == "__main__":
    main()