import random
from pathlib import Path

import cv2
import pandas as pd

ROOT = Path(r"D:\intezet\Pico_algae")

OG_DIR = ROOT / "data" / "raw" / "images_og"
LABEL_DIR = ROOT / "data" / "processed" / "labels_abs"
MANIFEST = ROOT / "data" / "processed" / "manifest.csv"
OUT_DIR = ROOT / "reports" / "debug_overlays"

OUT_DIR.mkdir(parents=True, exist_ok=True)


def read_labels(path):
    boxes = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls = int(parts[0])
            x1, y1, x2, y2 = map(float, parts[1:])
            boxes.append((cls, x1, y1, x2, y2))
    return boxes


df = pd.read_csv(MANIFEST)

# Pick 5 random images
samples = df.sample(min(5, len(df)), random_state=42)

for _, row in samples.iterrows():
    stem = row["stem"]
    img_path = OG_DIR / f"{stem}_og.png"
    label_path = LABEL_DIR / f"{stem}_combined.txt"

    img = cv2.imread(str(img_path))
    boxes = read_labels(label_path)

    for cls, x1, y1, x2, y2 in boxes:
        color = (0, 255, 0)
        if cls == 3:  # colony
            color = (0, 0, 255)

        cv2.rectangle(
            img,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            color,
            2,
        )

    out_path = OUT_DIR / f"{stem}_overlay.png"
    cv2.imwrite(str(out_path), img)

print("Overlay images saved to:", OUT_DIR)