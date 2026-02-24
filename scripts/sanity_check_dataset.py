from __future__ import annotations
import argparse
from pathlib import Path

import cv2
import numpy as np

from src.data.dataset_index import load_index
from src.data.pico_dataset import PicoOgDetectionDataset
from src.inference.visualize import draw_xyxy

# usage python scripts\sanity_check_dataset.py --index_csv data\processed\dataset_2048x1500_webp\index.csv --n 20

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_csv", required=True, type=str)
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()

    df = load_index(Path(args.index_csv))
    ds = PicoOgDetectionDataset(df)

    n = min(args.n, len(ds))
    for i in range(n):
        img_t, tgt = ds[i]
        img_rgb = (img_t.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)

        vis = draw_xyxy(img_rgb, tgt["boxes"].numpy(), tgt["labels"].numpy())
        bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        cv2.imshow("sanity_check_dataset", bgr)

        k = cv2.waitKey(0) & 0xFF
        if k in (27, ord("q")):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()