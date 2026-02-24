# cvat_combined_xml_to_abs_txt.py
# Converts CVAT "for images 1.1" XML annotations made on horizontally concatenated images (OG|RED)
# into per-image ABS pixel label files in single-view (W x H) coordinates.
#
# Output per image:  class_id x_min y_min x_max y_max
#
# Rule:
#   - If box is on left half (OG): keep x as-is
#   - If box is on right half (RED): subtract W_half from xtl/xbr
#   - If box crosses the boundary: drop (safest)
#
# Supports one XML file or a folder of XML batches.

from __future__ import annotations

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Optional


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def collect_xml_files(path: Path) -> List[Path]:
    if path.is_dir():
        xmls = sorted(path.glob("*.xml"))
        if not xmls:
            raise FileNotFoundError(f"No .xml files found in: {path}")
        return xmls
    if not path.exists():
        raise FileNotFoundError(path)
    return [path]


def parse_cvat_xml(xml_path: Path) -> List[dict]:
    """
    Parses CVAT 'CVAT for images 1.1' XML.
    Returns list of:
      {
        "name": str,
        "width": int,
        "height": int,
        "boxes": [(label, xtl, ytl, xbr, ybr), ...]
      }
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    images = []
    for img in root.findall("image"):
        name = img.get("name")
        width = int(img.get("width"))
        height = int(img.get("height"))

        boxes = []
        for box in img.findall("box"):
            label = box.get("label")
            xtl = float(box.get("xtl"))
            ytl = float(box.get("ytl"))
            xbr = float(box.get("xbr"))
            ybr = float(box.get("ybr"))
            boxes.append((label, xtl, ytl, xbr, ybr))

        images.append({"name": name, "width": width, "height": height, "boxes": boxes})

    return images


def parse_label_map_arg(s: str) -> Dict[str, int]:
    """
    Format: "EUK=0,FE=1,FC=2,colony=3"
    """
    mapping: Dict[str, int] = {}
    s = (s or "").strip()
    if not s:
        return mapping
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError("label-map must look like 'name=0,name2=1'")
        k, v = part.split("=", 1)
        mapping[k.strip()] = int(v.strip())
    return mapping


def infer_label_map(images: List[dict]) -> Dict[str, int]:
    labels = sorted({lbl for img in images for (lbl, *_rest) in img["boxes"]})
    return {lbl: i for i, lbl in enumerate(labels)}


def map_box_from_combined_to_single_view(
    xtl: float,
    xbr: float,
    w_half: int,
    policy: str,
) -> Optional[Tuple[float, float]]:
    """
    Maps x-coordinates from combined (2W x H) to single-view (W x H).
    policy:
      - "allow_both": accept boxes on either half, mapping them into [0, W]
      - "left_only": accept only left-half boxes
      - "right_only": accept only right-half boxes
    Returns (xtl_new, xbr_new) in single-view coordinates, or None if dropped.
    """
    # Left half: [0, w_half]
    if xbr <= w_half:
        if policy in ("allow_both", "left_only"):
            return xtl, xbr
        return None

    # Right half: [w_half, 2*w_half]
    if xtl >= w_half:
        if policy in ("allow_both", "right_only"):
            return xtl - w_half, xbr - w_half
        return None

    # Crosses the boundary -> drop (safest)
    return None


def write_abs_labels_per_image(
    images: List[dict],
    out_dir: Path,
    label_map: Dict[str, int],
    combined: bool,
    w_half_override: int,
    half_policy: str,
    drop_boundary_crossers: bool = True,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save label map for reproducibility
    (out_dir / "label_map.txt").write_text(
        "\n".join([f"{k}={v}" for k, v in sorted(label_map.items(), key=lambda x: x[1])]) + "\n",
        encoding="utf-8",
    )

    for img in images:
        name = img["name"]
        W = img["width"]
        H = img["height"]

        if combined:
            # Determine half width
            w_half = w_half_override if w_half_override > 0 else (W // 2)
        else:
            w_half = W  # not used

        lines: List[str] = []

        for (label, xtl, ytl, xbr, ybr) in img["boxes"]:
            if label not in label_map:
                continue
            cls_id = label_map[label]

            # Clamp Y to image bounds
            y1 = clamp(ytl, 0.0, H)
            y2 = clamp(ybr, 0.0, H)
            if y2 <= y1:
                continue

            if combined:
                mapped = map_box_from_combined_to_single_view(xtl, xbr, w_half, half_policy)
                if mapped is None:
                    continue
                x1, x2 = mapped

                # Clamp X to single-view bounds [0, w_half]
                x1 = clamp(x1, 0.0, w_half)
                x2 = clamp(x2, 0.0, w_half)
                if x2 <= x1:
                    continue
            else:
                # Non-combined: keep as is (single view)
                x1 = clamp(xtl, 0.0, W)
                x2 = clamp(xbr, 0.0, W)
                if x2 <= x1:
                    continue

            lines.append(f"{cls_id} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}")

        out_path = out_dir / Path(name).with_suffix(".txt").name
        out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True, help="Path to CVAT XML file OR folder of XML files.")
    ap.add_argument("--out", required=True, help="Output folder for per-image .txt label files.")
    ap.add_argument("--label-map", default="", help="Explicit mapping, e.g. 'EUK=0,FE=1,FC=2,colony=3'")
    ap.add_argument("--combined", action="store_true", help="Annotations were made on OG|RED concatenated images (2W x H).")
    ap.add_argument("--half-width", type=int, default=0, help="Optional OG width W in pixels. If 0, uses combined_width//2.")
    ap.add_argument(
        "--half-policy",
        default="allow_both",
        choices=["allow_both", "left_only", "right_only"],
        help="Which half boxes are accepted when --combined is set.",
    )
    args = ap.parse_args()

    xml_path = Path(args.xml)
    out_dir = Path(args.out)

    xml_files = collect_xml_files(xml_path)
    images: List[dict] = []
    for xf in xml_files:
        images.extend(parse_cvat_xml(xf))

    explicit_map = parse_label_map_arg(args.label_map)
    label_map = explicit_map if explicit_map else infer_label_map(images)

    write_abs_labels_per_image(
        images=images,
        out_dir=out_dir,
        label_map=label_map,
        combined=args.combined,
        w_half_override=args.half_width,
        half_policy=args.half_policy,
    )

    print(f"Done. Parsed images: {len(images)}")
    print(f"Wrote labels to: {out_dir}")
    print(f"Label map saved to: {out_dir / 'label_map.txt'}")


if __name__ == "__main__":
    main()
    
# python cvat_xml_to_txt.py --xml "D:\intezet\Pico_algae\data\raw\annotations\annotations1.xml" --out "D:\intezet\Pico_algae\data\processed\labels_abs" --label-map "EUK=0,FE=1,FC=2,colony=3" --combined --half-policy allow_both
