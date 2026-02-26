# collect_annotated_pairs.py
# Given CVAT "for images 1.1" XML batches where image names look like:
#   Image_5444_combined.png
# we extract the stem "Image_5444" and then copy/move:
#   Image_5444_og.png  and  Image_5444_red.png
# from a messy source folder into your project folders:
#   data/raw/images_og/
#   data/raw/images_red/
#
# Works with one XML file OR a folder of XML batches.

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Set, List


def collect_xml_files(path: Path) -> List[Path]:
    if path.is_dir():
        xmls = sorted(path.glob("*.xml"))
        if not xmls:
            raise FileNotFoundError(f"No .xml files found in: {path}")
        return xmls
    if not path.exists():
        raise FileNotFoundError(path)
    return [path]


def stems_from_cvat_combined(xml_file: Path) -> Set[str]:
    """
    Extracts stems like 'Image_5444' from CVAT <image name="Image_5444_combined.png" ...>
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    stems: Set[str] = set()
    for img in root.findall("image"):
        name = img.get("name") or ""
        p = Path(name)
        # remove extension
        base = p.stem  # e.g. "Image_5444_combined"
        if base.endswith("_combined"):
            stems.add(base[:-len("_combined")])  # "Image_5444"
        else:
            # If you ever have other naming, you can handle it here.
            # For now we assume _combined.
            stems.add(base)
    return stems


def copy_or_move(src: Path, dst: Path, do_move: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if do_move:
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(str(src), str(dst))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True, help="Path to CVAT XML file OR folder of XML files.")
    ap.add_argument("--src-images", required=True, help="Folder containing all images (messy pool).")
    ap.add_argument("--dst-og", required=True, help="Destination folder for *_og.png images.")
    ap.add_argument("--dst-red", required=True, help="Destination folder for *_red.png images.")
    ap.add_argument("--ext", default=".png", help="Image extension, e.g. .png or .jpg (default: .png)")
    ap.add_argument("--move", action="store_true", help="Move instead of copy.")
    ap.add_argument("--dry-run", action="store_true", help="Print actions without copying/moving.")
    args = ap.parse_args()

    xml_path = Path(args.xml)
    src_dir = Path(args.src_images)
    dst_og = Path(args.dst_og)
    dst_red = Path(args.dst_red)
    ext = args.ext if args.ext.startswith(".") else f".{args.ext}"

    if not src_dir.exists():
        raise FileNotFoundError(f"Source image folder not found: {src_dir}")

    xml_files = collect_xml_files(xml_path)

    stems: Set[str] = set()
    for xf in xml_files:
        stems |= stems_from_cvat_combined(xf)

    if not stems:
        print("No stems found in XML(s). Nothing to do.")
        return

    missing_og = []
    missing_red = []
    copied = 0

    for stem in sorted(stems):
        og_name = f"{stem}_og{ext}"
        red_name = f"{stem}_red{ext}"

        og_src = src_dir / og_name
        red_src = src_dir / red_name

        og_dst = dst_og / og_name
        red_dst = dst_red / red_name

        if not og_src.exists():
            missing_og.append(og_name)
        if not red_src.exists():
            missing_red.append(red_name)

        # Only act when the file exists
        if og_src.exists():
            if args.dry_run:
                print(f"{'MOVE' if args.move else 'COPY'}  {og_src} -> {og_dst}")
            else:
                copy_or_move(og_src, og_dst, do_move=args.move)
            copied += 1

        if red_src.exists():
            if args.dry_run:
                print(f"{'MOVE' if args.move else 'COPY'}  {red_src} -> {red_dst}")
            else:
                copy_or_move(red_src, red_dst, do_move=args.move)
            copied += 1

    print(f"Done. Stems: {len(stems)} | Files {'moved' if args.move else 'copied'}: {copied}")

    if missing_og:
        print(f"Missing OG images: {len(missing_og)}")
        print("  " + "\n  ".join(missing_og[:30]) + ("\n  ..." if len(missing_og) > 30 else ""))
    if missing_red:
        print(f"Missing RED images: {len(missing_red)}")
        print("  " + "\n  ".join(missing_red[:30]) + ("\n  ..." if len(missing_red) > 30 else ""))


if __name__ == "__main__":
    main()
    
# python src\prep\move_imgs.py --xml "D:\intezet\Pico_algae\data\raw\annotations" --src-images "D:\intezet\Alga_szam\data\collected_images" --dst-og "D:\intezet\Pico_algae\data\raw\images_og" --dst-red "D:\intezet\Pico_algae\data\raw\images_red"
