from __future__ import annotations

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET


def indent_xml(elem: ET.Element, level: int = 0) -> None:
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for child in elem:
            indent_xml(child, level + 1)
        if not elem[-1].tail or not elem[-1].tail.strip():
            elem[-1].tail = i
    elif level and (not elem.tail or not elem.tail.strip()):
        elem.tail = i


def collect_xml_files(in_dir: Path, out_file: Path) -> list[Path]:
    xml_files = sorted(p for p in in_dir.glob("*.xml") if p.resolve() != out_file.resolve())
    if not xml_files:
        raise FileNotFoundError(f"No .xml files found in: {in_dir}")
    return xml_files


def merge_cvat_xml(xml_files: list[Path]) -> ET.ElementTree:
    first_tree = ET.parse(xml_files[0])
    first_root = first_tree.getroot()

    merged_root = ET.Element("annotations")

    version = first_root.find("version")
    if version is not None:
        merged_root.append(version)

    meta = first_root.find("meta")
    if meta is not None:
        merged_root.append(meta)

    next_image_id = 0
    for xml_path in xml_files:
        root = ET.parse(xml_path).getroot()
        for image in root.findall("image"):
            image.set("id", str(next_image_id))
            merged_root.append(image)
            next_image_id += 1

    return ET.ElementTree(merged_root)


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    default_in_dir = project_root / "data" / "raw" / "annotations"
    default_out_file = default_in_dir / "annotations.xml"

    parser = argparse.ArgumentParser(
        description="Merge all CVAT XML files in a folder into one XML file."
    )
    parser.add_argument(
        "--in-dir",
        type=Path,
        default=default_in_dir,
        help=f"Input directory containing XML files (default: {default_in_dir})",
    )
    parser.add_argument(
        "--out-file",
        type=Path,
        default=default_out_file,
        help=f"Output merged XML file (default: {default_out_file})",
    )
    args = parser.parse_args()

    in_dir = args.in_dir.resolve()
    out_file = args.out_file.resolve()

    if not in_dir.exists() or not in_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {in_dir}")

    xml_files = collect_xml_files(in_dir, out_file)
    merged_tree = merge_cvat_xml(xml_files)
    indent_xml(merged_tree.getroot())

    out_file.parent.mkdir(parents=True, exist_ok=True)
    merged_tree.write(out_file, encoding="utf-8", xml_declaration=True)
    print(f"Merged {len(xml_files)} files into: {out_file}")


if __name__ == "__main__":
    main()
