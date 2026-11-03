"""Parse Pascal-VOC XML annotations (the format SSDD ships in)."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class VocAnnotation:
    filename: str
    width: int
    height: int
    boxes: np.ndarray  # (N, 4) float, xyxy in pixels
    names: list[str] = field(default_factory=list)


def parse_voc_xml(path: str | Path) -> VocAnnotation:
    """Read one VOC XML file. Width/height may be 0 if the XML omits them."""
    root = ET.parse(path).getroot()
    filename = (root.findtext("filename") or Path(path).with_suffix(".jpg").name).strip()
    size = root.find("size")
    width = int(float(size.findtext("width", "0"))) if size is not None else 0
    height = int(float(size.findtext("height", "0"))) if size is not None else 0

    boxes, names = [], []
    for obj in root.iter("object"):
        bnd = obj.find("bndbox")
        if bnd is None:  # e.g. rotated-box-only objects
            continue
        boxes.append([float(bnd.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")])
        names.append((obj.findtext("name") or "").strip())

    arr = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    return VocAnnotation(filename=filename, width=width, height=height, boxes=arr, names=names)
