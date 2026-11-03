"""Bounding-box format conversion and VOC -> YOLO dataset export."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from sarship.data.voc import VocAnnotation, parse_voc_xml


def xyxy_to_yolo(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    """Pixel (xmin, ymin, xmax, ymax) -> normalised (cx, cy, w, h) in [0, 1].

    Boxes are clipped to the image first, so slightly out-of-bounds
    annotations do not produce coordinates outside [0, 1].
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"Image size must be positive, got {width}x{height}")
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4).copy()
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, width)
    b[:, [1, 3]] = b[:, [1, 3]].clip(0, height)
    cx = (b[:, 0] + b[:, 2]) / 2 / width
    cy = (b[:, 1] + b[:, 3]) / 2 / height
    w = (b[:, 2] - b[:, 0]) / width
    h = (b[:, 3] - b[:, 1]) / height
    return np.stack([cx, cy, w, h], axis=1)


def yolo_to_xyxy(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    """Inverse of xyxy_to_yolo (without the clipping)."""
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    x1 = (b[:, 0] - b[:, 2] / 2) * width
    y1 = (b[:, 1] - b[:, 3] / 2) * height
    x2 = (b[:, 0] + b[:, 2] / 2) * width
    y2 = (b[:, 1] + b[:, 3] / 2) * height
    return np.stack([x1, y1, x2, y2], axis=1)


def valid_box_mask(boxes: np.ndarray, min_size: float = 1.0) -> np.ndarray:
    """True for boxes with positive width and height of at least `min_size` pixels."""
    b = np.asarray(boxes).reshape(-1, 4)
    return ((b[:, 2] - b[:, 0]) >= min_size) & ((b[:, 3] - b[:, 1]) >= min_size)


def format_yolo_lines(classes: np.ndarray, yolo_boxes: np.ndarray) -> str:
    lines = [
        f"{int(c)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
        for c, (cx, cy, w, h) in zip(classes, yolo_boxes, strict=True)
    ]
    return "\n".join(lines) + ("\n" if lines else "")


def with_image_size(ann: VocAnnotation, image_path: Path) -> VocAnnotation:
    """Fill in width/height from the image when the XML does not provide them."""
    if ann.width > 0 and ann.height > 0:
        return ann
    with Image.open(image_path) as im:
        ann.width, ann.height = im.size
    return ann


def export_split(
    pairs: list[tuple[Path, Path]],
    out_root: Path,
    split: str,
    min_box_size: float = 1.0,
) -> list[dict]:
    """Copy images and write YOLO labels for one split (single class 'ship' = 0).

    `pairs` is a list of (image_path, xml_path). Returns one record per image
    (used later for statistics). Degenerate boxes are dropped and counted.
    """
    img_dir = out_root / "images" / split
    lbl_dir = out_root / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for image_path, xml_path in pairs:
        ann = with_image_size(parse_voc_xml(xml_path), image_path)
        keep = valid_box_mask(ann.boxes, min_box_size)
        boxes = ann.boxes[keep]
        yolo = xyxy_to_yolo(boxes, ann.width, ann.height)
        classes = np.zeros(len(yolo), dtype=int)

        shutil.copy2(image_path, img_dir / image_path.name)
        (lbl_dir / f"{image_path.stem}.txt").write_text(format_yolo_lines(classes, yolo))
        records.append(
            {
                "split": split,
                "stem": image_path.stem,
                "width": ann.width,
                "height": ann.height,
                "boxes": boxes,
                "dropped": int((~keep).sum()),
            }
        )
    return records
