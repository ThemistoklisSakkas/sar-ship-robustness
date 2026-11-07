"""Bounding-box format conversion and VOC -> YOLO dataset export."""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2
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


def colour_fraction(image: np.ndarray, threshold: int = 16) -> float:
    """Fraction of pixels whose channels differ by more than `threshold` grey levels.

    A grey image stored as RGB has R == G == B (up to small JPEG noise), so the
    fraction is ~0. Pseudo-colour or colour composites give a clearly larger value.
    """
    if image.ndim == 2 or image.shape[2] == 1:
        return 0.0
    rgb = image[..., :3].astype(np.int16)
    spread = rgb.max(axis=2) - rgb.min(axis=2)
    return float((spread > threshold).mean())


def export_image(
    src: Path, dst_dir: Path, to_grayscale: bool, threshold: int, min_fraction: float
) -> tuple[Path, float, bool]:
    """Copy one image; colour images become single-channel PNG if `to_grayscale`.

    PNG is lossless, so the conversion adds no second round of JPEG artefacts.
    Returns (written path, colour fraction, is_colour).
    """
    image = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(src)
    fraction = colour_fraction(image, threshold)
    is_colour = fraction > min_fraction
    if is_colour and to_grayscale:
        dst = dst_dir / f"{src.stem}.png"
        cv2.imwrite(str(dst), cv2.cvtColor(image[..., :3], cv2.COLOR_BGR2GRAY))
    else:
        dst = dst_dir / src.name
        shutil.copy2(src, dst)
    return dst, fraction, is_colour


def export_split(
    pairs: list[tuple[Path, Path]],
    out_root: Path,
    split: str,
    min_box_size: float = 1.0,
    to_grayscale: bool = False,
    colour_threshold: int = 16,
    colour_min_fraction: float = 0.001,
) -> list[dict]:
    """Copy images and write YOLO labels for one split (single class 'ship' = 0).

    `pairs` is a list of (image_path, xml_path). Returns one record per image
    (used later for statistics). Degenerate boxes are dropped and counted;
    colour images are detected and, if requested, converted to grayscale.
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

        _, fraction, is_colour = export_image(
            image_path, img_dir, to_grayscale, colour_threshold, colour_min_fraction
        )
        (lbl_dir / f"{image_path.stem}.txt").write_text(format_yolo_lines(classes, yolo))
        records.append(
            {
                "split": split,
                "stem": image_path.stem,
                "width": ann.width,
                "height": ann.height,
                "boxes": boxes,
                "dropped": int((~keep).sum()),
                "source": image_path,
                "colour_fraction": fraction,
                "is_colour": is_colour,
            }
        )
    return records
