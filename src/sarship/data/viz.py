"""Visual checks: draw boxes from the *written YOLO labels* (not from the XML).

Drawing from the exported label files verifies the whole conversion chain:
if a box is shifted, scaled or has W/H swapped, it shows up immediately.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from sarship.data.convert import yolo_to_xyxy


def read_yolo_labels(label_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (classes, boxes cx,cy,w,h normalised) from a YOLO .txt file."""
    text = Path(label_path).read_text().strip()
    if not text:
        return np.zeros(0, dtype=int), np.zeros((0, 4))
    arr = np.loadtxt(text.splitlines(), ndmin=2)
    return arr[:, 0].astype(int), arr[:, 1:5]


def draw_boxes(
    image: np.ndarray, boxes_xyxy: np.ndarray, color=(0, 255, 0), thickness: int = 2
) -> np.ndarray:
    """Return a BGR copy of `image` with rectangles drawn (grayscale input is converted)."""
    out = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    for x1, y1, x2, y2 in np.round(boxes_xyxy).astype(int):
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
    return out


def render_label_overlay(image_path: Path, label_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)
    h, w = image.shape[:2]
    _, yolo = read_yolo_labels(label_path)
    return draw_boxes(image, yolo_to_xyxy(yolo, w, h))


def save_overlay_grid(
    image_paths: list[Path], label_root: Path, out_path: Path, cols: int = 4, title: str = ""
) -> Path:
    """Grid of images with their YOLO boxes (from `label_root/<stem>.txt`), saved as PNG."""
    pairs = [(p, label_root / f"{p.stem}.txt") for p in image_paths]
    return save_pairs_grid(pairs, out_path, cols, title)


def save_pairs_grid(
    pairs: list[tuple[Path, Path]], out_path: Path, cols: int = 4, title: str = ""
) -> Path:
    """Grid of (image, YOLO label file) pairs with the boxes drawn, saved as PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = max(1, int(np.ceil(len(pairs) / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (img_path, label_path) in zip(axes.flat, pairs, strict=False):
        overlay = render_label_overlay(img_path, label_path)
        n = len(read_yolo_labels(label_path)[0])
        ax.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{img_path.stem}  ({n} ships)", fontsize=9)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def save_size_histogram(
    areas_by_group: dict[str, np.ndarray], bins_px2: dict[str, list[float]], out_path: Path
) -> Path:
    """Histogram of sqrt(box area) per group, with the size-bin edges marked."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    edges = np.linspace(0, 200, 51)
    for name, areas in areas_by_group.items():
        ax.hist(np.sqrt(areas), bins=edges, alpha=0.55, label=f"{name} (n={len(areas)})")
    for low, _ in list(bins_px2.values())[1:]:
        ax.axvline(np.sqrt(float(low)), color="k", ls="--", lw=1)
    ax.set_xlabel("sqrt(box area) [px]   (dashed: small/medium/large edges)")
    ax.set_ylabel("ships")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path
