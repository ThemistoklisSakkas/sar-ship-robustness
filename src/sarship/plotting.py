"""Training curves and prediction grids."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

# Ultralytics results.csv column names (detection task)
TRAIN_LOSSES = ("train/box_loss", "train/cls_loss", "train/dfl_loss")
VAL_LOSSES = ("val/box_loss", "val/cls_loss", "val/dfl_loss")
MAP50, MAP50_95 = "metrics/mAP50(B)", "metrics/mAP50-95(B)"
PRECISION, RECALL = "metrics/precision(B)", "metrics/recall(B)"

# Colours (BGR) for the prediction grids
GT_COLOUR, TP_COLOUR, FP_COLOUR, FN_COLOUR = (0, 200, 0), (255, 140, 0), (0, 0, 255), (0, 220, 255)


def read_results_csv(path: Path) -> dict[str, np.ndarray]:
    """Columns of an Ultralytics results.csv as float arrays (header whitespace stripped)."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    header = [h.strip() for h in rows[0]]
    data = np.array([[float(v) for v in r] for r in rows[1:] if r], dtype=np.float64)
    return {name: data[:, i] for i, name in enumerate(header)}


def summarize_training(cols: dict[str, np.ndarray]) -> dict:
    """Key numbers for RESULTS.md and for diagnosing over/underfitting."""
    epochs = cols["epoch"].astype(int)
    best = int(np.argmax(cols[MAP50_95]))
    val_box = cols["val/box_loss"]
    return {
        "epochs_run": int(len(epochs)),
        "best_epoch": int(epochs[best]),
        "best_val_map50": float(cols[MAP50][best]),
        "best_val_map50_95": float(cols[MAP50_95][best]),
        "final_train_box_loss": float(cols["train/box_loss"][-1]),
        "final_val_box_loss": float(val_box[-1]),
        "min_val_box_loss_epoch": int(epochs[int(np.argmin(val_box))]),
    }


def plot_training_curves(cols: dict[str, np.ndarray], out_path: Path) -> Path:
    """Three panels: train vs val loss (sum of components), mAP, precision/recall."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = cols["epoch"]
    train_loss = sum(cols[k] for k in TRAIN_LOSSES)
    val_loss = sum(cols[k] for k in VAL_LOSSES)
    best = epochs[int(np.argmax(cols[MAP50_95]))]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].plot(epochs, train_loss, label="train")
    axes[0].plot(epochs, val_loss, label="val")
    axes[0].set_title("total loss (box + cls + dfl)")
    axes[1].plot(epochs, cols[MAP50], label="mAP@0.5")
    axes[1].plot(epochs, cols[MAP50_95], label="mAP@0.5:0.95")
    axes[1].set_title("validation mAP")
    axes[2].plot(epochs, cols[PRECISION], label="precision")
    axes[2].plot(epochs, cols[RECALL], label="recall")
    axes[2].set_title("validation precision / recall")
    for ax in axes:
        ax.axvline(best, color="k", ls="--", lw=1, label=f"best epoch {int(best)}")
        ax.set_xlabel("epoch")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def draw_prediction_overlay(item: dict) -> np.ndarray:
    """GT boxes thin green; predictions TP blue / FP red; missed GT (FN) thick yellow."""
    image = cv2.imread(str(item["image_path"]), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(item["image_path"])
    for box, found in zip(item["gt"], item["matched"], strict=True):
        colour, width = (GT_COLOUR, 1) if found else (FN_COLOUR, 3)
        x1, y1, x2, y2 = np.round(box).astype(int)
        cv2.rectangle(image, (x1, y1), (x2, y2), colour, width)
    for box, is_tp in zip(item["pred"], item["tp"], strict=True):
        x1, y1, x2, y2 = np.round(box).astype(int)
        cv2.rectangle(image, (x1, y1), (x2, y2), TP_COLOUR if is_tp else FP_COLOUR, 2)
    return image


def save_prediction_grid(items: list[dict], out_path: Path, title: str, cols: int = 4) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = max(1, int(np.ceil(len(items) / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, item in zip(axes.flat, items, strict=False):
        ax.imshow(cv2.cvtColor(draw_prediction_overlay(item), cv2.COLOR_BGR2RGB))
        ax.set_title(f"{Path(item['image_path']).stem}  TP {item['n_tp']} FP {item['n_fp']} "
                     f"FN {item['n_fn']}", fontsize=9)
    fig.suptitle(f"{title}\nblue = correct, red = false alarm, yellow = missed ship, "
                 "green = ground truth")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def _crop_window(box: np.ndarray, width: int, height: int, min_side: int = 80) -> tuple:
    """Square window around a box, three times its size, clipped to the image."""
    x1, y1, x2, y2 = box
    side = max(min_side, 3 * max(x2 - x1, y2 - y1))
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    left = int(np.clip(cx - side / 2, 0, max(0, width - side)))
    top = int(np.clip(cy - side / 2, 0, max(0, height - side)))
    return left, top, int(min(width, left + side)), int(min(height, top + side))


def save_crop_gallery(crops: list[dict], out_path: Path, title: str, cols: int = 6) -> Path:
    """Zoomed crops around single boxes. Each crop: image_path, box, gt, label, colour (BGR)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = max(1, int(np.ceil(len(crops) / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.9 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, c in zip(axes.flat, crops, strict=False):
        image = cv2.imread(str(c["image_path"]), cv2.IMREAD_COLOR)
        h, w = image.shape[:2]
        left, top, right, bottom = _crop_window(np.asarray(c["box"], dtype=float), w, h)
        for g in np.asarray(c.get("gt", np.zeros((0, 4)))).reshape(-1, 4):
            gx1, gy1, gx2, gy2 = np.round(g).astype(int)
            cv2.rectangle(image, (gx1, gy1), (gx2, gy2), GT_COLOUR, 1)
        x1, y1, x2, y2 = np.round(c["box"]).astype(int)
        cv2.rectangle(image, (x1, y1), (x2, y2), c["colour"], 2)
        ax.imshow(cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_BGR2RGB))
        ax.set_title(c["label"], fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def plot_recall_by_size(
    recall: dict[str, dict[str, int]], sizes: list[str], out_path: Path
) -> Path:
    """Grouped bars: recall per ship size, inshore vs offshore, with ship counts."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(sizes))
    for k, (scene, offset) in enumerate((("inshore", -0.2), ("offshore", 0.2))):
        vals, labels = [], []
        for s in sizes:
            g = recall.get(f"{scene}/{s}", {"found": 0, "total": 0})
            vals.append(g["found"] / g["total"] if g["total"] else 0.0)
            labels.append(f"{g['found']}/{g['total']}")
        bars = ax.bar(x + offset, vals, width=0.4, label=scene, color=f"C{k}")
        for bar, label in zip(bars, labels, strict=True):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005, label,
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, sizes)
    ax.set_ylim(min(0.6, ax.get_ylim()[0]), 1.05)
    ax.set_ylabel("recall at the operating point")
    ax.set_title("Ships found, by size and scene")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path
