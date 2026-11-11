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


def plot_robustness_curves(report: dict, cfg: dict, out_path: Path) -> Path:
    """One panel per degradation (mAP vs severity, CI band) plus an overlay of relative mAP."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = report["results"]
    clean = res["clean"]
    degradations = cfg["robustness"]["degradations"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    overlay = axes.flat[5]

    for k, (ax, (name, spec)) in enumerate(zip(axes.flat, degradations.items(), strict=False)):
        ids = ["clean"] + [f"{name}_{float(v):g}" for v in spec["values"]]
        x = np.arange(len(ids))
        b = [res[i]["all"]["map50_95"] for i in ids]
        point = np.array([m["point"] for m in b])
        ax.fill_between(x, [m["low"] for m in b], [m["high"] for m in b], alpha=0.2, color="C0")
        ax.plot(x, point, "o-", color="C0", label="mAP@0.5:0.95 (95% CI)")
        ax.plot(x, [res[i]["all"]["map50"]["point"] for i in ids], "s--", color="C3",
                label="mAP@0.5")
        for scene, colour in (("inshore", "C1"), ("offshore", "C2")):
            ax.plot(x, [res[i][scene]["map50_95"]["point"] for i in ids], ":", color=colour,
                    label=f"{scene} mAP@0.5:0.95")
        ax.set_xticks(x, ["clean"] + [f"{v:g}" for v in spec["values"]])
        ax.set_xlabel(f"{spec['param']} (severity increases to the right)")
        ax.set_title(name)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
        if k == 0:
            ax.legend(fontsize=8, loc="lower left")
        overlay.plot(x, point / clean["all"]["map50_95"]["point"], "o-", label=name)

    overlay.axhline(1 - cfg["robustness"]["relative_drop"], color="k", ls="--", lw=1,
                    label=f"-{int(100 * cfg['robustness']['relative_drop'])}%")
    overlay.set_title("All degradations: mAP@0.5:0.95 relative to clean")
    overlay.set_xlabel("severity level (0 = clean)")
    overlay.set_ylim(0, 1.05)
    overlay.grid(alpha=0.3)
    overlay.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def plot_degradation_examples(items: list[dict], cfg: dict, out_path: Path,
                              crop: int = 192) -> Path:
    """What each severity level looks like: one image chip, rows = degradations."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from sarship.degrade import apply_degradation
    from sarship.robustness import condition_id, image_rng

    offshore = [it for it in items if it["scene"] == "offshore" and len(it["gt"])] or items
    item = max(offshore, key=lambda it: len(it["gt"]))
    img = cv2.imread(str(item["path"]), cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    x1, y1, x2, y2 = item["gt"][0]
    left = int(np.clip((x1 + x2) / 2 - crop / 2, 0, max(0, w - crop)))
    top = int(np.clip((y1 + y2) / 2 - crop / 2, 0, max(0, h - crop)))

    degradations = cfg["robustness"]["degradations"]
    cols = 1 + max(len(s["values"]) for s in degradations.values())
    fig, axes = plt.subplots(len(degradations), cols, figsize=(2.3 * cols, 2.5 * len(degradations)),
                             squeeze=False)
    for row, (name, spec) in zip(axes, degradations.items(), strict=True):
        for ax in row:
            ax.axis("off")
        row[0].imshow(img[top:top + crop, left:left + crop], cmap="gray", vmin=0, vmax=255)
        row[0].set_title(f"{name}: clean", fontsize=9)
        for ax, v in zip(row[1:], spec["values"], strict=False):
            cid = condition_id(name, float(v))
            out = apply_degradation(name, img, float(v),
                                    image_rng(cfg["seed"], cid, item["stem"]))
            ax.imshow(out[top:top + crop, left:left + crop], cmap="gray", vmin=0, vmax=255)
            ax.set_title(f"{spec['param']}={float(v):g}", fontsize=9)
    fig.suptitle(f"Degradation ladder on test image {item['stem']} (crop {crop}x{crop} px)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path
