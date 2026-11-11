"""README hero figure: one SAR chip, clean and heavily degraded, seen by both models.

Usage (Colab, CPU is fine; reads the cached robustness predictions, no model needed):
    python -m sarship.hero --data-root DATA --runs-root RUNS --out DRIVE/report_figures/00_hero.png

The chip is chosen automatically as the test image where the degradation-aware model
recovers the most ships that the baseline loses (ties: more ships). It is an
illustration, so the caption says so; the full-test-set numbers are in RESULTS.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from sarship.analysis.ap import ImageDetections
from sarship.analysis.errors import classify_image
from sarship.analysis.run import load_split
from sarship.config import DEFAULT_CONFIG, load_config
from sarship.degrade import apply_degradation
from sarship.robustness import image_rng


def load_predictions(runs_root: Path, name: str, cid: str) -> dict:
    path = runs_root / name / "robustness" / "predictions" / f"{cid}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `python -m sarship.robustness` for {name}")
    return json.loads(path.read_text())["predictions"]


def outcome(item: dict, pred: dict, cfg: dict) -> dict:
    ex = cfg["eval"]["examples"]
    im = ImageDetections(stem=item["stem"], scene=item["scene"], width=item["width"],
                         height=item["height"], gt=item["gt"],
                         pred=np.asarray(pred["boxes"], dtype=np.float64).reshape(-1, 4),
                         scores=np.asarray(pred["scores"], dtype=np.float64))
    r = classify_image(im, ex["conf"], ex["iou_match"])
    return {"pred": r["pred"], "tp": r["tp"], "matched": r["matched"], "gt": item["gt"],
            "n_tp": int(r["tp"].sum()), "n_fp": len(r["fp"]), "n_fn": len(r["fn"])}


def pick_image(items: list[dict], preds: dict[tuple[str, str], dict], cfg: dict,
               min_ships: int = 3) -> dict:
    """Largest (robust − baseline) ships found under degradation, both perfect when clean."""
    best, best_key = None, None
    for it in items:
        if len(it["gt"]) < min_ships:
            continue
        o = {k: outcome(it, preds[k][it["stem"]], cfg) for k in preds}
        if o[("baseline", "clean")]["n_fn"] or o[("robust", "clean")]["n_fn"]:
            continue
        gain = o[("robust", "degraded")]["n_tp"] - o[("baseline", "degraded")]["n_tp"]
        key = (gain, len(it["gt"]), it["stem"])
        if best_key is None or key > best_key:
            best, best_key = it, key
    if best is None:
        raise ValueError("no test image satisfies the selection criteria")
    return best


def draw(image: np.ndarray, o: dict) -> np.ndarray:
    """Blue = correct, red = false alarm, yellow = missed ship (same colours as phase 2)."""
    from sarship.plotting import FN_COLOUR, FP_COLOUR, TP_COLOUR

    out = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for box, found in zip(o["gt"], o["matched"], strict=True):
        if not found:
            x1, y1, x2, y2 = np.round(box).astype(int)
            cv2.rectangle(out, (x1, y1), (x2, y2), FN_COLOUR, 3)
    for box, ok in zip(o["pred"], o["tp"], strict=True):
        x1, y1, x2, y2 = np.round(box).astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), TP_COLOUR if ok else FP_COLOUR, 2)
    return out


def make_hero(items: list[dict], runs_root: Path, cfg: dict, out_path: Path,
              baseline: str, robust: str, condition: str, min_ships: int = 3) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    name, value = condition.rsplit("_", 1)
    preds = {(who, state): load_predictions(runs_root, run, cid)
             for who, run in (("baseline", baseline), ("robust", robust))
             for state, cid in (("clean", "clean"), ("degraded", condition))}
    item = pick_image(items, preds, cfg, min_ships)
    clean = cv2.imread(str(item["path"]), cv2.IMREAD_GRAYSCALE)
    degraded = apply_degradation(name, clean, float(value),
                                 image_rng(cfg["seed"], condition, item["stem"]))
    param = cfg["robustness"]["degradations"][name]["param"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 11 * clean.shape[0] / clean.shape[1] + 1.2))
    for row, (who, label) in enumerate((("baseline", "Baseline YOLO11n"),
                                        ("robust", "Degradation-aware YOLO11n"))):
        for col, (state, img, title) in enumerate((
                ("clean", clean, "clean image"),
                ("degraded", degraded, f"heavy {name} ({param}={value})"))):
            o = outcome(item, preds[(who, state)][item["stem"]], cfg)
            ax = axes[row, col]
            ax.imshow(cv2.cvtColor(draw(img, o), cv2.COLOR_BGR2RGB))
            ax.set_title(f"{label} · {title}\n{o['n_tp']}/{len(item['gt'])} ships found, "
                         f"{o['n_fp']} false alarm{'s' if o['n_fp'] != 1 else ''}", fontsize=11)
            ax.axis("off")
    fig.suptitle("SAR ship detection: clean vs heavily degraded input\n"
                 "blue = ship found · yellow = ship missed · red = false alarm", fontsize=13)
    note = (f"Illustrative SSDD test image {item['stem']}, chosen automatically to show the "
            "effect. Full-test-set results: RESULTS.md.")
    fig.text(0.5, 0.005, note, ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--baseline", default="baseline_yolo11n")
    parser.add_argument("--robust", default="robust_yolo11n")
    parser.add_argument("--condition", default="speckle_1")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    cfg = load_config(args.config)
    items = load_split(args.data_root / cfg["dataset"]["yolo_dirname"], "test")
    path = make_hero(items, args.runs_root, cfg, args.out, args.baseline, args.robust,
                     args.condition)
    print(f"Hero figure written to {path}")


if __name__ == "__main__":
    main()
