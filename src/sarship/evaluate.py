"""Evaluate a trained detector: val + held-out test (all / inshore / offshore),
training curves, and good/bad prediction examples.

Usage (Colab):
    python -m sarship.evaluate --weights RUNS/baseline_yolo11n/weights/best.pt \
        --data-root DATA --runs-root RUNS --name baseline_yolo11n
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sarship.analysis.matching import match_detections
from sarship.config import DEFAULT_CONFIG, load_config
from sarship.data.convert import yolo_to_xyxy
from sarship.data.viz import read_yolo_labels
from sarship.plotting import (
    plot_training_curves,
    read_results_csv,
    save_prediction_grid,
    summarize_training,
)

# (row name, dataset yaml, split key inside the yaml)
SUBSETS = [
    ("val", "ssdd.yaml", "val"),
    ("test", "ssdd.yaml", "test"),
    ("test/inshore", "ssdd_test_inshore.yaml", "test"),
    ("test/offshore", "ssdd_test_offshore.yaml", "test"),
]


def metrics_row(results) -> dict:
    """Pull the numbers we report from an ultralytics DetMetrics object."""
    box = results.box
    return {
        "precision": float(box.mp),
        "recall": float(box.mr),
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "ms_per_image": float(sum(results.speed.values())),
    }


def metrics_markdown(rows: dict[str, dict]) -> str:
    lines = ["| subset | P | R | mAP@0.5 | mAP@0.5:0.95 | ms/img |", "|---|---|---|---|---|---|"]
    for name, m in rows.items():
        lines.append(f"| {name} | {m['precision']:.3f} | {m['recall']:.3f} | {m['map50']:.3f} | "
                     f"{m['map50_95']:.3f} | {m['ms_per_image']:.1f} |")
    return "\n".join(lines) + "\n"


def image_outcome(image_path: Path, label_path: Path, pred: np.ndarray, scores: np.ndarray,
                  shape_hw: tuple[int, int], iou_thr: float) -> dict:
    """Per-image TP/FP/FN at one operating point, plus the boxes for drawing."""
    h, w = shape_hw
    gt = yolo_to_xyxy(read_yolo_labels(label_path)[1], w, h)
    tp, matched = match_detections(pred, scores, gt, iou_thr)
    n_tp, n_fp, n_fn = int(tp.sum()), int((~tp).sum()), int((~matched).sum())
    denom = 2 * n_tp + n_fp + n_fn
    return {
        "image_path": image_path, "gt": gt, "pred": pred, "scores": scores,
        "tp": tp, "matched": matched, "n_tp": n_tp, "n_fp": n_fp, "n_fn": n_fn,
        "f1": 1.0 if denom == 0 else 2 * n_tp / denom,
    }


def select_examples(outcomes: list[dict], n: int) -> tuple[list[dict], list[dict]]:
    """Good = perfect or best F1, preferring busy images; bad = most errors."""
    good = sorted(outcomes, key=lambda o: (-o["f1"], -len(o["gt"]), str(o["image_path"])))
    bad = sorted(outcomes, key=lambda o: (-(o["n_fp"] + o["n_fn"]), o["f1"],
                                          str(o["image_path"])))
    return good[:n], bad[:n]


def predict_outcomes(model, image_dir: Path, label_dir: Path, cfg: dict) -> list[dict]:
    ex = cfg["eval"]["examples"]
    paths = sorted(image_dir.iterdir())
    outcomes = []
    results = model.predict(source=[str(p) for p in paths], conf=ex["conf"],
                            iou=cfg["eval"]["iou_nms"], imgsz=cfg["train"]["imgsz"],
                            stream=True, verbose=False)
    for path, r in zip(paths, results, strict=True):
        pred = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        outcomes.append(image_outcome(path, label_dir / f"{path.stem}.txt", pred, scores,
                                      r.orig_shape, ex["iou_match"]))
    return outcomes


def evaluate(cfg: dict, weights: Path, data_root: Path, runs_root: Path, name: str) -> dict:
    from ultralytics import YOLO

    ds_dir = data_root / cfg["dataset"]["yolo_dirname"]
    run_dir = runs_root / name
    out_dir = run_dir / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights))

    rows = {}
    for row_name, yaml_name, split in SUBSETS:
        res = model.val(data=str(ds_dir / yaml_name), split=split, imgsz=cfg["train"]["imgsz"],
                        batch=cfg["eval"]["batch"], conf=cfg["eval"]["conf"],
                        iou=cfg["eval"]["iou_nms"], project=str(out_dir),
                        name=row_name.replace("/", "_"), exist_ok=True, plots=True,
                        verbose=False)
        rows[row_name] = metrics_row(res)

    report = {"weights": str(weights), "metrics": rows}
    csv_path = run_dir / "results.csv"
    if csv_path.exists():
        cols = read_results_csv(csv_path)
        report["training"] = summarize_training(cols)
        plot_training_curves(cols, out_dir / "training_curves.png")

    ex = cfg["eval"]["examples"]
    outcomes = predict_outcomes(model, ds_dir / "images" / "test", ds_dir / "labels" / "test", cfg)
    good, bad = select_examples(outcomes, ex["n"])
    save_prediction_grid(good, out_dir / "examples_good.png",
                         f"test: best predictions (conf >= {ex['conf']})")
    save_prediction_grid(bad, out_dir / "examples_bad.png",
                         f"test: worst predictions (conf >= {ex['conf']})")
    totals = {k: int(sum(o[k] for o in outcomes)) for k in ("n_tp", "n_fp", "n_fn")}
    report["operating_point"] = {"conf": ex["conf"], "iou_match": ex["iou_match"], **totals}

    md = metrics_markdown(rows)
    (out_dir / "metrics.md").write_text(md, encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    report["markdown"] = md
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    report = evaluate(load_config(args.config, args.set), args.weights, args.data_root,
                      args.runs_root, args.name)
    print(report["markdown"])
    if "training" in report:
        print("Training:", json.dumps(report["training"], indent=2))
    print("Operating point (test):", json.dumps(report["operating_point"]))


if __name__ == "__main__":
    main()
