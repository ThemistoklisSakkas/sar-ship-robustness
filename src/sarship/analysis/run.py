"""Phase 3 error analysis of a trained detector on the official test set.

Usage (Colab, a GPU is NOT needed):
    python -m sarship.analysis.run --weights RUNS/baseline_yolo11n/weights/best.pt \
        --data-root DATA --runs-root RUNS --name baseline_yolo11n

Steps:
1. Predict once on the test images at conf 0.001 and cache the boxes (JSON).
2. Our own COCO-style mAP: all / inshore / offshore x all / small / medium / large,
   checked against the Ultralytics numbers of phase 2.
3. Bootstrap 95% confidence intervals (resampling images).
4. At the operating point (conf 0.25): recall per size/scene/crowding, and a reason
   for every missed ship and every false alarm.
5. Galleries of missed ships and of the most confident false alarms.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from sarship.analysis.ap import ALL_AREAS, ImageDetections, map_from_tables, match_all
from sarship.analysis.bootstrap import bootstrap_map
from sarship.analysis.errors import (
    FN_REASONS,
    FP_REASONS,
    classify_image,
    error_summary,
    recall_by_group,
)
from sarship.config import DEFAULT_CONFIG, load_config
from sarship.data.convert import yolo_to_xyxy
from sarship.data.viz import read_yolo_labels
from sarship.plotting import (
    FN_COLOUR,
    FP_COLOUR,
    plot_recall_by_size,
    save_crop_gallery,
)
from sarship.predict import cached_predictions, make_predictor, prediction_signature

SCENES = ("inshore", "offshore")


def load_scenes(ds_dir: Path, split: str) -> dict[str, str]:
    scenes = {}
    for scene in SCENES:
        for line in (ds_dir / f"{split}_{scene}.txt").read_text().splitlines():
            if line.strip():
                scenes[Path(line.strip()).stem] = scene
    return scenes


def load_split(ds_dir: Path, split: str) -> list[dict]:
    """Image path, size, ground-truth boxes (xyxy pixels) and scene for every image."""
    scenes = load_scenes(ds_dir, split)
    items = []
    for path in sorted((ds_dir / "images" / split).iterdir()):
        with Image.open(path) as im:
            width, height = im.size
        _, yolo = read_yolo_labels(ds_dir / "labels" / split / f"{path.stem}.txt")
        items.append({"path": path, "stem": path.stem, "width": width, "height": height,
                      "gt": yolo_to_xyxy(yolo, width, height), "scene": scenes[path.stem]})
    return items


def predict_cached(weights: Path, items: list[dict], cfg: dict, cache: Path) -> dict:
    """{stem: {"boxes": [...], "scores": [...]}}; re-predicts only if the weights changed."""
    return cached_predictions(
        cache, prediction_signature(weights, cfg), [it["stem"] for it in items],
        lambda: [it["path"] for it in items], lambda: make_predictor(weights, cfg),
    )


def build_images(items: list[dict], preds: dict) -> list[ImageDetections]:
    return [
        ImageDetections(
            stem=it["stem"], scene=it["scene"], width=it["width"], height=it["height"],
            gt=it["gt"],
            pred=np.asarray(preds[it["stem"]]["boxes"], dtype=np.float64).reshape(-1, 4),
            scores=np.asarray(preds[it["stem"]]["scores"], dtype=np.float64),
            path=str(it["path"]),
        )
        for it in items
    ]


def map_breakdown(images: list[ImageDetections], size_bins: dict) -> dict:
    ranges = {"all": ALL_AREAS, **{k: (float(lo), float(hi)) for k, (lo, hi) in size_bins.items()}}
    table = {}
    for scene in ("all", *SCENES):
        subset = [im for im in images if scene == "all" or im.scene == scene]
        for size, rng in ranges.items():
            table[f"{scene}/{size}"] = map_from_tables(match_all(subset, rng))
    return table


def galleries(images, results, out_dir: Path, n: int) -> None:
    fn_crops, fp_crops = [], []
    for im, r in zip(images, results, strict=True):
        for j, reason in r["fn"]:
            fn_crops.append({"image_path": im.path, "box": im.gt[j], "gt": im.gt,
                             "label": f"{im.stem} {im.scene[:3]} {reason}",
                             "colour": FN_COLOUR})
        for i, reason in r["fp"]:
            if reason in ("background", "border"):
                fp_crops.append({"image_path": im.path, "box": r["pred"][i], "gt": im.gt,
                                 "score": float(r["scores"][i]),
                                 "label": f"{im.stem} {im.scene[:3]} {reason} "
                                          f"{r['scores'][i]:.2f}",
                                 "colour": FP_COLOUR})
    fp_crops.sort(key=lambda c: -c["score"])
    if fn_crops:
        save_crop_gallery(fn_crops[:n], out_dir / "missed_ships.png",
                          "Missed ships (yellow) with the reason; green = other ground truth")
    if fp_crops:
        save_crop_gallery(fp_crops[:n], out_dir / "false_alarms_top.png",
                          "Most confident false alarms with no ship overlap (red): "
                          "clutter or unlabelled vessel?")


def report_markdown(rep: dict) -> str:
    lines = ["### mAP by scene and ship size (our COCO-style implementation)", "",
             "| subset | ships | mAP@0.5 | mAP@0.5:0.95 |", "|---|---|---|---|"]
    for key, m in rep["map"].items():
        lines.append(f"| {key} | {m['n_gt']} | {m['map50']:.3f} | {m['map50_95']:.3f} |")
    if rep.get("check"):
        c = rep["check"]
        lines += ["", f"Check against Ultralytics (test): mAP@0.5 {c['ours_map50']:.3f} vs "
                      f"{c['ultralytics_map50']:.3f}, mAP@0.5:0.95 {c['ours_map50_95']:.3f} vs "
                      f"{c['ultralytics_map50_95']:.3f}"]
    lines += ["", "### Bootstrap 95% confidence intervals (images resampled)", "",
              "| subset | images | mAP@0.5 [95% CI] | mAP@0.5:0.95 [95% CI] |", "|---|---|---|---|"]
    for key, b in rep["bootstrap"].items():
        lines.append(
            f"| {key} | {b['n_images']} | {b['map50']['point']:.3f} "
            f"[{b['map50']['low']:.3f}, {b['map50']['high']:.3f}] | "
            f"{b['map50_95']['point']:.3f} [{b['map50_95']['low']:.3f}, "
            f"{b['map50_95']['high']:.3f}] |")
    op = rep["operating_point"]
    lines += ["", f"### Operating point: conf >= {op['conf']}, TP if IoU >= {op['iou_match']}", "",
              "| group | found / ships | recall |", "|---|---|---|"]
    for key, g in rep["recall"].items():
        lines.append(f"| {key} | {g['found']}/{g['total']} | {g['found'] / g['total']:.3f} |")
    lines += ["", "| scene | TP | " + " | ".join(f"FN {r}" for r in FN_REASONS) + " |",
              "|" + "---|" * (2 + len(FN_REASONS))]
    for scene, s in rep["errors"].items():
        lines.append(f"| {scene} | {s['tp']} | " + " | ".join(str(s["fn"][r]) for r in FN_REASONS)
                     + " |")
    lines += ["", "| scene | " + " | ".join(f"FP {r}" for r in FP_REASONS) + " |",
              "|" + "---|" * (1 + len(FP_REASONS))]
    for scene, s in rep["errors"].items():
        lines.append(f"| {scene} | " + " | ".join(str(s["fp"][r]) for r in FP_REASONS) + " |")
    return "\n".join(lines) + "\n"


def analyse(images: list[ImageDetections], cfg: dict, out_dir: Path,
            ultralytics_test: dict | None = None) -> dict:
    a = cfg["analysis"]
    ex = cfg["eval"]["examples"]
    rep: dict = {"map": map_breakdown(images, a["size_bins"])}

    if ultralytics_test:
        rep["check"] = {"ours_map50": rep["map"]["all/all"]["map50"],
                        "ours_map50_95": rep["map"]["all/all"]["map50_95"],
                        "ultralytics_map50": ultralytics_test["map50"],
                        "ultralytics_map50_95": ultralytics_test["map50_95"]}

    rep["bootstrap"] = {}
    for scene in ("all", *SCENES):
        subset = [im for im in images if scene == "all" or im.scene == scene]
        rep["bootstrap"][scene] = bootstrap_map(match_all(subset), a["bootstrap_resamples"],
                                                a["bootstrap_ci"], cfg["seed"])

    results = [classify_image(im, ex["conf"], ex["iou_match"], a["overlap_iou"],
                              a["border_margin_px"]) for im in images]
    rep["operating_point"] = {"conf": ex["conf"], "iou_match": ex["iou_match"]}
    rep["recall"] = recall_by_group(images, results, a["size_bins"])
    rep["errors"] = error_summary(images, results)

    out_dir.mkdir(parents=True, exist_ok=True)
    plot_recall_by_size(rep["recall"], list(a["size_bins"]), out_dir / "recall_by_size.png")
    galleries(images, results, out_dir, a["gallery_n"])
    md = report_markdown(rep)
    (out_dir / "analysis.md").write_text(md, encoding="utf-8")
    (out_dir / "analysis.json").write_text(json.dumps(rep, indent=2))
    rep["markdown"] = md
    return rep


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
    cfg = load_config(args.config, args.set)

    ds_dir = args.data_root / cfg["dataset"]["yolo_dirname"]
    run_dir = args.runs_root / args.name
    out_dir = run_dir / "analysis"
    items = load_split(ds_dir, "test")
    preds = predict_cached(args.weights, items, cfg, out_dir / "predictions_test.json")

    ultra = None
    metrics = run_dir / "evaluation" / "metrics.json"
    if metrics.exists():
        ultra = json.loads(metrics.read_text())["metrics"].get("test")

    rep = analyse(build_images(items, preds), cfg, out_dir, ultra)
    print(rep["markdown"])
    print(f"Figures and tables in {out_dir}")


if __name__ == "__main__":
    main()
