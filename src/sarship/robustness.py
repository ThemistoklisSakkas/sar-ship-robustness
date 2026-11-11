"""Phase 4: how does the detector degrade under realistic field conditions?

Usage (Colab, a GPU is optional):
    python -m sarship.robustness --weights RUNS/baseline_yolo11n/weights/best.pt \
        --data-root DATA --runs-root RUNS --name baseline_yolo11n [--work-dir /content/degraded]

For every degradation and severity level the test images are degraded (seeded),
the detector is run once (predictions cached on Drive, so the study is resumable),
and mAP is computed with our own COCO-style code. Output: a table, robustness
curves with bootstrap CIs, and a figure showing what each level looks like.
"""

from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path

import cv2
import numpy as np

from sarship.analysis.ap import ImageDetections, map_from_tables, match_all
from sarship.analysis.bootstrap import bootstrap_map
from sarship.analysis.errors import classify_image
from sarship.analysis.run import build_images, load_split
from sarship.config import DEFAULT_CONFIG, load_config
from sarship.degrade import apply_degradation
from sarship.predict import Predictor, cached_predictions, make_predictor, prediction_signature

SCENES = ("inshore", "offshore")


def conditions(cfg: dict) -> list[tuple[str, float | None]]:
    """[("clean", None), ("speckle", 16), ...] in config order, mildest level first."""
    out: list[tuple[str, float | None]] = [("clean", None)]
    for name, spec in cfg["robustness"]["degradations"].items():
        out += [(name, float(v)) for v in spec["values"]]
    return out


def condition_id(name: str, value: float | None) -> str:
    return "clean" if value is None else f"{name}_{value:g}"


def image_rng(seed: int, cid: str, stem: str) -> np.random.Generator:
    """Independent, reproducible noise per (condition, image), whatever the processing order."""
    return np.random.default_rng([seed, zlib.crc32(cid.encode()), zlib.crc32(stem.encode())])


def degrade_file(src: Path, dst: Path, name: str, value: float, rng) -> Path:
    img = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    # PNG is lossless: the only compression artefacts are those of the "jpeg" condition
    cv2.imwrite(str(dst), apply_degradation(name, img, value, rng))
    return dst


def degraded_paths(items: list[dict], name: str, value: float, work_dir: Path,
                   seed: int) -> list[Path]:
    cid = condition_id(name, value)
    return [degrade_file(it["path"], work_dir / cid / f"{it['stem']}.png", name, value,
                         image_rng(seed, cid, it["stem"])) for it in items]


def operating_point(images: list[ImageDetections], cfg: dict) -> dict:
    ex = cfg["eval"]["examples"]
    tp = fp = fn = 0
    for im in images:
        r = classify_image(im, ex["conf"], ex["iou_match"])
        tp += int(r["tp"].sum())
        fp += len(r["fp"])
        fn += len(r["fn"])
    return {"tp": tp, "fp": fp, "fn": fn, "recall": tp / max(1, tp + fn),
            "fp_per_image": fp / max(1, len(images))}


def evaluate_condition(images: list[ImageDetections], cfg: dict) -> dict:
    r = cfg["robustness"]
    out = {"all": bootstrap_map(match_all(images), r["bootstrap_resamples"],
                                cfg["analysis"]["bootstrap_ci"], cfg["seed"])}
    for scene in SCENES:
        m = map_from_tables(match_all([im for im in images if im.scene == scene]))
        out[scene] = {"map50": {"point": m["map50"]}, "map50_95": {"point": m["map50_95"]}}
    out["operating_point"] = operating_point(images, cfg)
    return out


def breaking_points(results: dict, cfg: dict) -> dict:
    """First level per degradation whose mAP@0.5 falls more than `relative_drop` below clean."""
    clean = results["clean"]["all"]["map50"]["point"]
    limit = (1 - cfg["robustness"]["relative_drop"]) * clean
    out = {}
    for name, spec in cfg["robustness"]["degradations"].items():
        out[name] = next((v for v in spec["values"]
                          if results[condition_id(name, float(v))]["all"]["map50"]["point"]
                          < limit), None)
    return out


def run_study(items: list[dict], cfg: dict, out_dir: Path, work_dir: Path,
              signature: dict, predictor_fn) -> dict:
    """Evaluate every condition; `predictor_fn()` builds the model only on a cache miss."""
    predictor: list[Predictor] = []

    def lazy_predictor() -> Predictor:
        if not predictor:
            predictor.append(predictor_fn())
        return predictor[0]

    stems = [it["stem"] for it in items]
    results = {}
    for name, value in conditions(cfg):
        cid = condition_id(name, value)
        if value is None:
            def paths_fn():
                return [it["path"] for it in items]
        else:
            def paths_fn(name=name, value=value):
                return degraded_paths(items, name, value, work_dir, cfg["seed"])
        preds = cached_predictions(out_dir / "predictions" / f"{cid}.json",
                                   {**signature, "condition": cid, "seed": cfg["seed"]},
                                   stems, paths_fn, lazy_predictor)
        results[cid] = evaluate_condition(build_images(items, preds), cfg)
        m = results[cid]["all"]
        print(f"{cid:>16}: mAP@0.5 {m['map50']['point']:.3f}  "
              f"mAP@0.5:0.95 {m['map50_95']['point']:.3f}", flush=True)

    report = {"results": results, "breaking_points": breaking_points(results, cfg)}
    report["markdown"] = report_markdown(report, cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "robustness.md").write_text(report["markdown"], encoding="utf-8")
    (out_dir / "robustness.json").write_text(
        json.dumps({k: v for k, v in report.items() if k != "markdown"}, indent=2))
    return report


def report_markdown(report: dict, cfg: dict) -> str:
    res = report["results"]
    clean = res["clean"]["all"]["map50_95"]["point"]
    lines = ["| condition | mAP@0.5 [95% CI] | mAP@0.5:0.95 [95% CI] | Δ vs clean | "
             "inshore mAP@0.5:0.95 | offshore mAP@0.5:0.95 | recall @0.25 | FP/img |",
             "|---|---|---|---|---|---|---|---|"]
    for name, value in conditions(cfg):
        cid = condition_id(name, value)
        r = res[cid]
        a, b = r["all"]["map50"], r["all"]["map50_95"]
        label = cid if value is None else \
            f"{name} {cfg['robustness']['degradations'][name]['param']}={value:g}"
        lines.append(
            f"| {label} | {a['point']:.3f} [{a['low']:.3f}, {a['high']:.3f}] | "
            f"{b['point']:.3f} [{b['low']:.3f}, {b['high']:.3f}] | "
            f"{100 * (b['point'] - clean) / clean:+.1f}% | "
            f"{r['inshore']['map50_95']['point']:.3f} | {r['offshore']['map50_95']['point']:.3f} | "
            f"{r['operating_point']['recall']:.3f} | {r['operating_point']['fp_per_image']:.2f} |")
    drop = int(100 * cfg["robustness"]["relative_drop"])
    lines += ["", f"Breaking point (first level with mAP@0.5 more than {drop}% below clean):"]
    for name, v in report["breaking_points"].items():
        param = cfg["robustness"]["degradations"][name]["param"]
        lines.append(f"- {name}: " + ("not reached" if v is None else f"{param}={v:g}"))
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/sarship_degraded"),
                        help="scratch folder for degraded images (local disk is fastest)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)

    from sarship.plotting import plot_degradation_examples, plot_robustness_curves

    items = load_split(args.data_root / cfg["dataset"]["yolo_dirname"], "test")
    out_dir = args.runs_root / args.name / "robustness"
    report = run_study(items, cfg, out_dir, args.work_dir, prediction_signature(args.weights, cfg),
                       lambda: make_predictor(args.weights, cfg))
    plot_robustness_curves(report, cfg, out_dir / "robustness_curves.png")
    plot_degradation_examples(items, cfg, out_dir / "degradation_examples.png")
    print(report["markdown"])
    print(f"Figures and tables in {out_dir}")


if __name__ == "__main__":
    main()
