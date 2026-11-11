"""Phase 6: deployment benchmark — size, speed, and an ONNX export that must match PyTorch.

Usage (Colab, GPU runtime gives both GPU and CPU numbers):
    python -m sarship.benchmark --data-root DATA --runs-root RUNS \
        --names robust_yolo11n robust_yolo11s

For every run: params / GFLOPs / file sizes, ONNX (FP32) export, an equivalence check
(our test mAP from PyTorch vs ONNX predictions), and batch-1 latency for PyTorch on
GPU and CPU and ONNX Runtime on CPU. Accuracy columns come from each run's
robustness study (clean and mean over all degraded conditions).
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from sarship.analysis.ap import map_from_tables, match_all
from sarship.analysis.run import build_images, load_split
from sarship.config import DEFAULT_CONFIG, load_config
from sarship.predict import make_predictor

STAGES = ("preprocess", "inference", "postprocess")


def summarize_timings(speeds: list[dict[str, float]]) -> dict:
    """Median and 90th percentile per stage (ms) from Ultralytics `result.speed` dicts."""
    out = {}
    for stage in (*STAGES, "total"):
        if stage == "total":
            values = np.array([sum(s[k] for k in STAGES) for s in speeds])
        else:
            values = np.array([s[stage] for s in speeds])
        out[stage] = {"median": float(np.median(values)), "p90": float(np.quantile(values, 0.9))}
    return out


def accuracy_from_robustness(path: Path) -> dict | None:
    """Clean and mean-degraded mAP@0.5:0.95 from a run's robustness.json."""
    if not path.exists():
        return None
    results = json.loads(path.read_text())["results"]
    degraded = [r["all"]["map50_95"]["point"] for cid, r in results.items() if cid != "clean"]
    return {"clean": results["clean"]["all"]["map50_95"]["point"],
            "degraded_mean": float(np.mean(degraded))}


def equivalence(map_pt: dict, map_onnx: dict, tolerance: float) -> dict:
    diff = map_onnx["map50_95"] - map_pt["map50_95"]
    return {"pytorch_map50_95": map_pt["map50_95"], "onnx_map50_95": map_onnx["map50_95"],
            "pytorch_map50": map_pt["map50"], "onnx_map50": map_onnx["map50"],
            "diff_map50_95": diff, "tolerance": tolerance, "ok": abs(diff) <= tolerance}


def model_info(weights: Path, imgsz: int) -> dict:
    """Parameters counted directly from the network; GFLOPs via Ultralytics if available.

    (`YOLO.info(verbose=False)` returns None in Ultralytics 8.4, so it is not used.)
    """
    from ultralytics import YOLO

    net = YOLO(str(weights)).model
    params = sum(p.numel() for p in net.parameters())
    try:
        from ultralytics.utils.torch_utils import get_flops

        gflops = float(get_flops(net, imgsz))
    except Exception:  # GFLOPs are informative only; never fail the benchmark on them
        gflops = float("nan")
    return {"params_m": params / 1e6, "gflops": gflops,
            "pt_mb": weights.stat().st_size / 2**20}


def export_onnx(weights: Path, imgsz: int) -> Path:
    """FP32 ONNX next to the weights; reused if newer than the .pt file."""
    onnx = weights.with_suffix(".onnx")
    if onnx.exists() and onnx.stat().st_mtime >= weights.stat().st_mtime:
        return onnx
    from ultralytics import YOLO

    return Path(YOLO(str(weights)).export(format="onnx", imgsz=imgsz, simplify=True,
                                          dynamic=False, half=False))


def time_backend(weights: Path, paths: list[Path], cfg: dict, device: str, warmup: int) -> dict:
    """Batch-1 latency over single images, after `warmup` untimed runs."""
    from ultralytics import YOLO

    model = YOLO(str(weights), task="detect")
    kw = {"conf": cfg["eval"]["examples"]["conf"], "iou": cfg["eval"]["iou_nms"],
          "imgsz": cfg["train"]["imgsz"], "device": device, "verbose": False}
    for p in paths[:warmup]:
        model.predict(str(p), **kw)
    speeds = []
    start = time.perf_counter()
    for p in paths:
        speeds.append(model.predict(str(p), **kw)[0].speed)
    wall = 1000 * (time.perf_counter() - start) / len(paths)
    return {"device": device, "stages": summarize_timings(speeds), "wall_ms": wall,
            "fps": 1000 / wall}


def evaluate_test_map(weights: Path, items: list[dict], cfg: dict, device: str) -> dict:
    preds = make_predictor(weights, cfg, device)([it["path"] for it in items])
    return map_from_tables(match_all(build_images(items, dict(zip(
        [it["stem"] for it in items], preds, strict=True)))))


def benchmark_run(name: str, runs_root: Path, items: list[dict], cfg: dict) -> dict:
    import torch

    b = cfg["benchmark"]
    pt = runs_root / name / "weights" / "best.pt"
    gpu = "0" if torch.cuda.is_available() else None
    row = {"name": name, **model_info(pt, cfg["train"]["imgsz"])}
    onnx = export_onnx(pt, cfg["train"]["imgsz"])
    row["onnx_mb"] = onnx.stat().st_size / 2**20
    row["accuracy"] = accuracy_from_robustness(runs_root / name / "robustness" /
                                               "robustness.json")
    row["equivalence"] = equivalence(evaluate_test_map(pt, items, cfg, gpu or "cpu"),
                                     evaluate_test_map(onnx, items, cfg, "cpu"),
                                     b["equivalence_tolerance"])
    paths = [it["path"] for it in items[: b["n_images"]]]
    row["timing"] = {}
    if gpu is not None:
        row["timing"]["pytorch_gpu"] = time_backend(pt, paths, cfg, gpu, b["warmup"])
    row["timing"]["pytorch_cpu"] = time_backend(pt, paths, cfg, "cpu", b["warmup"])
    row["timing"]["onnx_cpu"] = time_backend(onnx, paths, cfg, "cpu", b["warmup"])
    return row


def benchmark_markdown(rows: list[dict], env: dict) -> str:
    lines = [f"Hardware: {env.get('gpu', 'no GPU')}; CPU: {env.get('cpu', '?')} "
             f"({env.get('cpu_threads', '?')} threads). Batch 1, imgsz 640, times in ms "
             "(median inference; total = pre + inference + post; FPS from wall clock).", "",
             "| model | params (M) | GFLOPs | .pt MB | .onnx MB | clean mAP@0.5:0.95 | "
             "degraded mean | ONNX Δ mAP | backend | inference ms | total ms | FPS |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        acc = r.get("accuracy") or {}
        eq = r["equivalence"]
        head = (f"| {r['name']} | {r['params_m']:.2f} | {r['gflops']:.1f} | {r['pt_mb']:.1f} | "
                f"{r['onnx_mb']:.1f} | {acc.get('clean', float('nan')):.3f} | "
                f"{acc.get('degraded_mean', float('nan')):.3f} | "
                f"{eq['diff_map50_95']:+.4f} {'ok' if eq['ok'] else 'FAIL'} ")
        for k, (backend, t) in enumerate(r["timing"].items()):
            prefix = head if k == 0 else "| | | | | | | | "
            lines.append(f"{prefix}| {backend} | {t['stages']['inference']['median']:.1f} | "
                         f"{t['stages']['total']['median']:.1f} | {t['fps']:.0f} |")
    return "\n".join(lines) + "\n"


def environment() -> dict:
    env = {"python": platform.python_version(), "cpu": platform.processor() or platform.machine()}
    try:
        import os

        import torch

        env["cpu_threads"] = os.cpu_count()
        env["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU"
        cpuinfo = Path("/proc/cpuinfo")
        if cpuinfo.exists():
            for line in cpuinfo.read_text().splitlines():
                if line.startswith("model name"):
                    env["cpu"] = line.split(":", 1)[1].strip()
                    break
    except ImportError:
        pass
    return env


def plot_accuracy_vs_speed(rows: list[dict], out_path: Path, backend: str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for r in rows:
        if not r.get("accuracy") or backend not in r["timing"]:
            continue
        ms = r["timing"][backend]["stages"]["total"]["median"]
        for key, marker, label in (("clean", "o", "clean"), ("degraded_mean", "s", "degraded")):
            ax.scatter(ms, r["accuracy"][key], marker=marker, s=60)
            ax.annotate(f"{r['name']} ({label})", (ms, r["accuracy"][key]),
                        textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel(f"latency per image, {backend} (ms, median, batch 1)")
    ax.set_ylabel("test mAP@0.5:0.95")
    ax.set_title("Accuracy vs speed (o = clean, square = mean over degraded conditions)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)

    items = load_split(args.data_root / cfg["dataset"]["yolo_dirname"], "test")
    rows = []
    for name in args.names:
        print(f"Benchmarking {name} ...", flush=True)
        rows.append(benchmark_run(name, args.runs_root, items, cfg))
    env = environment()
    md = benchmark_markdown(rows, env)
    out_dir = args.runs_root / "benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "benchmark.md").write_text(md, encoding="utf-8")
    (out_dir / "benchmark.json").write_text(json.dumps({"env": env, "rows": rows}, indent=2))
    # batch-1 GPU latency is overhead-bound (phase 6), so the trade-off plot uses CPU time
    backend = "pytorch_cpu"
    plot_accuracy_vs_speed(rows, out_dir / "accuracy_vs_speed.png", backend)
    print(md)
    print(f"Tables and figure in {out_dir}")


if __name__ == "__main__":
    main()
