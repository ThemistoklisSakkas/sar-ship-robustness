"""Train a YOLO ship detector on the prepared SSDD dataset.

Usage (Colab):
    python -m sarship.train --data-root DATA --runs-root RUNS [--name NAME] [--resume]
                            [--set train.epochs=50 model.weights=yolo11s.pt ...]

Everything comes from configs/default.yaml; --set overrides single values and the
final merged config is saved next to the weights, so every run is reproducible.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path

import yaml

from sarship.config import DEFAULT_CONFIG, REPO_ROOT, load_config
from sarship.seed import set_seed


def default_run_name(cfg: dict) -> str:
    return f"baseline_{Path(cfg['model']['weights']).stem}"


def build_train_args(cfg: dict, data_yaml: Path, runs_root: Path, name: str) -> dict:
    """Translate our config into keyword arguments for ultralytics `YOLO.train`."""
    t = cfg["train"]
    args = {
        "data": str(data_yaml),
        "epochs": t["epochs"],
        "imgsz": t["imgsz"],
        "batch": t["batch"],
        "patience": t["patience"],
        "optimizer": t["optimizer"],
        "workers": t["workers"],
        "close_mosaic": t["close_mosaic"],
        "seed": cfg["seed"],
        "deterministic": t["deterministic"],
        "project": str(runs_root),
        "name": name,
        "exist_ok": True,
        "pretrained": True,
        "plots": True,
    }
    overlap = set(args) & set(t.get("augment", {}))
    if overlap:
        raise ValueError(f"augment keys clash with core train args: {sorted(overlap)}")
    args.update(t.get("augment", {}))
    return args


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def environment_info() -> dict:
    info = {"python": platform.python_version(), "git_commit": git_commit()}
    try:
        import torch
        import ultralytics

        info.update(
            torch=torch.__version__,
            cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
            ultralytics=ultralytics.__version__,
        )
    except ImportError:
        pass
    return info


def train(cfg: dict, data_root: Path, runs_root: Path, name: str, resume: bool = False) -> Path:
    """Train (or resume) and return the run directory."""
    from ultralytics import YOLO

    set_seed(cfg["seed"], cfg["train"]["deterministic"])
    data_yaml = data_root / cfg["dataset"]["yolo_dirname"] / "ssdd.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"{data_yaml} not found: run `python -m sarship.data.prepare`")
    run_dir = runs_root / name
    last = run_dir / "weights" / "last.pt"

    if resume and last.exists():
        print(f"Resuming from {last}")
        YOLO(str(last)).train(resume=True)
    else:
        args = build_train_args(cfg, data_yaml, runs_root, name)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        YOLO(cfg["model"]["weights"]).train(**args)

    (run_dir / "run_info.json").write_text(json.dumps(environment_info(), indent=2))
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--name", default=None, help="run name (default: baseline_<model>)")
    parser.add_argument("--resume", action="store_true", help="continue from weights/last.pt")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    run_dir = train(cfg, args.data_root, args.runs_root, args.name or default_run_name(cfg),
                    args.resume)
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
