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
    prefix = "robust" if cfg.get("robust_train", {}).get("enabled") else "baseline"
    return f"{prefix}_{Path(cfg['model']['weights']).stem}"


def trainer_class(cfg: dict):
    """Custom trainer for degradation-aware runs, None (Ultralytics default) otherwise."""
    if not cfg.get("robust_train", {}).get("enabled"):
        return None
    from sarship.robust_train import make_trainer

    return make_trainer(cfg)


def config_for_resume(cfg: dict, run_dir: Path) -> dict:
    """A resumed run must use the configuration it was started with, not today's CLI."""
    saved = run_dir / "config_used.yaml"
    return yaml.safe_load(saved.read_text()) if saved.exists() else cfg


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


def check_run_dir(run_dir: Path, resume: bool, overwrite: bool) -> None:
    """Refuse actions that would silently destroy or fake a training run.

    A fresh run into a folder that already holds weights would overwrite them after
    the first epoch; --resume without a checkpoint would silently start from scratch.
    """
    last = run_dir / "weights" / "last.pt"
    if resume and not last.exists():
        raise RuntimeError(f"--resume given but {last} does not exist")
    if not resume and last.exists() and not overwrite:
        raise RuntimeError(
            f"{run_dir} already has weights. Use --resume to continue that run, "
            "--name for a new run, or --overwrite to replace it on purpose."
        )


def require_gpu() -> None:
    """Training on Colab's CPU takes ~20 min per epoch: fail fast instead."""
    try:
        import torch
    except ImportError as err:
        raise RuntimeError("PyTorch is not installed") from err
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No GPU available (Colab: Runtime > Change runtime type > T4 GPU). "
            "Use --allow-cpu only for a tiny smoke test."
        )


def train(
    cfg: dict,
    data_root: Path,
    runs_root: Path,
    name: str,
    resume: bool = False,
    overwrite: bool = False,
    allow_cpu: bool = False,
) -> Path:
    """Train (or resume) and return the run directory."""
    run_dir = runs_root / name
    last = run_dir / "weights" / "last.pt"
    check_run_dir(run_dir, resume, overwrite)
    data_yaml = data_root / cfg["dataset"]["yolo_dirname"] / "ssdd.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"{data_yaml} not found: run `python -m sarship.data.prepare`")
    if not allow_cpu:
        require_gpu()

    from ultralytics import YOLO

    set_seed(cfg["seed"], cfg["train"]["deterministic"])
    if resume:
        print(f"Resuming from {last}")
        YOLO(str(last)).train(resume=True, trainer=trainer_class(config_for_resume(cfg, run_dir)))
    else:
        args = build_train_args(cfg, data_yaml, runs_root, name)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        if cfg.get("robust_train", {}).get("enabled"):
            print("Degradation-aware training:", cfg["robust_train"])
        YOLO(cfg["model"]["weights"]).train(trainer=trainer_class(cfg), **args)

    (run_dir / "run_info.json").write_text(json.dumps(environment_info(), indent=2))
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--name", default=None,
                        help="run name (default: baseline_<model> or robust_<model>)")
    parser.add_argument("--resume", action="store_true", help="continue from weights/last.pt")
    parser.add_argument("--overwrite", action="store_true",
                        help="start a fresh run even if the run folder already has weights")
    parser.add_argument("--allow-cpu", action="store_true", help="permit training without GPU")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    try:
        run_dir = train(cfg, args.data_root, args.runs_root,
                        args.name or default_run_name(cfg), args.resume, args.overwrite,
                        args.allow_cpu)
    except RuntimeError as err:
        raise SystemExit(f"STOPPED: {err}") from err
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
