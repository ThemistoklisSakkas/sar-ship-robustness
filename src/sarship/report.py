"""Phase 7: gather the figures used by the README into one folder with stable names.

Usage (Colab, CPU is fine):
    python -m sarship.report --drive-root /content/drive/MyDrive/sar-ship-robustness

Writes <drive-root>/report_figures/*.png; download that folder and unzip it into
docs/figures/ of the repository.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

# (source path relative to the Drive project root, name used in the README)
FIGURES = [
    ("data/ssdd_yolo/figures/size_hist.png", "01_size_hist.png"),
    ("data/ssdd_yolo/figures/gt_test_inshore.png", "02_gt_test_inshore.png"),
    ("data/ssdd_yolo/figures/colour_images_raw.png", "03_colour_images_raw.png"),
    ("runs/baseline_yolo11n/evaluation/training_curves.png", "04_baseline_training_curves.png"),
    ("runs/baseline_yolo11n/evaluation/examples_good.png", "05_baseline_examples_good.png"),
    ("runs/baseline_yolo11n/evaluation/examples_bad.png", "06_baseline_examples_bad.png"),
    ("runs/baseline_yolo11n/analysis/recall_by_size.png", "07_recall_by_size.png"),
    ("runs/baseline_yolo11n/analysis/missed_ships.png", "08_missed_ships.png"),
    ("runs/baseline_yolo11n/analysis/false_alarms_top.png", "09_false_alarms_top.png"),
    ("runs/baseline_yolo11n/robustness/degradation_examples.png", "10_degradation_examples.png"),
    ("runs/baseline_yolo11n/robustness/robustness_curves.png", "11_baseline_robustness_curves.png"),
    ("runs/robust_yolo11n/comparison/comparison.png", "12_robust_vs_baseline.png"),
    ("runs/robust_yolo11s/comparison/comparison.png", "13_small_vs_nano.png"),
    ("runs/benchmark/accuracy_vs_speed.png", "14_accuracy_vs_speed.png"),
]


def collect(drive_root: Path, out_dir: Path) -> tuple[list[str], list[str]]:
    """Copy every figure that exists; return (copied names, missing sources)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    copied, missing = [], []
    for src, name in FIGURES:
        path = drive_root / src
        if path.exists():
            shutil.copy2(path, out_dir / name)
            copied.append(name)
        else:
            missing.append(src)
    return copied, missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--drive-root", type=Path, required=True)
    args = parser.parse_args()
    out_dir = args.drive_root / "report_figures"
    copied, missing = collect(args.drive_root, out_dir)
    print(f"Copied {len(copied)} figures to {out_dir}")
    for src in missing:
        print(f"  missing: {src}")


if __name__ == "__main__":
    main()
