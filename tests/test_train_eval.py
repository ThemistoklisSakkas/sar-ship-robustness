from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from sarship.config import load_config
from sarship.evaluate import image_outcome, metrics_markdown, metrics_row, select_examples
from sarship.plotting import (
    plot_training_curves,
    read_results_csv,
    save_prediction_grid,
    summarize_training,
)
from sarship.train import build_train_args, default_run_name


@pytest.fixture
def cfg():
    return load_config()


def test_train_args_come_from_config(cfg, tmp_path):
    args = build_train_args(cfg, tmp_path / "ssdd.yaml", tmp_path / "runs", "baseline")
    assert args["epochs"] == cfg["train"]["epochs"] and args["seed"] == 42
    assert args["deterministic"] is True and args["project"] == str(tmp_path / "runs")
    # colour jitter is disabled for single-channel SAR
    assert args["hsv_h"] == 0.0 and args["hsv_s"] == 0.0


def test_train_args_override_and_clash(cfg, tmp_path):
    cfg2 = load_config(overrides=["train.epochs=5", "train.augment.fliplr=0.0"])
    args = build_train_args(cfg2, tmp_path / "d.yaml", tmp_path, "x")
    assert args["epochs"] == 5 and args["fliplr"] == 0.0
    cfg["train"]["augment"]["epochs"] = 1
    with pytest.raises(ValueError, match="clash"):
        build_train_args(cfg, tmp_path / "d.yaml", tmp_path, "x")


def test_default_run_name(cfg):
    assert default_run_name(cfg) == "baseline_yolo11n"


def test_metrics_row_and_markdown():
    fake = SimpleNamespace(box=SimpleNamespace(mp=0.9, mr=0.8, map50=0.85, map=0.5),
                           speed={"preprocess": 0.5, "inference": 3.0, "postprocess": 1.0})
    row = metrics_row(fake)
    assert row == {"precision": 0.9, "recall": 0.8, "map50": 0.85, "map50_95": 0.5,
                   "ms_per_image": 4.5}
    md = metrics_markdown({"test": row})
    assert "| test | 0.900 | 0.800 | 0.850 | 0.500 | 4.5 |" in md


def test_image_outcome_counts(tmp_path):
    label = tmp_path / "a.txt"
    # two ships in a 100x100 image: [10,10,30,30] and [60,60,80,80]
    label.write_text("0 0.2 0.2 0.2 0.2\n0 0.7 0.7 0.2 0.2\n")
    pred = np.array([[10, 10, 30, 30], [40, 0, 50, 10]], dtype=float)  # 1 TP + 1 false alarm
    o = image_outcome(Path("a.png"), label, pred, np.array([0.9, 0.6]), (100, 100), 0.5)
    assert (o["n_tp"], o["n_fp"], o["n_fn"]) == (1, 1, 1)
    assert o["f1"] == pytest.approx(0.5)


def test_image_outcome_empty_is_perfect(tmp_path):
    label = tmp_path / "b.txt"
    label.write_text("")
    o = image_outcome(Path("b.png"), label, np.zeros((0, 4)), np.zeros(0), (50, 50), 0.5)
    assert o["f1"] == 1.0 and o["n_fp"] == 0


def test_select_examples():
    def o(name, f1, n_gt, fp, fn):
        return {"image_path": name, "f1": f1, "gt": [0] * n_gt, "n_fp": fp, "n_fn": fn}

    outs = [o("a", 1.0, 1, 0, 0), o("b", 1.0, 5, 0, 0), o("c", 0.2, 4, 3, 2), o("d", 0.5, 2, 1, 0)]
    good, bad = select_examples(outs, 2)
    assert [g["image_path"] for g in good] == ["b", "a"]
    assert [b["image_path"] for b in bad] == ["c", "d"]


RESULTS_CSV = (
    "      epoch,  train/box_loss,  train/cls_loss,  train/dfl_loss,  metrics/precision(B),"
    "  metrics/recall(B),  metrics/mAP50(B),  metrics/mAP50-95(B),"
    "  val/box_loss,  val/cls_loss,  val/dfl_loss\n"
) + """1,2.0,3.0,1.5,0.3,0.2,0.25,0.10,2.2,3.1,1.6
2,1.6,2.0,1.3,0.6,0.5,0.60,0.30,1.9,2.2,1.4
3,1.3,1.4,1.2,0.8,0.7,0.80,0.45,1.8,1.6,1.3
4,1.1,1.0,1.1,0.8,0.7,0.79,0.44,1.9,1.7,1.35
"""


def test_results_csv_summary_and_plot(tmp_path):
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(RESULTS_CSV)
    cols = read_results_csv(csv_path)
    assert "metrics/mAP50-95(B)" in cols  # header whitespace stripped
    s = summarize_training(cols)
    assert s["epochs_run"] == 4 and s["best_epoch"] == 3
    assert s["best_val_map50_95"] == pytest.approx(0.45)
    assert s["min_val_box_loss_epoch"] == 3  # val loss rises again in epoch 4
    assert plot_training_curves(cols, tmp_path / "curves.png").stat().st_size > 0


def test_prediction_grid_renders(tmp_path):
    img = tmp_path / "x.png"
    cv2.imwrite(str(img), np.zeros((60, 80), dtype=np.uint8))
    item = {"image_path": img, "gt": np.array([[5, 5, 20, 20], [40, 40, 60, 55]]),
            "matched": np.array([True, False]), "pred": np.array([[5, 5, 21, 20], [0, 0, 3, 3]]),
            "tp": np.array([True, False]), "n_tp": 1, "n_fp": 1, "n_fn": 1}
    assert save_prediction_grid([item], tmp_path / "grid.png", "t").stat().st_size > 0
