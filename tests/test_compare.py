"""End-to-end model comparison with two fake detectors (no Ultralytics)."""

import numpy as np
import pytest
from test_robustness import FakeDetector, make_dataset

from sarship.analysis.run import load_split
from sarship.compare import (
    ALL_DEGRADED,
    HELD_LEVEL,
    HELD_TYPE,
    SEEN,
    compare,
    condition_groups,
    plot_comparison,
)
from sarship.config import load_config
from sarship.robustness import run_study


@pytest.fixture
def cfg():
    c = load_config(overrides=["robustness.bootstrap_resamples=10",
                               "compare.bootstrap_resamples=100",
                               "compare.baseline=base", "compare.candidate=robust"])
    c["robustness"]["degradations"] = {"blur": {"param": "sigma", "values": [1, 2, 3]},
                                       "jpeg": {"param": "quality", "values": [50, 10]}}
    c["robust_train"]["degradations"] = {"blur": [0.0, 2.0]}
    return c


class RobustDetector(FakeDetector):
    """Same as FakeDetector but its boxes drift only half as much."""

    def __call__(self, paths):
        out = super().__call__(paths)
        for p, o in zip(paths, out, strict=True):
            half = self.severity.get(p.parent.name, 0) / 2
            o["boxes"] = (np.asarray(o["boxes"]) - half).tolist()
        return out


def test_groups_split_seen_and_held_out(cfg):
    g = condition_groups(cfg)
    assert g[SEEN] == ["blur_1", "blur_2"]
    assert g[HELD_LEVEL] == ["blur_3"]
    assert g[HELD_TYPE] == ["jpeg_50", "jpeg_10"]
    assert len(g[ALL_DEGRADED]) == 5


def test_compare_end_to_end(tmp_path, cfg):
    items = load_split(make_dataset(tmp_path), "test")
    runs = tmp_path / "runs"
    for name, detector in (("base", FakeDetector(items, cfg)),
                           ("robust", RobustDetector(items, cfg))):
        run_study(items, cfg, runs / name / "robustness", tmp_path / "work" / name,
                  {"weights": name}, lambda d=detector: d)

    report = compare(cfg, items, runs)
    diff = report["differences"]
    assert diff["clean"]["map50_95"]["diff"] == pytest.approx(0.0)
    assert diff["blur_2"]["map50_95"]["diff"] > 0
    assert diff[ALL_DEGRADED]["map50_95"]["low"] > 0
    assert report["decision"] == {"clean_ok": True, "degraded_gain_ok": True, "accept": True}
    assert "candidate is BETTER" in report["markdown"]
    assert plot_comparison(report, cfg, tmp_path / "cmp.png").stat().st_size > 0


def test_missing_predictions_are_reported(tmp_path, cfg):
    items = load_split(make_dataset(tmp_path), "test")
    with pytest.raises(FileNotFoundError, match="sarship.robustness"):
        compare(cfg, items, tmp_path / "runs")
