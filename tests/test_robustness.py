"""End-to-end robustness study with a fake detector that gets worse with severity."""

import json

import cv2
import numpy as np
import pytest

from sarship.analysis.run import load_split
from sarship.config import load_config
from sarship.plotting import plot_degradation_examples, plot_robustness_curves
from sarship.robustness import condition_id, conditions, image_rng, run_study

SIGNATURE = {"weights": "fake.pt", "conf": 0.001}


@pytest.fixture
def cfg():
    c = load_config(overrides=["robustness.bootstrap_resamples=20"])
    # replace (not merge) the ladder: a small study keeps the test fast
    c["robustness"]["degradations"] = {"blur": {"param": "sigma", "values": [1, 2, 3]},
                                       "speckle": {"param": "looks", "values": [4, 1]}}
    return c


def make_dataset(root):
    ds = root / "ssdd_yolo"
    for d in ("images/test", "labels/test"):
        (ds / d).mkdir(parents=True)
    rng = np.random.default_rng(0)
    lines = {"inshore": [], "offshore": []}
    for i in range(6):
        stem = f"{i:06d}"
        cv2.imwrite(str(ds / "images" / "test" / f"{stem}.jpg"),
                    rng.integers(0, 255, (120, 160), dtype=np.uint8))
        (ds / "labels" / "test" / f"{stem}.txt").write_text(
            "0 0.25 0.25 0.125 0.1\n0 0.7 0.6 0.25 0.3\n")
        lines["inshore" if i < 2 else "offshore"].append(f"./images/test/{stem}.jpg")
    for scene, entries in lines.items():
        (ds / f"test_{scene}.txt").write_text("\n".join(entries) + "\n")
    return ds


class FakeDetector:
    """Perfect on clean images; boxes drift by `severity` px on degraded ones."""

    def __init__(self, items, cfg):
        self.gt = {it["stem"]: it["gt"] for it in items}
        self.severity = {}
        for name, spec in cfg["robustness"]["degradations"].items():
            for k, v in enumerate(spec["values"]):
                self.severity[condition_id(name, float(v))] = 2 * (k + 1)
        self.calls = 0

    def __call__(self, paths):
        self.calls += 1
        out = []
        for p in paths:
            img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            assert img is not None and img.ndim == 2  # degraded files are grayscale
            shift = self.severity.get(p.parent.name, 0)
            boxes = self.gt[p.stem] + np.array([shift, shift, shift, shift])
            out.append({"boxes": boxes.tolist(), "scores": [0.9] * len(boxes)})
        return out


def test_conditions_order_and_ids(cfg):
    assert conditions(cfg)[0] == ("clean", None)
    assert [condition_id(n, v) for n, v in conditions(cfg)] == [
        "clean", "blur_1", "blur_2", "blur_3", "speckle_4", "speckle_1"]


def test_image_rng_independent_of_order():
    a = image_rng(42, "speckle_1", "000001").random(3)
    image_rng(42, "speckle_1", "000002").random(5)
    np.testing.assert_array_equal(a, image_rng(42, "speckle_1", "000001").random(3))


def test_study_end_to_end_and_cache(tmp_path, cfg):
    items = load_split(make_dataset(tmp_path), "test")
    detector = FakeDetector(items, cfg)
    out_dir, work_dir = tmp_path / "robustness", tmp_path / "work"

    report = run_study(items, cfg, out_dir, work_dir, SIGNATURE, lambda: detector)
    res = report["results"]
    assert res["clean"]["all"]["map50_95"]["point"] == pytest.approx(1.0)
    curve = [res[c]["all"]["map50_95"]["point"] for c in ("clean", "blur_1", "blur_2", "blur_3")]
    assert curve == sorted(curve, reverse=True) and curve[-1] < curve[0]
    assert report["breaking_points"]["blur"] is not None
    assert (work_dir / "speckle_1" / "000003.png").exists()
    assert detector.calls == 6  # one call per condition
    assert "| clean |" in report["markdown"] and "Breaking point" in report["markdown"]
    assert json.loads((out_dir / "robustness.json").read_text())["results"]["blur_2"]

    # second run: everything comes from the cache, the model is never built
    again = run_study(items, cfg, out_dir, work_dir, SIGNATURE,
                      lambda: pytest.fail("predictor must not be built on a cache hit"))
    assert again["results"]["blur_3"] == report["results"]["blur_3"]

    assert plot_robustness_curves(report, cfg, out_dir / "curves.png").stat().st_size > 0
    assert plot_degradation_examples(items, cfg, out_dir / "examples.png").stat().st_size > 0
