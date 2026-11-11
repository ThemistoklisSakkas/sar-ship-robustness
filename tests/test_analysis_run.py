"""End-to-end phase-3 analysis on a tiny fake dataset with cached predictions (no model)."""

import json

import cv2
import numpy as np
import pytest

from sarship.analysis.run import analyse, build_images, load_split, predict_cached
from sarship.config import load_config
from sarship.predict import prediction_signature


@pytest.fixture
def cfg():
    return load_config(overrides=["analysis.bootstrap_resamples=50", "analysis.gallery_n=4"])


def make_dataset(root):
    ds = root / "ssdd_yolo"
    rng = np.random.default_rng(0)
    for d in ("images/test", "labels/test"):
        (ds / d).mkdir(parents=True)
    inshore, offshore = [], []
    for i in range(6):
        stem = f"{i:06d}"
        name = f"{stem}.png" if i == 0 else f"{stem}.jpg"  # one grayscale-converted PNG
        cv2.imwrite(str(ds / "images" / "test" / name),
                    rng.integers(0, 255, (100, 200), dtype=np.uint8))
        # two ships per image: 20x10 px (small) and 40x30 px (medium)
        (ds / "labels" / "test" / f"{stem}.txt").write_text(
            "0 0.25 0.25 0.1 0.1\n0 0.7 0.6 0.2 0.3\n")
        (inshore if i < 2 else offshore).append(f"./images/test/{name}")
    (ds / "test_inshore.txt").write_text("\n".join(inshore) + "\n")
    (ds / "test_offshore.txt").write_text("\n".join(offshore) + "\n")
    return ds


def fake_predictions(items):
    preds = {}
    for k, it in enumerate(items):
        boxes = it["gt"].copy().tolist()
        scores = [0.9, 0.8]
        if k == 0:
            boxes = boxes[:1]  # inshore image 0 misses the medium ship
            scores = scores[:1]
        if k == 1:
            boxes.append([0, 80, 15, 100])  # confident false alarm on the border
            scores.append(0.95)
        preds[it["stem"]] = {"boxes": boxes, "scores": scores}
    return preds


def test_full_analysis_from_cached_predictions(tmp_path, cfg):
    ds = make_dataset(tmp_path)
    items = load_split(ds, "test")
    assert [it["scene"] for it in items] == ["inshore"] * 2 + ["offshore"] * 4
    np.testing.assert_allclose(items[0]["gt"][0], [40, 20, 60, 30])

    weights = tmp_path / "best.pt"
    weights.write_bytes(b"w")
    cache = tmp_path / "analysis" / "predictions_test.json"
    signature = prediction_signature(weights, cfg)
    cache.parent.mkdir()
    cache.write_text(json.dumps({"signature": signature, "predictions": fake_predictions(items)}))
    preds = predict_cached(weights, items, cfg, cache)  # cache hit: no ultralytics needed

    rep = analyse(build_images(items, preds), cfg, tmp_path / "analysis",
                  {"map50": 0.9, "map50_95": 0.7})
    assert rep["map"]["offshore/all"]["map50"] == pytest.approx(1.0)
    assert rep["map"]["inshore/all"]["map50"] < 1.0
    assert rep["map"]["all/small"]["n_gt"] == 6 and rep["map"]["all/medium"]["n_gt"] == 6
    assert rep["recall"]["inshore/medium"] == {"found": 1, "total": 2}
    assert rep["errors"]["inshore"]["fn"]["missed"] == 1
    assert rep["errors"]["inshore"]["fp"]["border"] == 1
    assert "Check against Ultralytics" in rep["markdown"]
    for fig in ("recall_by_size.png", "missed_ships.png", "false_alarms_top.png"):
        assert (tmp_path / "analysis" / fig).stat().st_size > 0
    assert json.loads((tmp_path / "analysis" / "analysis.json").read_text())["map"]
