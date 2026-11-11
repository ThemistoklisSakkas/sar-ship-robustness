import numpy as np
import pytest

from sarship.analysis.ap import ImageDetections, match_all
from sarship.analysis.bootstrap import bootstrap_map, paired_bootstrap


def make_images(n=40, seed=0):
    rng = np.random.default_rng(seed)
    images = []
    for i in range(n):
        xy = rng.uniform(0, 300, (3, 2))
        gt = np.hstack([xy, xy + rng.uniform(10, 40, (3, 2))])
        pred = gt + rng.normal(0, 2.5, gt.shape)
        # add one false alarm in half of the images
        if i % 2:
            pred = np.vstack([pred, [[400, 400, 420, 420]]])
        images.append(ImageDetections(stem=str(i), scene="offshore", width=500, height=500,
                                      gt=gt, pred=pred, scores=rng.uniform(0.2, 1, len(pred))))
    return images


def test_ci_brackets_point_and_is_repeatable():
    tables = match_all(make_images())
    a = bootstrap_map(tables, n_resamples=200, seed=1)
    b = bootstrap_map(tables, n_resamples=200, seed=1)
    assert a == b
    for key in ("map50", "map50_95"):
        assert a[key]["low"] <= a[key]["point"] <= a[key]["high"]
        assert a[key]["high"] - a[key]["low"] > 0


def test_identical_images_give_zero_width():
    one = make_images(1)[0]
    tables = match_all([one] * 10)
    r = bootstrap_map(tables, n_resamples=50, seed=0)
    assert r["map50"]["low"] == r["map50"]["high"] == r["map50"]["point"]


def test_fewer_images_give_wider_interval():
    tables = match_all(make_images(n=200, seed=3))
    wide = bootstrap_map(tables[:20], n_resamples=300, seed=0)["map50_95"]
    narrow = bootstrap_map(tables, n_resamples=300, seed=0)["map50_95"]
    assert (wide["high"] - wide["low"]) > (narrow["high"] - narrow["low"])


def shifted(images, px):
    out = []
    for im in images:
        out.append(ImageDetections(stem=im.stem, scene=im.scene, width=im.width,
                                   height=im.height, gt=im.gt, pred=im.pred + px,
                                   scores=im.scores))
    return out


def test_paired_identical_models_give_zero():
    t = match_all(make_images(30))
    r = paired_bootstrap({"clean": (t, t)}, n_resamples=50, seed=0)
    assert r["clean"]["map50_95"] == {"diff": 0.0, "low": 0.0, "high": 0.0}


def test_paired_detects_consistent_improvement_and_groups():
    images = make_images(40, seed=4)
    worse = match_all(shifted(images, 3.0))   # model A: boxes off by 3 px
    better = match_all(images)                # model B: original boxes
    r = paired_bootstrap({"c1": (worse, better), "c2": (better, better)},
                         groups={"mean": ["c1", "c2"]}, n_resamples=200, seed=1)
    assert r["c1"]["map50_95"]["low"] > 0          # B is better, CI excludes 0
    assert r["mean"]["map50_95"]["diff"] == pytest.approx(r["c1"]["map50_95"]["diff"] / 2)


def test_paired_rejects_mismatched_images():
    t = match_all(make_images(10))
    with pytest.raises(ValueError):
        paired_bootstrap({"c": (t, t[:5])}, n_resamples=5)
