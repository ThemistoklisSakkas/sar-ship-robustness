import numpy as np
import pytest

from sarship.analysis.ap import (
    ImageDetections,
    average_precision,
    dataset_map,
    match_image,
)


def img(gt, pred, scores, scene="offshore"):
    return ImageDetections(stem="x", scene=scene, width=500, height=500,
                           gt=np.array(gt, dtype=float).reshape(-1, 4),
                           pred=np.array(pred, dtype=float).reshape(-1, 4),
                           scores=np.array(scores, dtype=float))


def test_perfect_predictions_give_ap_one():
    gt = [[0, 0, 10, 10], [50, 50, 70, 60]]
    m = dataset_map([img(gt, gt, [0.9, 0.8])])
    assert m["map50"] == pytest.approx(1.0) and m["map50_95"] == pytest.approx(1.0)


def test_no_correct_prediction_gives_zero():
    m = dataset_map([img([[0, 0, 10, 10]], [[100, 100, 110, 110]], [0.9])])
    assert m["map50"] == 0.0


def test_hand_computed_101_point_ap():
    # 2 ships; predictions by score: TP, FP, TP
    # recall/precision: (0.5, 1), (0.5, 0.5), (1.0, 2/3)
    # envelope: 1.0 for recall <= 0.5 (51 points), 2/3 above (50 points)
    gt = [[0, 0, 10, 10], [50, 50, 60, 60]]
    pred = [[0, 0, 10, 10], [200, 200, 210, 210], [50, 50, 60, 60]]
    t = match_image(np.array(pred), np.array([0.9, 0.8, 0.7]), np.array(gt), 0.5)
    assert t.tp.tolist() == [True, False, True]
    assert average_precision([t]) == pytest.approx((51 * 1.0 + 50 * 2 / 3) / 101)


def test_score_order_across_images_is_global():
    a = img([[0, 0, 10, 10]], [[0, 0, 10, 10]], [0.3])
    b = img([[0, 0, 10, 10]], [[100, 100, 110, 110]], [0.9])  # confident false alarm
    # sorted globally: FP(0.9), TP(0.3) -> precision 0.5 at recall 0.5, recall never reaches 1
    m = dataset_map([a, b])
    assert m["map50"] == pytest.approx(51 * 0.5 / 101)


def test_localisation_lowers_map50_95_only():
    gt = [[0, 0, 100, 100]]
    pred = [[0, 0, 100, 80]]  # IoU 0.8
    m = dataset_map([img(gt, pred, [0.9])])
    assert m["map50"] == pytest.approx(1.0)
    # TP at thresholds 0.50..0.80 (7 of 10), FP at 0.85..0.95
    assert m["map50_95"] == pytest.approx(0.7)


def test_size_range_ignores_other_sizes():
    small_gt, large_gt = [0, 0, 20, 20], [100, 100, 300, 300]  # areas 400 and 40000
    stray_large_fp = [400, 400, 499, 499]  # area 9801, no ship there
    im = img([small_gt, large_gt], [large_gt, stray_large_fp], [0.9, 0.8])
    small = dataset_map([im], (0, 1024))
    large = dataset_map([im], (9216, float("inf")))
    assert small["n_gt"] == 1 and small["map50"] == 0.0  # the small ship was missed
    # the large ship is found; the large stray box is an FP for the large range
    assert large["n_gt"] == 1 and large["map50"] == pytest.approx(1.0)
    # for the small range the large boxes are ignored, not counted as FP
    t = match_image(im.pred, im.scores, im.gt, 0.5, (0, 1024))
    assert t.ignore.tolist() == [True, True]


def test_all_range_equals_unrestricted():
    rng = np.random.default_rng(0)
    gt = rng.uniform(0, 400, (20, 2))
    gt = np.hstack([gt, gt + rng.uniform(5, 80, (20, 2))])
    pred = gt + rng.normal(0, 3, gt.shape)
    images = [img(gt, pred, rng.uniform(0, 1, 20))]
    assert dataset_map(images)["map50_95"] == pytest.approx(
        dataset_map(images, (0, float("inf")))["map50_95"]
    )


def test_image_without_ships_only_adds_false_alarms():
    a = img([[0, 0, 10, 10]], [[0, 0, 10, 10]], [0.5])
    empty = img([], [[30, 30, 40, 40]], [0.9])
    assert dataset_map([a])["map50"] == pytest.approx(1.0)
    assert dataset_map([a, empty])["map50"] < 1.0
