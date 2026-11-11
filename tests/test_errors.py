import numpy as np

from sarship.analysis.ap import ImageDetections
from sarship.analysis.errors import (
    classify_image,
    coverage,
    crowded_gt,
    error_summary,
    recall_by_group,
)

BINS = {"small": [0, 1024], "medium": [1024, 9216], "large": [9216, float("inf")]}


def img(gt, pred, scores, scene="inshore", size=(200, 200)):
    return ImageDetections(stem="x", scene=scene, width=size[0], height=size[1],
                           gt=np.array(gt, dtype=float).reshape(-1, 4),
                           pred=np.array(pred, dtype=float).reshape(-1, 4),
                           scores=np.array(scores, dtype=float))


def fn_reasons(r):
    return [reason for _, reason in r["fn"]]


def fp_reasons(r):
    return [reason for _, reason in r["fp"]]


def test_coverage():
    cov = coverage(np.array([[0, 0, 20, 10]]), np.array([[0, 0, 10, 10], [15, 0, 25, 10]]))
    np.testing.assert_allclose(cov, [[1.0, 0.5]])


def test_merged_neighbours():
    # two moored ships side by side; one wide box covers both and matches the first
    gt = [[50, 50, 60, 90], [60, 50, 70, 90]]
    r = classify_image(img(gt, [[50, 50, 68, 90]], [0.9]), conf=0.25)
    assert r["tp"].tolist() == [True]
    assert fn_reasons(r) == ["merged"]


def test_merged_box_false_alarm():
    # a box spanning two ships equally matches neither (IoU 0.5 each needs >= 0.5)
    gt = [[50, 50, 60, 90], [62, 50, 72, 90]]
    r = classify_image(img(gt, [[48, 50, 74, 90]], [0.9]), conf=0.25)
    assert fp_reasons(r) == ["merged_box"]
    assert sorted(fn_reasons(r)) == ["localisation", "localisation"]


def test_low_score_and_missed():
    gt = [[10, 10, 30, 20], [100, 100, 120, 110]]
    pred = [[10, 10, 30, 20]]  # right box, score below the operating threshold
    r = classify_image(img(gt, pred, [0.1]), conf=0.25)
    assert fn_reasons(r) == ["low_score", "missed"]


def test_duplicate_localisation_border_background():
    gt = [[50, 50, 90, 70]]
    pred = [
        [50, 50, 90, 70],    # TP
        [51, 50, 91, 70],    # duplicate
        [50, 50, 90, 120],   # IoU ~0.29: localisation
        [0, 150, 10, 160],   # touches left edge: border
        [140, 140, 150, 150],  # open water: background
    ]
    r = classify_image(img(gt, pred, [0.9, 0.8, 0.7, 0.6, 0.5]), conf=0.25)
    assert fp_reasons(r) == ["duplicate", "localisation", "border", "background"]
    assert r["fn"] == []


def test_crowded_flags_touching_boxes_only():
    gt = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [100, 100, 110, 110]])
    assert crowded_gt(gt).tolist() == [True, True, False]


def test_summary_and_recall_groups():
    a = img([[0, 0, 10, 10]], [[0, 0, 10, 10], [150, 150, 160, 160]], [0.9, 0.9], "offshore")
    b = img([[0, 0, 100, 100], [150, 150, 160, 160]], [[0, 0, 100, 100]], [0.9], "inshore")
    results = [classify_image(i, 0.25) for i in (a, b)]
    s = error_summary([a, b], results)
    assert s["offshore"]["fp"]["background"] == 1 and s["offshore"]["tp"] == 1
    assert s["inshore"]["fn"]["missed"] == 1
    rec = recall_by_group([a, b], results, BINS)
    assert rec["inshore/large"] == {"found": 1, "total": 1}
    assert rec["all/small"] == {"found": 1, "total": 2}
    assert rec["isolated"]["total"] == 3
