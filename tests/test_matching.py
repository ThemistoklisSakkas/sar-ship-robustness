import numpy as np
import pytest

from sarship.analysis.matching import iou_matrix, match_detections


def test_iou_identical_disjoint_and_half():
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[0, 0, 10, 10], [20, 20, 30, 30], [5, 0, 15, 10]])
    np.testing.assert_allclose(iou_matrix(a, b), [[1.0, 0.0, 50 / 150]])


def test_small_box_shifted_three_pixels():
    # The phase-0 example: a 10x10 box shifted by 3 px has IoU 70/130 ~ 0.54
    iou = iou_matrix(np.array([[0, 0, 10, 10]]), np.array([[3, 0, 13, 10]]))[0, 0]
    assert iou == pytest.approx(70 / 130)


def test_iou_shape_and_empty():
    assert iou_matrix(np.zeros((0, 4)), np.zeros((3, 4))).shape == (0, 3)


def test_duplicate_prediction_is_one_tp_one_fp():
    gt = np.array([[0, 0, 10, 10]])
    pred = np.array([[0, 0, 10, 10], [1, 0, 11, 10]])
    tp, matched = match_detections(pred, np.array([0.9, 0.8]), gt)
    assert tp.tolist() == [True, False] and matched.tolist() == [True]


def test_higher_score_wins_the_ground_truth():
    gt = np.array([[0, 0, 10, 10]])
    pred = np.array([[1, 0, 11, 10], [0, 0, 10, 10]])  # the better box has the lower score
    tp, _ = match_detections(pred, np.array([0.9, 0.3]), gt)
    assert tp.tolist() == [True, False]


def test_below_threshold_is_fp_and_fn():
    tp, matched = match_detections(
        np.array([[0, 0, 10, 10]]), np.array([0.9]), np.array([[6, 0, 16, 10]]), iou_thr=0.5
    )
    assert tp.tolist() == [False] and matched.tolist() == [False]


def test_two_adjacent_ships_both_found():
    gt = np.array([[0, 0, 10, 30], [10, 0, 20, 30]])
    pred = np.array([[10, 0, 20, 30], [0, 0, 10, 30]])
    tp, matched = match_detections(pred, np.array([0.7, 0.9]), gt)
    assert tp.all() and matched.all()


def test_empty_inputs():
    tp, matched = match_detections(np.zeros((0, 4)), np.zeros(0), np.array([[0, 0, 1, 1]]))
    assert tp.size == 0 and matched.tolist() == [False]
