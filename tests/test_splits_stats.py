import numpy as np
import pytest

from sarship.data.splits import assert_disjoint, split_train_val
from sarship.data.stats import box_areas, size_category, summarize

BINS = {"small": [0, 1024], "medium": [1024, 9216], "large": [9216, float("inf")]}


def test_split_is_deterministic_and_order_independent():
    stems = [f"{i:06d}" for i in range(100)]
    a = split_train_val(stems, 0.15, seed=42)
    b = split_train_val(list(reversed(stems)), 0.15, seed=42)
    assert a == b
    assert len(a[1]) == 15 and len(a[0]) == 85


def test_split_covers_everything_without_overlap():
    stems = [f"{i:06d}" for i in range(37)]
    train, val = split_train_val(stems, 0.2, seed=1)
    assert sorted(train + val) == sorted(stems)
    assert_disjoint(train=train, val=val)


def test_assert_disjoint_detects_leak():
    with pytest.raises(AssertionError, match="train and test"):
        assert_disjoint(train=["a", "b"], val=["c"], test=["b"])


def test_bad_val_fraction():
    with pytest.raises(ValueError):
        split_train_val(["a", "b"], 1.0, seed=0)


def test_size_category_boundaries():
    areas = box_areas(np.array([[0, 0, 31, 31], [0, 0, 32, 32], [0, 0, 96, 96]]))
    assert size_category(areas, BINS).tolist() == ["small", "medium", "large"]


def test_summarize_counts():
    records = [
        {"split": "train", "boxes": np.array([[0, 0, 10, 10], [0, 0, 50, 50]]), "dropped": 1},
        {"split": "train", "boxes": np.zeros((0, 4)), "dropped": 0},
        {"split": "test", "boxes": np.array([[0, 0, 200, 100]]), "dropped": 0},
    ]
    s = summarize(records, BINS)
    assert s["train"]["images"] == 2
    assert s["train"]["ships"] == 2
    assert s["train"]["empty_images"] == 1
    assert s["train"]["dropped_boxes"] == 1
    assert s["train"]["size_bins"] == {"small": 1, "medium": 1, "large": 0}
    assert s["test"]["size_bins"]["large"] == 1
