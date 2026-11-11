from collections import Counter

import numpy as np
import pytest

from sarship.degrade import RandomDegradation

RANGES = {"speckle": [2, 16], "blur": [0.0, 2.0], "contrast": [0.3, 1.0],
          "resolution": [1.0, 4.0]}


def test_probability_and_coverage():
    aug = RandomDegradation(RANGES, p=0.5)
    rng = np.random.default_rng(0)
    draws = [aug.sample(rng) for _ in range(4000)]
    applied = [d for d in draws if d is not None]
    assert 0.47 < len(applied) / len(draws) < 0.53
    counts = Counter(name for name, _ in applied)
    assert set(counts) == set(RANGES)
    assert min(counts.values()) > 0.2 * len(applied)  # roughly uniform over 4 types
    for name, value in applied:
        low, high = RANGES[name]
        assert low <= value <= high


def test_speckle_is_log_uniform():
    aug = RandomDegradation({"speckle": [2, 16]}, p=1.0)
    rng = np.random.default_rng(1)
    looks = np.array([aug.sample(rng)[1] for _ in range(4000)])
    # log-uniform on [2, 16]: the geometric midpoint sqrt(32) ~ 5.66 splits the mass in half
    assert 0.46 < (looks < np.sqrt(32)).mean() < 0.54


def test_p_zero_and_one():
    img = np.full((32, 32), 100, dtype=np.uint8)
    out, choice = RandomDegradation(RANGES, p=0.0)(img, np.random.default_rng(0))
    assert choice is None and out is img
    out, choice = RandomDegradation({"contrast": [0.5, 0.5]}, p=1.0)(img, np.random.default_rng(0))
    assert choice == ("contrast", 0.5) and out.shape == img.shape


def test_reproducible_with_same_seed():
    aug = RandomDegradation(RANGES, p=0.7)
    a = [aug.sample(np.random.default_rng(s)) for s in range(50)]
    b = [aug.sample(np.random.default_rng(s)) for s in range(50)]
    assert a == b


def test_validation():
    with pytest.raises(ValueError, match="unknown"):
        RandomDegradation({"fog": [0, 1]}, 0.5)
    with pytest.raises(ValueError):
        RandomDegradation(RANGES, 1.5)
    with pytest.raises(ValueError, match="low > high"):
        RandomDegradation({"blur": [2, 1]}, 0.5)
