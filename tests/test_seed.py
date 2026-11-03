import random

import numpy as np

from sarship.seed import make_rng, set_seed


def test_set_seed_makes_python_and_numpy_repeatable():
    set_seed(123)
    first = (random.random(), np.random.rand())
    set_seed(123)
    second = (random.random(), np.random.rand())
    assert first == second


def test_make_rng_is_independent_and_repeatable():
    a = make_rng(7).normal(size=5)
    b = make_rng(7).normal(size=5)
    np.testing.assert_array_equal(a, b)
