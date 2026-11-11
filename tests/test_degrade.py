import numpy as np
import pytest

from sarship.degrade import (
    apply_degradation,
    gaussian_blur,
    jpeg_compress,
    reduce_contrast,
    reduce_resolution,
    speckle,
)


def textured(seed=0, shape=(256, 256)):
    """Speckle-like test chip: dark sea with a few bright 'ships'."""
    rng = np.random.default_rng(seed)
    img = rng.gamma(2.0, 20.0, size=shape)
    img[60:70, 40:90] = 220
    img[150:190, 180:190] = 200
    return np.clip(img, 0, 255).astype(np.uint8)


def mse(a, b):
    return float(np.mean((a.astype(float) - b.astype(float)) ** 2))


@pytest.mark.parametrize("name,value", [("speckle", 2), ("blur", 1.5), ("contrast", 0.5),
                                        ("resolution", 3), ("jpeg", 20)])
def test_shape_and_dtype_are_preserved(name, value):
    img = textured()
    out = apply_degradation(name, img, value, np.random.default_rng(0))
    assert out.shape == img.shape and out.dtype == np.uint8


def test_rejects_colour_and_float_input():
    with pytest.raises(ValueError):
        gaussian_blur(np.zeros((8, 8, 3), dtype=np.uint8), 1.0)
    with pytest.raises(ValueError):
        gaussian_blur(np.zeros((8, 8), dtype=np.float32), 1.0)


def test_speckle_is_multiplicative_and_mean_preserving():
    img = np.full((400, 400), 100, dtype=np.uint8)
    out = speckle(img, looks=4, rng=np.random.default_rng(1)).astype(float)
    intensity_ratio = (out ** 2) / 100.0 ** 2
    assert intensity_ratio.mean() == pytest.approx(1.0, abs=0.02)      # E[n] = 1
    assert intensity_ratio.var() == pytest.approx(1 / 4, rel=0.08)     # Var[n] = 1/L
    # multiplicative: a black pixel stays black whatever the noise
    assert speckle(np.zeros((50, 50), np.uint8), 1, np.random.default_rng(2)).max() == 0


def test_speckle_fewer_looks_is_noisier_and_seeded():
    img = np.full((300, 300), 80, dtype=np.uint8)
    a = speckle(img, 1, np.random.default_rng(3))
    b = speckle(img, 16, np.random.default_rng(3))
    assert a.std() > 2 * b.std()
    np.testing.assert_array_equal(a, speckle(img, 1, np.random.default_rng(3)))
    with pytest.raises(ValueError):
        apply_degradation("speckle", img, 2)  # no rng


def test_blur_keeps_flat_image_and_grows_with_sigma():
    flat = np.full((64, 64), 77, dtype=np.uint8)
    np.testing.assert_array_equal(gaussian_blur(flat, 2.0), flat)
    img = textured()
    assert mse(img, gaussian_blur(img, 0.5)) < mse(img, gaussian_blur(img, 2.0))


def test_contrast_scales_spread_and_keeps_mean():
    img = textured()
    out = reduce_contrast(img, 0.5)
    assert out.mean() == pytest.approx(img.mean(), abs=1.0)
    assert out.std() == pytest.approx(0.5 * img.std(), rel=0.03)
    np.testing.assert_array_equal(reduce_contrast(img, 1.0), img)


def test_resolution_removes_fine_detail():
    checker = (np.indices((64, 64)).sum(axis=0) % 2 * 255).astype(np.uint8)
    out = reduce_resolution(checker, 2)
    assert out.std() < 10  # a 1-pixel pattern cannot survive half resolution
    np.testing.assert_array_equal(reduce_resolution(checker, 1), checker)
    with pytest.raises(ValueError):
        reduce_resolution(checker, 0.5)


def test_jpeg_lower_quality_costs_more():
    img = textured()
    assert mse(img, jpeg_compress(img, 90)) < mse(img, jpeg_compress(img, 10))


@pytest.mark.parametrize("name,values", [
    ("speckle", [16, 8, 4, 2, 1]),
    ("blur", [0.5, 1.0, 1.5, 2.0, 3.0]),
    ("contrast", [0.8, 0.6, 0.4, 0.3, 0.2]),
    ("resolution", [1.5, 2, 3, 4, 6]),
    ("jpeg", [50, 30, 20, 10, 5]),
])
def test_severity_levels_are_monotonic(name, values):
    """Each step of the configured severity ladder moves the image further from the original."""
    img = textured(seed=5, shape=(320, 320))
    errors = [mse(img, apply_degradation(name, img, v, np.random.default_rng(0)))
              for v in values]
    assert errors == sorted(errors)


def test_unknown_degradation():
    with pytest.raises(ValueError, match="unknown"):
        apply_degradation("fog", textured(), 1)
