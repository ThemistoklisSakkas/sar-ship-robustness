"""Field degradations for SAR image chips, each controlled by one severity parameter.

All functions take and return uint8 grayscale arrays of shape (H, W) and never move
objects, so the ground-truth boxes stay valid. Pixel values are treated as SAR
*amplitude* (display scale); intensity = amplitude^2.

Physical / operational meaning:
  speckle     extra multiplicative speckle, Gamma(L, 1/L) on intensity: fewer looks,
              e.g. a single-look high-resolution product instead of a multilooked one
  blur        Gaussian blur: defocus from uncompensated platform or target motion,
              a wider point-spread function
  contrast    contrast squeezed toward the image mean: lower target-to-clutter ratio
              (rough sea, poor calibration, compressed dynamic range)
  resolution  down-sample then up-sample back: a coarser sensor mode
              (e.g. a wide-swath mode instead of stripmap)
  jpeg        lossy JPEG compression: a bandwidth-limited datalink or dissemination chain
"""

from __future__ import annotations

import cv2
import numpy as np


def _check(img: np.ndarray) -> None:
    if img.ndim != 2 or img.dtype != np.uint8:
        raise ValueError(f"expected uint8 (H, W) grayscale, got {img.dtype} {img.shape}")


def _to_uint8(x: np.ndarray) -> np.ndarray:
    return np.clip(np.round(x), 0, 255).astype(np.uint8)


def speckle(img: np.ndarray, looks: float, rng: np.random.Generator) -> np.ndarray:
    """Multiply intensity by Gamma(L, 1/L) noise (mean 1, variance 1/L).

    Amplitude is scaled by sqrt(noise), so E[intensity] is preserved before clipping.
    The images already contain speckle; this lowers the equivalent number of looks.
    """
    _check(img)
    if looks <= 0:
        raise ValueError("looks must be > 0")
    noise = rng.gamma(shape=looks, scale=1.0 / looks, size=img.shape)
    return _to_uint8(img.astype(np.float64) * np.sqrt(noise))


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    _check(img)
    if sigma <= 0:
        return img.copy()
    return cv2.GaussianBlur(img, (0, 0), sigmaX=float(sigma), borderType=cv2.BORDER_REFLECT)


def reduce_contrast(img: np.ndarray, factor: float) -> np.ndarray:
    """Scale deviations from the image mean by `factor` (1 = unchanged, 0 = flat grey)."""
    _check(img)
    if not 0 <= factor <= 1:
        raise ValueError("factor must be in [0, 1]")
    mean = img.astype(np.float64).mean()
    return _to_uint8(mean + factor * (img.astype(np.float64) - mean))


def reduce_resolution(img: np.ndarray, factor: float) -> np.ndarray:
    """Coarser ground sampling by `factor` (2 = half the resolution), back to the same size."""
    _check(img)
    if factor < 1:
        raise ValueError("factor must be >= 1")
    if factor == 1:
        return img.copy()
    h, w = img.shape
    small = (max(1, round(w / factor)), max(1, round(h / factor)))
    down = cv2.resize(img, small, interpolation=cv2.INTER_AREA)  # area = sensor integration
    return cv2.resize(down, (w, h), interpolation=cv2.INTER_LINEAR)


def jpeg_compress(img: np.ndarray, quality: float) -> np.ndarray:
    _check(img)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)


def apply_degradation(
    name: str, img: np.ndarray, value: float, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Dispatch by name; only `speckle` is random and needs `rng`."""
    if name == "speckle":
        if rng is None:
            raise ValueError("speckle needs an rng for reproducibility")
        return speckle(img, value, rng)
    functions = {"blur": gaussian_blur, "contrast": reduce_contrast,
                 "resolution": reduce_resolution, "jpeg": jpeg_compress}
    if name not in functions:
        raise ValueError(f"unknown degradation {name!r}")
    return functions[name](img, value)


class RandomDegradation:
    """Training-time augmentation: with probability `p`, one degradation at a random severity.

    `ranges` maps a degradation name to [low, high]. Speckle looks are sampled
    log-uniformly (L in [2, 4] is as likely as L in [8, 16]: each halving of the looks
    is one equal step of severity); the other parameters uniformly; JPEG quality as
    an integer.
    """

    NAMES = ("speckle", "blur", "contrast", "resolution", "jpeg")

    def __init__(self, ranges: dict[str, list[float]], p: float):
        unknown = set(ranges) - set(self.NAMES)
        if unknown:
            raise ValueError(f"unknown degradations: {sorted(unknown)}")
        if not 0 <= p <= 1:
            raise ValueError("p must be in [0, 1]")
        for name, (low, high) in ranges.items():
            if low > high:
                raise ValueError(f"{name}: low > high")
        self.ranges = {k: (float(v[0]), float(v[1])) for k, v in ranges.items()}
        self.names = list(self.ranges)
        self.p = p

    def sample(self, rng: np.random.Generator) -> tuple[str, float] | None:
        if not self.names or rng.random() >= self.p:
            return None
        name = self.names[int(rng.integers(len(self.names)))]
        low, high = self.ranges[name]
        if name == "speckle":
            value = float(np.exp(rng.uniform(np.log(low), np.log(high))))
        elif name == "jpeg":
            value = float(rng.integers(int(low), int(high) + 1))
        else:
            value = float(rng.uniform(low, high))
        return name, value

    def __call__(
        self, img: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, tuple[str, float] | None]:
        choice = self.sample(rng)
        if choice is None:
            return img, None
        return apply_degradation(choice[0], img, choice[1], rng), choice
