"""Bootstrap confidence intervals for mAP: how much would the number move with a
different test set of the same size?

Images (not boxes) are resampled with replacement, because errors inside one image
are correlated (same scene, same clutter). Matching is done once per image, so each
resample only concatenates and re-sorts scores.
"""

from __future__ import annotations

import numpy as np

from sarship.analysis.ap import MatchTable, map_from_tables


def bootstrap_map(
    per_image: list[dict[float, MatchTable]],
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict:
    """Percentile CIs for mAP@0.5 and mAP@0.5:0.95. Returns point, low, high per metric."""
    point = map_from_tables(per_image)
    rng = np.random.default_rng(seed)
    n = len(per_image)
    samples = {"map50": [], "map50_95": []}
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        m = map_from_tables([per_image[i] for i in idx])
        for key in samples:
            samples[key].append(m[key])

    alpha = (1 - ci) / 2
    out = {"n_images": n, "n_resamples": n_resamples, "ci": ci}
    for key, values in samples.items():
        v = np.asarray(values, dtype=np.float64)
        v = v[~np.isnan(v)]
        out[key] = {
            "point": point[key],
            "low": float(np.quantile(v, alpha)),
            "high": float(np.quantile(v, 1 - alpha)),
        }
    return out
