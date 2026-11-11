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


def paired_bootstrap(
    pairs: dict[str, tuple[list[dict[float, MatchTable]], list[dict[float, MatchTable]]]],
    groups: dict[str, list[str]] | None = None,
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict:
    """CIs for the difference (model B - model A) on the *same* images.

    `pairs` maps a condition to (tables_A, tables_B), both in the same image order.
    In every resample one set of image indices is shared by both models and by all
    conditions, so image difficulty cancels out and group means (e.g. "all degraded
    conditions") get a proper joint CI.
    """
    groups = groups or {}
    names = list(pairs)
    n = len(next(iter(pairs.values()))[0])
    for a, b in pairs.values():
        if len(a) != n or len(b) != n:
            raise ValueError("all conditions and both models need the same images")

    def diffs(idx):
        out = {}
        for c in names:
            a = map_from_tables([pairs[c][0][i] for i in idx])
            b = map_from_tables([pairs[c][1][i] for i in idx])
            out[c] = {k: b[k] - a[k] for k in ("map50", "map50_95")}
        return out

    def with_groups(d):
        for g, members in groups.items():
            d[g] = {k: float(np.mean([d[m][k] for m in members])) for k in ("map50", "map50_95")}
        return d

    point = with_groups(diffs(np.arange(n)))
    rng = np.random.default_rng(seed)
    samples = {key: {"map50": [], "map50_95": []} for key in point}
    for _ in range(n_resamples):
        d = with_groups(diffs(rng.integers(0, n, size=n)))
        for key, v in d.items():
            for metric in ("map50", "map50_95"):
                samples[key][metric].append(v[metric])

    alpha = (1 - ci) / 2
    result = {}
    for key, metrics in samples.items():
        result[key] = {}
        for metric, values in metrics.items():
            v = np.asarray(values)
            v = v[~np.isnan(v)]
            result[key][metric] = {"diff": float(point[key][metric]),
                                   "low": float(np.quantile(v, alpha)),
                                   "high": float(np.quantile(v, 1 - alpha))}
    return result
