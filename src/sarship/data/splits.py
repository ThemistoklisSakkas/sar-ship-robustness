"""Leakage-safe split helpers."""

from __future__ import annotations

import numpy as np


def split_train_val(stems: list[str], val_fraction: float, seed: int) -> tuple[list, list]:
    """Deterministic random train/val split of image stems.

    Only applied *inside* the official training set; the official test set is
    never touched, so test leakage is impossible by construction.
    """
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be in (0, 1)")
    ordered = sorted(stems)  # independent of filesystem listing order
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(ordered))
    n_val = max(1, round(len(ordered) * val_fraction))
    val = sorted(ordered[i] for i in idx[:n_val])
    train = sorted(ordered[i] for i in idx[n_val:])
    return train, val


def assert_disjoint(**splits: list[str]) -> None:
    """Raise if any image stem appears in more than one split."""
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            common = set(splits[a]) & set(splits[b])
            if common:
                sample = sorted(common)[:5]
                raise AssertionError(f"{len(common)} images in both {a} and {b}: {sample}")
