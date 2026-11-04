"""Leakage-safe split helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path

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


def stratified_train_val(
    groups: dict[str, str], val_fraction: float, seed: int
) -> tuple[list, list]:
    """Split {stem: group} so every group (e.g. inshore/offshore) keeps its ratio in val."""
    train, val = [], []
    for i, group in enumerate(sorted(set(groups.values()))):
        stems = [s for s, g in groups.items() if g == group]
        t, v = split_train_val(stems, val_fraction, seed + i)
        train += t
        val += v
    return sorted(train), sorted(val)


def assert_disjoint(**splits: list[str]) -> None:
    """Raise if any image stem appears in more than one split."""
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            common = set(splits[a]) & set(splits[b])
            if common:
                sample = sorted(common)[:5]
                raise AssertionError(f"{len(common)} images in both {a} and {b}: {sample}")


def file_md5(path: Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def duplicate_content(paths_a: list[Path], paths_b: list[Path]) -> list[tuple[str, str]]:
    """Pairs (name_a, name_b) whose bytes are identical: leakage that stems alone cannot catch."""
    by_hash: dict[str, list[str]] = {}
    for p in paths_a:
        by_hash.setdefault(file_md5(p), []).append(Path(p).name)
    dups = []
    for p in paths_b:
        for name_a in by_hash.get(file_md5(p), []):
            dups.append((name_a, Path(p).name))
    return dups
