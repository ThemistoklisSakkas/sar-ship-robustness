"""Seed every random number generator we use, for reproducible runs."""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and (if installed) PyTorch.

    Note: even with deterministic=True, some CUDA kernels are not bit-exact,
    so GPU training is reproducible only approximately. We report this.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_rng(seed: int) -> np.random.Generator:
    """Independent NumPy generator, preferred over the global state in library code."""
    return np.random.default_rng(seed)
