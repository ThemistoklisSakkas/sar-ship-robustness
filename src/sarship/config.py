"""Load YAML experiment configs with optional overrides."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return a new dict where `override` is merged recursively into `base`."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def parse_dotted(overrides: list[str]) -> dict[str, Any]:
    """Turn CLI-style ["train.imgsz=512", "seed=0"] into a nested dict.

    Values are parsed with YAML, so numbers, booleans and lists get proper types.
    """
    result: dict[str, Any] = {}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must look like key.sub=value, got: {item!r}")
        dotted_key, raw_value = item.split("=", 1)
        node = result
        *parents, leaf = dotted_key.strip().split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = yaml.safe_load(raw_value)
    return result


def load_config(
    path: str | Path = DEFAULT_CONFIG,
    overrides: dict[str, Any] | list[str] | None = None,
) -> dict[str, Any]:
    """Load a YAML config and apply overrides (nested dict or dotted strings)."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if overrides:
        if isinstance(overrides, list):
            overrides = parse_dotted(overrides)
        cfg = deep_merge(cfg, overrides)
    return cfg
