import math

import pytest

from sarship.config import DEFAULT_CONFIG, deep_merge, load_config, parse_dotted


def test_default_config_loads():
    cfg = load_config(DEFAULT_CONFIG)
    assert cfg["seed"] == 42
    assert cfg["dataset"]["class_names"] == ["ship"]
    assert math.isinf(cfg["analysis"]["size_bins"]["large"][1])


def test_deep_merge_keeps_untouched_keys_and_does_not_mutate():
    base = {"a": 1, "b": {"c": 2, "d": 3}}
    out = deep_merge(base, {"b": {"c": 20}})
    assert out == {"a": 1, "b": {"c": 20, "d": 3}}
    assert base["b"]["c"] == 2


def test_parse_dotted_types():
    out = parse_dotted(["train.imgsz=512", "seed=0", "train.deterministic=false"])
    assert out == {"train": {"imgsz": 512, "deterministic": False}, "seed": 0}


def test_parse_dotted_rejects_bad_input():
    with pytest.raises(ValueError):
        parse_dotted(["no_equals_sign"])


def test_load_config_with_list_overrides():
    cfg = load_config(DEFAULT_CONFIG, ["train.imgsz=320"])
    assert cfg["train"]["imgsz"] == 320
    assert cfg["model"]["weights"] == "yolo11n.pt"
