import numpy as np
import yaml

from sarship.config import load_config
from sarship.robust_train import DegradeTransform, attach, insert_before_format
from sarship.train import config_for_resume, default_run_name, trainer_class


class Format:  # stands in for ultralytics.data.augment.Format (matched by class name)
    def __call__(self, labels):
        return labels


class Step:
    def __call__(self, labels):
        return labels


class FakeCompose:
    def __init__(self, transforms):
        self.transforms = transforms


class FakeDataset:
    """Mimics YOLODataset: close_mosaic rebuilds the transform list from scratch."""

    def __init__(self):
        self.transforms = FakeCompose([Step(), Step(), Format()])

    def close_mosaic(self, hyp):
        self.transforms = FakeCompose([Step(), Format()])


def test_insert_before_format_once():
    t = DegradeTransform({"blur": [1, 1]}, p=1.0)
    compose = FakeCompose([Step(), Format()])
    insert_before_format(compose, t)
    insert_before_format(compose, t)  # idempotent
    assert [type(x).__name__ for x in compose.transforms] == ["Step", "DegradeTransform", "Format"]


def test_transform_survives_close_mosaic():
    ds = FakeDataset()
    t = DegradeTransform({"blur": [1, 1]}, p=1.0)
    attach(ds, t)
    assert t in ds.transforms.transforms
    ds.close_mosaic(hyp=None)
    names = [type(x).__name__ for x in ds.transforms.transforms]
    assert names == ["Step", "DegradeTransform", "Format"]


def test_degrade_transform_keeps_three_identical_channels_and_boxes():
    rng = np.random.default_rng(0)
    gray = rng.integers(0, 255, (64, 64), dtype=np.uint8)
    labels = {"img": np.dstack([gray] * 3), "instances": "untouched"}
    np.random.seed(0)
    out = DegradeTransform({"blur": [2, 2]}, p=1.0)(labels)
    img = out["img"]
    assert img.shape == (64, 64, 3) and img.dtype == np.uint8
    assert (img[..., 0] == img[..., 1]).all() and (img[..., 1] == img[..., 2]).all()
    assert not np.array_equal(img[..., 0], gray)  # it was blurred
    assert out["instances"] == "untouched"


def test_degrade_transform_p_zero_is_identity():
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    out = DegradeTransform({"blur": [2, 2]}, p=0.0)({"img": img})
    assert out["img"] is img


def test_run_names_and_default_trainer():
    cfg = load_config()
    assert default_run_name(cfg) == "baseline_yolo11n"
    assert trainer_class(cfg) is None  # plain Ultralytics trainer for the baseline
    robust = load_config(overrides=["robust_train.enabled=true"])
    assert default_run_name(robust) == "robust_yolo11n"


def test_resume_uses_saved_config(tmp_path):
    saved = load_config(overrides=["robust_train.enabled=true"])
    (tmp_path / "config_used.yaml").write_text(yaml.safe_dump(saved))
    today = load_config()  # e.g. the user forgot --set on resume
    assert config_for_resume(today, tmp_path)["robust_train"]["enabled"] is True
    assert config_for_resume(today, tmp_path / "missing") is today
