"""Degradation-aware training: plug RandomDegradation into the Ultralytics data pipeline.

Where it goes: Ultralytics builds each training sample as
    mosaic/affine -> mixup -> albumentations -> HSV -> flips -> Format (to tensor).
Our transform is inserted just before `Format`, so it sees the final augmented
640x640 image, like a sensor would deliver it. Boxes are not touched.

Two traps handled here:
* `close_mosaic` rebuilds the dataset transforms for the last epochs, which would
  silently drop our transform: the dataset's `close_mosaic` is wrapped to re-insert it.
* Resuming builds a fresh trainer: `train.py` passes the same trainer class again,
  rebuilt from the run's saved config.
"""

from __future__ import annotations

import cv2
import numpy as np

from sarship.degrade import RandomDegradation


class DegradeTransform:
    """Ultralytics-style transform: takes and returns the `labels` dict of one sample."""

    def __init__(self, ranges: dict[str, list[float]], p: float):
        self.aug = RandomDegradation(ranges, p)

    def __call__(self, labels: dict) -> dict:
        img = labels["img"]
        # DataLoader workers seed NumPy's global state per worker; draw a fresh
        # generator from it so workers do not repeat each other's noise.
        rng = np.random.default_rng(np.random.randint(0, 2**31 - 1))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        out, choice = self.aug(gray, rng)
        if choice is not None:
            labels["img"] = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR) if img.ndim == 3 else out
        return labels


def insert_before_format(compose, transform) -> None:
    """Put `transform` right before Ultralytics' `Format` step (or last if absent)."""
    items = compose.transforms
    if any(t is transform for t in items):
        return
    idx = next((i for i, t in enumerate(items) if type(t).__name__ == "Format"), len(items))
    items.insert(idx, transform)


def attach(dataset, transform) -> None:
    """Add the transform now and again whenever `close_mosaic` rebuilds the transforms."""
    insert_before_format(dataset.transforms, transform)
    original_close = dataset.close_mosaic

    def close_mosaic(hyp):
        original_close(hyp)
        insert_before_format(dataset.transforms, transform)

    dataset.close_mosaic = close_mosaic


def make_trainer(cfg: dict) -> type:
    """A DetectionTrainer subclass whose training dataset gets the degradation transform."""
    from ultralytics.models.yolo.detect import DetectionTrainer

    spec = cfg["robust_train"]

    class RobustTrainer(DetectionTrainer):
        def build_dataset(self, img_path, mode="train", batch=None):
            dataset = super().build_dataset(img_path, mode, batch)
            if mode == "train":
                attach(dataset, DegradeTransform(spec["degradations"], spec["p"]))
            return dataset

    return RobustTrainer
