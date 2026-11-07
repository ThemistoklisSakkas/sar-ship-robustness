"""IoU and prediction-to-ground-truth matching (the core of every detection metric)."""

from __future__ import annotations

import numpy as np


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between boxes a (N, 4) and b (M, 4), both xyxy. Returns (N, M)."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def match_detections(
    pred: np.ndarray, scores: np.ndarray, gt: np.ndarray, iou_thr: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """Greedy one-to-one matching, highest score first (the COCO/VOC rule).

    Each prediction takes the unmatched ground-truth box it overlaps most, if that
    IoU >= iou_thr. Returns (tp mask over predictions, matched mask over ground truth).
    A second prediction on an already matched ship is a false positive (a duplicate).
    """
    pred = np.asarray(pred, dtype=np.float64).reshape(-1, 4)
    gt = np.asarray(gt, dtype=np.float64).reshape(-1, 4)
    tp = np.zeros(len(pred), dtype=bool)
    matched = np.zeros(len(gt), dtype=bool)
    if len(pred) == 0 or len(gt) == 0:
        return tp, matched
    ious = iou_matrix(pred, gt)
    for i in np.argsort(-np.asarray(scores), kind="stable"):
        candidates = np.where(~matched, ious[i], -1.0)
        j = int(np.argmax(candidates))
        if candidates[j] >= iou_thr:
            tp[i] = True
            matched[j] = True
    return tp, matched
