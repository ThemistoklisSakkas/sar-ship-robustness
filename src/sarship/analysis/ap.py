"""COCO-style average precision, written from scratch so every number can be explained.

Why our own implementation: it lets us (1) restrict AP to a ship-size range the COCO
way, (2) bootstrap over images cheaply, and (3) show in an interview exactly how mAP
is computed. It is validated against the Ultralytics numbers of the same run.

Matching is image-local, so each image is matched once per IoU threshold and the
results are concatenated afterwards; bootstrap resampling then only re-sorts scores.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sarship.analysis.matching import iou_matrix

IOU_THRESHOLDS = np.round(np.arange(0.50, 0.951, 0.05), 2)  # 0.50, 0.55, ..., 0.95
RECALL_POINTS = np.linspace(0.0, 1.0, 101)  # COCO 101-point interpolation
ALL_AREAS = (0.0, float("inf"))


@dataclass
class ImageDetections:
    """Ground truth and predictions of one image, in original pixel coordinates."""

    stem: str
    scene: str
    width: int
    height: int
    gt: np.ndarray  # (N, 4) xyxy
    pred: np.ndarray  # (M, 4) xyxy
    scores: np.ndarray  # (M,)
    path: str = ""
    extra: dict = field(default_factory=dict)


@dataclass
class MatchTable:
    """One image matched at one IoU threshold, predictions sorted by score."""

    scores: np.ndarray
    tp: np.ndarray  # bool
    ignore: np.ndarray  # bool: excluded from AP (outside the size range)
    n_pos: int  # ground-truth boxes that count


def box_area(boxes: np.ndarray) -> np.ndarray:
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    return (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])


def in_range(areas: np.ndarray, area_range: tuple[float, float]) -> np.ndarray:
    low, high = area_range
    return (areas >= low) & (areas < high)


def match_image(
    pred: np.ndarray,
    scores: np.ndarray,
    gt: np.ndarray,
    iou_thr: float,
    area_range: tuple[float, float] = ALL_AREAS,
) -> MatchTable:
    """COCO `evaluateImg` logic for one image and one IoU threshold.

    Ground truth outside `area_range` is *ignored*: matching a prediction to it is
    neither a TP nor an FP. Unmatched predictions outside the range are ignored too,
    so a "small ships" AP is not polluted by false alarms of large size.
    """
    pred = np.asarray(pred, dtype=np.float64).reshape(-1, 4)
    gt = np.asarray(gt, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)

    gt_ignore = ~in_range(box_area(gt), area_range)
    gt_order = np.argsort(gt_ignore, kind="stable")  # counted GT first, ignored last
    gt, gt_ignore = gt[gt_order], gt_ignore[gt_order]
    dt_order = np.argsort(-scores, kind="stable")
    pred, scores = pred[dt_order], scores[dt_order]

    tp = np.zeros(len(pred), dtype=bool)
    ignore = np.zeros(len(pred), dtype=bool)
    gt_taken = np.zeros(len(gt), dtype=bool)
    ious = iou_matrix(pred, gt)
    pred_out_of_range = ~in_range(box_area(pred), area_range)

    for d in range(len(pred)):
        best, m = min(iou_thr, 1 - 1e-10), -1
        for g in range(len(gt)):
            if gt_taken[g]:
                continue
            if m > -1 and not gt_ignore[m] and gt_ignore[g]:
                break  # already matched a counted GT; do not trade it for an ignored one
            if ious[d, g] < best:
                continue
            best, m = ious[d, g], g
        if m == -1:
            ignore[d] = pred_out_of_range[d]
            continue
        gt_taken[m] = True
        tp[d] = True
        ignore[d] = gt_ignore[m]

    return MatchTable(scores=scores, tp=tp, ignore=ignore, n_pos=int((~gt_ignore).sum()))


def average_precision(tables: list[MatchTable]) -> float:
    """AP from the match tables of several images (COCO 101-point interpolation)."""
    n_pos = sum(t.n_pos for t in tables)
    if n_pos == 0:
        return float("nan")
    keep = [~t.ignore for t in tables]
    scores = np.concatenate([t.scores[k] for t, k in zip(tables, keep, strict=True)])
    tp = np.concatenate([t.tp[k] for t, k in zip(tables, keep, strict=True)])
    order = np.argsort(-scores, kind="mergesort")
    tp = tp[order]

    tps = np.cumsum(tp)
    fps = np.cumsum(~tp)
    recall = tps / n_pos
    precision = tps / np.maximum(tps + fps, np.finfo(np.float64).eps)
    # precision envelope: best precision achievable at this recall or higher
    precision = np.maximum.accumulate(precision[::-1])[::-1]

    idx = np.searchsorted(recall, RECALL_POINTS, side="left")
    sampled = np.zeros_like(RECALL_POINTS)
    valid = idx < len(precision)
    sampled[valid] = precision[idx[valid]]
    return float(sampled.mean())


def match_all(
    images: list[ImageDetections],
    area_range: tuple[float, float] = ALL_AREAS,
    thresholds: np.ndarray = IOU_THRESHOLDS,
) -> list[dict[float, MatchTable]]:
    """Per image, a {iou_threshold: MatchTable} dict."""
    return [
        {float(t): match_image(im.pred, im.scores, im.gt, float(t), area_range) for t in thresholds}
        for im in images
    ]


def map_from_tables(per_image: list[dict[float, MatchTable]]) -> dict:
    """mAP@0.5 and mAP@0.5:0.95 (single class, so mAP == AP) from pre-matched images."""
    if not per_image:
        return {"map50": float("nan"), "map50_95": float("nan"), "n_gt": 0}
    thresholds = list(per_image[0])
    aps = [average_precision([im[t] for im in per_image]) for t in thresholds]
    return {
        "map50": aps[0],
        "map50_95": float(np.mean(aps)),
        "n_gt": int(sum(im[thresholds[0]].n_pos for im in per_image)),
    }


def dataset_map(
    images: list[ImageDetections], area_range: tuple[float, float] = ALL_AREAS
) -> dict:
    return map_from_tables(match_all(images, area_range))
