"""Explain every error at an operating point: *why* was a ship missed, *what* is a false alarm.

Rules are applied in order; the first that fits wins.

Missed ship (FN), ground-truth box j not matched:
  merged        a TP box (matched to another ship) overlaps j: j was absorbed by a neighbour
  localisation  an unmatched prediction overlaps j but with IoU < 0.5: box there, wrong extent
  low_score     a prediction below the operating threshold has IoU >= 0.5: found, not confident
  missed        nothing anywhere near j

False alarm (FP), prediction p not matched:
  duplicate     IoU >= 0.5 with a ship that another box already claimed
  merged_box    p covers at least half of two or more ships: one box for several ships
  localisation  max IoU with any ship in [overlap_iou, 0.5): right place, wrong extent
  border        no ship overlap and p touches the image edge: often a cut-off object
  background    no ship overlap: clutter (coast, sea) or an unlabelled vessel
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from sarship.analysis.ap import ImageDetections, box_area
from sarship.analysis.matching import iou_matrix, match_detections

FN_REASONS = ("merged", "localisation", "low_score", "missed")
FP_REASONS = ("duplicate", "merged_box", "localisation", "border", "background")


def coverage(boxes: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Fraction of each target's area covered by each box: (len(boxes), len(targets))."""
    a = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(targets, dtype=np.float64).reshape(-1, 4)
    w = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]),
                0, None)
    h = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]),
                0, None)
    area_b = box_area(b)
    return np.divide(w * h, area_b[None, :], out=np.zeros((len(a), len(b))),
                     where=area_b[None, :] > 0)


def touches_border(boxes: np.ndarray, width: int, height: int, margin: float) -> np.ndarray:
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    return ((b[:, 0] <= margin) | (b[:, 1] <= margin) |
            (b[:, 2] >= width - margin) | (b[:, 3] >= height - margin))


def crowded_gt(gt: np.ndarray) -> np.ndarray:
    """True for ships whose box overlaps another ship's box (dense berths, groups)."""
    ious = iou_matrix(gt, gt)
    np.fill_diagonal(ious, 0.0)
    return (ious > 0).any(axis=1) if len(gt) else np.zeros(0, dtype=bool)


def classify_image(
    im: ImageDetections,
    conf: float,
    iou_match: float = 0.5,
    overlap_iou: float = 0.1,
    border_margin: float = 2.0,
) -> dict:
    """Per-image TP/FP/FN at `conf`, with a reason for every FP and FN."""
    keep = im.scores >= conf
    pred, scores = im.pred[keep], im.scores[keep]
    low_pred = im.pred[~keep]
    tp, matched = match_detections(pred, scores, im.gt, iou_match)

    ious = iou_matrix(pred, im.gt)  # (P, G)
    fn_reasons = []
    for j in np.where(~matched)[0]:
        if np.any(tp & (ious[:, j] >= overlap_iou)):
            fn_reasons.append((int(j), "merged"))
        elif np.any(~tp & (ious[:, j] >= overlap_iou)):
            fn_reasons.append((int(j), "localisation"))
        elif len(low_pred) and iou_matrix(low_pred, im.gt[j:j + 1]).max() >= iou_match:
            fn_reasons.append((int(j), "low_score"))
        else:
            fn_reasons.append((int(j), "missed"))

    cov = coverage(pred, im.gt)
    edge = touches_border(pred, im.width, im.height, border_margin)
    fp_reasons = []
    for i in np.where(~tp)[0]:
        max_iou = ious[i].max() if ious.shape[1] else 0.0
        if max_iou >= iou_match:
            reason = "duplicate"
        elif (cov[i] >= 0.5).sum() >= 2:
            reason = "merged_box"
        elif max_iou >= overlap_iou:
            reason = "localisation"
        elif edge[i]:
            reason = "border"
        else:
            reason = "background"
        fp_reasons.append((int(i), reason))

    return {
        "pred": pred, "scores": scores, "tp": tp, "matched": matched,
        "fn": fn_reasons, "fp": fp_reasons, "crowded": crowded_gt(im.gt),
    }


def error_summary(images: list[ImageDetections], results: list[dict]) -> dict:
    """Counts of FN and FP reasons, split by scene."""
    out: dict[str, dict] = {}
    for im, r in zip(images, results, strict=True):
        s = out.setdefault(im.scene, {"tp": 0, "fn": Counter(), "fp": Counter()})
        s["tp"] += int(r["tp"].sum())
        s["fn"].update(reason for _, reason in r["fn"])
        s["fp"].update(reason for _, reason in r["fp"])
    return {
        scene: {
            "tp": s["tp"],
            "fn": {k: s["fn"].get(k, 0) for k in FN_REASONS},
            "fp": {k: s["fp"].get(k, 0) for k in FP_REASONS},
        }
        for scene, s in sorted(out.items())
    }


def recall_by_group(
    images: list[ImageDetections], results: list[dict], size_bins: dict[str, list[float]]
) -> dict[str, dict[str, int]]:
    """Found / total ships at the operating point, per (scene, size) and crowded vs isolated."""
    groups: dict[str, list[int]] = {}

    def add(key: str, found: bool) -> None:
        g = groups.setdefault(key, [0, 0])
        g[0] += int(found)
        g[1] += 1

    for im, r in zip(images, results, strict=True):
        areas = box_area(im.gt)
        for j, found in enumerate(r["matched"]):
            size = next(name for name, (lo, hi) in size_bins.items()
                        if float(lo) <= areas[j] < float(hi))
            add(f"{im.scene}/{size}", found)
            add(f"all/{size}", found)
            add("crowded" if r["crowded"][j] else "isolated", found)
    return {k: {"found": v[0], "total": v[1]} for k, v in sorted(groups.items())}
