"""Dataset statistics: box counts and size distribution."""

from __future__ import annotations

from collections import Counter

import numpy as np


def box_areas(boxes: np.ndarray) -> np.ndarray:
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    return (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])


def size_category(areas: np.ndarray, bins: dict[str, list[float]]) -> np.ndarray:
    """Map each area to a bin name. Bins are [low, high) in pixels^2."""
    areas = np.asarray(areas, dtype=np.float64)
    out = np.full(areas.shape, "", dtype=object)
    for name, (low, high) in bins.items():
        out[(areas >= float(low)) & (areas < float(high))] = name
    if (out == "").any():
        raise ValueError("Some areas fall outside every size bin")
    return out


def summarize(records: list[dict], bins: dict[str, list[float]]) -> dict[str, dict]:
    """Per-split summary: images, ships, empty images, ships/image, size-bin counts."""
    summary: dict[str, dict] = {}
    for split in sorted({r["split"] for r in records}):
        recs = [r for r in records if r["split"] == split]
        counts = np.array([len(r["boxes"]) for r in recs])
        all_boxes = (
            np.concatenate([r["boxes"] for r in recs if len(r["boxes"])])
            if counts.sum()
            else np.zeros((0, 4))
        )
        cats = Counter(size_category(box_areas(all_boxes), bins)) if len(all_boxes) else {}
        summary[split] = {
            "images": len(recs),
            "ships": int(counts.sum()),
            "empty_images": int((counts == 0).sum()),
            "max_ships_per_image": int(counts.max()) if len(counts) else 0,
            "dropped_boxes": int(sum(r.get("dropped", 0) for r in recs)),
            "size_bins": {name: int(cats.get(name, 0)) for name in bins},
        }
    return summary
