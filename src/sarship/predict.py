"""Run a trained detector over image files and cache the raw boxes as JSON.

Shared by the error analysis (phase 3) and the robustness study (phase 4), so both
use exactly the same prediction settings.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

Predictions = dict[str, dict[str, list]]  # {stem: {"boxes": [[x1,y1,x2,y2]], "scores": [...]}}
Predictor = Callable[[list[Path]], list[dict[str, list]]]


def weights_signature(weights: Path) -> dict:
    stat = Path(weights).stat()
    return {"weights": str(weights), "size": stat.st_size, "mtime": int(stat.st_mtime)}


def prediction_signature(weights: Path, cfg: dict) -> dict:
    return {**weights_signature(weights), "conf": cfg["analysis"]["predict_conf"],
            "iou": cfg["eval"]["iou_nms"], "imgsz": cfg["train"]["imgsz"]}


def make_predictor(weights: Path, cfg: dict, device: str | None = None) -> Predictor:
    """Ultralytics-backed predictor for .pt or .onnx weights (imported lazily for tests)."""
    from ultralytics import YOLO

    model = YOLO(str(weights), task="detect")
    extra = {} if device is None else {"device": device}

    def predict(paths: list[Path]) -> list[dict[str, list]]:
        results = model.predict(source=[str(p) for p in paths],
                                conf=cfg["analysis"]["predict_conf"],
                                iou=cfg["eval"]["iou_nms"], imgsz=cfg["train"]["imgsz"],
                                stream=True, verbose=False, **extra)
        return [{"boxes": r.boxes.xyxy.cpu().numpy().round(2).tolist(),
                 "scores": r.boxes.conf.cpu().numpy().round(5).tolist()} for r in results]

    return predict


def cached_predictions(
    cache: Path, signature: dict, stems: list[str], paths_fn: Callable[[], list[Path]],
    predictor_fn: Callable[[], Predictor],
) -> Predictions:
    """Load predictions if the cache matches `signature`, otherwise predict and save.

    `paths_fn` and `predictor_fn` are only called on a cache miss, so neither the
    (possibly degraded) images nor the model are needed when the cache is valid.
    """
    if cache.exists():
        data = json.loads(cache.read_text())
        if data.get("signature") == signature:
            return data["predictions"]
    outputs = predictor_fn()(paths_fn())
    preds = dict(zip(stems, outputs, strict=True))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"signature": signature, "predictions": preds}))
    return preds
