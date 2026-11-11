import json

import numpy as np
from test_robustness import make_dataset

from sarship.analysis.run import load_split
from sarship.config import load_config
from sarship.hero import make_hero, outcome, pick_image


def write_preds(runs, name, cid, preds):
    d = runs / name / "robustness" / "predictions"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{cid}.json").write_text(json.dumps({"signature": {}, "predictions": preds}))


def perfect(items):
    return {it["stem"]: {"boxes": it["gt"].tolist(), "scores": [0.9] * len(it["gt"])}
            for it in items}


def test_pick_image_and_figure(tmp_path):
    cfg = load_config()
    items = load_split(make_dataset(tmp_path), "test")
    runs = tmp_path / "runs"
    target = items[3]["stem"]
    baseline_degraded = perfect(items)
    baseline_degraded[target] = {"boxes": [], "scores": []}  # baseline loses both ships here
    write_preds(runs, "base", "clean", perfect(items))
    write_preds(runs, "base", "speckle_1", baseline_degraded)
    write_preds(runs, "rob", "clean", perfect(items))
    write_preds(runs, "rob", "speckle_1", perfect(items))

    o = outcome(items[0], perfect(items)[items[0]["stem"]], cfg)
    assert (o["n_tp"], o["n_fp"], o["n_fn"]) == (2, 0, 0)

    preds = {("baseline", "clean"): perfect(items), ("baseline", "degraded"): baseline_degraded,
             ("robust", "clean"): perfect(items), ("robust", "degraded"): perfect(items)}
    assert pick_image(items, preds, cfg, min_ships=2)["stem"] == target

    out = make_hero(items, runs, cfg, tmp_path / "hero.png", "base", "rob", "speckle_1",
                    min_ships=2)
    assert out.stat().st_size > 0
    assert np.asarray(__import__("PIL.Image").Image.open(out)).ndim == 3
