"""Build the YOLO-format SSDD dataset from the official release.

Usage (Colab):
    python -m sarship.data.prepare --data-root /content/drive/MyDrive/sar-ship-robustness/data

Produces under <data_root>/<yolo_dirname>/:
    images/{train,val,test}/, labels/{train,val,test}/
    test_inshore.txt, test_offshore.txt, val_inshore.txt, val_offshore.txt
    ssdd.yaml (+ ssdd_test_inshore.yaml, ssdd_test_offshore.yaml)
    stats.json, stats.md, figures/*.png
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import yaml

from sarship.config import DEFAULT_CONFIG, load_config
from sarship.data.convert import export_split
from sarship.data.download import SSDD_DIRNAME
from sarship.data.splits import assert_disjoint, duplicate_content, stratified_train_val
from sarship.data.ssdd import LAYOUT, find_pairs, find_voc_root, scene_labels
from sarship.data.stats import box_areas, summarize
from sarship.data.viz import save_overlay_grid, save_size_histogram
from sarship.seed import make_rng

SCENES = ("inshore", "offshore")


def write_dataset_yamls(out: Path, class_names: list[str]) -> list[Path]:
    """Ultralytics dataset files; the variants differ only in which test subset they use."""
    base = {
        "path": str(out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": dict(enumerate(class_names)),
    }
    written = []
    for name, test in [("ssdd", "images/test")] + [
        (f"ssdd_test_{s}", f"test_{s}.txt") for s in SCENES
    ]:
        path = out / f"{name}.yaml"
        path.write_text(yaml.safe_dump({**base, "test": test}, sort_keys=False))
        written.append(path)
    return written


def write_scene_lists(out: Path, split: str, scenes: dict[str, str]) -> None:
    """Image lists per scene; './' paths are resolved by Ultralytics relative to the list file."""
    for scene in SCENES:
        names = sorted(p.name for p in (out / "images" / split).iterdir()
                       if scenes.get(p.stem) == scene)
        (out / f"{split}_{scene}.txt").write_text(
            "".join(f"./images/{split}/{n}\n" for n in names)
        )


def stats_markdown(
    summary: dict, image_sizes: dict, duplicates: list, excluded: list[str]
) -> str:
    bins = list(next(iter(summary.values()))["size_bins"])
    head = ["subset", "images", "ships", "empty imgs", "max ships/img", *bins, "dropped boxes"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for name, s in summary.items():
        row = [name, s["images"], s["ships"], s["empty_images"], s["max_ships_per_image"],
               *s["size_bins"].values(), s["dropped_boxes"]]
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    lines.append("")
    for split, (wmin, wmax, hmin, hmax) in image_sizes.items():
        lines.append(f"- {split} image size: width {wmin}–{wmax} px, height {hmin}–{hmax} px")
    lines.append(f"- Byte-identical train/test image pairs (official split): {len(duplicates)}")
    for train_name, test_name in duplicates:
        lines.append(f"  - train {train_name} == test {test_name}")
    lines.append(f"- Train images excluded because of those duplicates: {len(excluded)}")
    return "\n".join(lines) + "\n"


def prepare_ssdd(cfg: dict, data_root: Path) -> dict:
    ds = cfg["dataset"]
    seed = cfg["seed"]
    voc = find_voc_root(data_root / SSDD_DIRNAME, ds["raw_subdir"])
    out = data_root / ds["yolo_dirname"]
    if out.exists():
        shutil.rmtree(out)  # always rebuild from the raw release: no stale files

    pairs = {split: find_pairs(voc / img, voc / ann) for split, (img, ann) in LAYOUT.items()}
    stems = {split: [img.stem for img, _ in p] for split, p in pairs.items()}
    scenes = {split: scene_labels(voc, split, stems[split]) for split in LAYOUT}

    # The official split contains byte-identical train/test images. The test set
    # stays official (comparable with the literature); the copies leave the train pool.
    duplicates = duplicate_content(
        [img for img, _ in pairs["train"]], [img for img, _ in pairs["test"]]
    )
    excluded = sorted({Path(name).stem for name, _ in duplicates})
    if ds["drop_train_test_duplicates"]:
        train_pool = {s: g for s, g in scenes["train"].items() if s not in set(excluded)}
    else:
        excluded, train_pool = [], scenes["train"]

    train_stems, val_stems = stratified_train_val(train_pool, ds["val_fraction"], seed)
    assert_disjoint(train=train_stems, val=val_stems, test=stems["test"])

    train_set, val_set = set(train_stems), set(val_stems)
    split_pairs = {
        "train": [(img, xml) for img, xml in pairs["train"] if img.stem in train_set],
        "val": [(img, xml) for img, xml in pairs["train"] if img.stem in val_set],
        "test": pairs["test"],
    }
    scene_of = {**scenes["train"], **scenes["test"]}

    records = []
    for split, sp in split_pairs.items():
        for rec in export_split(sp, out, split, ds["min_box_size"]):
            rec["scene"] = scene_of[rec["stem"]]
            records.append(rec)

    for split in ("val", "test"):
        write_scene_lists(out, split, scene_of)
    yamls = write_dataset_yamls(out, ds["class_names"])

    bins = cfg["analysis"]["size_bins"]
    summary = summarize(records, bins)
    summary.update(summarize([{**r, "split": f"{r['split']}/{r['scene']}"} for r in records], bins))
    image_sizes = {}
    for split in split_pairs:
        ws = [r["width"] for r in records if r["split"] == split]
        hs = [r["height"] for r in records if r["split"] == split]
        image_sizes[split] = (min(ws), max(ws), min(hs), max(hs))

    md = stats_markdown(summary, image_sizes, duplicates, excluded)
    (out / "stats.md").write_text(md, encoding="utf-8")
    (out / "stats.json").write_text(json.dumps(
        {"summary": summary, "image_sizes": image_sizes, "duplicates": duplicates,
         "excluded_train_stems": excluded,
         "seed": seed, "val_fraction": ds["val_fraction"]}, indent=2))

    figures = out / "figures"
    rng = make_rng(seed)
    n = ds["viz_samples"]
    for split in split_pairs:
        imgs = sorted((out / "images" / split).iterdir())
        pick = [imgs[i] for i in sorted(rng.choice(len(imgs), size=min(n, len(imgs)),
                                                   replace=False))]
        save_overlay_grid(pick, out / "labels" / split, figures / f"gt_{split}.png",
                          title=f"{split}: ground-truth boxes from YOLO labels")
    test_inshore = sorted(p for p in (out / "images" / "test").iterdir()
                          if scene_of[p.stem] == "inshore")
    save_overlay_grid(test_inshore[:n], out / "labels" / "test", figures / "gt_test_inshore.png",
                      title="test / inshore: ground truth")
    areas = {split: np.concatenate([box_areas(r["boxes"]) for r in records if r["split"] == split])
             for split in split_pairs}
    save_size_histogram(areas, bins, figures / "size_hist.png")

    return {"out": out, "yamls": yamls, "markdown": md, "duplicates": duplicates,
            "excluded": excluded}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    report = prepare_ssdd(load_config(args.config), args.data_root)
    print(report["markdown"])
    print("Dataset files:", *report["yamls"], sep="\n  ")


if __name__ == "__main__":
    main()
