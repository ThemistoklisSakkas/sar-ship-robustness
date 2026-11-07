"""End-to-end test of the SSDD preparation on a tiny fake copy of the official layout."""

import shutil

import numpy as np
import pytest
import yaml
from PIL import Image

from sarship.config import load_config
from sarship.data.prepare import prepare_ssdd
from sarship.data.splits import duplicate_content, stratified_train_val
from sarship.data.ssdd import find_pairs, find_voc_root, scene_labels

XML = """<annotation><filename>{stem}.jpg</filename>
<size><width>{w}</width><height>{h}</height><depth>3</depth></size>
<object><name>ship</name><bndbox><xmin>10</xmin><ymin>10</ymin><xmax>{x2}</xmax>
<ymax>{y2}</ymax></bndbox></object></annotation>"""


def make_fake_ssdd(root, n_train=20, n_test=10):
    voc = root / "ssdd_raw" / "Official-SSDD-OPEN" / "BBox_SSDD" / "voc_style"
    rng = np.random.default_rng(0)
    for split, n, offset in [("train", n_train, 0), ("test", n_test, 1000)]:
        img_dir = voc / f"JPEGImages_{split}"
        ann_dir = voc / f"Annotations_{split}"
        for d in (img_dir, ann_dir, voc / f"JPEGImages_{split}_inshore",
                  voc / f"JPEGImages_{split}_offshore"):
            d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            stem = f"{offset + i:06d}"
            w, h = 120 + i, 100
            # unique pixel content per image, so no accidental duplicates
            Image.fromarray(rng.integers(0, 255, (h, w), dtype=np.uint8)).save(
                img_dir / f"{stem}.jpg"
            )
            (ann_dir / f"{stem}.xml").write_text(XML.format(stem=stem, w=w, h=h, x2=40, y2=50))
            scene = "inshore" if i % 4 == 0 else "offshore"
            shutil.copy(img_dir / f"{stem}.jpg", voc / f"JPEGImages_{split}_{scene}")
    return voc


@pytest.fixture
def cfg():
    return load_config(overrides={"dataset": {"viz_samples": 4}})


def test_find_voc_root_through_top_folder(tmp_path):
    voc = make_fake_ssdd(tmp_path, 4, 4)
    assert find_voc_root(tmp_path / "ssdd_raw") == voc


def test_find_pairs_reports_missing_xml(tmp_path):
    voc = make_fake_ssdd(tmp_path, 4, 4)
    (voc / "Annotations_train" / "000002.xml").unlink()
    with pytest.raises(ValueError, match="1 images without XML"):
        find_pairs(voc / "JPEGImages_train", voc / "Annotations_train")


def test_scene_labels_reject_image_in_both_scenes(tmp_path):
    voc = make_fake_ssdd(tmp_path, 4, 4)
    shutil.copy(voc / "JPEGImages_train_offshore" / "000001.jpg", voc / "JPEGImages_train_inshore")
    with pytest.raises(ValueError, match="in both"):
        scene_labels(voc, "train", [f"{i:06d}" for i in range(4)])


def test_stratified_split_keeps_scene_ratio():
    groups = {f"{i:04d}": ("inshore" if i < 200 else "offshore") for i in range(1000)}
    _, val = stratified_train_val(groups, 0.15, seed=42)
    inshore_val = sum(groups[s] == "inshore" for s in val)
    assert inshore_val == 30 and len(val) == 150


def test_duplicate_content_finds_identical_bytes(tmp_path):
    a, b, c = tmp_path / "a.jpg", tmp_path / "b.jpg", tmp_path / "c.jpg"
    a.write_bytes(b"same")
    b.write_bytes(b"same")
    c.write_bytes(b"other")
    assert duplicate_content([a], [b, c]) == [("a.jpg", "b.jpg")]


def test_prepare_end_to_end(tmp_path, cfg):
    make_fake_ssdd(tmp_path, n_train=20, n_test=10)
    report = prepare_ssdd(cfg, tmp_path)
    out = report["out"]

    n = {s: len(list((out / "images" / s).iterdir())) for s in ("train", "val", "test")}
    assert n["train"] + n["val"] == 20 and n["test"] == 10 and n["val"] == 3
    for s in ("train", "val", "test"):
        imgs = {p.stem for p in (out / "images" / s).iterdir()}
        lbls = {p.stem for p in (out / "labels" / s).iterdir()}
        assert imgs == lbls

    inshore = (out / "test_inshore.txt").read_text().splitlines()
    offshore = (out / "test_offshore.txt").read_text().splitlines()
    assert len(inshore) + len(offshore) == 10
    assert not set(inshore) & set(offshore)
    assert all(line.startswith("./images/test/") for line in inshore)

    data_yaml = yaml.safe_load((out / "ssdd.yaml").read_text())
    assert data_yaml["names"] == {0: "ship"} and data_yaml["val"] == "images/val"
    inshore_yaml = yaml.safe_load((out / "ssdd_test_inshore.yaml").read_text())
    assert inshore_yaml["test"] == "test_inshore.txt"

    assert report["duplicates"] == []
    assert "| test/inshore |" in report["markdown"]
    for fig in ("gt_train.png", "gt_val.png", "gt_test.png", "gt_test_inshore.png",
                "size_hist.png"):
        assert (out / "figures" / fig).stat().st_size > 0


def test_prepare_is_deterministic(tmp_path, cfg):
    make_fake_ssdd(tmp_path)
    def val_names():
        return sorted(p.name for p in (prepare_ssdd(cfg, tmp_path)["out"] / "images/val").iterdir())

    assert val_names() == val_names()


def test_train_copy_of_test_image_is_excluded(tmp_path, cfg):
    voc = make_fake_ssdd(tmp_path, n_train=20, n_test=10)
    # plant a leak: test image 001003 also appears in train under a new name
    test_img = voc / "JPEGImages_test" / "001003.jpg"
    shutil.copy(test_img, voc / "JPEGImages_train" / "000099.jpg")
    shutil.copy(test_img, voc / "JPEGImages_train_offshore" / "000099.jpg")
    shutil.copy(voc / "Annotations_test" / "001003.xml", voc / "Annotations_train" / "000099.xml")

    report = prepare_ssdd(cfg, tmp_path)
    out = report["out"]
    assert report["duplicates"] == [("000099.jpg", "001003.jpg")]
    assert report["excluded"] == ["000099"]
    kept = {p.stem for s in ("train", "val") for p in (out / "images" / s).iterdir()}
    assert "000099" not in kept and len(kept) == 20
    assert "train 000099.jpg == test 001003.jpg" in report["markdown"]


def test_colour_image_is_counted_converted_and_listed(tmp_path, cfg):
    voc = make_fake_ssdd(tmp_path, n_train=20, n_test=10)
    grey = np.asarray(Image.open(voc / "JPEGImages_test" / "001000.jpg"))
    rgb = np.dstack([grey, grey, grey])
    rgb[:30, :30] = (255, 220, 0)  # yellow patch, like the pseudo-colour in SSDD 000069
    Image.fromarray(rgb).save(voc / "JPEGImages_test" / "001000.jpg", quality=95)

    report = prepare_ssdd(cfg, tmp_path)
    out = report["out"]
    assert [c["stem"] for c in report["colour"]] == ["001000"]
    converted = out / "images" / "test" / "001000.png"
    assert converted.exists() and Image.open(converted).mode == "L"
    assert not (out / "images" / "test" / "001000.jpg").exists()
    listed = (out / "test_inshore.txt").read_text()  # 001000 is inshore (i % 4 == 0)
    assert "./images/test/001000.png" in listed
    assert "Colour images found: 1, converted to grayscale" in report["markdown"]
    assert (out / "figures" / "colour_images_raw.png").stat().st_size > 0


def test_fake_grey_dataset_has_no_colour_images(tmp_path, cfg):
    make_fake_ssdd(tmp_path, n_train=8, n_test=4)
    assert prepare_ssdd(cfg, tmp_path)["colour"] == []
