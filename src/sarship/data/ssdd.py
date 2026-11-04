"""Knowledge of the official SSDD folder layout (BBox_SSDD/voc_style)."""

from __future__ import annotations

from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

# Folder names inside BBox_SSDD/voc_style of the official release.
LAYOUT = {
    "train": ("JPEGImages_train", "Annotations_train"),
    "test": ("JPEGImages_test", "Annotations_test"),
}
SCENE_FOLDERS = {
    ("train", "inshore"): "JPEGImages_train_inshore",
    ("train", "offshore"): "JPEGImages_train_offshore",
    ("test", "inshore"): "JPEGImages_test_inshore",
    ("test", "offshore"): "JPEGImages_test_offshore",
}


def find_voc_root(raw_root: Path, subdir: str = "BBox_SSDD/voc_style") -> Path:
    """Locate e.g. BBox_SSDD/voc_style anywhere below raw_root (archive top folders vary)."""
    direct = raw_root / subdir
    if direct.is_dir():
        return direct
    tail = "/" + Path(subdir).as_posix()
    matches = sorted(p for p in raw_root.rglob("*") if p.is_dir() and p.as_posix().endswith(tail))
    if not matches:
        raise FileNotFoundError(f"Could not find '{subdir}' under {raw_root}")
    return matches[0]


def image_stems(folder: Path) -> dict[str, Path]:
    return {p.stem: p for p in sorted(folder.iterdir()) if p.suffix.lower() in IMAGE_EXTS}


def find_pairs(image_dir: Path, ann_dir: Path) -> list[tuple[Path, Path]]:
    """Match images to XML annotations by file stem; fail loudly on any mismatch."""
    images = image_stems(image_dir)
    xmls = {p.stem: p for p in sorted(ann_dir.glob("*.xml"))}
    missing_xml = sorted(set(images) - set(xmls))
    missing_img = sorted(set(xmls) - set(images))
    if missing_xml or missing_img:
        raise ValueError(
            f"Unpaired files in {image_dir.name}/{ann_dir.name}: "
            f"{len(missing_xml)} images without XML {missing_xml[:5]}, "
            f"{len(missing_img)} XML without image {missing_img[:5]}"
        )
    return [(images[s], xmls[s]) for s in sorted(images)]


def scene_labels(voc_root: Path, split: str, stems: list[str]) -> dict[str, str]:
    """Return {stem: 'inshore'|'offshore'} using the official scene folders.

    Raises if a stem is in both scenes or in neither (the protocol says each
    image belongs to exactly one).
    """
    inshore = set(image_stems(voc_root / SCENE_FOLDERS[(split, "inshore")]))
    offshore = set(image_stems(voc_root / SCENE_FOLDERS[(split, "offshore")]))
    both = inshore & offshore
    neither = set(stems) - inshore - offshore
    extra = (inshore | offshore) - set(stems)
    if both or neither or extra:
        raise ValueError(
            f"{split}: scene labels inconsistent — in both: {sorted(both)[:5]}, "
            f"in neither: {sorted(neither)[:5]}, not in split: {sorted(extra)[:5]}"
        )
    return {s: ("inshore" if s in inshore else "offshore") for s in stems}
