"""Download and unpack the official SSDD release, then print its layout.

Usage (Colab):  python -m sarship.data.download --data-root /content/drive/MyDrive/.../data
"""

from __future__ import annotations

import argparse
import zipfile
from collections import Counter
from pathlib import Path

# Official release, linked from https://github.com/TianwenZhang0825/Official-SSDD
SSDD_GDRIVE_ID = "1glNJUGotrbEyk43twwB9556AdngJsynZ"
SSDD_DIRNAME = "ssdd_raw"


def download_ssdd(data_root: Path) -> Path:
    """Download (once) and extract SSDD into data_root/ssdd_raw. Returns that folder."""
    import gdown  # only needed here, installed with the [train] extra

    data_root.mkdir(parents=True, exist_ok=True)
    archive = data_root / "ssdd_official.zip"
    target = data_root / SSDD_DIRNAME
    if not archive.exists():
        gdown.download(id=SSDD_GDRIVE_ID, output=str(archive), quiet=False)
    if not target.exists():
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(target)
    return target


def describe_tree(root: Path, max_depth: int = 4) -> str:
    """Folder tree with file counts per extension (no individual file names)."""
    lines = []
    for folder in sorted([root, *[p for p in root.rglob("*") if p.is_dir()]]):
        depth = len(folder.relative_to(root).parts)
        if depth > max_depth:
            continue
        exts = Counter(p.suffix.lower() for p in folder.iterdir() if p.is_file())
        counts = ", ".join(f"{n} {e or '(none)'}" for e, n in sorted(exts.items()))
        lines.append(f"{'  ' * depth}{folder.name}/  {counts}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    args = parser.parse_args()
    target = download_ssdd(args.data_root)
    print(describe_tree(target))


if __name__ == "__main__":
    main()
