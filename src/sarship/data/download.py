"""Download and unpack the official SSDD release, then print its layout.

Usage (Colab):  python -m sarship.data.download --data-root /content/drive/MyDrive/.../data
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tarfile
import zipfile
from collections import Counter
from pathlib import Path

# Official release, linked from https://github.com/TianwenZhang0825/Official-SSDD
SSDD_GDRIVE_ID = "1glNJUGotrbEyk43twwB9556AdngJsynZ"
SSDD_DIRNAME = "ssdd_raw"
# The file is saved without trusting any extension; the real format is sniffed.
ARCHIVE_NAME = "ssdd_official.archive"
LEGACY_ARCHIVE_NAME = "ssdd_official.zip"  # name used by the first version of this script

MAGIC = [
    (b"PK\x03\x04", "zip"),
    (b"Rar!\x1a\x07", "rar"),
    (b"7z\xbc\xaf\x27\x1c", "7z"),
    (b"\x1f\x8b", "gztar"),
]


def detect_archive_format(path: Path) -> str:
    """Identify an archive from its first bytes (file extensions can lie)."""
    with open(path, "rb") as f:
        head = f.read(512)
    for signature, name in MAGIC:
        if head.startswith(signature):
            return name
    if head[257:262] == b"ustar":
        return "tar"
    return "unknown"


def _run_tool(cmd: list[str], package: str) -> None:
    if shutil.which(cmd[0]) is None:
        raise RuntimeError(f"'{cmd[0]}' not found. On Colab run: !apt-get -qq install {package}")
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)


def extract_archive(archive: Path, target: Path) -> None:
    """Extract into `target`, via a temporary folder so a failure leaves no half-extracted data."""
    fmt = detect_archive_format(archive)
    partial = target.with_name(target.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)

    if fmt == "zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(partial)
    elif fmt in ("tar", "gztar"):
        with tarfile.open(archive) as tf:
            tf.extractall(partial, filter="data")
    elif fmt == "rar":
        _run_tool(["unrar", "x", "-o+", "-idq", str(archive), f"{partial}/"], "unrar")
    elif fmt == "7z":
        _run_tool(["7z", "x", "-y", f"-o{partial}", str(archive)], "p7zip-full")
    else:
        with open(archive, "rb") as f:
            head = f.read(16)
        shutil.rmtree(partial)
        raise ValueError(f"Unknown archive format, first bytes: {head!r}")

    partial.rename(target)


def download_ssdd(data_root: Path) -> Path:
    """Download (once) and extract SSDD into data_root/ssdd_raw. Returns that folder."""
    data_root.mkdir(parents=True, exist_ok=True)
    archive = data_root / ARCHIVE_NAME
    legacy = data_root / LEGACY_ARCHIVE_NAME
    if not archive.exists() and legacy.exists():
        legacy.rename(archive)  # reuse an earlier 1 GB download instead of fetching again
    if not archive.exists():
        import gdown  # only needed here, installed with the [train] extra

        gdown.download(id=SSDD_GDRIVE_ID, output=str(archive), quiet=False)

    target = data_root / SSDD_DIRNAME
    if not target.exists():
        print(f"Archive format: {detect_archive_format(archive)} — extracting...")
        extract_archive(archive, target)
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
