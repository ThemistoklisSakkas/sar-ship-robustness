import tarfile
import zipfile

import pytest

from sarship.data.download import describe_tree, detect_archive_format, extract_archive


@pytest.mark.parametrize(
    ("head", "expected"),
    [
        (b"PK\x03\x04rest", "zip"),
        (b"Rar!\x1a\x07\x01\x00", "rar"),
        (b"7z\xbc\xaf\x27\x1crest", "7z"),
        (b"\x1f\x8b\x08rest", "gztar"),
        (b"<!DOCTYPE html>", "unknown"),  # e.g. a Google Drive error page
    ],
)
def test_detect_archive_format_from_magic_bytes(tmp_path, head, expected):
    f = tmp_path / "file.zip"  # extension deliberately misleading
    f.write_bytes(head)
    assert detect_archive_format(f) == expected


def test_detect_plain_tar(tmp_path):
    (tmp_path / "a.txt").write_text("x")
    archive = tmp_path / "a.bin"
    with tarfile.open(archive, "w") as tf:
        tf.add(tmp_path / "a.txt", arcname="a.txt")
    assert detect_archive_format(archive) == "tar"


def test_extract_zip_and_describe(tmp_path):
    archive = tmp_path / "data.archive"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("SSDD/images/000001.jpg", b"img")
        zf.writestr("SSDD/images/000002.jpg", b"img")
        zf.writestr("SSDD/ann/000001.xml", b"<a/>")
    target = tmp_path / "out"
    extract_archive(archive, target)
    assert (target / "SSDD" / "images" / "000001.jpg").exists()
    assert not (tmp_path / "out.partial").exists()
    tree = describe_tree(target)
    assert "images/  2 .jpg" in tree
    assert "ann/  1 .xml" in tree


def test_unknown_format_leaves_no_partial_folder(tmp_path):
    archive = tmp_path / "bad.archive"
    archive.write_bytes(b"<html>quota exceeded</html>")
    with pytest.raises(ValueError, match="Unknown archive format"):
        extract_archive(archive, tmp_path / "out")
    assert not (tmp_path / "out").exists()
    assert not (tmp_path / "out.partial").exists()
