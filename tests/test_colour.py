import cv2
import numpy as np

from sarship.data.convert import colour_fraction, export_image, export_split

XML = """<annotation><filename>x.jpg</filename>
<size><width>64</width><height>48</height><depth>3</depth></size>
<object><name>ship</name><bndbox><xmin>5</xmin><ymin>5</ymin><xmax>20</xmax>
<ymax>15</ymax></bndbox></object></annotation>"""


def grey_as_bgr(h=48, w=64, seed=0):
    grey = np.random.default_rng(seed).integers(0, 255, (h, w), dtype=np.uint8)
    return np.dstack([grey, grey, grey])


def test_single_channel_and_grey_rgb_are_not_colour():
    assert colour_fraction(grey_as_bgr()[..., 0]) == 0.0
    assert colour_fraction(grey_as_bgr()) == 0.0


def test_small_jpeg_like_channel_noise_is_ignored():
    img = grey_as_bgr().astype(np.int16)
    img[..., 2] += np.random.default_rng(1).integers(-3, 4, img.shape[:2])
    assert colour_fraction(np.clip(img, 0, 255).astype(np.uint8)) == 0.0


def test_coloured_patch_is_measured():
    img = grey_as_bgr()
    img[:12, :16] = (0, 220, 255)  # yellow patch in BGR: 192 of 3072 pixels
    assert colour_fraction(img) == 192 / (48 * 64)


def test_export_image_converts_colour_to_single_channel_png(tmp_path):
    img = grey_as_bgr()
    img[:12, :16] = (255, 0, 0)
    src = tmp_path / "src.jpg"
    cv2.imwrite(str(src), img, [cv2.IMWRITE_JPEG_QUALITY, 100])
    out = tmp_path / "out"
    out.mkdir()

    dst, fraction, is_colour = export_image(src, out, True, 16, 0.001)
    assert is_colour and fraction > 0.01
    assert dst == out / "src.png" and not (out / "src.jpg").exists()
    written = cv2.imread(str(dst), cv2.IMREAD_UNCHANGED)
    assert written.ndim == 2 and written.shape == (48, 64)


def test_export_image_copies_grey_and_keeps_colour_when_disabled(tmp_path):
    grey_src = tmp_path / "grey.png"
    cv2.imwrite(str(grey_src), grey_as_bgr())
    colour_img = grey_as_bgr()
    colour_img[:, :8] = (0, 0, 255)
    colour_src = tmp_path / "colour.png"
    cv2.imwrite(str(colour_src), colour_img)
    out = tmp_path / "out"
    out.mkdir()

    assert export_image(grey_src, out, True, 16, 0.001)[2] is False
    assert (out / "grey.png").read_bytes() == grey_src.read_bytes()  # untouched copy
    dst, _, is_colour = export_image(colour_src, out, False, 16, 0.001)
    assert is_colour and cv2.imread(str(dst), cv2.IMREAD_UNCHANGED).ndim == 3


def test_export_split_records_colour_and_labels_by_stem(tmp_path):
    img = grey_as_bgr()
    img[:, :16] = (0, 220, 255)
    src = tmp_path / "000007.jpg"
    cv2.imwrite(str(src), img)
    xml = tmp_path / "000007.xml"
    xml.write_text(XML)

    [rec] = export_split([(src, xml)], tmp_path / "ds", "test", to_grayscale=True)
    assert rec["is_colour"] and rec["source"] == src
    assert (tmp_path / "ds" / "images" / "test" / "000007.png").exists()
    assert (tmp_path / "ds" / "labels" / "test" / "000007.txt").read_text().startswith("0 ")
