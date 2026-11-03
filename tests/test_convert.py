import numpy as np
import pytest
from PIL import Image

from sarship.data.convert import (
    export_split,
    format_yolo_lines,
    valid_box_mask,
    xyxy_to_yolo,
    yolo_to_xyxy,
)
from sarship.data.voc import parse_voc_xml

VOC_XML = """<annotation>
  <filename>000001.jpg</filename>
  <size><width>{w}</width><height>{h}</height><depth>3</depth></size>
  <object><name>ship</name>
    <bndbox><xmin>10</xmin><ymin>20</ymin><xmax>30</xmax><ymax>60</ymax></bndbox></object>
  <object><name>ship</name>
    <bndbox><xmin>50</xmin><ymin>50</ymin><xmax>50</xmax><ymax>70</ymax></bndbox></object>
</annotation>"""


def test_xyxy_to_yolo_known_values():
    # 20x40 box centred at (20, 40) in a 100x200 image
    out = xyxy_to_yolo(np.array([[10, 20, 30, 60]]), width=100, height=200)
    np.testing.assert_allclose(out, [[0.2, 0.2, 0.2, 0.2]])


def test_width_and_height_are_not_swapped_for_non_square_images():
    out = xyxy_to_yolo(np.array([[0, 0, 50, 10]]), width=100, height=20)
    np.testing.assert_allclose(out, [[0.25, 0.25, 0.5, 0.5]])


def test_round_trip():
    rng = np.random.default_rng(0)
    x1y1 = rng.uniform(0, 400, size=(50, 2))
    wh = rng.uniform(1, 100, size=(50, 2))
    boxes = np.hstack([x1y1, x1y1 + wh])
    back = yolo_to_xyxy(xyxy_to_yolo(boxes, 500, 500), 500, 500)
    np.testing.assert_allclose(back, boxes, atol=1e-9)


def test_out_of_bounds_boxes_are_clipped():
    out = xyxy_to_yolo(np.array([[-5, -5, 110, 50]]), width=100, height=100)
    assert out.min() >= 0 and out.max() <= 1
    np.testing.assert_allclose(out, [[0.5, 0.25, 1.0, 0.5]])


def test_invalid_image_size_raises():
    with pytest.raises(ValueError):
        xyxy_to_yolo(np.array([[0, 0, 1, 1]]), width=0, height=10)


def test_valid_box_mask_drops_degenerate():
    mask = valid_box_mask(np.array([[0, 0, 5, 5], [3, 3, 3, 8], [0, 0, 0.5, 5]]))
    assert mask.tolist() == [True, False, False]


def test_format_yolo_lines():
    text = format_yolo_lines(np.array([0]), np.array([[0.5, 0.25, 0.1, 0.2]]))
    assert text == "0 0.500000 0.250000 0.100000 0.200000\n"
    assert format_yolo_lines(np.array([]), np.zeros((0, 4))) == ""


def test_parse_voc_xml(tmp_path):
    xml = tmp_path / "000001.xml"
    xml.write_text(VOC_XML.format(w=100, h=200))
    ann = parse_voc_xml(xml)
    assert (ann.width, ann.height) == (100, 200)
    assert ann.boxes.shape == (2, 4)
    assert ann.names == ["ship", "ship"]


def test_export_split_writes_labels_and_drops_degenerate(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    Image.new("L", (100, 200)).save(src / "000001.jpg")
    # width/height 0 in XML -> must be read from the image
    (src / "000001.xml").write_text(VOC_XML.format(w=0, h=0))

    records = export_split([(src / "000001.jpg", src / "000001.xml")], tmp_path / "out", "train")

    label = (tmp_path / "out" / "labels" / "train" / "000001.txt").read_text()
    assert label == "0 0.200000 0.200000 0.200000 0.200000\n"
    assert (tmp_path / "out" / "images" / "train" / "000001.jpg").exists()
    assert records[0]["dropped"] == 1
    assert (records[0]["width"], records[0]["height"]) == (100, 200)
