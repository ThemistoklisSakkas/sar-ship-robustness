from sarship.report import FIGURES, collect


def test_collect_copies_existing_and_reports_missing(tmp_path):
    first_src, first_name = FIGURES[0]
    (tmp_path / first_src).parent.mkdir(parents=True)
    (tmp_path / first_src).write_bytes(b"png")
    copied, missing = collect(tmp_path, tmp_path / "out")
    assert copied == [first_name]
    assert (tmp_path / "out" / first_name).read_bytes() == b"png"
    assert len(missing) == len(FIGURES) - 1


def test_figure_names_are_unique_and_ordered():
    names = [name for _, name in FIGURES]
    assert len(set(names)) == len(names) and names == sorted(names)
