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


def test_figure_already_in_output_folder_is_not_copied_onto_itself(tmp_path):
    out = tmp_path / "report_figures"
    out.mkdir()
    (out / "00_hero.png").write_bytes(b"hero")
    copied, _ = collect(tmp_path, out)
    assert "00_hero.png" in copied and (out / "00_hero.png").read_bytes() == b"hero"
