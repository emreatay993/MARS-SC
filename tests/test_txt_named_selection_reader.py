"""
Tests for TXT nodal named-selection imports.
"""

import pytest

from file_io.txt_named_selection_reader import TXTNamedSelectionReader


def test_txt_named_selection_reader_parses_tab_file_and_converts_units(tmp_path):
    txt_file = tmp_path / "nodes.txt"
    txt_file.write_text(
        "Node Number\tX Location (cm)\tY Location (m)\tZ Location (mm)\t\n"
        "11175\t0.\t1.5e-3\t50.\t\n"
        "11176\t-8.\t2.0e-3\t25.\t\n",
        encoding="utf-8",
    )

    reader = TXTNamedSelectionReader.from_file(str(txt_file), "Imported_NS")

    assert reader.get_named_selections() == ["Imported_NS"]
    assert reader.get_named_selection_locations() == {"Imported_NS": "nodal"}
    assert reader.get_named_selection_sources() == {"Imported_NS": "txt"}

    selection = reader.selections["Imported_NS"]
    assert selection.ids == [11175, 11176]
    assert selection.coordinates_mm[11175] == (0.0, 1.5, 50.0)
    assert selection.coordinates_mm[11176] == (-80.0, 2.0, 25.0)


def test_txt_named_selection_reader_rejects_missing_location_units(tmp_path):
    txt_file = tmp_path / "nodes.txt"
    txt_file.write_text(
        "Node Number\tX Location\tY Location (mm)\tZ Location (mm)\n"
        "1\t0\t0\t0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must include a unit"):
        TXTNamedSelectionReader.from_file(str(txt_file), "Imported_NS")


def test_txt_named_selection_reader_rejects_invalid_name(tmp_path):
    txt_file = tmp_path / "nodes.txt"
    txt_file.write_text(
        "Node Number\tX Location (mm)\tY Location (mm)\tZ Location (mm)\n"
        "1\t0\t0\t0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="start with a letter"):
        TXTNamedSelectionReader.from_file(str(txt_file), "1 Invalid")
