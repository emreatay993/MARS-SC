"""
Tests for CDB named-selection/component parsing.
"""

from pathlib import Path

from file_io.cdb_reader import CDBNamedSelectionReader


def test_parse_node_and_element_cmblocks(tmp_path: Path):
    cdb_file = tmp_path / "model.cdb"
    cdb_file.write_text(
        "\n".join(
            [
                "CMBLOCK,NS_NODES,NODE,4",
                "(8i10)",
                "         1         2         3",
                "         4",
                "CMBLOCK,NS_ELEMS,ELEM,2",
                "(8i10)",
                "        10        11",
            ]
        ),
        encoding="utf-8",
    )

    reader = CDBNamedSelectionReader.from_file(str(cdb_file))

    assert reader.get_named_selections() == ["NS_NODES", "NS_ELEMS"]
    assert reader.get_named_selection_locations() == {
        "NS_NODES": "nodal",
        "NS_ELEMS": "elemental",
    }
    assert reader.selections["NS_NODES"].ids == [1, 2, 3, 4]
    assert reader.selections["NS_ELEMS"].ids == [10, 11]


def test_parse_cmblock_ignores_unsupported_entities(tmp_path: Path):
    cdb_file = tmp_path / "model.cdb"
    cdb_file.write_text(
        "\n".join(
            [
                "CMBLOCK,ALL_NODES,NODE,2",
                "(8i10)",
                "1 2",
                "CMBLOCK,MATERIAL_COMP,MAT,2",
                "(8i10)",
                "7 8",
            ]
        ),
        encoding="utf-8",
    )

    reader = CDBNamedSelectionReader.from_file(str(cdb_file))

    assert reader.get_named_selections() == ["ALL_NODES"]
    assert "MATERIAL_COMP" not in reader.selections


def test_cdb_reader_converts_scoping_through_dpf_reader_adapter(tmp_path: Path):
    cdb_file = tmp_path / "model.cdb"
    cdb_file.write_text(
        "\n".join(
            [
                "CMBLOCK,NS_NODES,NODE,2",
                "(8i10)",
                "1 2",
                "CMBLOCK,NS_ELEMS,ELEM,2",
                "(8i10)",
                "10 11",
            ]
        ),
        encoding="utf-8",
    )

    class ReaderAdapter:
        def create_nodal_scoping_from_node_ids(self, node_ids):
            return ("nodes", list(node_ids))

        def create_nodal_scoping_from_element_ids(self, element_ids):
            return ("elements", list(element_ids))

    reader = CDBNamedSelectionReader.from_file(str(cdb_file))

    assert reader.get_nodal_scoping_from_named_selection("NS_NODES", ReaderAdapter()) == (
        "nodes",
        [1, 2],
    )
    assert reader.get_nodal_scoping_from_named_selection("NS_ELEMS", ReaderAdapter()) == (
        "elements",
        [10, 11],
    )


def test_cdb_reader_sanitizes_cmblock_name(tmp_path: Path):
    cdb_file = tmp_path / "model.cdb"
    cdb_file.write_text(
        "\n".join(
            [
                "CMBLOCK,123 bad-name$,NODE,2",
                "(8i10)",
                "1 2",
            ]
        ),
        encoding="utf-8",
    )

    reader = CDBNamedSelectionReader.from_file(str(cdb_file))

    assert reader.get_named_selections() == ["NS_123_bad_name"]
    assert reader.selections["NS_123_bad_name"].ids == [1, 2]


def test_cdb_reader_renames_conflicting_cmblock_names(tmp_path: Path):
    cdb_file = tmp_path / "model.cdb"
    cdb_file.write_text(
        "\n".join(
            [
                "CMBLOCK,RAW_NAME,NODE,2",
                "(8i10)",
                "1 2",
            ]
        ),
        encoding="utf-8",
    )

    reader = CDBNamedSelectionReader.from_file(str(cdb_file))
    reader.rename_conflicting_selections({"RAW_NAME", "RAW_NAME_2"})

    assert reader.get_named_selections() == ["RAW_NAME_3"]
    assert reader.selections["RAW_NAME_3"].ids == [1, 2]
