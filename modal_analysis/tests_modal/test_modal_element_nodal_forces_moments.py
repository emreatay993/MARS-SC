"""Smoke tests for modal ENFO/ENMO extraction."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from modal_gui import dpf_modal_extractor
from tests_modal._helpers import RST_PATH, RST_WITH_ENF, read_csv, require_dpf


def test_modal_element_nodal_forces_extracts_csv(tmp_path: Path) -> None:
    require_dpf()
    assert RST_WITH_ENF.exists(), "Missing ENF sample RST file."

    output_path = tmp_path / "modal_element_nodal_forces_w_coords.csv"
    dpf_modal_extractor.extract_modal_element_nodal_forces_csv(
        rst_path=str(RST_WITH_ENF),
        output_csv_path=str(output_path),
        named_selection="All Nodes",
        mode_count=1,
    )

    header, data = read_csv(output_path)
    assert header[:4] == ["NodeID", "X", "Y", "Z"]
    assert header[4:7] == ["enfox_Mode1", "enfoy_Mode1", "enfoz_Mode1"]
    assert data.shape[1] == 7
    assert data.shape[0] > 0
    assert np.count_nonzero(np.linalg.norm(data[:, 4:7], axis=1) > 0.0) > 0


def test_modal_element_nodal_moments_extracts_csv(tmp_path: Path) -> None:
    require_dpf()
    assert RST_WITH_ENF.exists(), "Missing ENF sample RST file."

    output_path = tmp_path / "modal_element_nodal_moments_w_coords.csv"
    dpf_modal_extractor.extract_modal_element_nodal_moments_csv(
        rst_path=str(RST_WITH_ENF),
        output_csv_path=str(output_path),
        named_selection="All Nodes",
        mode_count=1,
    )

    header, data = read_csv(output_path)
    assert header[:4] == ["NodeID", "X", "Y", "Z"]
    assert header[4:7] == ["enmox_Mode1", "enmoy_Mode1", "enmoz_Mode1"]
    assert data.shape[1] == 7
    assert data.shape[0] > 0


def test_modal_element_nodal_forces_reports_missing_result(tmp_path: Path) -> None:
    require_dpf()
    assert RST_PATH.exists(), "Missing baseline modal RST file."

    output_path = tmp_path / "modal_element_nodal_forces_w_coords.csv"
    with pytest.raises(dpf_modal_extractor.ModalExtractionError, match="not available"):
        dpf_modal_extractor.extract_modal_element_nodal_forces_csv(
            rst_path=str(RST_PATH),
            output_csv_path=str(output_path),
            named_selection="All Nodes",
            mode_count=1,
        )


def test_modal_element_nodal_forces_moments_extracts_single_csv(tmp_path: Path) -> None:
    require_dpf()
    assert RST_WITH_ENF.exists(), "Missing ENF sample RST file."

    output_path = tmp_path / "modal_element_nodal_forces_moments_w_coords.csv"
    dpf_modal_extractor.extract_modal_element_nodal_forces_moments_csv(
        rst_path=str(RST_WITH_ENF),
        output_csv_path=str(output_path),
        named_selection="All Nodes",
        mode_count=1,
    )

    header, data = read_csv(output_path)
    assert header[:4] == ["NodeID", "X", "Y", "Z"]
    assert header[4:10] == [
        "enfox_Mode1",
        "enfoy_Mode1",
        "enfoz_Mode1",
        "enmox_Mode1",
        "enmoy_Mode1",
        "enmoz_Mode1",
    ]
    assert data.shape[1] == 10
    assert data.shape[0] > 0
    assert np.count_nonzero(np.linalg.norm(data[:, 4:7], axis=1) > 0.0) > 0


def test_modal_element_nodal_forces_moments_reports_missing_result(tmp_path: Path) -> None:
    require_dpf()
    assert RST_PATH.exists(), "Missing baseline modal RST file."

    output_path = tmp_path / "modal_element_nodal_forces_moments_w_coords.csv"
    with pytest.raises(dpf_modal_extractor.ModalExtractionError, match="not available"):
        dpf_modal_extractor.extract_modal_element_nodal_forces_moments_csv(
            rst_path=str(RST_PATH),
            output_csv_path=str(output_path),
            named_selection="All Nodes",
            mode_count=1,
        )
