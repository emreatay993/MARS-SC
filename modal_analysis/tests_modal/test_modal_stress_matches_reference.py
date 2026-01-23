"""Validate modal stress extraction against reference CSV."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from modal_gui import dpf_modal_extractor
from tests_modal._helpers import (
    RST_PATH,
    STRESS_REF,
    find_named_selection,
    infer_mode_count,
    read_csv,
    read_node_order,
    require_dpf,
    row_count,
)


def test_modal_stress_matches_reference(tmp_path: Path) -> None:
    require_dpf()
    assert RST_PATH.exists(), "Missing example RST file."
    assert STRESS_REF.exists(), "Missing reference stress CSV."

    ref_header, ref_data = read_csv(STRESS_REF)
    expected_rows = row_count(STRESS_REF)
    mode_count = infer_mode_count(ref_header, components=6)
    node_order = read_node_order(STRESS_REF)

    ns_name = find_named_selection(RST_PATH, expected_rows)

    output_path = tmp_path / "modal_stress_tensor_w_coords.csv"
    dpf_modal_extractor.extract_modal_stress_csv(
        rst_path=str(RST_PATH),
        output_csv_path=str(output_path),
        named_selection=ns_name,
        mode_count=mode_count,
        node_order=node_order,
    )

    out_header, out_data = read_csv(output_path)

    assert out_header == ref_header
    assert out_data.shape == ref_data.shape

    np.testing.assert_array_equal(out_data[:, 0].astype(int), ref_data[:, 0].astype(int))
    np.testing.assert_allclose(out_data[:, 1:], ref_data[:, 1:], rtol=1e-5, atol=1.0)
