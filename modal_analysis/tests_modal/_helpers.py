"""Test helpers for modal extraction."""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest

from modal_gui import dpf_modal_extractor


BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
RST_PATH = BASE_DIR / "example_rst_file_modal" / "file.rst"
STRESS_REF = BASE_DIR / "example_tensor_outputs" / "modal_stress_tensor_w_coords_2.csv"
DISP_REF = BASE_DIR / "example_tensor_outputs" / "modal_directional_deformation_w_coords_2.csv"


def require_dpf() -> None:
    pytest.importorskip("ansys.dpf.core")


def read_csv(path: Path) -> Tuple[List[str], np.ndarray]:
    with path.open("r", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        data = np.array([[float(x) for x in row] for row in reader], dtype=float)
    return header, data


def read_node_order(path: Path) -> List[int]:
    with path.open("r", newline="") as handle:
        reader = csv.reader(handle)
        _ = next(reader)
        return [int(row[0]) for row in reader]


def row_count(path: Path) -> int:
    with path.open("r") as handle:
        return sum(1 for _ in handle) - 1


def infer_mode_count(header: List[str], components: int) -> int:
    return max(1, (len(header) - 4) // components)


def find_named_selection(rst_path: Path, expected_rows: int) -> str:
    names = dpf_modal_extractor.list_named_selections(str(rst_path))
    candidates = ["All Nodes"] + list(names)

    for name in candidates:
        try:
            ids = dpf_modal_extractor.get_nodal_scoping_ids(str(rst_path), name)
        except Exception:
            continue
        if len(ids) == expected_rows:
            return name

    return "All Nodes"
