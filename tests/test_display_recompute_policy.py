import os
import sys
from types import SimpleNamespace

import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from core.data_models import CombinationResult
from ui.handlers.display_contour_sync_handler import DisplayContourSyncHandler
from ui.handlers.display_contour_types import ContourType
from ui.handlers.display_recompute_policy import (
    get_recompute_button_text,
    get_recompute_cached_note_text,
    get_recompute_note_text,
    is_stress_on_demand_recompute_available,
)


def _make_stress_result(result_type: str, all_combo_results=None, metadata=None) -> CombinationResult:
    result = CombinationResult(
        node_ids=np.array([1, 2]),
        node_coords=np.zeros((2, 3)),
        max_over_combo=np.array([10.0, 20.0]),
        min_over_combo=np.array([5.0, 8.0]),
        combo_of_max=np.array([0, 1]),
        combo_of_min=np.array([1, 0]),
        result_type=result_type,
        all_combo_results=all_combo_results,
    )
    if metadata is not None:
        result.metadata = metadata
    return result


def test_stress_recompute_available_for_chunked_max_principal():
    result = _make_stress_result("max_principal", all_combo_results=None)
    assert is_stress_on_demand_recompute_available(result) is True


def test_stress_recompute_available_for_chunked_min_principal():
    result = _make_stress_result("min_principal", all_combo_results=None)
    assert is_stress_on_demand_recompute_available(result) is True


def test_stress_recompute_available_for_chunked_von_mises_without_plasticity_metadata():
    result = _make_stress_result("von_mises", all_combo_results=None, metadata={})
    assert is_stress_on_demand_recompute_available(result) is True


def test_stress_recompute_unavailable_when_full_combo_results_exist():
    result = _make_stress_result(
        "von_mises",
        all_combo_results=np.array([[10.0, 20.0], [12.0, 18.0]]),
    )
    assert is_stress_on_demand_recompute_available(result) is False


def test_stress_recompute_unavailable_for_unknown_result_type():
    result = _make_stress_result("equivalent_stress", all_combo_results=None)
    assert is_stress_on_demand_recompute_available(result) is False


def test_recompute_labels_switch_based_on_plasticity_metadata():
    plain_result = _make_stress_result("von_mises", all_combo_results=None, metadata={})
    corrected_result = _make_stress_result(
        "von_mises",
        all_combo_results=None,
        metadata={"plasticity": {"method": "neuber"}},
    )

    assert get_recompute_button_text(plain_result) == "Recompute This Combination"
    assert (
        get_recompute_note_text(plain_result)
        == "Chunked stress run: recompute selected combo on demand."
    )
    assert (
        get_recompute_cached_note_text(plain_result)
        == "Selected combination has cached recomputed values."
    )

    assert get_recompute_button_text(corrected_result) == "Recompute This Combination (Corrected)"
    assert (
        get_recompute_note_text(corrected_result)
        == "Chunked stress run: recompute selected combo to get corrected values."
    )
    assert (
        get_recompute_cached_note_text(corrected_result)
        == "Selected combination has cached corrected values."
    )


def test_stress_family_supports_specific_when_on_demand_recompute_is_available():
    tab = SimpleNamespace(
        all_combo_results=None,
        nodal_forces_result=None,
        deformation_result=None,
        is_stress_on_demand_recompute_available=lambda: True,
    )
    handler = DisplayContourSyncHandler(tab=tab, state=SimpleNamespace())

    assert handler._family_supports_specific(ContourType.STRESS) is True


def test_stress_family_requires_combo_data_or_on_demand_support():
    tab = SimpleNamespace(
        all_combo_results=None,
        nodal_forces_result=None,
        deformation_result=None,
    )
    handler = DisplayContourSyncHandler(tab=tab, state=SimpleNamespace())

    assert handler._family_supports_specific(ContourType.STRESS) is False
