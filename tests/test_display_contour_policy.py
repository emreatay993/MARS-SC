import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ui.handlers.display_contour_policy import (
    get_available_contour_types,
    should_show_contour_type_selector,
    resolve_contour_type,
    get_scalar_display_items,
)
from ui.handlers.display_contour_types import ContourType


def test_available_contour_types_detection():
    assert get_available_contour_types(True, False, False) == [ContourType.STRESS]
    assert get_available_contour_types(False, True, False) == [ContourType.FORCES]
    assert get_available_contour_types(False, False, True) == [ContourType.DEFORMATION]
    assert get_available_contour_types(True, True, True) == [
        ContourType.STRESS,
        ContourType.FORCES,
        ContourType.DEFORMATION,
    ]


def test_contour_selector_visibility_rule():
    assert should_show_contour_type_selector([]) is False
    assert should_show_contour_type_selector([ContourType.STRESS]) is False
    assert should_show_contour_type_selector([ContourType.STRESS, ContourType.DEFORMATION]) is True


def test_resolve_contour_type_prefers_current_then_inferred_then_fallback():
    available = [ContourType.FORCES, ContourType.DEFORMATION]

    # Keep current if still valid
    assert resolve_contour_type("Forces", available, inferred="Deformation") == ContourType.FORCES

    # Use inferred if current is invalid
    assert resolve_contour_type("Stress", available, inferred="Deformation") == ContourType.DEFORMATION

    # Use fallback order if neither current nor inferred is valid
    assert resolve_contour_type("Stress", available, inferred="Stress") == ContourType.FORCES


def test_scalar_display_items_include_min_when_available():
    assert get_scalar_display_items(False) == ["Max Value", "Combo # of Max"]
    assert get_scalar_display_items(True) == [
        "Max Value",
        "Combo # of Max",
        "Min Value",
        "Combo # of Min",
    ]
