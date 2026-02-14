from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ui.handlers.display_contour_policy import get_available_contour_types
from ui.handlers.display_contour_types import ContourType


@dataclass
class DisplayContourContext:
    """Snapshot of display-tab state used by contour sync rules."""

    has_stress: bool
    has_forces: bool
    has_deformation: bool
    has_min_stress: bool
    has_min_forces: bool
    has_min_deformation: bool
    available_types: list[ContourType]
    is_envelope_view: bool
    view_combination_index: int
    selected_combination_index: int
    scalar_display_text: str
    force_component_index: int
    displacement_component_index: int
    active_scalar_name: Optional[str]
    inferred_active_type: Optional[ContourType]


def _safe_current_index(widget, default: int = 0) -> int:
    if widget is None:
        return default
    try:
        return int(widget.currentIndex())
    except Exception:
        return default


def _safe_current_text(widget, default: str = "") -> str:
    if widget is None:
        return default
    try:
        return str(widget.currentText())
    except Exception:
        return default


def build_contour_context(tab, mesh=None) -> DisplayContourContext:
    """Build current contour context from DisplayTab state and active mesh."""
    active_mesh = mesh if mesh is not None else tab.current_mesh
    array_names = set(active_mesh.array_names) if active_mesh is not None else set()

    has_stress_arrays = (
        "Max_Stress" in array_names
        or "Min_Stress" in array_names
        or any(name.endswith("_Stress") and name.startswith("Combo_") for name in array_names)
    )
    has_forces_arrays = (
        "Max_Force_Magnitude" in array_names
        or "Min_Force_Magnitude" in array_names
        or "Force_Magnitude" in array_names
        or "FX" in array_names
    )
    has_deformation_arrays = (
        "Def_Max_U_mag" in array_names
        or "Def_Min_U_mag" in array_names
        or "Max_U_mag" in array_names
        or "Min_U_mag" in array_names
        or "U_mag" in array_names
    )

    has_stress = has_stress_arrays or tab.all_combo_results is not None
    has_forces = has_forces_arrays or tab.nodal_forces_result is not None
    has_deformation = has_deformation_arrays or tab.deformation_result is not None

    current_result_type = tab.current_result_type

    has_min_stress = (
        "Min_Stress" in array_names and current_result_type == "min_principal"
    )

    forces_result = tab.nodal_forces_result
    has_min_forces = (
        "Min_Force_Magnitude" in array_names
        or (forces_result is not None and forces_result.min_magnitude_over_combo is not None)
    )

    deformation_result = tab.deformation_result
    has_min_deformation = (
        "Def_Min_U_mag" in array_names
        or "Min_U_mag" in array_names
        or (deformation_result is not None and deformation_result.min_magnitude_over_combo is not None)
    )

    available_types = get_available_contour_types(
        has_stress=has_stress,
        has_forces=has_forces,
        has_deformation=has_deformation,
    )

    view_index = _safe_current_index(tab.view_combination_combo, default=0)
    is_envelope_view = view_index == 0

    selected_combo_idx = view_index - 1

    active_scalar_name = None
    if active_mesh is not None:
        active_scalar_name = active_mesh.active_scalars_name

    inferred_active_type = None
    if active_scalar_name:
        if active_scalar_name.startswith("Def_") or active_scalar_name in {"U_mag", "UX", "UY", "UZ", "Max_U_mag", "Min_U_mag"}:
            inferred_active_type = ContourType.DEFORMATION
        elif active_scalar_name in {
            "Force_Magnitude", "Max_Force_Magnitude", "Min_Force_Magnitude",
            "FX", "FY", "FZ", "Shear_XY", "Shear_XZ", "Shear_YZ", "Shear_Force",
        } or active_scalar_name.startswith("Max_F") or active_scalar_name.startswith("Min_F"):
            inferred_active_type = ContourType.FORCES
        elif active_scalar_name in {"Max_Stress", "Min_Stress"} or (
            active_scalar_name.endswith("_Stress") and active_scalar_name.startswith("Combo_")
        ):
            inferred_active_type = ContourType.STRESS

    return DisplayContourContext(
        has_stress=has_stress,
        has_forces=has_forces,
        has_deformation=has_deformation,
        has_min_stress=has_min_stress,
        has_min_forces=has_min_forces,
        has_min_deformation=has_min_deformation,
        available_types=available_types,
        is_envelope_view=is_envelope_view,
        view_combination_index=view_index,
        selected_combination_index=selected_combo_idx,
        scalar_display_text=_safe_current_text(tab.scalar_display_combo, default="Max Value"),
        force_component_index=_safe_current_index(tab.force_component_combo, default=0),
        displacement_component_index=_safe_current_index(
            tab.displacement_component_combo,
            default=0,
        ),
        active_scalar_name=active_scalar_name,
        inferred_active_type=inferred_active_type,
    )
