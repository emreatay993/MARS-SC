from __future__ import annotations

from typing import Iterable, Optional

from ui.handlers.display_contour_types import ContourType


def _normalize_contour_type(value: object) -> Optional[ContourType]:
    """Convert a string/enum value into ContourType."""
    if value is None:
        return None
    if isinstance(value, ContourType):
        return value
    if isinstance(value, str):
        for contour_type in ContourType:
            if value == contour_type.value:
                return contour_type
    return None


def get_available_contour_types(
    has_stress: bool,
    has_forces: bool,
    has_deformation: bool,
) -> list[ContourType]:
    """Return contour families that are currently usable."""
    available: list[ContourType] = []
    if has_stress:
        available.append(ContourType.STRESS)
    if has_forces:
        available.append(ContourType.FORCES)
    if has_deformation:
        available.append(ContourType.DEFORMATION)
    return available


def should_show_contour_type_selector(available_types: Iterable[ContourType]) -> bool:
    """Show contour selector only when two or more families are available."""
    return len(list(available_types)) >= 2


def resolve_contour_type(
    current: object,
    available_types: Iterable[ContourType],
    inferred: object = None,
) -> Optional[ContourType]:
    """Resolve the active contour type using current, inferred, then fixed fallback order."""
    available = list(available_types)
    if not available:
        return None

    current_type = _normalize_contour_type(current)
    if current_type in available:
        return current_type

    inferred_type = _normalize_contour_type(inferred)
    if inferred_type in available:
        return inferred_type

    fallback_order = [
        ContourType.STRESS,
        ContourType.FORCES,
        ContourType.DEFORMATION,
    ]
    for contour_type in fallback_order:
        if contour_type in available:
            return contour_type

    return available[0]


def get_scalar_display_items(has_min: bool) -> list[str]:
    """Get scalar-display dropdown items for envelope mode."""
    items = ["Max Value", "Combo # of Max"]
    if has_min:
        items.extend(["Min Value", "Combo # of Min"])
    return items
