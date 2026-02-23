"""
Policy helpers for stress on-demand recompute behavior in Display tab.
"""

from __future__ import annotations

from typing import Any


SUPPORTED_STRESS_RESULT_TYPES = {
    "von_mises",
    "max_principal",
    "min_principal",
}


def is_stress_on_demand_recompute_available(stress_result: Any) -> bool:
    """
    Return True when single-combination stress recompute can be offered in Display.

    This is intended for chunked stress envelope runs where full per-combination
    stress arrays are not retained in memory.
    """
    if stress_result is None:
        return False

    # Standard (non-chunked) runs already have all combinations available.
    if getattr(stress_result, "all_combo_results", None) is not None:
        return False

    result_type = str(getattr(stress_result, "result_type", "") or "").strip().lower()
    if result_type not in SUPPORTED_STRESS_RESULT_TYPES:
        return False

    return True


def has_plasticity_correction_metadata(stress_result: Any) -> bool:
    """Return True when stress metadata indicates plasticity-corrected values."""
    metadata = getattr(stress_result, "metadata", None) or {}
    return isinstance(metadata.get("plasticity"), dict)


def get_recompute_button_text(stress_result: Any) -> str:
    """Get button label for recompute action."""
    if has_plasticity_correction_metadata(stress_result):
        return "Recompute This Combination (Corrected)"
    return "Recompute This Combination"


def get_recompute_note_text(stress_result: Any) -> str:
    """Get helper note shown near recompute controls."""
    if has_plasticity_correction_metadata(stress_result):
        return "Chunked stress run: recompute selected combo to get corrected values."
    return "Chunked stress run: recompute selected combo on demand."


def get_recompute_cached_note_text(stress_result: Any) -> str:
    """Get helper note shown when selected combination was already recomputed."""
    if has_plasticity_correction_metadata(stress_result):
        return "Selected combination has cached corrected values."
    return "Selected combination has cached recomputed values."

