"""
Utilities for preparing plasticity solver inputs.

This module bridges UI/domain data models (material profiles, temperature
fields) with the low-level plasticity engine data structures.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from core.data_models import MaterialProfileData, TemperatureFieldData
from solver.plasticity_engine import MaterialDB


class PlasticityDataError(ValueError):
    """Raised when required plasticity data is missing or inconsistent."""


def _get_column(df: pd.DataFrame, candidates: Sequence[str], label: str) -> pd.Series:
    """Return the first matching column from ``candidates`` or raise."""
    for name in candidates:
        if name in df.columns:
            return df[name]
    raise PlasticityDataError(f"Expected column for {label} not found. Tried: {', '.join(candidates)}")


def _sorted_numeric(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float64)
    order = np.argsort(arr)
    return arr[order]


def _resample_curve(strain_src: np.ndarray, stress_src: np.ndarray, strain_target: np.ndarray) -> np.ndarray:
    """Piecewise-linear resample with linear extrapolation at the ends."""
    stress_target = np.interp(strain_target, strain_src, stress_src)

    # Linear extrapolation on the low end if needed
    mask_low = strain_target < strain_src[0]
    if mask_low.any() and strain_src.size >= 2:
        slope = (stress_src[1] - stress_src[0]) / (strain_src[1] - strain_src[0] + 1e-12)
        stress_target[mask_low] = stress_src[0] + slope * (strain_target[mask_low] - strain_src[0])

    # Linear extrapolation on the high end if needed
    mask_high = strain_target > strain_src[-1]
    if mask_high.any() and strain_src.size >= 2:
        slope = (stress_src[-1] - stress_src[-2]) / (strain_src[-1] - strain_src[-2] + 1e-12)
        stress_target[mask_high] = stress_src[-1] + slope * (strain_target[mask_high] - strain_src[-1])

    return stress_target


def build_material_db_from_profile(profile: MaterialProfileData) -> MaterialDB:
    """
    Convert a :class:`MaterialProfileData` instance into a ``MaterialDB``.

    The helper validates that:
      * At least one plastic curve exists.
      * Each plastic curve provides both stress and plastic strain columns.
      * Young's modulus data is available for interpolation across all
        temperatures represented by the plastic curves.
    """
    if profile is None or not profile.has_data:
        raise PlasticityDataError("Material profile is empty; plasticity correction cannot proceed.")
    if not profile.plastic_curves:
        raise PlasticityDataError("Material profile does not contain any plastic curves.")

    temperatures = _sorted_numeric(profile.plastic_curves.keys())
    if temperatures.size == 0:
        raise PlasticityDataError("Material profile does not contain any plastic curve datasets.")

    # Build SIG/EPSP arrays
    curve_records = []
    max_points = -1
    target_strain = None

    for temp in temperatures:
        curve_df = profile.plastic_curves[temp]
        if curve_df.empty:
            raise PlasticityDataError(f"Plastic curve for temperature {temp} °C is empty.")

        stress_series = _get_column(
            curve_df,
            ["True Stress [MPa]", "True Stress", "Stress [MPa]"],
            "plastic curve stress",
        )
        strain_series = _get_column(
            curve_df,
            ["Plastic Strain", "Equivalent Plastic Strain"],
            "plastic curve strain",
        )

        stress = np.asarray(stress_series, dtype=np.float64)
        strain = np.asarray(strain_series, dtype=np.float64)
        if stress.shape != strain.shape:
            raise PlasticityDataError(f"Stress/strain length mismatch for plastic curve at {temp} °C.")
        if stress.size < 2:
            raise PlasticityDataError(f"Plastic curve at {temp} °C must have at least two points.")
        if not np.all(np.diff(strain) > 0):
            raise PlasticityDataError(f"Plastic strain values must be strictly increasing for temperature {temp} °C.")

        if stress.size > max_points:
            max_points = stress.size
            target_strain = strain

        curve_records.append((temp, strain, stress))

    if target_strain is None:
        raise PlasticityDataError("No valid plastic curves found in material profile.")

    sig_rows = []
    for temp, strain, stress in curve_records:
        if strain.shape[0] != target_strain.shape[0] or not np.allclose(strain, target_strain):
            stress = _resample_curve(strain, stress, target_strain)
        sig_rows.append(stress)

    SIG = np.vstack(sig_rows)
    EPSP = np.vstack([target_strain for _ in curve_records])

    # Interpolate Young's modulus values for each plastic curve temperature.
    youngs_df = profile.youngs_modulus
    if youngs_df.empty:
        raise PlasticityDataError("Young's modulus table is required for plasticity correction.")

    temp_series = _get_column(
        youngs_df,
        ["Temperature (°C)", "Temperature", "Temp"],
        "Young's modulus temperature",
    )
    modulus_series = _get_column(
        youngs_df,
        ["Young's Modulus [MPa]", "Young's Modulus", "E [MPa]"],
        "Young's modulus values",
    )

    youngs_temps = np.asarray(temp_series, dtype=np.float64)
    youngs_vals = np.asarray(modulus_series, dtype=np.float64)
    order = np.argsort(youngs_temps)
    youngs_temps = youngs_temps[order]
    youngs_vals = youngs_vals[order]

    if youngs_temps.size == 0:
        raise PlasticityDataError("Young's modulus table does not contain any entries.")

    E_tab = np.interp(temperatures, youngs_temps, youngs_vals)

    return MaterialDB.from_arrays(temperatures, E_tab, SIG, EPSP)


def infer_temperature_column(temperature_data: TemperatureFieldData) -> str:
    """
    Pick a sensible default temperature column from :class:`TemperatureFieldData`.
    """
    df = temperature_data.dataframe
    candidates = [
        col for col in df.columns
        if col.lower() not in {"node number", "node", "node_id", "nodeid"}
    ]
    if not candidates:
        raise PlasticityDataError("No temperature columns found in temperature field file.")

    preferred = [c for c in candidates if "temp" in c.lower()]
    return preferred[0] if preferred else candidates[0]


def map_temperature_field_to_nodes(
    temperature_data: TemperatureFieldData,
    node_ids: np.ndarray,
    column_name: Optional[str] = None,
    default_temperature: Optional[float] = None,
) -> np.ndarray:
    """
    Create a temperature array aligned with ``node_ids``.

    Args:
        temperature_data: Parsed temperature field table.
        node_ids: Array of node IDs used by the solver.
        column_name: Optional explicit column to read. When omitted the first
            non-node column (preferring anything containing “temp”) is used.
        default_temperature: Constant value used for nodes missing from the
            table. When ``None`` any missing node triggers an error.
    """
    df = temperature_data.dataframe.copy()
    node_column_name = _get_column(df, ["Node Number", "Node", "NodeID", "Node Id"], "node identifiers").name
    
    df.drop_duplicates(subset=[node_column_name], keep="last", inplace=True)
    df = df.set_index(df[node_column_name].astype(int))

    column = column_name or infer_temperature_column(temperature_data)
    if column not in df.columns:
        raise PlasticityDataError(f"Temperature column '{column}' not present in temperature field data.")

    series = df[column].astype(float)
    aligned = series.reindex(node_ids, copy=False)
    missing_mask = aligned.isna()
    if missing_mask.any():
        if default_temperature is None:
            missing_ids = node_ids[missing_mask.to_numpy()]
            preview = ", ".join(str(int(n)) for n in missing_ids[:5])
            raise PlasticityDataError(
                f"Temperature values are missing for {missing_mask.sum()} node(s): {preview} ..."
            )
        aligned = aligned.fillna(float(default_temperature))

    return aligned.to_numpy(dtype=np.float64)


def extract_poisson_ratio(profile: MaterialProfileData, default: float = 0.3) -> float:
    """
    Retrieve a representative Poisson ratio from the material profile.

    Returns ``default`` when no Poisson data is supplied.
    """
    if profile is None or profile.poisson_ratio.empty:
        return float(default)

    df = profile.poisson_ratio
    ratio_series = _get_column(
        df,
        ["Poisson's Ratio", "Poisson Ratio", "Nu"],
        "Poisson's ratio",
    )
    values = np.asarray(ratio_series, dtype=np.float64)
    if values.size == 0:
        return float(default)
    return float(np.clip(values.mean(), 0.0, 0.4999))


__all__ = [
    "PlasticityDataError",
    "build_material_db_from_profile",
    "map_temperature_field_to_nodes",
    "infer_temperature_column",
    "extract_poisson_ratio",
]
