"""
Runtime file loading helpers for MARS-SC.

This module intentionally contains only actively used loading paths:
- temperature field loading
- material profile loading
"""

import json
import os
from typing import Optional, Tuple

import pandas as pd

from core.data_models import TemperatureFieldData, MaterialProfileData


def _validate_table(
    section: Optional[dict],
    section_name: str,
    expected_columns,
) -> Tuple[bool, Optional[str]]:
    """Validate one tabular section of a material profile payload."""
    if section is None:
        # Missing section is treated as an empty table.
        return True, None

    if not isinstance(section, dict):
        return False, f"{section_name} section must be an object."

    columns = section.get("columns")
    data = section.get("data")
    if columns is None or data is None:
        return False, f"{section_name} section must include 'columns' and 'data'."
    if not isinstance(columns, list) or not isinstance(data, list):
        return False, f"{section_name} section must use list-based 'columns' and 'data'."

    try:
        df = pd.DataFrame(data, columns=columns)
    except Exception as exc:
        return False, f"Unable to parse {section_name} data: {exc}"

    rename_map = {
        columns[i]: expected_columns[i]
        for i in range(min(len(columns), len(expected_columns)))
    }
    df = df.rename(columns=rename_map)

    missing = [col for col in expected_columns if col not in df.columns]
    if missing:
        return False, f"{section_name} is missing columns: {', '.join(missing)}."

    df = df[expected_columns]
    try:
        for column in expected_columns:
            if df[column].empty:
                continue
            pd.to_numeric(df[column], errors="raise")
    except Exception as exc:
        return False, f"{section_name} contains non-numeric values: {exc}"

    return True, None


def _validate_material_profile_payload(payload: dict) -> Tuple[bool, Optional[str]]:
    """Validate the structure of a material profile JSON payload."""
    if not isinstance(payload, dict):
        return False, "Material profile JSON must be an object."

    youngs_valid, youngs_error = _validate_table(
        payload.get("youngs_modulus"),
        "Young's modulus",
        ["Temperature (\u00B0C)", "Young's Modulus [MPa]"],
    )
    if not youngs_valid:
        return False, youngs_error

    poisson_valid, poisson_error = _validate_table(
        payload.get("poisson_ratio"),
        "Poisson's ratio",
        ["Temperature (\u00B0C)", "Poisson's Ratio"],
    )
    if not poisson_valid:
        return False, poisson_error

    plastic_data = payload.get("plastic_curves", [])
    if plastic_data is None:
        plastic_data = []
    if not isinstance(plastic_data, list):
        return False, "Plastic curves section must be a list."

    for entry in plastic_data:
        if not isinstance(entry, dict):
            return False, "Each plastic curve entry must be an object."

        temperature = entry.get("temperature")
        if temperature is None:
            return False, "Plastic curve entries must include a temperature value."
        try:
            float(temperature)
        except (TypeError, ValueError):
            return False, f"Invalid temperature value '{temperature}' in plastic curve entry."

        curve_valid, curve_error = _validate_table(
            entry,
            f"Plastic curve (@ {temperature})",
            ["Plastic Strain", "True Stress [MPa]"],
        )
        if not curve_valid:
            return False, curve_error

    return True, None


def load_temperature_field(filename: str) -> TemperatureFieldData:
    """Load nodal temperature field data from a TXT/CSV file."""
    try:
        df = pd.read_csv(filename, sep="\t", engine="python")
        if df.shape[1] <= 1:
            df = pd.read_csv(filename, sep=r"\s+", engine="python")
    except Exception as exc:
        raise ValueError(f"Failed to parse temperature field file: {exc}") from exc

    if df.empty:
        raise ValueError("Temperature field file is empty.")

    df.columns = [col.strip() for col in df.columns]
    if "Node Number" not in df.columns:
        raise ValueError("Temperature field file must contain a 'Node Number' column.")

    return TemperatureFieldData(dataframe=df)


def _build_material_profile_dataframe(section: Optional[dict], expected_columns) -> pd.DataFrame:
    """Build a normalized numeric dataframe from a JSON table section."""
    if section is None:
        return pd.DataFrame(columns=expected_columns)

    columns = section.get("columns", expected_columns)
    data = section.get("data", [])
    df = pd.DataFrame(data, columns=columns)

    rename_map = {
        columns[i]: expected_columns[i]
        for i in range(min(len(columns), len(expected_columns)))
    }
    df = df.rename(columns=rename_map)

    missing = [col for col in expected_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {', '.join(missing)}")

    df = df[expected_columns]
    for column in expected_columns:
        if df[column].empty:
            continue
        df[column] = pd.to_numeric(df[column], errors="raise")

    return df


def load_material_profile(filename: str) -> MaterialProfileData:
    """Load a material profile JSON file into a MaterialProfileData object."""
    if not os.path.exists(filename):
        raise ValueError("File does not exist.")

    try:
        with open(filename, "r", encoding="utf-8-sig") as fh:
            payload = json.load(fh)
    except Exception as exc:
        raise ValueError(f"Failed to read material profile: {exc}") from exc

    is_valid, error = _validate_material_profile_payload(payload)
    if not is_valid:
        raise ValueError(f"Invalid material profile: {error}")

    youngs_df = _build_material_profile_dataframe(
        payload.get("youngs_modulus"),
        ["Temperature (\u00B0C)", "Young's Modulus [MPa]"],
    )
    poisson_df = _build_material_profile_dataframe(
        payload.get("poisson_ratio"),
        ["Temperature (\u00B0C)", "Poisson's Ratio"],
    )

    plastic_curves = {}
    for entry in payload.get("plastic_curves", []):
        temperature = float(entry.get("temperature"))
        curve_df = _build_material_profile_dataframe(
            entry,
            ["Plastic Strain", "True Stress [MPa]"],
        )
        plastic_curves[temperature] = curve_df

    return MaterialProfileData(
        youngs_modulus=youngs_df,
        poisson_ratio=poisson_df,
        plastic_curves=plastic_curves,
    )
