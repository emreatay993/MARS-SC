from __future__ import annotations

from typing import Optional

import numpy as np


def _flatten(values) -> np.ndarray:
    return np.asarray(values).reshape(-1)


def _mesh_and_size_ok(mesh, values) -> bool:
    if mesh is None:
        return False
    if values is None:
        return False
    try:
        return int(mesh.n_points) == int(_flatten(values).shape[0])
    except Exception:
        return False


def attach_force_component_arrays(mesh, forces_result, combo_idx: int) -> bool:
    """Attach force component arrays for a specific combination."""
    if mesh is None or forces_result is None:
        return False
    if getattr(forces_result, "all_combo_fx", None) is None:
        return False

    num_combos = int(forces_result.all_combo_fx.shape[0])
    if combo_idx < 0 or combo_idx >= num_combos:
        return False

    fx = _flatten(forces_result.all_combo_fx[combo_idx, :])
    fy = _flatten(forces_result.all_combo_fy[combo_idx, :])
    fz = _flatten(forces_result.all_combo_fz[combo_idx, :])

    if not _mesh_and_size_ok(mesh, fx):
        return False

    mesh["FX"] = fx
    mesh["FY"] = fy
    mesh["FZ"] = fz
    mesh["Force_Magnitude"] = np.sqrt(fx ** 2 + fy ** 2 + fz ** 2)

    mesh["Shear_XY"] = np.sqrt(fx ** 2 + fy ** 2)
    mesh["Shear_XZ"] = np.sqrt(fx ** 2 + fz ** 2)
    mesh["Shear_YZ"] = np.sqrt(fy ** 2 + fz ** 2)
    mesh["Shear_Force"] = mesh["Shear_YZ"]

    return True


def attach_deformation_specific_arrays(mesh, deformation_result, combo_idx: int) -> bool:
    """Attach deformation component arrays for a specific combination."""
    if mesh is None or deformation_result is None:
        return False
    if getattr(deformation_result, "all_combo_ux", None) is None:
        return False

    num_combos = int(deformation_result.all_combo_ux.shape[0])
    if combo_idx < 0 or combo_idx >= num_combos:
        return False

    ux = _flatten(deformation_result.all_combo_ux[combo_idx, :])
    uy = _flatten(deformation_result.all_combo_uy[combo_idx, :])
    uz = _flatten(deformation_result.all_combo_uz[combo_idx, :])

    if not _mesh_and_size_ok(mesh, ux):
        return False

    mesh["UX"] = ux
    mesh["UY"] = uy
    mesh["UZ"] = uz
    mesh["U_mag"] = np.sqrt(ux ** 2 + uy ** 2 + uz ** 2)

    return True


def attach_deformation_envelope_arrays(mesh, deformation_result) -> bool:
    """Attach deformation envelope arrays (Def_*) to mesh."""
    if mesh is None or deformation_result is None:
        return False

    max_mag = getattr(deformation_result, "max_magnitude_over_combo", None)
    min_mag = getattr(deformation_result, "min_magnitude_over_combo", None)
    combo_of_max_mag = getattr(deformation_result, "combo_of_max", None)
    combo_of_min_mag = getattr(deformation_result, "combo_of_min", None)

    if max_mag is not None and _mesh_and_size_ok(mesh, max_mag):
        mesh["Def_Max_U_mag"] = _flatten(max_mag)
    if min_mag is not None and _mesh_and_size_ok(mesh, min_mag):
        mesh["Def_Min_U_mag"] = _flatten(min_mag)
    if combo_of_max_mag is not None and _mesh_and_size_ok(mesh, combo_of_max_mag):
        mesh["Def_Combo_of_Max_U_mag"] = _flatten(combo_of_max_mag).astype(int)
    if combo_of_min_mag is not None and _mesh_and_size_ok(mesh, combo_of_min_mag):
        mesh["Def_Combo_of_Min_U_mag"] = _flatten(combo_of_min_mag).astype(int)

    all_ux = getattr(deformation_result, "all_combo_ux", None)
    all_uy = getattr(deformation_result, "all_combo_uy", None)
    all_uz = getattr(deformation_result, "all_combo_uz", None)

    if all_ux is None or all_uy is None or all_uz is None:
        return True

    all_ux = np.asarray(all_ux)
    all_uy = np.asarray(all_uy)
    all_uz = np.asarray(all_uz)

    if all_ux.ndim != 2 or all_uy.ndim != 2 or all_uz.ndim != 2:
        return False

    if all_ux.shape[1] != int(mesh.n_points):
        return False

    # Signed directional extrema for UX/UY/UZ (not absolute extrema)
    mesh["Def_Max_UX"] = np.max(all_ux, axis=0)
    mesh["Def_Min_UX"] = np.min(all_ux, axis=0)
    mesh["Def_Combo_of_Max_UX"] = np.argmax(all_ux, axis=0).astype(int)
    mesh["Def_Combo_of_Min_UX"] = np.argmin(all_ux, axis=0).astype(int)

    mesh["Def_Max_UY"] = np.max(all_uy, axis=0)
    mesh["Def_Min_UY"] = np.min(all_uy, axis=0)
    mesh["Def_Combo_of_Max_UY"] = np.argmax(all_uy, axis=0).astype(int)
    mesh["Def_Combo_of_Min_UY"] = np.argmin(all_uy, axis=0).astype(int)

    mesh["Def_Max_UZ"] = np.max(all_uz, axis=0)
    mesh["Def_Min_UZ"] = np.min(all_uz, axis=0)
    mesh["Def_Combo_of_Max_UZ"] = np.argmax(all_uz, axis=0).astype(int)
    mesh["Def_Combo_of_Min_UZ"] = np.argmin(all_uz, axis=0).astype(int)

    return True


def build_deformation_component_payload_from_result(deformation_result) -> Optional[dict[str, np.ndarray]]:
    """Build deformation component envelope payload from DeformationResult."""
    if deformation_result is None:
        return None

    all_ux = getattr(deformation_result, "all_combo_ux", None)
    all_uy = getattr(deformation_result, "all_combo_uy", None)
    all_uz = getattr(deformation_result, "all_combo_uz", None)

    if all_ux is None or all_uy is None or all_uz is None:
        return None

    all_ux = np.asarray(all_ux)
    all_uy = np.asarray(all_uy)
    all_uz = np.asarray(all_uz)

    if all_ux.ndim != 2 or all_uy.ndim != 2 or all_uz.ndim != 2:
        return None

    return {
        "max_ux": np.max(all_ux, axis=0),
        "min_ux": np.min(all_ux, axis=0),
        "combo_of_max_ux": np.argmax(all_ux, axis=0),
        "combo_of_min_ux": np.argmin(all_ux, axis=0),
        "max_uy": np.max(all_uy, axis=0),
        "min_uy": np.min(all_uy, axis=0),
        "combo_of_max_uy": np.argmax(all_uy, axis=0),
        "combo_of_min_uy": np.argmin(all_uy, axis=0),
        "max_uz": np.max(all_uz, axis=0),
        "min_uz": np.min(all_uz, axis=0),
        "combo_of_max_uz": np.argmax(all_uz, axis=0),
        "combo_of_min_uz": np.argmin(all_uz, axis=0),
    }
