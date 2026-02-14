import os
import sys

import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.data_models import DeformationResult
from ui.handlers.display_mesh_arrays import (
    attach_deformation_envelope_arrays,
    attach_deformation_specific_arrays,
)


class DummyMesh:
    def __init__(self, n_points: int):
        self.n_points = n_points
        self._arrays = {}

    @property
    def array_names(self):
        return list(self._arrays.keys())

    def __setitem__(self, key, value):
        self._arrays[key] = np.asarray(value)

    def __getitem__(self, key):
        return self._arrays[key]


def _build_result() -> DeformationResult:
    # Shape: (num_combinations=3, num_nodes=4)
    ux = np.array([
        [1.0, -2.0, 0.5, -4.0],
        [3.0, -1.0, -0.5, -2.0],
        [2.0, -3.0, 1.5, -6.0],
    ])
    uy = np.array([
        [0.0, -1.0, 2.0, -2.0],
        [1.0, -2.0, 1.0, -1.0],
        [2.0, -3.0, 0.0, -4.0],
    ])
    uz = np.array([
        [5.0, -1.0, 0.0, -3.0],
        [4.0, -2.0, 1.0, -5.0],
        [6.0, -0.5, -1.0, -4.0],
    ])
    mag = np.sqrt(ux ** 2 + uy ** 2 + uz ** 2)

    return DeformationResult(
        node_ids=np.array([10, 20, 30, 40]),
        node_coords=np.zeros((4, 3)),
        max_magnitude_over_combo=np.max(mag, axis=0),
        min_magnitude_over_combo=np.min(mag, axis=0),
        combo_of_max=np.argmax(mag, axis=0),
        combo_of_min=np.argmin(mag, axis=0),
        all_combo_ux=ux,
        all_combo_uy=uy,
        all_combo_uz=uz,
        displacement_unit="mm",
    )


def test_attach_deformation_envelope_arrays_exact_def_names_and_values():
    result = _build_result()
    mesh = DummyMesh(n_points=4)

    ok = attach_deformation_envelope_arrays(mesh, result)
    assert ok is True

    expected_names = {
        "Def_Max_U_mag",
        "Def_Min_U_mag",
        "Def_Combo_of_Max_U_mag",
        "Def_Combo_of_Min_U_mag",
        "Def_Max_UX",
        "Def_Min_UX",
        "Def_Combo_of_Max_UX",
        "Def_Combo_of_Min_UX",
        "Def_Max_UY",
        "Def_Min_UY",
        "Def_Combo_of_Max_UY",
        "Def_Combo_of_Min_UY",
        "Def_Max_UZ",
        "Def_Min_UZ",
        "Def_Combo_of_Max_UZ",
        "Def_Combo_of_Min_UZ",
    }

    assert expected_names.issubset(set(mesh.array_names))


def test_attach_deformation_envelope_uses_signed_extrema_for_components():
    result = _build_result()
    mesh = DummyMesh(n_points=4)

    attach_deformation_envelope_arrays(mesh, result)

    # Signed max/min for UX (not absolute extrema)
    np.testing.assert_array_equal(mesh["Def_Max_UX"], np.max(result.all_combo_ux, axis=0))
    np.testing.assert_array_equal(mesh["Def_Min_UX"], np.min(result.all_combo_ux, axis=0))

    # Signed max/min for UY
    np.testing.assert_array_equal(mesh["Def_Max_UY"], np.max(result.all_combo_uy, axis=0))
    np.testing.assert_array_equal(mesh["Def_Min_UY"], np.min(result.all_combo_uy, axis=0))

    # Signed max/min for UZ
    np.testing.assert_array_equal(mesh["Def_Max_UZ"], np.max(result.all_combo_uz, axis=0))
    np.testing.assert_array_equal(mesh["Def_Min_UZ"], np.min(result.all_combo_uz, axis=0))


def test_attach_deformation_envelope_component_combo_indices_are_correct():
    result = _build_result()
    mesh = DummyMesh(n_points=4)

    attach_deformation_envelope_arrays(mesh, result)

    np.testing.assert_array_equal(mesh["Def_Combo_of_Max_UX"], np.argmax(result.all_combo_ux, axis=0))
    np.testing.assert_array_equal(mesh["Def_Combo_of_Min_UX"], np.argmin(result.all_combo_ux, axis=0))
    np.testing.assert_array_equal(mesh["Def_Combo_of_Max_UY"], np.argmax(result.all_combo_uy, axis=0))
    np.testing.assert_array_equal(mesh["Def_Combo_of_Min_UY"], np.argmin(result.all_combo_uy, axis=0))
    np.testing.assert_array_equal(mesh["Def_Combo_of_Max_UZ"], np.argmax(result.all_combo_uz, axis=0))
    np.testing.assert_array_equal(mesh["Def_Combo_of_Min_UZ"], np.argmin(result.all_combo_uz, axis=0))


def test_attach_deformation_specific_arrays_for_selected_combination():
    result = _build_result()
    mesh = DummyMesh(n_points=4)

    ok = attach_deformation_specific_arrays(mesh, result, combo_idx=1)
    assert ok is True

    np.testing.assert_array_equal(mesh["UX"], result.all_combo_ux[1, :])
    np.testing.assert_array_equal(mesh["UY"], result.all_combo_uy[1, :])
    np.testing.assert_array_equal(mesh["UZ"], result.all_combo_uz[1, :])

    expected_umag = np.sqrt(
        result.all_combo_ux[1, :] ** 2
        + result.all_combo_uy[1, :] ** 2
        + result.all_combo_uz[1, :] ** 2
    )
    np.testing.assert_array_equal(mesh["U_mag"], expected_umag)
