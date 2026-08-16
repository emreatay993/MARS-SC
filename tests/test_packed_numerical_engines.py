import numpy as np

from core.data_models import CombinationTableData
from solver.deformation_engine import DeformationCombinationEngine
from solver.nodal_forces_engine import NodalForcesCombinationEngine
from solver.stress_engine import StressCombinationEngine


def _table():
    return CombinationTableData(
        combination_names=["positive", "same", "negative"],
        combination_types=["Linear"] * 3,
        analysis1_coeffs=np.array([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]]),
        analysis2_coeffs=np.array([[0.5], [0.5], [-0.5]]),
        analysis1_step_ids=[10, 20],
        analysis2_step_ids=[30],
    )


def _vector_engine(engine_type, packed):
    engine = engine_type.__new__(engine_type)
    engine.table = _table()
    engine._node_ids = np.arange(1, packed.shape[2] + 1)
    engine._node_coords = np.zeros((packed.shape[2], 3))
    engine._active_step_keys, engine._active_coefficients = engine.table.get_active_step_matrix()
    if engine_type is NodalForcesCombinationEngine:
        engine._force_data = packed
        engine._force_unit = "N"
        engine._node_element_types = None
        engine._has_beam_nodes = False
        engine.rotate_to_global = True
        engine._field_cache = {}
    else:
        engine._displacement_data = packed
        engine._displacement_unit = "mm"
        engine.cylindrical_cs_id = None
    return engine


def test_active_step_matrix_is_ordered_float64_and_contiguous():
    keys, coefficients = _table().get_active_step_matrix()
    assert keys == [(1, 10), (2, 30)]
    assert coefficients.dtype == np.float64
    assert coefficients.flags.c_contiguous
    np.testing.assert_array_equal(
        coefficients,
        [[1.0, 0.5], [1.0, 0.5], [-1.0, -0.5]],
    )


def test_packed_stress_matches_step_order_reference_and_chunk_path():
    table = _table()
    packed = np.arange(2 * 6 * 5, dtype=np.float64).reshape(2, 6, 5) / 7.0
    engine = StressCombinationEngine.__new__(StressCombinationEngine)
    engine.table = table
    engine._node_ids = np.arange(1, 6)
    engine._stress_data = packed

    expected_tensors = np.empty((3, 6, 5))
    _, coefficients = table.get_active_step_matrix()
    for combo_index in range(3):
        expected_tensors[combo_index] = 0.0
        for step_index in range(2):
            expected_tensors[combo_index] += coefficients[combo_index, step_index] * packed[step_index]
    expected = StressCombinationEngine.compute_von_mises(
        *(expected_tensors[:, component, :] for component in range(6))
    )

    actual = engine.compute_all_combinations("von_mises")
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    cache = {
        key: (engine._node_ids, *packed[index])
        for index, key in enumerate(table.get_active_step_matrix()[0])
    }
    chunked = engine._compute_chunk_combinations(cache, 5, "von_mises")
    np.testing.assert_allclose(chunked, expected, rtol=1e-12, atol=1e-12)


def test_batched_principal_stresses_handle_repeated_eigenvalues():
    sx = np.array([[2.0, 3.0], [4.0, 4.0]])
    sy = np.array([[2.0, 1.0], [4.0, 4.0]])
    sz = np.array([[2.0, 2.0], [4.0, 4.0]])
    zeros = np.zeros_like(sx)
    s1, s2, s3 = StressCombinationEngine.compute_principal_stresses_numpy(
        sx, sy, sz, zeros, zeros, zeros
    )
    np.testing.assert_array_equal(s1, [[2.0, 3.0], [4.0, 4.0]])
    np.testing.assert_array_equal(s2, [[2.0, 2.0], [4.0, 4.0]])
    np.testing.assert_array_equal(s3, [[2.0, 1.0], [4.0, 4.0]])


def test_vector_engines_keep_components_and_first_index_ties_without_magnitude_matrix():
    packed = np.array(
        [
            [[3.0, 6.0], [4.0, 8.0], [0.0, 0.0]],
            [[2.0, 4.0], [0.0, 0.0], [0.0, 0.0]],
        ]
    )
    coefficients = _table().get_active_step_matrix()[1]
    expected_components = tuple(
        coefficients @ packed[:, component, :] for component in range(3)
    )
    expected_magnitude = np.sqrt(sum(component**2 for component in expected_components))

    force = _vector_engine(NodalForcesCombinationEngine, packed).compute_full_analysis(
        auto_cleanup=False
    )
    deformation = _vector_engine(DeformationCombinationEngine, packed).compute_full_analysis(
        auto_cleanup=False
    )

    for actual_components in (
        (force.all_combo_fx, force.all_combo_fy, force.all_combo_fz),
        (deformation.all_combo_ux, deformation.all_combo_uy, deformation.all_combo_uz),
    ):
        for actual, expected in zip(actual_components, expected_components):
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    for result in (force, deformation):
        np.testing.assert_allclose(
            result.max_magnitude_over_combo,
            expected_magnitude[0],
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            result.min_magnitude_over_combo,
            expected_magnitude[0],
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_array_equal(result.combo_of_max, 0)
        np.testing.assert_array_equal(result.combo_of_min, 0)


def test_force_cancellation_is_recomputed_in_legacy_step_order():
    combined = np.array([[1.0]])
    coefficients = np.array([[1.0, 1.0]])
    step_values = np.array([[1.0e16], [-1.0e16]])

    NodalForcesCombinationEngine._restore_cancellation_order(
        combined,
        coefficients,
        step_values,
    )

    np.testing.assert_array_equal(combined, [[0.0]])
