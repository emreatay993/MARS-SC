from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from core.data_models import CombinationTableData
from file_io.dpf_reader import DPFAnalysisReader
from solver.deformation_engine import DeformationCombinationEngine
from solver.nodal_forces_engine import NodalForcesCombinationEngine


def _reader_with_results(*results):
    def operator(unit):
        field = SimpleNamespace(unit=unit, data=np.ones((1, 3)))
        return SimpleNamespace(
            inputs=SimpleNamespace(
                time_scoping=Mock(),
                requested_location=Mock(),
                bool_rotate_to_global=Mock(),
            ),
            outputs=SimpleNamespace(fields_container=Mock(return_value=[field])),
        )

    reader = DPFAnalysisReader.__new__(DPFAnalysisReader)
    reader.model = SimpleNamespace(
        metadata=SimpleNamespace(
            result_info=SimpleNamespace(available_results=list(results))
        ),
        results=SimpleNamespace(
            stress=Mock(side_effect=lambda: operator("MPa")),
            element_nodal_forces=Mock(side_effect=lambda: operator("N")),
            displacement=Mock(side_effect=lambda: operator("mm")),
        ),
    )
    reader.get_load_step_ids = Mock(return_value=[1])
    reader._stress_unit = None
    reader._stress_conversion_factor = None
    reader._nodal_forces_available = None
    reader._displacement_available = None
    reader._force_unit = None
    reader._nodal_force_read_sequence_prepared = False
    return reader


def _table():
    return CombinationTableData(
        combination_names=["one"],
        combination_types=["Linear"],
        analysis1_coeffs=np.array([[1.0]]),
        analysis2_coeffs=np.array([[1.0]]),
        analysis1_step_ids=[1],
        analysis2_step_ids=[1],
    )


def test_align_nodal_data_handles_sorted_missing_and_unsorted_ids():
    values = np.array([[10.0, 11.0], [20.0, 21.0], [40.0, 41.0]])
    ids, aligned = DPFAnalysisReader._align_nodal_data(
        np.array([1, 2, 4]), values, np.array([4, 3, 1])
    )
    np.testing.assert_array_equal(ids, [4, 3, 1])
    np.testing.assert_array_equal(aligned, [[40.0, 41.0], [0.0, 0.0], [10.0, 11.0]])

    ids, aligned = DPFAnalysisReader._align_nodal_data(
        np.array([4, 1, 2]), values[[2, 0, 1]], np.array([2, 4, 1])
    )
    np.testing.assert_array_equal(ids, [2, 4, 1])
    np.testing.assert_array_equal(aligned, [[20.0, 21.0], [40.0, 41.0], [10.0, 11.0]])

    _, aligned = DPFAnalysisReader._align_nodal_data(
        np.array([1, 2, 2, 4]),
        np.array([[10.0], [20.0], [22.0], [40.0]]),
        np.array([2, 1]),
    )
    np.testing.assert_array_equal(aligned, [[22.0], [10.0]])


def test_units_and_availability_are_cached():
    stress = SimpleNamespace(name="stress", unit="MPa")
    force = SimpleNamespace(name="element_nodal_forces", unit="N")
    displacement = SimpleNamespace(name="displacement", unit="mm")
    reader = _reader_with_results(stress, force, displacement)

    assert reader.stress_unit == "MPa"
    assert reader.check_nodal_forces_available() is True
    assert reader.get_force_unit() == "N"
    assert reader.check_displacement_available() is True

    call_counts = (
        reader.model.results.stress.call_count,
        reader.model.results.element_nodal_forces.call_count,
        reader.model.results.displacement.call_count,
    )
    assert call_counts == (0, 0, 0)

    reader.model.metadata.result_info.available_results = []
    assert reader.stress_unit == "MPa"
    assert reader.check_nodal_forces_available() is True
    assert reader.get_force_unit() == "N"
    assert reader.check_displacement_available() is True
    assert call_counts == (
        reader.model.results.stress.call_count,
        reader.model.results.element_nodal_forces.call_count,
        reader.model.results.displacement.call_count,
    )


def test_force_prepare_replays_legacy_probe_order_once():
    stress = SimpleNamespace(name="stress", unit="MPa")
    force = SimpleNamespace(name="element_nodal_forces", unit="N")
    displacement = SimpleNamespace(name="displacement", unit="mm")
    reader = _reader_with_results(stress, force, displacement)

    events = []
    stress_factory = reader.model.results.stress.side_effect
    force_factory = reader.model.results.element_nodal_forces.side_effect
    displacement_factory = reader.model.results.displacement.side_effect
    reader.model.results.stress.side_effect = lambda: (
        events.append("stress"), stress_factory()
    )[1]
    reader.model.results.element_nodal_forces.side_effect = lambda: (
        events.append("force"), force_factory()
    )[1]
    reader.model.results.displacement.side_effect = lambda: (
        events.append("displacement"), displacement_factory()
    )[1]

    reader.prepare_nodal_forces_for_solve()
    reader.prepare_nodal_forces_for_solve()

    assert events == ["force", "displacement", "stress", "force"]
    assert reader._nodal_force_read_sequence_prepared is True


def test_force_validation_preserves_dpf_read_sequence_but_displacement_preflight_is_cached():
    readers = []
    for _ in range(2):
        reader = Mock()
        reader.check_nodal_forces_available.return_value = True
        reader.check_displacement_available.return_value = True
        readers.append(reader)

    force_engine = NodalForcesCombinationEngine(readers[0], readers[1], object(), _table())
    deformation_engine = DeformationCombinationEngine(readers[0], readers[1], object(), _table())

    assert force_engine.validate_nodal_forces_availability() == (True, "")
    assert deformation_engine.validate_displacement_availability() == (True, "")
    for reader in readers:
        reader.read_nodal_forces_for_loadstep.assert_called_once()
        reader.read_displacement_for_loadstep.assert_not_called()
