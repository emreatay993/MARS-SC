"""
Execution-handler tests for single-node history fast-path routing.
"""

import numpy as np

from core.data_models import SolverConfig
from ui.handlers.solver_engine_factory import SolverEngineFactory
from ui.handlers.solver_run_execution_handler import SolverRunExecutionHandler


class _FakeConsole:
    def __init__(self):
        self.messages = []

    def append(self, message):
        self.messages.append(message)


class _FakeFileHandler:
    base_reader = None
    combine_reader = None


class _FakeTab:
    def __init__(self):
        self.console_textbox = _FakeConsole()
        self.file_handler = _FakeFileHandler()

    def get_nodal_scoping_for_selected_named_selection(self):
        return None

    def get_combination_table_data(self):
        return None


class _FakeReader:
    @staticmethod
    def create_single_node_scoping(node_id, _full_scoping):
        return type("Scoping", (), {"ids": [int(node_id)]})()


class _FakeNodalForcesEngineHistoryFast:
    def __init__(self):
        self.reader1 = _FakeReader()
        self.scoping = type("Scoping", (), {"ids": [10, 20, 30]})()
        self.force_unit = "N"
        self.preload_called = False
        self.fast_called = False
        self.validated_scopes = []

    def validate_nodal_forces_availability(self, nodal_scoping=None):
        self.validated_scopes.append(nodal_scoping)
        return True, ""

    def preload_force_data(self, progress_callback=None):
        self.preload_called = True

    def compute_single_node_history_fast(self, node_id, progress_callback=None):
        self.fast_called = True
        return (
            np.array([0, 1], dtype=int),
            np.array([1.0, 2.0], dtype=np.float64),
            np.array([3.0, 4.0], dtype=np.float64),
            np.array([5.0, 6.0], dtype=np.float64),
            np.array([6.0, 7.0], dtype=np.float64),
        )


class _FakeDeformationEngineHistoryCartesianFast:
    def __init__(self):
        self.reader1 = _FakeReader()
        self.scoping = type("Scoping", (), {"ids": [11, 22, 33]})()
        self.displacement_unit = "mm"
        self.uses_cylindrical_cs = False
        self.preload_called = False
        self.fast_called = False
        self.legacy_called = False
        self.validated_scopes = []

    def validate_displacement_availability(self, nodal_scoping=None):
        self.validated_scopes.append(nodal_scoping)
        return True, ""

    def preload_displacement_data(self, progress_callback=None):
        self.preload_called = True

    def compute_single_node_history_fast(self, node_id, progress_callback=None):
        self.fast_called = True
        return (
            np.array([0, 1], dtype=int),
            np.array([0.1, 0.2], dtype=np.float64),
            np.array([0.3, 0.4], dtype=np.float64),
            np.array([0.5, 0.6], dtype=np.float64),
            np.array([0.7, 0.8], dtype=np.float64),
        )

    def compute_single_node_history(self, node_id):
        self.legacy_called = True
        return (
            np.array([0, 1], dtype=int),
            np.array([0.0, 0.0], dtype=np.float64),
            np.array([0.0, 0.0], dtype=np.float64),
            np.array([0.0, 0.0], dtype=np.float64),
            np.array([0.0, 0.0], dtype=np.float64),
        )


class _FakeDeformationEngineHistoryCylindrical:
    def __init__(self):
        self.reader1 = _FakeReader()
        self.scoping = type("Scoping", (), {"ids": [11, 22, 33]})()
        self.displacement_unit = "mm"
        self.uses_cylindrical_cs = True
        self.preload_called = False
        self.fast_called = False
        self.legacy_called = False
        self.validate_cs_called = False
        self.validated_scopes = []

    def validate_displacement_availability(self, nodal_scoping=None):
        self.validated_scopes.append(nodal_scoping)
        return True, ""

    def validate_cylindrical_cs(self):
        self.validate_cs_called = True
        return True, ""

    def preload_displacement_data(self, progress_callback=None):
        self.preload_called = True

    def compute_single_node_history_fast(self, node_id, progress_callback=None):
        self.fast_called = True
        return (
            np.array([0], dtype=int),
            np.array([0.0], dtype=np.float64),
            np.array([0.0], dtype=np.float64),
            np.array([0.0], dtype=np.float64),
            np.array([0.0], dtype=np.float64),
        )

    def compute_single_node_history(self, node_id):
        self.legacy_called = True
        return (
            np.array([0, 1], dtype=int),
            np.array([1.0, 2.0], dtype=np.float64),
            np.array([3.0, 4.0], dtype=np.float64),
            np.array([5.0, 6.0], dtype=np.float64),
            np.array([7.0, 8.0], dtype=np.float64),
        )


def test_run_nodal_forces_history_uses_fast_path(monkeypatch):
    tab = _FakeTab()
    handler = SolverRunExecutionHandler(tab, SolverEngineFactory())
    fake_engine = _FakeNodalForcesEngineHistoryFast()

    monkeypatch.setattr(handler, "_create_nodal_forces_engine", lambda _config: fake_engine)

    config = SolverConfig(
        calculate_nodal_forces=True,
        combination_history_mode=True,
        selected_node_id=42,
        nodal_forces_rotate_to_global=False,
    )

    result = handler.run_nodal_forces_analysis(config, lambda *_: None)

    assert fake_engine.fast_called is True
    assert fake_engine.preload_called is False
    assert fake_engine.validated_scopes[0] is not None
    assert list(fake_engine.validated_scopes[0].ids) == [42]
    assert result.metadata["node_id"] == 42
    np.testing.assert_allclose(result.metadata["fx"], np.array([1.0, 2.0]))


def test_run_deformation_history_cartesian_uses_fast_path(monkeypatch):
    tab = _FakeTab()
    handler = SolverRunExecutionHandler(tab, SolverEngineFactory())
    fake_engine = _FakeDeformationEngineHistoryCartesianFast()

    monkeypatch.setattr(handler, "_create_deformation_engine", lambda _config: fake_engine)

    config = SolverConfig(
        calculate_deformation=True,
        combination_history_mode=True,
        selected_node_id=77,
    )

    result = handler.run_deformation_analysis(config, lambda *_: None)

    assert fake_engine.fast_called is True
    assert fake_engine.preload_called is False
    assert fake_engine.legacy_called is False
    assert fake_engine.validated_scopes[0] is not None
    assert list(fake_engine.validated_scopes[0].ids) == [77]
    assert result.metadata["node_id"] == 77
    np.testing.assert_allclose(result.metadata["ux"], np.array([0.1, 0.2]))


def test_run_deformation_history_cylindrical_keeps_existing_path(monkeypatch):
    tab = _FakeTab()
    handler = SolverRunExecutionHandler(tab, SolverEngineFactory())
    fake_engine = _FakeDeformationEngineHistoryCylindrical()

    monkeypatch.setattr(handler, "_create_deformation_engine", lambda _config: fake_engine)

    config = SolverConfig(
        calculate_deformation=True,
        combination_history_mode=True,
        selected_node_id=33,
        deformation_cylindrical_cs_id=5,
    )

    result = handler.run_deformation_analysis(config, lambda *_: None)

    assert fake_engine.fast_called is False
    assert fake_engine.preload_called is True
    assert fake_engine.legacy_called is True
    assert fake_engine.validate_cs_called is True
    assert fake_engine.validated_scopes[0] is None
    assert result.metadata["node_id"] == 33
    np.testing.assert_allclose(result.metadata["magnitude"], np.array([7.0, 8.0]))
