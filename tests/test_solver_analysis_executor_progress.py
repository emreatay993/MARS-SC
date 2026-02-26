"""Tests for progress mapping behavior in SolverAnalysisExecutor."""

import os
import sys

import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from core.data_models import SolverConfig
from ui.handlers.solver_analysis_executor import SolverAnalysisExecutor
from ui.handlers.solver_engine_factory import SolverEngineFactory


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


class _FakeStressEngineHistoryProgress:
    def check_memory_available(
        self,
        raise_on_insufficient=False,
        calculate_scalar_plasticity=False,
    ):
        _ = (raise_on_insufficient, calculate_scalar_plasticity)
        return True, {"available_bytes": 8_000_000_000}

    def compute_single_node_history_fast(self, node_id, stress_type, progress_callback=None):
        _ = (node_id, stress_type)
        if progress_callback is not None:
            progress_callback(0, 100, "loading")
            progress_callback(100, 100, "combination complete")
        return (
            np.array([0, 1], dtype=int),
            np.array([10.0, 20.0], dtype=np.float64),
        )


class _FakeReader:
    @staticmethod
    def create_single_node_scoping(node_id, _full_scoping):
        return type("Scoping", (), {"ids": [int(node_id)]})()


class _FakeDeformationEngineCylHistoryProgress:
    def __init__(self):
        self.reader1 = _FakeReader()
        self.scoping = type("Scoping", (), {"ids": [11, 22, 33]})()
        self.displacement_unit = "mm"
        self.uses_cylindrical_cs = True

    def validate_displacement_availability(self, nodal_scoping=None):
        _ = nodal_scoping
        return True, ""

    def validate_cylindrical_cs(self):
        return True, ""

    def preload_displacement_data(self, progress_callback=None):
        if progress_callback is not None:
            progress_callback(0, 10, "preload start")
            progress_callback(10, 10, "preload done")

    def compute_single_node_history(self, node_id):
        _ = node_id
        return (
            np.array([0, 1], dtype=int),
            np.array([1.0, 2.0], dtype=np.float64),
            np.array([3.0, 4.0], dtype=np.float64),
            np.array([5.0, 6.0], dtype=np.float64),
            np.array([7.0, 8.0], dtype=np.float64),
        )


def test_run_stress_history_progress_mapping_is_monotonic(monkeypatch):
    tab = _FakeTab()
    executor = SolverAnalysisExecutor(tab, SolverEngineFactory())
    fake_engine = _FakeStressEngineHistoryProgress()

    monkeypatch.setattr(executor, "_create_stress_engine", lambda: fake_engine)
    monkeypatch.setattr(executor, "_should_use_chunked_processing", lambda *_args, **_kwargs: False)

    events = []
    config = SolverConfig(
        calculate_von_mises=True,
        combination_history_mode=True,
        selected_node_id=101,
    )
    executor.run_stress_analysis(config, "von_mises", lambda c, t, m: events.append((c, t, m)))

    percents = [int((current / total) * 100) for current, total, _ in events if total > 0]
    assert percents == sorted(percents)
    assert percents[-1] == 100
    assert any(8 <= pct <= 95 for pct in percents)


def test_run_deformation_cylindrical_history_progress_mapping_is_monotonic(monkeypatch):
    tab = _FakeTab()
    executor = SolverAnalysisExecutor(tab, SolverEngineFactory())
    fake_engine = _FakeDeformationEngineCylHistoryProgress()
    monkeypatch.setattr(executor, "_create_deformation_engine", lambda _config: fake_engine)

    events = []
    config = SolverConfig(
        calculate_deformation=True,
        combination_history_mode=True,
        selected_node_id=33,
        deformation_cylindrical_cs_id=7,
    )
    executor.run_deformation_analysis(config, lambda c, t, m: events.append((c, t, m)))

    percents = [int((current / total) * 100) for current, total, _ in events if total > 0]
    assert percents == sorted(percents)
    assert percents[-1] == 100
    assert any("Computed displacement history for node" in message for _, _, message in events)

