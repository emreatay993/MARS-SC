"""Tests for staged solve progress orchestration in SolveRunController."""

import os
import sys
import types

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

def _ensure_pyqt5_stubs() -> None:
    pyqt5 = sys.modules.get("PyQt5")
    if pyqt5 is None:
        pyqt5 = types.ModuleType("PyQt5")
        pyqt5.__path__ = []
        sys.modules["PyQt5"] = pyqt5
    elif not hasattr(pyqt5, "__path__"):
        pyqt5.__path__ = []

    qtcore = sys.modules.get("PyQt5.QtCore")
    if qtcore is None:
        qtcore = types.ModuleType("PyQt5.QtCore")
        sys.modules["PyQt5.QtCore"] = qtcore

    qtwidgets = sys.modules.get("PyQt5.QtWidgets")
    if qtwidgets is None:
        qtwidgets = types.ModuleType("PyQt5.QtWidgets")
        sys.modules["PyQt5.QtWidgets"] = qtwidgets

    class _FakeQApplication:
        @staticmethod
        def processEvents(*_args, **_kwargs):
            return None

    class _FakeQMessageBox:
        @staticmethod
        def critical(*_args, **_kwargs):
            return None

        @staticmethod
        def warning(*_args, **_kwargs):
            return None

    class _FakeQEventLoop:
        ExcludeUserInputEvents = 0

    class _FakeQTimer:
        @staticmethod
        def singleShot(_ms, callback):
            callback()

    qtcore.QEventLoop = _FakeQEventLoop
    qtcore.QTimer = _FakeQTimer
    qtwidgets.QApplication = _FakeQApplication
    qtwidgets.QMessageBox = _FakeQMessageBox
    pyqt5.QtCore = qtcore
    pyqt5.QtWidgets = qtwidgets


_ensure_pyqt5_stubs()

from core.data_models import SolverConfig
from ui.handlers.solve_run_controller import SolveRunController


class _FakeTab:
    def _build_solver_config(self):
        return SolverConfig()


class _FakeValidator:
    def validate_inputs(self, _config):
        return True

    def get_selected_stress_type(self, _config):
        return "von_mises"


class _FakeLifecycle:
    def __init__(self):
        self.begin_calls = 0
        self.stage_messages = []
        self.progress_events = []
        self.completed = False
        self.finished_without_results = False

    def begin_solve(self, _config):
        self.begin_calls += 1

    def announce_stage(self, message):
        self.stage_messages.append(message)

    def update_progress(self, current, total, message):
        self.progress_events.append((current, total, message))

    def complete_solve(self, **_kwargs):
        self.completed = True

    def finish_without_results(self):
        self.finished_without_results = True

    def handle_engine_creation_error(self, *_args, **_kwargs):
        raise AssertionError("Unexpected engine creation error in test.")

    def handle_nodal_forces_unavailable(self, *_args, **_kwargs):
        raise AssertionError("Unexpected nodal-forces unavailable error in test.")

    def handle_displacement_unavailable(self, *_args, **_kwargs):
        raise AssertionError("Unexpected displacement unavailable error in test.")

    def handle_cylindrical_cs_error(self, *_args, **_kwargs):
        raise AssertionError("Unexpected cylindrical CS error in test.")

    def handle_memory_error(self, *_args, **_kwargs):
        raise AssertionError("Unexpected memory error in test.")

    def fail_solve(self, *_args, **_kwargs):
        raise AssertionError("Unexpected generic solve failure in test.")


class _FakeExecutor:
    def run_stress_analysis(self, config, stress_type, progress_callback):
        _ = (config, stress_type)
        progress_callback(0, 100, "starting")
        progress_callback(100, 100, "done")
        return {"stress": True}

    def run_nodal_forces_analysis(self, config, progress_callback):
        _ = config
        progress_callback(0, 100, "starting")
        progress_callback(100, 100, "done")
        return {"forces": True}

    def run_deformation_analysis(self, config, progress_callback):
        _ = config
        progress_callback(0, 100, "starting")
        progress_callback(100, 100, "done")
        return {"deformation": True}

    def get_stress_engine(self):
        return None


def test_solve_run_controller_uses_global_monotonic_stage_progress():
    controller = SolveRunController(_FakeTab())
    controller.input_validator = _FakeValidator()
    lifecycle = _FakeLifecycle()
    controller.lifecycle_handler = lifecycle
    controller.execution_handler = _FakeExecutor()

    config = SolverConfig(
        calculate_von_mises=True,
        calculate_nodal_forces=True,
        calculate_deformation=True,
    )
    controller.solve(config)

    assert lifecycle.begin_calls == 1
    assert lifecycle.completed is True
    assert lifecycle.finished_without_results is False

    percents = [
        int((current / total) * 100)
        for current, total, _message in lifecycle.progress_events
        if total > 0
    ]
    assert percents == sorted(percents)
    assert percents[-1] == 100
    assert any(msg.startswith("Stress:") for _, _, msg in lifecycle.progress_events)
    assert any(msg.startswith("Nodal Forces:") for _, _, msg in lifecycle.progress_events)
    assert any(msg.startswith("Deformation:") for _, _, msg in lifecycle.progress_events)


def test_solve_run_controller_ignores_reentrant_solve_request():
    controller = SolveRunController(_FakeTab())
    controller.input_validator = _FakeValidator()
    lifecycle = _FakeLifecycle()
    controller.lifecycle_handler = lifecycle
    controller.execution_handler = _FakeExecutor()
    controller._solve_in_progress = True

    controller.solve(SolverConfig())

    assert lifecycle.begin_calls == 0
    assert lifecycle.stage_messages
    assert "already running" in lifecycle.stage_messages[-1]
