"""Tests for staged solve progress orchestration in SolveRunController."""

import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

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
    def __init__(self):
        self.calls = []

    def prepare_nodal_forces_for_solve(self):
        self.calls.append("prepare_forces")

    def run_stress_analysis(self, config, stress_type, progress_callback):
        _ = (config, stress_type)
        self.calls.append("stress")
        progress_callback(0, 100, "starting")
        progress_callback(100, 100, "done")
        return {"stress": True}

    def run_nodal_forces_analysis(self, config, progress_callback):
        _ = config
        self.calls.append("forces")
        progress_callback(0, 100, "starting")
        progress_callback(100, 100, "done")
        return {"forces": True}

    def run_deformation_analysis(self, config, progress_callback):
        _ = config
        self.calls.append("deformation")
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
    executor = _FakeExecutor()
    controller.execution_handler = executor

    config = SolverConfig(
        calculate_von_mises=True,
        calculate_nodal_forces=True,
        calculate_deformation=True,
    )
    controller.solve(config)

    assert lifecycle.begin_calls == 1
    assert lifecycle.completed is True
    assert lifecycle.finished_without_results is False
    assert executor.calls == ["prepare_forces", "stress", "forces", "deformation"]

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
