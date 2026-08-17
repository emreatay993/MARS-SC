"""Tests for solve-run UI progress behavior."""

import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from core.data_models import SolverConfig
import ui.handlers.solver_run_ui_handler as ui_handler_module
from ui.handlers.solver_run_ui_handler import SolverRunUiHandler


class _FakeQApplication:
    @staticmethod
    def processEvents(*_args, **_kwargs):
        return None


class _FakeQMessageBox:
    @staticmethod
    def critical(*_args, **_kwargs):
        return None


class _FakeQEventLoop:
    ExcludeUserInputEvents = 0


class _FakeQTimer:
    @staticmethod
    def singleShot(_ms, callback):
        callback()


# Keep the Qt substitutes local to the module under test. Installing fake
# packages in sys.modules during collection poisons unrelated Qt tests.
ui_handler_module.QApplication = _FakeQApplication
ui_handler_module.QMessageBox = _FakeQMessageBox
ui_handler_module.QEventLoop = _FakeQEventLoop
ui_handler_module.QTimer = _FakeQTimer


class _FakeConsole:
    def __init__(self):
        self.messages = []

    def append(self, message):
        self.messages.append(message)


class _FakeProgressBar:
    def __init__(self):
        self.visible = False
        self.value = 0
        self.format_text = ""
        self.range = (0, 100)
        self.operations = []
        self.updates_enabled = True

    def setVisible(self, visible):
        self.visible = bool(visible)

    def setValue(self, value):
        self.value = int(value)
        self.operations.append(("value", self.value))

    def setFormat(self, text):
        self.format_text = str(text)
        self.operations.append(("format", self.format_text))

    def setRange(self, minimum, maximum):
        self.range = (int(minimum), int(maximum))
        self.operations.append(("range", self.range))

    def setUpdatesEnabled(self, enabled):
        self.updates_enabled = bool(enabled)
        self.operations.append(("updates", self.updates_enabled))


class _FakeTab:
    def __init__(self):
        self.progress_bar = _FakeProgressBar()
        self.console_textbox = _FakeConsole()
        self._history_popup_requested = False
        self.enabled = True
        self.combination_result = None
        self.nodal_forces_result = None
        self.deformation_result = None

    def setEnabled(self, enabled):
        self.enabled = bool(enabled)


def test_update_progress_is_monotonic_and_handles_indeterminate(monkeypatch):
    tab = _FakeTab()
    handler = SolverRunUiHandler(tab)

    process_calls = []
    monkeypatch.setattr(
        ui_handler_module.QApplication,
        "processEvents",
        lambda *args, **kwargs: process_calls.append((args, kwargs)),
    )

    handler.begin_solve(SolverConfig())

    handler.update_progress(80, 100, "phase one")
    assert tab.progress_bar.value == 80

    # Backward updates should be clamped.
    handler.update_progress(20, 100, "phase one rewind")
    assert tab.progress_bar.value == 80

    # Unknown total switches to indeterminate.
    handler.update_progress(0, 0, "waiting")
    assert tab.progress_bar.range == (0, 0)

    # Returning to determinate restores range and continues forward.
    handler.update_progress(90, 100, "phase two")
    assert tab.progress_bar.range == (0, 100)
    assert tab.progress_bar.value == 90
    assert process_calls, "Expected UI event pumping calls."


def test_complete_solve_shows_100_and_hides_after_delay(monkeypatch):
    tab = _FakeTab()
    handler = SolverRunUiHandler(tab)

    monkeypatch.setattr(ui_handler_module.QApplication, "processEvents", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        ui_handler_module.QTimer,
        "singleShot",
        lambda _ms, callback: callback(),
    )

    config = SolverConfig()
    handler.begin_solve(config)
    assert tab.progress_bar.visible is True

    handler.complete_solve(
        stress_result=None,
        config=config,
        forces_result=None,
        deformation_result=None,
    )

    assert tab.enabled is True
    assert tab.progress_bar.visible is False


def test_progress_state_is_atomic_and_event_pumping_is_throttled(monkeypatch):
    tab = _FakeTab()
    handler = SolverRunUiHandler(tab)
    times = iter((1.0, 1.01, 1.02, 1.11))
    process_calls = []
    monkeypatch.setattr(ui_handler_module.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(
        ui_handler_module.QApplication,
        "processEvents",
        lambda *_args, **_kwargs: process_calls.append(True),
    )

    handler.begin_solve(SolverConfig())
    tab.progress_bar.operations.clear()
    process_calls.clear()

    handler.update_progress(10, 100, "phase one")
    assert tab.progress_bar.operations == [
        ("updates", False),
        ("format", "phase one (10%)"),
        ("value", 10),
        ("updates", True),
    ]
    assert process_calls == []

    handler.update_progress(20, 100, "phase two")
    assert process_calls == []
    handler.update_progress(30, 100, "phase three")
    assert process_calls == [True]
