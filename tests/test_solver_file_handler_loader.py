from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("PyQt5.QtCore")
pytest.importorskip("PyQt5.QtWidgets")

from ui.handlers import file_handler
from file_io.dpf_reader import DPFAnalysisReader
from ui.handlers.file_handler import RSTLoaderThread, SolverFileHandler


class _FakeTab:
    def __init__(self):
        self.enabled_states = []
        self.base_loaded = None
        self.combine_loaded = None
        self.console_textbox = MagicMock()

    def setEnabled(self, enabled):
        self.enabled_states.append(enabled)

    def on_base_rst_loaded(self, analysis_data, filename):
        self.base_loaded = (analysis_data, filename)

    def on_combine_rst_loaded(self, analysis_data, filename):
        self.combine_loaded = (analysis_data, filename)

    def window(self):
        return self


def test_base_rst_loaded_reuses_loader_reader(monkeypatch):
    def fail_if_constructed(*args, **kwargs):
        raise AssertionError("DPFAnalysisReader should not be reopened in load callback")

    monkeypatch.setattr(file_handler, "DPFAnalysisReader", fail_if_constructed)
    tab = _FakeTab()
    handler = SolverFileHandler(tab)
    analysis_data = object()
    reader = object()

    handler._on_base_rst_loaded(analysis_data, "base.rst", reader)

    assert handler.base_reader is reader
    assert tab.enabled_states == [True]
    assert tab.base_loaded == (analysis_data, "base.rst")


def test_combine_rst_loaded_reuses_loader_reader(monkeypatch):
    def fail_if_constructed(*args, **kwargs):
        raise AssertionError("DPFAnalysisReader should not be reopened in load callback")

    monkeypatch.setattr(file_handler, "DPFAnalysisReader", fail_if_constructed)
    tab = _FakeTab()
    handler = SolverFileHandler(tab)
    analysis_data = object()
    reader = object()

    handler._on_combine_rst_loaded(analysis_data, "combine.rst", reader)

    assert handler.combine_reader is reader
    assert tab.enabled_states == [True]
    assert tab.combine_loaded == (analysis_data, "combine.rst")


def test_analysis_data_reports_real_metadata_phases():
    reader = SimpleNamespace(
        rst_path="sample.rst",
        cdb_named_selection_reader=None,
        check_nodal_forces_available=lambda: False,
        check_displacement_available=lambda: False,
        get_load_step_ids=lambda: [1, 2],
        get_time_values=lambda set_ids=None: [0.0, 1.0],
        get_named_selections=lambda: ["SCOPE"],
        get_named_selection_locations=lambda: {"SCOPE": "nodal"},
        get_named_selection_sources=lambda: {"SCOPE": "rst"},
        unit_system="MKS",
        stress_unit="Pa",
        stress_conversion_factor=1e-6,
    )
    progress = []
    data = DPFAnalysisReader.get_analysis_data(reader, progress_callback=progress.append)

    assert data.load_step_ids == [1, 2]
    assert progress == [
        "Checking available result types...",
        "Reading load-step and time metadata...",
        "Reading named selections...",
        "Reading result units...",
        "Finalizing RST metadata...",
    ]


def test_rst_loader_thread_forwards_reader_progress(monkeypatch):
    analysis_data = object()

    class _Reader:
        def __init__(self, rst_path):
            assert rst_path == "sample.rst"

        def get_analysis_data(self, skip_substeps=False, progress_callback=None):
            assert skip_substeps is True
            progress_callback("Reading named selections...")
            return analysis_data

    monkeypatch.setattr(file_handler, "DPFAnalysisReader", _Reader)
    loader = RSTLoaderThread("sample.rst", skip_substeps=True)
    progress = []
    finished = []
    loader.progress.connect(progress.append)
    loader.finished.connect(lambda data, path, reader: finished.append((data, path, reader)))

    loader.run()

    assert progress == [
        "Starting DPF and opening the result file...",
        "Reading named selections...",
    ]
    assert finished[0][:2] == (analysis_data, "sample.rst")
    assert isinstance(finished[0][2], _Reader)


def _mock_progress_widgets(monkeypatch):
    delayed = []
    timer = MagicMock()
    dialog = MagicMock()
    dialog.windowFlags.return_value = file_handler.Qt.WindowCloseButtonHint

    class _TimerFactory:
        def __new__(cls):
            return timer

        @staticmethod
        def singleShot(delay, callback):
            delayed.append((delay, callback))

    monkeypatch.setattr(file_handler, "QTimer", _TimerFactory)
    monkeypatch.setattr(file_handler, "QProgressDialog", MagicMock(return_value=dialog))
    return delayed, dialog, timer


def test_slow_rst_load_shows_detailed_dialog_and_cleans_up(monkeypatch):
    delayed, dialog, timer = _mock_progress_widgets(monkeypatch)
    clock = {"value": 100.0}
    monkeypatch.setattr(file_handler.time, "monotonic", lambda: clock["value"])
    monkeypatch.setattr(file_handler.os.path, "getsize", lambda _path: 3 * 1024 ** 3)
    monkeypatch.setattr(file_handler.QMessageBox, "warning", lambda *_args: None)
    tab = _FakeTab()
    handler = SolverFileHandler(tab)
    loader = MagicMock()
    loader.isRunning.return_value = True

    handler._begin_rst_progress("C:/results/large.rst", "Base Analysis", loader)
    handler._on_rst_load_progress("Reading named selections...", loader)

    assert delayed[0][0] == 5000
    dialog.show.assert_not_called()

    clock["value"] = 106.0
    delayed[0][1]()
    label = dialog.setLabelText.call_args.args[0]

    dialog.show.assert_called_once()
    assert "File: large.rst" in label
    assert "Size: 3,072.0 MB" in label
    assert "Elapsed: 00:06" in label
    assert "Current phase: Reading named selections..." in label

    handler._on_rst_load_error("bad file", "Base Analysis")

    dialog.close.assert_called_once()
    dialog.deleteLater.assert_called_once()
    timer.stop.assert_called_once()
    timer.deleteLater.assert_called_once()
    assert handler._active_rst_loader_thread is None


def test_fast_rst_load_never_opens_delayed_dialog(monkeypatch):
    delayed, dialog, _timer = _mock_progress_widgets(monkeypatch)
    monkeypatch.setattr(file_handler.time, "monotonic", lambda: 100.0)
    tab = _FakeTab()
    handler = SolverFileHandler(tab)
    loader = MagicMock()
    loader.isRunning.return_value = True
    analysis_data = object()
    reader = object()

    handler._begin_rst_progress("base.rst", "Base Analysis", loader)
    handler._on_base_rst_loaded(analysis_data, "base.rst", reader)
    delayed[0][1]()

    dialog.show.assert_not_called()
    assert tab.base_loaded == (analysis_data, "base.rst")
