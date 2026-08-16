import pytest

pytest.importorskip("PyQt5")

from ui.handlers import file_handler
from ui.handlers.file_handler import SolverFileHandler


class _FakeTab:
    def __init__(self):
        self.enabled_states = []
        self.base_loaded = None
        self.combine_loaded = None

    def setEnabled(self, enabled):
        self.enabled_states.append(enabled)

    def on_base_rst_loaded(self, analysis_data, filename):
        self.base_loaded = (analysis_data, filename)

    def on_combine_rst_loaded(self, analysis_data, filename):
        self.combine_loaded = (analysis_data, filename)


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
