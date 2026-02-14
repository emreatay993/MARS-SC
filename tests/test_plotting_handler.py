"""
Regression tests for Plotly web view loading behavior.
"""

import os
import sys
import types

def _ensure_fake_plotly():
    """Install a lightweight plotly shim when dependency is absent."""
    if "plotly" in sys.modules and "plotly.io" in sys.modules:
        return
    fake_plotly = types.ModuleType("plotly")
    fake_plotly_io = types.ModuleType("plotly.io")
    fake_plotly_io.to_html = lambda *_args, **_kwargs: "<html><body>plot</body></html>"
    fake_plotly.io = fake_plotly_io
    sys.modules["plotly"] = fake_plotly
    sys.modules["plotly.io"] = fake_plotly_io


def _ensure_fake_pyqt5():
    """Install a lightweight QtCore shim when PyQt5 is absent."""
    if "PyQt5" in sys.modules and "PyQt5.QtCore" in sys.modules:
        return

    class _FakeQUrl:
        def __init__(self, path):
            self.path = path

        @staticmethod
        def fromLocalFile(path):
            return _FakeQUrl(path)

    fake_pyqt5 = types.ModuleType("PyQt5")
    fake_qtcore = types.ModuleType("PyQt5.QtCore")
    fake_qtcore.QUrl = _FakeQUrl
    fake_pyqt5.QtCore = fake_qtcore
    sys.modules["PyQt5"] = fake_pyqt5
    sys.modules["PyQt5.QtCore"] = fake_qtcore


try:
    from ui.handlers import plotting_handler as plotting_handler_module
except ModuleNotFoundError:
    _ensure_fake_plotly()
    _ensure_fake_pyqt5()
    from ui.handlers import plotting_handler as plotting_handler_module


class _FakeWebView:
    """Small test double for QWebEngineView."""

    def __init__(self):
        self.url = None
        self.html = None
        self.show_called = False

    def setUrl(self, url):
        self.url = url

    def setHtml(self, html):
        self.html = html

    def show(self):
        self.show_called = True


def test_load_fig_to_webview_uses_local_file_url():
    """Primary render path should use local-file URL instead of setHtml."""
    handler = plotting_handler_module.PlottingHandler()
    web_view = _FakeWebView()
    fig = object()

    original_to_html = plotting_handler_module.pio.to_html
    plotting_handler_module.pio.to_html = lambda *_args, **_kwargs: "<html><body>plot</body></html>"

    try:
        handler.load_fig_to_webview(fig, web_view)
    finally:
        plotting_handler_module.pio.to_html = original_to_html

    assert web_view.url is not None
    assert web_view.html is None
    assert web_view.show_called is True
    assert len(handler.temp_files) == 1
    assert os.path.exists(handler.temp_files[0])

    handler.cleanup_temp_files()
