"""
Handles plotting operations, such as loading Plotly figures into
WebViews and managing temporary files.
"""

import os
import traceback
from tempfile import NamedTemporaryFile

import plotly.io as pio
from PyQt5.QtCore import QUrl


class PlottingHandler:
    """Manages plotting and temporary file cleanup."""

    def __init__(self):
        """Initialize the plotting handler."""
        self.temp_files = []

    def load_fig_to_webview(self, fig, web_view):
        """Load Plotly figure into web view widget."""
        try:
            plotly_fig = fig

            html_content = pio.to_html(
                plotly_fig,
                full_html=True,
                include_plotlyjs=True,
                config={'responsive': True}
            )

            with NamedTemporaryFile(
                    mode='w', suffix='.html', delete=False, encoding='utf-8'
            ) as tmp_file:
                tmp_file.write(html_content)
                file_path = tmp_file.name
                self.temp_files.append(file_path)

            web_view.setUrl(QUrl.fromLocalFile(file_path))
            web_view.show()

        except Exception as e:
            print(f"Error loading figure to webview: {e}")
            traceback.print_exc()
            error_html = (
                f"<html><body><h1>Error loading plot</h1>"
                f"<pre>{e}</pre><pre>{traceback.format_exc()}</pre></body></html>"
            )
            try:
                web_view.setHtml(error_html)
            except Exception:
                pass

    def cleanup_temp_files(self):
        """Remove temporary files created during session."""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                print(f"Error removing temp file {temp_file}: {e}")
        self.temp_files.clear()