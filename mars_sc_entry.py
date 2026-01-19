"""
PyInstaller entry point for MARS-SC application.

This wrapper ensures proper module paths are set before importing the application.
"""

import sys
import os

# Determine the base path
if getattr(sys, 'frozen', False):
    # Running as a PyInstaller bundle
    base_path = sys._MEIPASS
else:
    # Running as a normal Python script
    base_path = os.path.dirname(os.path.abspath(__file__))
    # Add src directory for development
    src_path = os.path.join(base_path, 'src')
    if os.path.exists(src_path):
        base_path = src_path

# Ensure base path is in sys.path
if base_path not in sys.path:
    sys.path.insert(0, base_path)

# Now import and run the main application
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication
from ui.application_controller import ApplicationController


def main():
    """Main entry point for the application."""
    # Enable high DPI scaling
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    # Create application
    app = QApplication(sys.argv)

    # Create and show main window
    main_window = ApplicationController()
    main_window.showMaximized()

    # Run application
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
