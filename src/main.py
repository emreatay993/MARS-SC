"""
Entry point for the MARS-SC: Solution Combination application.

MARS-SC performs linear combination of stress results from two static analysis
RST files and computes stress envelopes over combinations.

--- Run in Pycharm via the command:
...PycharmProjects/Solution_Combination_ANSYS> .\\venv\\Scripts\\Activate.ps1
...PycharmProjects/Solution_Combination_ANSYS/src> python -m main

Initialises the Qt application and launches the main window.
"""

import sys

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
