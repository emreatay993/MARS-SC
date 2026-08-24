"""
Handles user interactions with the File Navigator dock, such as
selecting project directories and opening files.
"""

import subprocess
from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import QFileDialog, QMenu


class NavigatorHandler:
    """Manages file navigator interactions."""

    def __init__(self, file_model, tree_view, solver_tab):
        """
        Initialize the navigator handler.

        Args:
            file_model (QFileSystemModel): The model used by the tree view.
            tree_view (QTreeView): The tree view widget.
            solver_tab (SolverTab): The solver tab to update.
        """
        self.file_model = file_model
        self.tree_view = tree_view
        self.solver_tab = solver_tab
        self.project_directory = None

    def select_project_directory(self, parent_window):
        """Open dialog to select project directory."""
        dir_path = QFileDialog.getExistingDirectory(
            parent_window, "Select Project Directory"
        )
        if dir_path:
            self.project_directory = dir_path
            print(f"Project directory selected: {self.project_directory}")

            # Update solver tab's project directory
            self.solver_tab.project_directory = self.project_directory

            # Update navigator
            self.file_model.setRootPath(self.project_directory)
            self.tree_view.setRootIndex(
                self.file_model.index(self.project_directory)
            )

    def open_navigator_file(self, index):
        """Open file from navigator in default application."""
        if self.file_model.isDir(index):
            return

        file_path = self.file_model.filePath(index)
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(file_path)):
            print(f"Error opening file '{file_path}'.")

    def show_navigator_context_menu(self, position):
        """Show native file actions for the clicked navigator file."""
        index = self.tree_view.indexAt(position)
        if not index.isValid() or self.file_model.isDir(index):
            return

        menu = QMenu(self.tree_view)
        show_action = menu.addAction("Show in File Explorer")
        show_action.triggered.connect(
            lambda _checked=False: self.show_in_file_explorer(index)
        )
        open_action = menu.addAction("Open")
        open_action.triggered.connect(
            lambda _checked=False: self.open_navigator_file(index)
        )
        open_with_action = menu.addAction("Open With...")
        open_with_action.triggered.connect(
            lambda _checked=False: self.open_with(index)
        )
        menu.exec_(self.tree_view.viewport().mapToGlobal(position))

    def show_in_file_explorer(self, index):
        """Select the navigator file in Windows File Explorer."""
        file_path = self.file_model.filePath(index)
        try:
            subprocess.Popen(["explorer.exe", f"/select,{file_path}"])
        except OSError as error:
            print(f"Error showing file '{file_path}' in File Explorer: {error}")

    def open_with(self, index):
        """Show the Windows Open With dialog for the navigator file."""
        file_path = self.file_model.filePath(index)
        try:
            subprocess.Popen([
                "rundll32.exe",
                "shell32.dll,OpenAs_RunDLL",
                file_path,
            ])
        except OSError as error:
            print(f"Error opening Open With for '{file_path}': {error}")
