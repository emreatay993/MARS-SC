"""
Handles user interactions with the File Navigator dock, such as
selecting project directories and opening files.
"""

import subprocess
from PyQt5.QtWidgets import QFileDialog


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

        try:
            subprocess.run(['cmd', '/c', 'start', '/max', '', file_path], shell=True)
        except Exception as e:
            print(f"Error opening file '{file_path}': {e}")