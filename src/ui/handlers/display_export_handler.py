"""
Export-related functionality for the Display tab.
"""

from PyQt5.QtWidgets import QFileDialog, QMessageBox

from file_io.exporters import export_mesh_to_csv, export_apdl_ic
from ui.handlers.display_base_handler import DisplayBaseHandler


class DisplayExportHandler(DisplayBaseHandler):
    """Handles data export operations from the Display tab."""

    def save_time_point_results(self) -> None:
        """Export the currently displayed scalar field to CSV."""
        mesh = self.state.current_mesh or self.tab.current_mesh
        if mesh is None:
            QMessageBox.warning(self.tab, "No Data", "No visualization data to save.")
            return

        active_scalar_name = mesh.active_scalars_name
        if not active_scalar_name:
            QMessageBox.warning(
                self.tab, "No Active Data",
                "The current mesh does not have an active scalar field to save."
            )
            return

        base_name = active_scalar_name.split(" ")[0]
        selected_time = self.tab.time_point_spinbox.value()
        default_filename = f"{base_name}_T_{selected_time:.5f}.csv".replace('.', '_')

        file_name, _ = QFileDialog.getSaveFileName(
            self.tab, "Save Time Point Results", default_filename, "CSV Files (*.csv)"
        )

        if not file_name:
            return

        try:
            export_mesh_to_csv(mesh, active_scalar_name, file_name)
            QMessageBox.information(
                self.tab, "Save Successful",
                f"Time point results saved successfully to:\n{file_name}"
            )
        except Exception as exc:
            QMessageBox.critical(
                self.tab, "Save Error",
                f"An error occurred while saving the file: {exc}"
            )

    def extract_initial_conditions(self) -> None:
        """Export velocity initial conditions in APDL format."""
        mesh = self.state.current_mesh or self.tab.current_mesh
        if mesh is None:
            QMessageBox.warning(
                self.tab, "No Data",
                "No visualization data available. Please compute velocity at a time point first."
            )
            return

        required_arrays = {'vel_x', 'vel_y', 'vel_z'}
        if not required_arrays.issubset(set(mesh.array_names)):
            QMessageBox.warning(
                self.tab, "Missing Velocity Data",
                "Velocity components not found in current mesh.\n\n"
                "Please compute velocity for a time point first using the Update button."
            )
            return

        if 'NodeID' not in mesh.array_names:
            QMessageBox.warning(
                self.tab, "Missing Node IDs",
                "Node IDs not found in mesh. Cannot export initial conditions."
            )
            return

        node_ids = mesh['NodeID']
        vel_x = mesh['vel_x'].flatten()
        vel_y = mesh['vel_y'].flatten()
        vel_z = mesh['vel_z'].flatten()

        file_name, _ = QFileDialog.getSaveFileName(
            self.tab,
            "Save Initial Conditions",
            "initial_conditions.inp",
            "APDL Files (*.inp);;All Files (*)"
        )

        if not file_name:
            return

        try:
            export_apdl_ic(node_ids, vel_x, vel_y, vel_z, file_name)
            QMessageBox.information(
                self.tab, "Export Successful",
                f"Initial conditions exported successfully to:\n{file_name}"
            )
        except Exception as exc:
            QMessageBox.critical(
                self.tab, "Export Error",
                f"Failed to export initial conditions:\n{exc}"
            )
