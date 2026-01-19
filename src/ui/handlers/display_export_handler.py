"""
Export-related functionality for the Display tab.
"""

from PyQt5.QtWidgets import QFileDialog, QMessageBox

from file_io.exporters import (
    export_mesh_to_csv, export_apdl_ic, 
    export_nodal_forces_single_combination, export_deformation_single_combination
)
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

    def export_forces_csv(self) -> None:
        """
        Export nodal forces for the currently selected combination to CSV.
        
        Exports FX, FY, FZ components, magnitude, shear (if beams present),
        element type, and coordinate system information.
        """
        # Check if nodal forces result is available
        if self.tab.nodal_forces_result is None:
            QMessageBox.warning(
                self.tab, "No Forces Data",
                "No nodal forces data available.\n\n"
                "Please run a combination analysis with 'Nodal Forces' output enabled."
            )
            return
        
        result = self.tab.nodal_forces_result
        
        # Get the currently selected combination from view_combination_combo
        view_combo_idx = self.tab.view_combination_combo.currentIndex()
        
        if view_combo_idx == 0:
            # Envelope view selected - ask user to select a specific combination
            QMessageBox.information(
                self.tab, "Select Combination",
                "Please select a specific combination from the 'View Combination' dropdown "
                "to export its nodal forces.\n\n"
                "The envelope view cannot be exported as forces CSV."
            )
            return
        
        # Get the actual combination index (0-based)
        combo_idx = view_combo_idx - 1  # Account for "(Envelope View)" at index 0
        
        # Validate combination index
        if combo_idx < 0 or combo_idx >= result.num_combinations:
            QMessageBox.warning(
                self.tab, "Invalid Combination",
                f"Invalid combination index: {combo_idx}.\n"
                f"Available combinations: 0 to {result.num_combinations - 1}"
            )
            return
        
        # Get combination name
        combo_name = ""
        if self.tab.combination_names and combo_idx < len(self.tab.combination_names):
            combo_name = self.tab.combination_names[combo_idx]
        else:
            combo_name = f"Combination_{combo_idx + 1}"
        
        # Get force components for this combination
        fx = result.all_combo_fx[combo_idx, :]
        fy = result.all_combo_fy[combo_idx, :]
        fz = result.all_combo_fz[combo_idx, :]
        
        # Generate default filename
        safe_name = combo_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
        default_filename = f"nodal_forces_{safe_name}.csv"
        
        # Show file save dialog
        file_name, _ = QFileDialog.getSaveFileName(
            self.tab,
            "Export Nodal Forces",
            default_filename,
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if not file_name:
            return
        
        try:
            # Export using the updated function
            export_nodal_forces_single_combination(
                filename=file_name,
                node_ids=result.node_ids,
                node_coords=result.node_coords,
                fx=fx,
                fy=fy,
                fz=fz,
                combination_index=combo_idx,
                combination_name=combo_name,
                force_unit=result.force_unit,
                coordinate_system=result.coordinate_system,
                node_element_types=result.node_element_types,
                include_shear=result.has_beam_nodes
            )
            
            QMessageBox.information(
                self.tab, "Export Successful",
                f"Nodal forces exported successfully to:\n{file_name}\n\n"
                f"Combination: {combo_name}\n"
                f"Coordinate System: {result.coordinate_system}\n"
                f"Nodes: {result.num_nodes:,}"
            )
            
        except Exception as exc:
            QMessageBox.critical(
                self.tab, "Export Error",
                f"Failed to export nodal forces:\n{exc}"
            )

    def export_deformation_csv(self) -> None:
        """
        Export deformation (displacement) for the currently selected combination to CSV.
        
        Exports UX, UY, UZ components and magnitude (U_mag).
        """
        # Check if deformation result is available
        if self.tab.deformation_result is None:
            QMessageBox.warning(
                self.tab, "No Deformation Data",
                "No deformation data available.\n\n"
                "Please run a combination analysis with 'Deformation' output enabled."
            )
            return
        
        result = self.tab.deformation_result
        
        # Get the currently selected combination from view_combination_combo
        view_combo_idx = self.tab.view_combination_combo.currentIndex()
        
        if view_combo_idx == 0:
            # Envelope view selected - ask user to select a specific combination
            QMessageBox.information(
                self.tab, "Select Combination",
                "Please select a specific combination from the 'View Combination' dropdown "
                "to export its displacement data.\n\n"
                "The envelope view cannot be exported as displacement CSV."
            )
            return
        
        # Get the actual combination index (0-based)
        combo_idx = view_combo_idx - 1  # Account for "(Envelope View)" at index 0
        
        # Validate combination index
        if combo_idx < 0 or combo_idx >= result.num_combinations:
            QMessageBox.warning(
                self.tab, "Invalid Combination",
                f"Invalid combination index: {combo_idx}.\n"
                f"Available combinations: 0 to {result.num_combinations - 1}"
            )
            return
        
        # Get combination name
        combo_name = ""
        if self.tab.combination_names and combo_idx < len(self.tab.combination_names):
            combo_name = self.tab.combination_names[combo_idx]
        else:
            combo_name = f"Combination_{combo_idx + 1}"
        
        # Get displacement components for this combination
        ux = result.all_combo_ux[combo_idx, :]
        uy = result.all_combo_uy[combo_idx, :]
        uz = result.all_combo_uz[combo_idx, :]
        
        # Generate default filename
        safe_name = combo_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
        default_filename = f"deformation_{safe_name}.csv"
        
        # Show file save dialog
        file_name, _ = QFileDialog.getSaveFileName(
            self.tab,
            "Export Deformation",
            default_filename,
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if not file_name:
            return
        
        try:
            # Export using the updated function
            export_deformation_single_combination(
                filename=file_name,
                node_ids=result.node_ids,
                node_coords=result.node_coords,
                ux=ux,
                uy=uy,
                uz=uz,
                combination_index=combo_idx,
                combination_name=combo_name,
                displacement_unit=result.displacement_unit
            )
            
            QMessageBox.information(
                self.tab, "Export Successful",
                f"Deformation exported successfully to:\n{file_name}\n\n"
                f"Combination: {combo_name}\n"
                f"Displacement Unit: {result.displacement_unit}\n"
                f"Nodes: {result.num_nodes:,}"
            )
            
        except Exception as exc:
            QMessageBox.critical(
                self.tab, "Export Error",
                f"Failed to export deformation:\n{exc}"
            )
