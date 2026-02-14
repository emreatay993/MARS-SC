"""
Export-related functionality for the Display tab.
"""

import os
from typing import Optional, List, Tuple

import numpy as np
from PyQt5.QtWidgets import QFileDialog, QMessageBox

from file_io.exporters import (
    export_mesh_to_csv, export_apdl_ic, 
    export_nodal_forces_single_combination, export_deformation_single_combination,
    export_envelope_results, export_single_combination,
    export_nodal_forces_envelope, export_deformation_envelope
)
from ui.handlers.display_mesh_arrays import build_deformation_component_payload_from_result
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
                    include_shear=True
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

    def export_output_csv(self) -> None:
        """
        Export results to CSV based on current view and available data.
        
        This unified export method handles all output types:
        - Stress (von Mises, Max Principal, Min Principal)
        - Nodal Forces
        - Deformation
        
        Behavior:
        - If envelope view (index 0): exports envelope CSV for each result type
        - If single combination: exports single combination CSV for each result type
        - Rule: Deformation results are ALWAYS exported to a separate file from
          stress/forces, even when exported together
        """
        # Get references to available result data
        stress_result = self._get_stress_result()
        forces_result = self.tab.nodal_forces_result
        deformation_result = self.tab.deformation_result
        
        # Check if any data is available
        if stress_result is None and forces_result is None and deformation_result is None:
            QMessageBox.warning(
                self.tab, "No Data",
                "No analysis results available to export.\n\n"
                "Please run a combination analysis first."
            )
            return
        
        # Get view combination index to determine export mode
        view_combo_idx = self.tab.view_combination_combo.currentIndex()
        is_envelope_view = (view_combo_idx == 0)
        
        # Get combination names for metadata
        combination_names = self.tab.combination_names or []
        
        if is_envelope_view:
            self._export_envelope_view(
                stress_result, forces_result, deformation_result, combination_names
            )
        else:
            # Single combination export
            combo_idx = view_combo_idx - 1  # Account for "(Envelope View)" at index 0
            self._export_single_combination_view(
                stress_result, forces_result, deformation_result,
                combo_idx, combination_names
            )

    def _get_stress_result(self):
        """
        Get cached stress result from display state.
        
        Returns:
            CombinationResult or None if not available.
        """
        return self.tab.stress_result

    def _get_envelope_export_mode(self) -> Tuple[bool, bool]:
        """
        Determine which envelope data to export based on scalar display selection.
        
        Returns:
            Tuple of (export_max, export_min) booleans indicating which data to export.
        """
        # Get current selection from scalar display combo
        selected_text = self.tab.scalar_display_combo.currentText()
        
        # Determine export mode based on selection
        # "Max Value" or "Combo # of Max" -> export max-related data
        # "Min Value" or "Combo # of Min" -> export min-related data
        export_max = "Max" in selected_text
        export_min = "Min" in selected_text
        
        # If neither (shouldn't happen), default to max
        if not export_max and not export_min:
            export_max = True

        return export_max, export_min

    def _build_deformation_component_payload(self, deformation_result) -> Optional[dict]:
        """
        Build optional deformation component-envelope payload for CSV export.

        Priority:
        1) Use Def_* arrays already attached to the active mesh.
        2) Fallback to compute from deformation_result all-combination arrays.
        """
        mesh = self.state.current_mesh or self.tab.current_mesh
        payload: dict[str, np.ndarray] = {}

        if mesh is not None:
            mesh_map = {
                "max_ux": "Def_Max_UX",
                "min_ux": "Def_Min_UX",
                "combo_of_max_ux": "Def_Combo_of_Max_UX",
                "combo_of_min_ux": "Def_Combo_of_Min_UX",
                "max_uy": "Def_Max_UY",
                "min_uy": "Def_Min_UY",
                "combo_of_max_uy": "Def_Combo_of_Max_UY",
                "combo_of_min_uy": "Def_Combo_of_Min_UY",
                "max_uz": "Def_Max_UZ",
                "min_uz": "Def_Min_UZ",
                "combo_of_max_uz": "Def_Combo_of_Max_UZ",
                "combo_of_min_uz": "Def_Combo_of_Min_UZ",
            }

            for key, array_name in mesh_map.items():
                if array_name in mesh.array_names:
                    payload[key] = np.asarray(mesh[array_name]).reshape(-1)

        if payload:
            return payload

        return build_deformation_component_payload_from_result(deformation_result)

    def _export_envelope_view(
        self,
        stress_result,
        forces_result,
        deformation_result,
        combination_names: List[str]
    ) -> None:
        """
        Export envelope results for all available result types.
        
        The export respects the user's selection in the scalar display dropdown:
        - "Max Value" or "Combo # of Max" -> exports max envelope data
        - "Min Value" or "Combo # of Min" -> exports min envelope data
        
        Args:
            stress_result: CombinationResult or None
            forces_result: NodalForcesResult or None
            deformation_result: DeformationResult or None
            combination_names: List of combination names
        """
        # Build list of available result types
        available_types = []
        if stress_result is not None:
            available_types.append("stress")
        if forces_result is not None:
            available_types.append("forces")
        if deformation_result is not None:
            available_types.append("deformation")
        
        if not available_types:
            QMessageBox.warning(
                self.tab, "No Data",
                "No envelope data available to export."
            )
            return
        
        # Determine which envelope data to export based on user selection
        export_max, export_min = self._get_envelope_export_mode()
        
        # Create suffix for filename based on selection
        if export_max and not export_min:
            envelope_suffix = "max_envelope"
        elif export_min and not export_max:
            envelope_suffix = "min_envelope"
        else:
            envelope_suffix = "envelope"
        
        # Ask user for output directory
        output_dir = QFileDialog.getExistingDirectory(
            self.tab,
            "Select Output Directory for Envelope CSV Files",
            "",
            QFileDialog.ShowDirsOnly
        )
        
        if not output_dir:
            return
        
        exported_files = []
        errors = []
        
        # Export stress envelope
        if stress_result is not None:
            try:
                result_type = stress_result.result_type or "von_mises"
                safe_type = result_type.replace(" ", "_")
                filename = os.path.join(output_dir, f"{safe_type}_{envelope_suffix}.csv")
                
                export_envelope_results(
                    filename=filename,
                    node_ids=stress_result.node_ids,
                    node_coords=stress_result.node_coords,
                    max_values=stress_result.max_over_combo if export_max else None,
                    min_values=stress_result.min_over_combo if export_min else None,
                    combo_of_max=stress_result.combo_of_max if export_max else None,
                    combo_of_min=stress_result.combo_of_min if export_min else None,
                    result_type=result_type,
                    combination_names=combination_names
                )
                exported_files.append(os.path.basename(filename))
            except Exception as e:
                errors.append(f"Stress: {str(e)}")
        
        # Export forces envelope
        if forces_result is not None:
            try:
                filename = os.path.join(output_dir, f"nodal_forces_{envelope_suffix}.csv")
                
                export_nodal_forces_envelope(
                    filename=filename,
                    node_ids=forces_result.node_ids,
                    node_coords=forces_result.node_coords,
                    max_magnitude=forces_result.max_magnitude_over_combo if export_max else None,
                    min_magnitude=forces_result.min_magnitude_over_combo if export_min else None,
                    combo_of_max=forces_result.combo_of_max if export_max else None,
                    combo_of_min=forces_result.combo_of_min if export_min else None,
                    combination_names=combination_names,
                    force_unit=forces_result.force_unit,
                    all_combo_fx=forces_result.all_combo_fx,
                    all_combo_fy=forces_result.all_combo_fy,
                    all_combo_fz=forces_result.all_combo_fz,
                    include_shear_variants=True,
                    include_component_envelopes=True,
                    include_component_combo_indices=True
                )
                exported_files.append(os.path.basename(filename))
            except Exception as e:
                errors.append(f"Forces: {str(e)}")
        
        # Export deformation envelope (always separate file)
        if deformation_result is not None:
            try:
                filename = os.path.join(output_dir, f"deformation_{envelope_suffix}.csv")
                component_payload = self._build_deformation_component_payload(deformation_result)
                
                export_deformation_envelope(
                    filename=filename,
                    node_ids=deformation_result.node_ids,
                    node_coords=deformation_result.node_coords,
                    max_magnitude=deformation_result.max_magnitude_over_combo if export_max else None,
                    min_magnitude=deformation_result.min_magnitude_over_combo if export_min else None,
                    combo_of_max=deformation_result.combo_of_max if export_max else None,
                    combo_of_min=deformation_result.combo_of_min if export_min else None,
                    combination_names=combination_names,
                    displacement_unit=deformation_result.displacement_unit,
                    component_payload=component_payload
                )
                exported_files.append(os.path.basename(filename))
            except Exception as e:
                errors.append(f"Deformation: {str(e)}")
        
        # Show result message
        self._show_export_result_message(exported_files, errors, output_dir)

    def _export_single_combination_view(
        self,
        stress_result,
        forces_result,
        deformation_result,
        combo_idx: int,
        combination_names: List[str]
    ) -> None:
        """
        Export single combination results for all available result types.
        
        Args:
            stress_result: CombinationResult or None
            forces_result: NodalForcesResult or None
            deformation_result: DeformationResult or None
            combo_idx: Index of the combination to export (0-based)
            combination_names: List of combination names
        """
        # Get combination name
        if combination_names and combo_idx < len(combination_names):
            combo_name = combination_names[combo_idx]
        else:
            combo_name = f"Combination_{combo_idx + 1}"
        
        safe_name = combo_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
        
        # Build list of available result types
        available_types = []
        if stress_result is not None and stress_result.all_combo_results is not None:
            if combo_idx < stress_result.num_combinations:
                available_types.append("stress")
        if forces_result is not None and forces_result.all_combo_fx is not None:
            if combo_idx < forces_result.all_combo_fx.shape[0]:
                available_types.append("forces")
        if deformation_result is not None and deformation_result.all_combo_ux is not None:
            if combo_idx < deformation_result.num_combinations:
                available_types.append("deformation")
        
        if not available_types:
            QMessageBox.warning(
                self.tab, "No Data",
                f"No data available for combination {combo_idx + 1}."
            )
            return
        
        # Determine if we need separate files
        # Rule: deformation is ALWAYS separate from stress/forces
        has_deformation = "deformation" in available_types
        has_other = "stress" in available_types or "forces" in available_types
        
        # Ask user for output location
        if has_deformation and has_other:
            # Multiple files needed - ask for directory
            output_dir = QFileDialog.getExistingDirectory(
                self.tab,
                f"Select Output Directory for {combo_name} CSV Files",
                "",
                QFileDialog.ShowDirsOnly
            )
            if not output_dir:
                return
        else:
            # Single file - ask for file path
            output_dir = None
        
        exported_files = []
        errors = []
        
        # Export stress results
        if "stress" in available_types:
            try:
                result_type = stress_result.result_type or "von_mises"
                safe_type = result_type.replace(" ", "_")
                
                if output_dir:
                    filename = os.path.join(output_dir, f"{safe_name}_{safe_type}.csv")
                else:
                    filename, _ = QFileDialog.getSaveFileName(
                        self.tab,
                        f"Export {combo_name} Stress Results",
                        f"{safe_name}_{safe_type}.csv",
                        "CSV Files (*.csv);;All Files (*)"
                    )
                    if not filename:
                        return
                
                # Get stress values for this combination
                stress_values = stress_result.all_combo_results[combo_idx, :]
                
                export_single_combination(
                    filename=filename,
                    node_ids=stress_result.node_ids,
                    node_coords=stress_result.node_coords,
                    stress_values=stress_values,
                    combination_index=combo_idx,
                    combination_name=combo_name,
                    result_type=result_type
                )
                exported_files.append(os.path.basename(filename))
            except Exception as e:
                errors.append(f"Stress: {str(e)}")
        
        # Export forces results
        if "forces" in available_types:
            try:
                if output_dir:
                    filename = os.path.join(output_dir, f"{safe_name}_forces.csv")
                elif "stress" not in available_types:
                    # Forces only - ask for file
                    filename, _ = QFileDialog.getSaveFileName(
                        self.tab,
                        f"Export {combo_name} Forces Results",
                        f"{safe_name}_forces.csv",
                        "CSV Files (*.csv);;All Files (*)"
                    )
                    if not filename:
                        return
                else:
                    # Stress already asked for directory but no deformation
                    # This case shouldn't happen with current logic, but handle it
                    filename = os.path.join(
                        os.path.dirname(exported_files[0]) if exported_files else ".",
                        f"{safe_name}_forces.csv"
                    )
                
                fx = forces_result.all_combo_fx[combo_idx, :]
                fy = forces_result.all_combo_fy[combo_idx, :]
                fz = forces_result.all_combo_fz[combo_idx, :]
                
                export_nodal_forces_single_combination(
                    filename=filename,
                    node_ids=forces_result.node_ids,
                    node_coords=forces_result.node_coords,
                    fx=fx,
                    fy=fy,
                    fz=fz,
                    combination_index=combo_idx,
                    combination_name=combo_name,
                    force_unit=forces_result.force_unit,
                    coordinate_system=forces_result.coordinate_system,
                    node_element_types=forces_result.node_element_types,
                    include_shear=True
                )
                exported_files.append(os.path.basename(filename))
            except Exception as e:
                errors.append(f"Forces: {str(e)}")
        
        # Export deformation results (ALWAYS separate file)
        if "deformation" in available_types:
            try:
                if output_dir:
                    filename = os.path.join(output_dir, f"{safe_name}_deformation.csv")
                else:
                    # Deformation only - ask for file
                    filename, _ = QFileDialog.getSaveFileName(
                        self.tab,
                        f"Export {combo_name} Deformation Results",
                        f"{safe_name}_deformation.csv",
                        "CSV Files (*.csv);;All Files (*)"
                    )
                    if not filename:
                        return
                
                ux = deformation_result.all_combo_ux[combo_idx, :]
                uy = deformation_result.all_combo_uy[combo_idx, :]
                uz = deformation_result.all_combo_uz[combo_idx, :]
                
                export_deformation_single_combination(
                    filename=filename,
                    node_ids=deformation_result.node_ids,
                    node_coords=deformation_result.node_coords,
                    ux=ux,
                    uy=uy,
                    uz=uz,
                    combination_index=combo_idx,
                    combination_name=combo_name,
                    displacement_unit=deformation_result.displacement_unit
                )
                exported_files.append(os.path.basename(filename))
            except Exception as e:
                errors.append(f"Deformation: {str(e)}")
        
        # Show result message
        output_location = output_dir if output_dir else os.path.dirname(filename)
        self._show_export_result_message(exported_files, errors, output_location)

    def _show_export_result_message(
        self,
        exported_files: List[str],
        errors: List[str],
        output_location: str
    ) -> None:
        """
        Show a message box with export results.
        
        Args:
            exported_files: List of successfully exported filenames
            errors: List of error messages
            output_location: Path to the output directory
        """
        if exported_files and not errors:
            # All successful
            file_list = "\n".join(f"  - {f}" for f in exported_files)
            QMessageBox.information(
                self.tab, "Export Successful",
                f"Successfully exported {len(exported_files)} file(s) to:\n"
                f"{output_location}\n\n"
                f"Files:\n{file_list}"
            )
        elif exported_files and errors:
            # Partial success
            file_list = "\n".join(f"  - {f}" for f in exported_files)
            error_list = "\n".join(f"  - {e}" for e in errors)
            QMessageBox.warning(
                self.tab, "Export Partially Successful",
                f"Exported {len(exported_files)} file(s) to:\n"
                f"{output_location}\n\n"
                f"Files:\n{file_list}\n\n"
                f"Errors:\n{error_list}"
            )
        else:
            # All failed
            error_list = "\n".join(f"  - {e}" for e in errors)
            QMessageBox.critical(
                self.tab, "Export Failed",
                f"Failed to export results.\n\n"
                f"Errors:\n{error_list}"
            )
