"""
UI State Handler for the SolverTab.

This class encapsulates all logic related to managing the state of UI components
(enabling/disabling, showing/hiding, etc.) based on user interaction.
"""

import math
import numpy as np


class SolverUIHandler:
    """Manages the UI state for the SolverTab."""

    def __init__(self, tab):
        """
        Initialize the UI handler.

        Args:
            tab (SolverTab): The parent SolverTab instance.
        """
        self.tab = tab

    def update_output_checkboxes_state(self):
        """Enable/disable output checkboxes based on loaded files."""
        # Stress-related outputs
        stress_enabled = self.tab.coord_loaded and self.tab.stress_loaded
        for cb in self.tab._coord_stress_outputs:
            cb.setEnabled(stress_enabled)
            if not stress_enabled:
                cb.setChecked(False)

        # Deformation-related outputs
        deformations_enabled = (
                self.tab.coord_loaded and
                self.tab.deformations_checkbox.isChecked() and
                self.tab.deformation_loaded
        )
        for cb in self.tab._deformation_outputs:
            cb.setEnabled(deformations_enabled)
            if not deformations_enabled:
                cb.setChecked(False)

        # Ensure dependent controls track Von Mises selection state
        self._update_plasticity_state()
        self._update_damage_index_state()

    def toggle_steady_state_stress_inputs(self, is_checked):
        """Show/hide steady-state stress file controls."""
        self.tab.steady_state_file_button.setVisible(is_checked)
        self.tab.steady_state_file_path.setVisible(is_checked)
        if not is_checked:
            self.tab.steady_state_file_path.clear()

    def toggle_deformations_inputs(self, is_checked):
        """Show/hide deformation file controls."""
        self.tab.deformations_file_button.setVisible(is_checked)
        self.tab.deformations_file_path.setVisible(is_checked)
        self.update_output_checkboxes_state()
        if not is_checked:
            self.tab.deformations_file_path.clear()
            self.tab.deformation_loaded = False

    def toggle_damage_index_checkbox_visibility(self, is_checked=None):
        """Keep damage index checkbox hidden until benchmarking completes."""
        _ = is_checked  # Maintains signal compatibility while checkbox stays hidden
        self.tab.damage_index_checkbox.setChecked(False)
        self.tab.damage_index_checkbox.setVisible(False)  # TODO: Re-enable visibility once damage index is verified

    def toggle_fatigue_params_visibility(self, checked):
        """Show/hide fatigue parameters group."""
        self.tab.fatigue_params_group.setVisible(checked)

    def toggle_plasticity_options_visibility(self, is_checked):
        """Show/hide plasticity options group based on Plasticity checkbox."""
        try:
            self.tab.plasticity_options_group.setVisible(bool(is_checked))
            if is_checked:
                self.on_plasticity_iteration_inputs_changed()
            else:
                self.clear_plasticity_warning()
        except Exception as e:
            print(f"Error toggling plasticity options visibility: {e}")

    def toggle_single_node_solution_group(self, is_checked):
        """Show/hide single node selection group."""
        try:
            if is_checked:
                # Connect exclusive handlers
                for cb in self.tab.time_history_exclusive_outputs:
                    cb.toggled.connect(
                        lambda checked, a_checkbox=cb:
                        self.on_exclusive_output_toggled(checked, a_checkbox)
                    )

                self.tab.single_node_group.setVisible(True)
                self.tab.show_output_tab_widget.setTabVisible(
                    self.tab.show_output_tab_widget.indexOf(self.tab.plot_single_node_tab),
                    True
                )
            else:
                # Disconnect exclusive handlers
                for checkbox in self.tab.time_history_exclusive_outputs:
                    try:
                        checkbox.toggled.disconnect(self.on_exclusive_output_toggled)
                    except TypeError:
                        pass

                self.tab.single_node_group.setVisible(False)
                self.tab.show_output_tab_widget.setTabVisible(
                    self.tab.show_output_tab_widget.indexOf(self.tab.plot_single_node_tab),
                    False
                )
        except Exception as e:
            print(f"Error toggling single node group visibility: {e}")

    def _on_time_history_toggled(self, is_checked):
        """Handle time history mode toggle."""
        if is_checked:
            all_output_checkboxes = self.tab._coord_stress_outputs + self.tab._deformation_outputs
            for checkbox in all_output_checkboxes:
                if checkbox is self.tab.time_history_checkbox:
                    continue
                checkbox.blockSignals(True)
                checkbox.setChecked(False)
                checkbox.blockSignals(False)
            # Ensure dependent UI reflects new states
            self._update_plasticity_state()

    def _update_damage_index_state(self, checked=False):
        """Update damage index checkbox state."""
        self.tab.damage_index_checkbox.setEnabled(False)  # TODO: Enable after damage index benchmarks
        self.tab.damage_index_checkbox.setChecked(False)
        self.tab.damage_index_checkbox.setVisible(False)

    def _update_plasticity_state(self, checked=False):
        """Enable Plasticity Correction only when Von Mises is selected.

        Remains visible at all times; disables and unchecks when Von Mises is not selected.
        """
        is_von_mises_selected = self.tab.von_mises_checkbox.isChecked()
        # Keep visible; only toggle enabled state
        self.tab.plasticity_correction_checkbox.setEnabled(is_von_mises_selected)
        if not is_von_mises_selected:
            self.tab.plasticity_correction_checkbox.setChecked(False)
            # Also hide options when dependency not satisfied
            self.toggle_plasticity_options_visibility(False)
            self.clear_plasticity_warning()

    def on_exclusive_output_toggled(self, is_checked, sender_checkbox):
        """Ensure only one output is selected in time history mode."""
        if self.tab.time_history_checkbox.isChecked() and is_checked:
            for checkbox in self.tab.time_history_exclusive_outputs:
                if checkbox is not sender_checkbox:
                    checkbox.blockSignals(True)
                    checkbox.setChecked(False)
                    checkbox.blockSignals(False)

    # ========== Plot Management Methods ==========

    def update_single_node_plot(self):
        """Update placeholder plot."""
        x = np.linspace(0, 10, 100)
        y = np.zeros(100)
        self.tab.plot_single_node_tab.update_plot(x, y)

    def update_single_node_plot_based_on_checkboxes(self, checked=False):
        """Update plot based on checkbox states."""
        try:
            x_data = [1, 2, 3, 4, 5]
            y_data = [0, 0, 0, 0, 0]

            self.tab.plot_single_node_tab.update_plot(
                x_data, y_data, None,
                is_max_principal_stress=self.tab.max_principal_stress_checkbox.isChecked(),
                is_min_principal_stress=self.tab.min_principal_stress_checkbox.isChecked(),
                is_von_mises=self.tab.von_mises_checkbox.isChecked(),
                is_deformation=self.tab.deformation_checkbox.isChecked(),
                is_velocity=self.tab.velocity_checkbox.isChecked(),
                is_acceleration=self.tab.acceleration_checkbox.isChecked()
            )
        except Exception as e:
            print(f"Error updating plot based on checkbox states: {e}")

    def _update_max_min_plots(self, checked=False):
        """
        Update max/min over time plots when checkboxes are toggled.

        This method rebuilds the plots to reflect the current checkbox selections,
        hiding tabs if no relevant outputs are selected.
        """
        # Only update if solver has run and data exists
        if self.tab.analysis_engine.solver is None:
            return

        # Don't update in time history mode
        if self.tab.time_history_checkbox.isChecked():
            return

        solver = self.tab.analysis_engine.solver

        # Build max traces based on current checkbox states AND available data
        max_traces = []

        if (self.tab.von_mises_checkbox.isChecked() and
                hasattr(solver, 'max_over_time_svm') and solver.max_over_time_svm is not None):
            max_traces.append({
                'name': 'Von Mises (MPa)',
                'data': solver.max_over_time_svm
            })

        if (self.tab.max_principal_stress_checkbox.isChecked() and
                hasattr(solver, 'max_over_time_s1') and solver.max_over_time_s1 is not None):
            max_traces.append({
                'name': 'S1 (MPa)',
                'data': solver.max_over_time_s1
            })

        if (self.tab.deformation_checkbox.isChecked() and
                hasattr(solver, 'max_over_time_def') and solver.max_over_time_def is not None):
            max_traces.append({
                'name': 'Deformation (mm)',
                'data': solver.max_over_time_def
            })

        if (self.tab.velocity_checkbox.isChecked() and
                hasattr(solver, 'max_over_time_vel') and solver.max_over_time_vel is not None):
            max_traces.append({
                'name': 'Velocity (mm/s)',
                'data': solver.max_over_time_vel
            })

        if (self.tab.acceleration_checkbox.isChecked() and
                hasattr(solver, 'max_over_time_acc') and solver.max_over_time_acc is not None):
            max_traces.append({
                'name': 'Acceleration (mm/s²)',
                'data': solver.max_over_time_acc
            })

        # Update or hide max tab
        if max_traces and self.tab.plot_max_over_time_tab is not None:
            self.tab.plot_max_over_time_tab.update_plot(
                self.tab.modal_data.time_values,
                traces=max_traces
            )
            self.tab.show_output_tab_widget.setTabVisible(
                self.tab.show_output_tab_widget.indexOf(self.tab.plot_max_over_time_tab),
                True
            )
        elif self.tab.plot_max_over_time_tab is not None:
            self.tab.plot_max_over_time_tab.clear_plot()
            self.tab.show_output_tab_widget.setTabVisible(
                self.tab.show_output_tab_widget.indexOf(self.tab.plot_max_over_time_tab),
                False
            )

        # Update or hide min tab
        if (self.tab.min_principal_stress_checkbox.isChecked() and
                hasattr(solver, 'min_over_time_s3') and
                solver.min_over_time_s3 is not None and
                self.tab.plot_min_over_time_tab is not None):
            min_traces = [{
                'name': 'S3 (MPa)',
                'data': solver.min_over_time_s3
            }]
            self.tab.plot_min_over_time_tab.update_plot(
                self.tab.modal_data.time_values,
                traces=min_traces
            )
            self.tab.show_output_tab_widget.setTabVisible(
                self.tab.show_output_tab_widget.indexOf(self.tab.plot_min_over_time_tab),
                True
            )
        elif self.tab.plot_min_over_time_tab is not None:
            # Min principal unchecked - hide the tab
            self.tab.plot_min_over_time_tab.clear_plot()
            self.tab.show_output_tab_widget.setTabVisible(
                self.tab.show_output_tab_widget.indexOf(self.tab.plot_min_over_time_tab),
                False
            )

    def on_plasticity_iteration_inputs_changed(self, _text=None):
        """React to iteration parameter edits and display warnings if needed."""
        if not self.tab.plasticity_options_group.isVisible():
            return
        self._update_plasticity_iteration_warning()

    def _update_plasticity_iteration_warning(self):
        warning_label = getattr(self.tab, 'plasticity_warning_label', None)
        if warning_label is None:
            return

        max_iter_text = self.tab.plasticity_max_iter_input.text().strip()
        tolerance_text = self.tab.plasticity_tolerance_input.text().strip()

        default_max_iter = 60
        default_tolerance = 1e-10

        show_warning = False

        try:
            max_iter_value = default_max_iter if max_iter_text == '' else int(max_iter_text)
        except ValueError:
            warning_label.setText("Warning: Invalid iteration count entered.")
            warning_label.setVisible(True)
            return

        try:
            tolerance_value = default_tolerance if tolerance_text == '' else float(tolerance_text)
        except ValueError:
            warning_label.setText("Warning: Invalid tolerance entered.")
            warning_label.setVisible(True)
            return

        warning_label.setText("Warning: Relaxed iteration settings may impact accuracy.")

        if max_iter_value != default_max_iter:
            show_warning = True
        if not math.isclose(tolerance_value, default_tolerance, rel_tol=0.0, abs_tol=1e-15):
            show_warning = True

        warning_label.setVisible(show_warning)

    def clear_plasticity_warning(self):
        warning_label = getattr(self.tab, 'plasticity_warning_label', None)
        if warning_label is not None:
            warning_label.setVisible(False)

    def _update_skip_modes_combo(self, num_modes):
        """Update skip modes combo box."""
        self.tab.skip_modes_combo.clear()
        self.tab.skip_modes_combo.addItems([str(i) for i in range(num_modes + 1)])
        self.tab.skip_modes_label.setVisible(True)
        self.tab.skip_modes_combo.setVisible(True)

    def on_skip_modes_changed(self, text):
        """Handle skip modes selection change."""
        try:
            if not text or not text.isdigit():
                return
            num_skipped = int(text)
            message = (
                f"\n[INFO] Skip Modes option is set to {num_skipped}. "
                f"The first {num_skipped} modes will be excluded from the next calculation.\n"
            )
            if self.tab.stress_data and self.tab.stress_data.modal_sx is not None:
                total_modes = self.tab.stress_data.num_modes
                modes_used = total_modes - num_skipped
                message += (
                    f"       - Modes to be used: {modes_used} "
                    f"(from mode {num_skipped + 1} to {total_modes})\n"
                )
            self.tab.console_textbox.append(message)
            self.tab.console_textbox.verticalScrollBar().setValue(
                self.tab.console_textbox.verticalScrollBar().maximum()
            )
        except (ValueError, TypeError) as e:
            self.tab.console_textbox.append(
                f"\n[DEBUG] Could not parse skip modes value: {text}. Error: {e}"
            )

    def _update_solve_button_state(self):
        """Enable/disable solve button based on loaded files and selected outputs."""
        files_loaded = self.tab.coord_loaded and self.tab.stress_loaded
        
        # Check if at least one result output is selected
        result_outputs = [
            self.tab.max_principal_stress_checkbox,
            self.tab.min_principal_stress_checkbox,
            self.tab.von_mises_checkbox,
            self.tab.deformation_checkbox,
            self.tab.velocity_checkbox,
            self.tab.acceleration_checkbox,
            self.tab.damage_index_checkbox,
            self.tab.plasticity_correction_checkbox
        ]
        
        output_selected = any(cb.isChecked() for cb in result_outputs)
        
        self.tab.solve_button.setEnabled(files_loaded and output_selected)

    def _hide_plot_tabs(self):
        """Hide all plot tabs."""
        self.tab.show_output_tab_widget.setTabVisible(
            self.tab.show_output_tab_widget.indexOf(self.tab.plot_modal_coords_tab), False
        )
        if self.tab.plot_max_over_time_tab is not None:
            self.tab.plot_max_over_time_tab.clear_plot()
            self.tab.show_output_tab_widget.setTabVisible(
                self.tab.show_output_tab_widget.indexOf(self.tab.plot_max_over_time_tab), False
            )
        if self.tab.plot_min_over_time_tab is not None:
            self.tab.plot_min_over_time_tab.clear_plot()
            self.tab.show_output_tab_widget.setTabVisible(
                self.tab.show_output_tab_widget.indexOf(self.tab.plot_min_over_time_tab), False
            )

    def _show_modal_coords_tab(self):
        """Show modal coordinates plot tab."""
        self.tab.show_output_tab_widget.setTabVisible(
            self.tab.show_output_tab_widget.indexOf(self.tab.plot_modal_coords_tab), True
        )
