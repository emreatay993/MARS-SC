"""
Output-selection and UI-state handling for the solver tab.

This module owns output checkbox availability, solve-button enablement,
and output-related UI toggle behavior.
"""

from utils.constants import MSG_NODAL_FORCES_ANSYS
from ui.handlers.solver_output_availability import evaluate_output_availability


class SolverOutputStateHandler:
    """Encapsulate output and UI state behavior for ``SolverTab``."""

    def __init__(self, tab):
        self.tab = tab

    def enable_output_checkboxes(self) -> None:
        """Enable output checkboxes when both RST files are loaded."""
        enabled = self.tab.base_rst_loaded and self.tab.combine_rst_loaded
        availability = evaluate_output_availability(
            self.tab.analysis1_data,
            self.tab.analysis2_data,
        )

        self.tab.von_mises_checkbox.setEnabled(enabled)
        self.tab.max_principal_stress_checkbox.setEnabled(enabled)
        self.tab.min_principal_stress_checkbox.setEnabled(enabled)
        self.tab.combination_history_checkbox.setEnabled(enabled)
        self.tab.plasticity_correction_checkbox.setEnabled(enabled)

        self.tab.nodal_forces_checkbox.setEnabled(availability.nodal_forces)
        if not availability.nodal_forces and enabled:
            self.tab.nodal_forces_checkbox.setToolTip(
                "Nodal forces not available.\n"
                "At least one RST file does not contain nodal forces.\n"
                + MSG_NODAL_FORCES_ANSYS
            )
        else:
            self.tab.nodal_forces_checkbox.setToolTip(
                "Combine nodal forces from both analyses.\n"
                "Requires 'Write element nodal forces' to be enabled in ANSYS Output Controls."
            )

        self.tab.deformation_checkbox.setEnabled(availability.displacement)
        if not availability.displacement and enabled:
            self.tab.deformation_checkbox.setToolTip(
                "Displacement results not available.\n"
                "At least one RST file does not contain displacement results."
            )
        else:
            self.tab.deformation_checkbox.setToolTip(
                "Calculate combined displacement/deformation (UX, UY, UZ, U_mag).\n"
                "Can be selected alongside stress outputs.\n"
                "Enables deformed mesh visualization with scale control."
            )

    def update_solve_button_state(self) -> None:
        """Enable/disable solve button based on current tab state."""
        can_solve = (
            self.tab.base_rst_loaded
            and self.tab.combine_rst_loaded
            and self.tab.combo_table.rowCount() > 0
            and any(
                [
                    self.tab.von_mises_checkbox.isChecked(),
                    self.tab.max_principal_stress_checkbox.isChecked(),
                    self.tab.min_principal_stress_checkbox.isChecked(),
                    self.tab.nodal_forces_checkbox.isChecked(),
                    self.tab.deformation_checkbox.isChecked(),
                ]
            )
        )
        self.tab.solve_button.setEnabled(can_solve)

    def toggle_combination_history_mode(self, checked: bool) -> None:
        """Toggle combination-history mode visibility and solve-state refresh."""
        self.tab.single_node_group.setVisible(checked)
        self.update_solve_button_state()

    def toggle_plasticity_options(self, checked: bool) -> None:
        """Toggle plasticity options visibility."""
        self.tab.plasticity_options_group.setVisible(checked)

    def toggle_nodal_forces_csys_combo(self, checked: bool) -> None:
        """Toggle nodal-forces coordinate-system picker visibility."""
        self.tab.nodal_forces_csys_combo.setVisible(checked)

    def toggle_deformation_csys_options(self, checked: bool) -> None:
        """Toggle deformation coordinate-system controls visibility."""
        self.tab.deformation_csys_combo.setVisible(checked)
        if checked:
            is_cylindrical = self.tab.deformation_csys_combo.currentIndex() == 1
            self.tab.deformation_cs_id_label.setVisible(is_cylindrical)
            self.tab.deformation_cs_input.setVisible(is_cylindrical)
        else:
            self.tab.deformation_cs_id_label.setVisible(False)
            self.tab.deformation_cs_input.setVisible(False)
            self.tab.deformation_csys_combo.setCurrentIndex(0)
            self.tab.deformation_cs_input.clear()

    def on_deformation_csys_changed(self, index: int) -> None:
        """Handle deformation coordinate-system selection change."""
        is_cylindrical = index == 1
        self.tab.deformation_cs_id_label.setVisible(is_cylindrical)
        self.tab.deformation_cs_input.setVisible(is_cylindrical)
        if not is_cylindrical:
            self.tab.deformation_cs_input.clear()

    def on_output_checkbox_toggled(self, source_checkbox, checked: bool) -> None:
        """
        Enforce mutual exclusivity across stress/forces output checkboxes.

        Note: deformation output is intentionally not part of this exclusive group.
        """
        if not checked:
            self.update_solve_button_state()
            return

        output_checkboxes = [
            self.tab.von_mises_checkbox,
            self.tab.max_principal_stress_checkbox,
            self.tab.min_principal_stress_checkbox,
            self.tab.nodal_forces_checkbox,
        ]

        for checkbox in output_checkboxes:
            checkbox.blockSignals(True)

        try:
            for checkbox in output_checkboxes:
                if checkbox is not source_checkbox:
                    checkbox.setChecked(False)
        finally:
            for checkbox in output_checkboxes:
                checkbox.blockSignals(False)

        self.update_solve_button_state()
