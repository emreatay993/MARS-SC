"""
Input validation helpers for Solver analysis runs.

This module keeps UI-level validation logic out of the main analysis handler.
"""

from typing import Optional

from PyQt5.QtWidgets import QMessageBox

from core.data_models import SolverConfig


class SolverInputValidator:
    """Validate solver-tab inputs before analysis execution."""

    def __init__(self, tab):
        self.tab = tab

    def validate_inputs(self, config: SolverConfig) -> bool:
        """Validate inputs before running analysis."""
        if not self.tab.base_rst_loaded:
            QMessageBox.warning(
                self.tab, "Missing Input",
                "Please load a Base Analysis RST file."
            )
            return False

        if not self.tab.combine_rst_loaded:
            QMessageBox.warning(
                self.tab, "Missing Input",
                "Please load an Analysis to Combine RST file."
            )
            return False

        # Check named selection
        ns_name = self.tab.get_selected_named_selection()
        if ns_name is None:
            QMessageBox.warning(
                self.tab, "Missing Input",
                "Please select a valid Named Selection."
            )
            return False

        # At least one output must be selected.
        if not any([
            config.calculate_von_mises,
            config.calculate_max_principal_stress,
            config.calculate_min_principal_stress,
            config.calculate_nodal_forces,
            config.calculate_deformation,
        ]):
            QMessageBox.warning(
                self.tab,
                "No Output Selected",
                "Please select at least one output type (stress, deformation, or nodal forces)."
            )
            return False

        # Validate nodal forces availability if selected.
        if config.calculate_nodal_forces:
            a1 = self.tab.analysis1_data
            a2 = self.tab.analysis2_data
            if a1 is None or a2 is None or not (a1.nodal_forces_available and a2.nodal_forces_available):
                QMessageBox.warning(
                    self.tab, "Nodal Forces Not Available",
                    "Nodal forces are not available in one or both RST files.\n\n"
                    "To enable nodal forces output, ensure 'Write element nodal forces' "
                    "is enabled in ANSYS Output Controls before running the analysis."
                )
                return False

        # Check combination table
        combo_table = self.tab.get_combination_table_data()
        if combo_table is None or combo_table.num_combinations == 0:
            QMessageBox.warning(
                self.tab, "Invalid Table",
                "Please enter at least one combination."
            )
            return False
        self.tab.combination_table = combo_table

        # Validate combination table (check for all-zero coefficients)
        is_valid, error_msg = combo_table.validate()
        if not is_valid:
            QMessageBox.warning(
                self.tab, "Invalid Combination Coefficients",
                error_msg
            )
            return False

        # Validate stress outputs are not selected for beam-element named selections.
        stress_outputs_selected = any([
            config.calculate_von_mises,
            config.calculate_max_principal_stress,
            config.calculate_min_principal_stress,
        ])
        if stress_outputs_selected:
            try:
                scoping_reader, source_label = self.tab.get_scoping_reader_for_named_selection(ns_name)
                has_beams = scoping_reader.check_named_selection_has_beam_elements(ns_name)
                if has_beams:
                    QMessageBox.warning(
                        self.tab, "Stress Output Not Supported",
                        f"The selected Named Selection '{ns_name}' contains beam elements.\n\n"
                        f"Scoping source: {source_label}\n\n"
                        "Stress tensor output (Von Mises, Max Principal, Min Principal) is not "
                        "supported for beam elements.\n\n"
                        "Please either:\n"
                        "  - Select a Named Selection that contains only solid/shell elements, or\n"
                        "  - Use 'Nodal Forces' output instead (if available)"
                    )
                    return False
            except Exception as error:
                # Log the error but don't block - let downstream operations handle it.
                self.tab.console_textbox.append(
                    f"[Warning] Could not verify element types for named selection: {error}"
                )

        # Validate node ID for history mode.
        if config.combination_history_mode:
            node_id = config.selected_node_id

            if node_id is None:
                node_text = self.tab.node_line_edit.text().strip()
                if not node_text:
                    QMessageBox.warning(
                        self.tab, "Missing Node ID",
                        "Please enter a Node ID for combination history mode."
                    )
                    return False
                try:
                    node_id = int(node_text)
                except ValueError:
                    node_id = None

            if node_id is None or int(node_id) <= 0:
                QMessageBox.warning(
                    self.tab, "Invalid Node ID",
                    f"'{self.tab.node_line_edit.text().strip()}' is not a valid Node ID.\n\n"
                    "Please enter a positive integer."
                )
                return False
            node_id = int(node_id)
            config.selected_node_id = node_id

            # Check if node exists in current named-selection scoping.
            try:
                scoping_reader, source_label = self.tab.get_scoping_reader_for_named_selection(ns_name)
                scoping = scoping_reader.get_nodal_scoping_from_named_selection(ns_name)
                scoping_ids = list(scoping.ids)
                if node_id not in scoping_ids:
                    QMessageBox.warning(
                        self.tab, "Node Not Found",
                        f"Node ID {node_id} was not found in Named Selection '{ns_name}'.\n\n"
                        f"Scoping source: {source_label}\n"
                        f"The selected Named Selection contains {len(scoping_ids):,} nodes.\n"
                        f"Please enter a valid Node ID from this selection."
                    )
                    return False
            except Exception as error:
                # Log but do not block; downstream checks can still fail with details.
                self.tab.console_textbox.append(
                    f"[Warning] Could not validate node ID: {error}"
                )

        return True

    @staticmethod
    def get_selected_stress_type(config: SolverConfig) -> Optional[str]:
        """Return the stress result type requested by the current config."""
        if config.calculate_von_mises:
            return "von_mises"
        if config.calculate_max_principal_stress:
            return "max_principal"
        if config.calculate_min_principal_stress:
            return "min_principal"
        return None
