"""
Solver tab: combine stress (and optionally forces/deformation) from two static RST files
via linear combination coefficients.
"""

import os
import sys
from typing import Optional

from PyQt5.QtCore import pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import (
    QMessageBox, QWidget, QDialog, QVBoxLayout,
    QTableWidgetItem
)


# Import builders and managers
from ui.builders.solver_ui import SolverTabUIBuilder
from ui.handlers.file_handler import SolverFileHandler
from ui.handlers.solve_run_controller import SolveRunController
from ui.handlers.solver_result_payload_handler import SolverResultPayloadHandler
from ui.handlers.solver_combination_table_handler import SolverCombinationTableHandler
from ui.handlers.solver_named_selection_handler import SolverNamedSelectionHandler
from ui.handlers.solver_output_state_handler import SolverOutputStateHandler
from ui.handlers.log_handler import SolverLogHandler
from ui.dialogs.material_profile_dialog import MaterialProfileDialog
from ui.widgets.console import Logger
from core.data_models import (
    AnalysisData, CombinationTableData, CombinationResult, NodalForcesResult,
    DeformationResult, TemperatureFieldData, MaterialProfileData, SolverConfig
)


class SolverTab(QWidget):
    """
    Solver tab for MARS-SC solution combination analysis.

    Manages RST file loading, combination table configuration, and analysis
    execution for combining stress results from two static analyses.

    Signals:
        initial_data_loaded: Emitted when initial RST data is loaded
        display_payload_ready: Emitted when display results payload is ready
    """
    
    # Signals
    initial_data_loaded = pyqtSignal(object)
    display_payload_ready = pyqtSignal(object)
    
    def __init__(self, parent=None):
        """Initialize the Solver Tab."""
        super().__init__(parent)
        
        # Initialize state
        self.project_directory = None
        
        # RST file data
        self.analysis1_data: Optional[AnalysisData] = None  # Base analysis
        self.analysis2_data: Optional[AnalysisData] = None  # Analysis to combine
        
        # Combination data
        self.combination_table: Optional[CombinationTableData] = None
        self.combination_result: Optional[CombinationResult] = None
        self.nodal_forces_result: Optional[NodalForcesResult] = None
        self.deformation_result: Optional[DeformationResult] = None
        
        # Temperature and material data (for plasticity)
        self.temperature_field_data: Optional[TemperatureFieldData] = None
        self.material_profile_data: MaterialProfileData = MaterialProfileData.empty()
        
        # Handlers
        self.file_handler = SolverFileHandler(self)
        self.solve_run_controller = SolveRunController(self)
        self.result_payload_handler = SolverResultPayloadHandler(self)
        self.combination_table_handler = SolverCombinationTableHandler(self)
        self.named_selection_handler = SolverNamedSelectionHandler(self)
        self.output_state_handler = SolverOutputStateHandler(self)
        self.log_handler = SolverLogHandler(self)
        
        # Flags for loaded data
        self.base_rst_loaded = False
        self.combine_rst_loaded = False

        # For plotting
        self.plot_dialog = None
        
        # Build UI
        self._build_ui()
        
        # Setup logger
        self.logger = Logger(self.console_textbox)
        sys.stdout = self.logger
        
        # Enable drag and drop
        self.setAcceptDrops(True)
        
        # Initial state
        self._update_solve_button_state()
    
    def _build_ui(self):
        """Build the UI using the UI builder."""
        builder = SolverTabUIBuilder()
        
        # Set window palette
        builder.set_window_palette(self)
        
        # Build layout and get components
        layout, self.components = builder.build_complete_layout()
        self.setLayout(layout)
        
        # Store commonly used components as direct attributes
        self._setup_component_references()
        
        # Connect signals
        self._connect_signals()
    
    def _setup_component_references(self):
        """Create direct references to frequently used components."""
        # RST File controls
        self.base_rst_button = self.components['base_rst_button']
        self.base_rst_path = self.components['base_rst_path']
        self.base_info_label = self.components['base_info_label']
        self.combine_rst_button = self.components['combine_rst_button']
        self.combine_rst_path = self.components['combine_rst_path']
        self.combine_info_label = self.components['combine_info_label']
        self.named_selection_source_combo = self.components['named_selection_source_combo']
        self.named_selection_combo = self.components['named_selection_combo']
        self.refresh_ns_button = self.components['refresh_ns_button']
        self.skip_substeps_checkbox = self.components['skip_substeps_checkbox']
        
        # Combination table
        self.combo_table = self.components['combo_table']
        self.import_csv_btn = self.components['import_csv_btn']
        self.export_csv_btn = self.components['export_csv_btn']
        self.add_row_btn = self.components['add_row_btn']
        self.delete_row_btn = self.components['delete_row_btn']
        
        # Collapsible group boxes (for programmatic collapse/expand)
        self.file_input_group = self.components.get('file_input_group')
        self.combo_table_group = self.components.get('combo_table_group')
        
        # Setup table delegates for validation
        self._setup_table_delegates()
        
        # Output checkboxes
        self.combination_history_checkbox = self.components['combination_history_checkbox']
        self.von_mises_checkbox = self.components['von_mises_checkbox']
        self.max_principal_stress_checkbox = self.components['max_principal_stress_checkbox']
        self.min_principal_stress_checkbox = self.components['min_principal_stress_checkbox']
        self.nodal_forces_checkbox = self.components['nodal_forces_checkbox']
        self.nodal_forces_csys_combo = self.components['nodal_forces_csys_combo']
        self.deformation_checkbox = self.components['deformation_checkbox']
        self.deformation_csys_combo = self.components['deformation_csys_combo']
        self.deformation_cs_id_label = self.components['deformation_cs_id_label']
        self.deformation_cs_input = self.components['deformation_cs_input']
        self.plasticity_correction_checkbox = self.components['plasticity_correction_checkbox']
        
        # Single node controls
        self.node_line_edit = self.components['node_line_edit']
        self.single_node_group = self.components['single_node_group']
        
        # Plasticity options
        self.plasticity_options_group = self.components['plasticity_options_group']
        self.material_profile_button = self.components['material_profile_button']
        self.temperature_field_button = self.components['temperature_field_button']
        self.temperature_field_file_path = self.components['temperature_field_path']
        self.plasticity_method_combo = self.components['plasticity_method_combo']
        self.plasticity_max_iter_input = self.components['plasticity_max_iter_input']
        self.plasticity_tolerance_input = self.components['plasticity_tolerance_input']
        self.plasticity_warning_label = self.components['plasticity_warning_label']
        self.plasticity_extrapolation_combo = self.components['plasticity_extrapolation_combo']
        
        # Console and plots
        self.console_textbox = self.components['console_textbox']
        self.show_output_tab_widget = self.components['show_output_tab_widget']
        self.plot_combo_history_tab = self.components['plot_combo_history_tab']
        self.plot_max_combo_tab = self.components['plot_max_combo_tab']
        self.plot_min_combo_tab = self.components['plot_min_combo_tab']
        
        # Progress and solve
        self.progress_bar = self.components['progress_bar']
        self.solve_button = self.components['solve_button']
    
    def _connect_signals(self):
        """Connect UI signals to their handlers."""
        # RST File loading
        self.base_rst_button.clicked.connect(self.file_handler.select_base_rst_file)
        self.combine_rst_button.clicked.connect(self.file_handler.select_combine_rst_file)
        self.refresh_ns_button.clicked.connect(self.file_handler.refresh_named_selections)
        self.named_selection_source_combo.currentIndexChanged.connect(self._on_named_selection_source_changed)
        
        # Combination table controls
        self.import_csv_btn.clicked.connect(self.file_handler.import_combination_table)
        self.export_csv_btn.clicked.connect(self.file_handler.export_combination_table)
        self.add_row_btn.clicked.connect(self._add_table_row)
        self.delete_row_btn.clicked.connect(self._delete_table_row)
        
        # Connect cell change signal for coefficient highlighting
        self.combo_table.cellChanged.connect(self._on_coefficient_cell_changed)
        
        # Output checkboxes - mutual exclusivity for stress output types only
        self.von_mises_checkbox.toggled.connect(
            lambda checked: self._on_output_checkbox_toggled(self.von_mises_checkbox, checked))
        self.max_principal_stress_checkbox.toggled.connect(
            lambda checked: self._on_output_checkbox_toggled(self.max_principal_stress_checkbox, checked))
        self.min_principal_stress_checkbox.toggled.connect(
            lambda checked: self._on_output_checkbox_toggled(self.min_principal_stress_checkbox, checked))
        self.nodal_forces_checkbox.toggled.connect(
            lambda checked: self._on_output_checkbox_toggled(self.nodal_forces_checkbox, checked))
        self.nodal_forces_checkbox.toggled.connect(self._toggle_nodal_forces_csys_combo)
        
        # Deformation checkbox - NOT mutually exclusive, can be selected alongside stress
        self.deformation_checkbox.toggled.connect(self._update_solve_button_state)
        self.deformation_checkbox.toggled.connect(self._toggle_deformation_csys_options)
        self.deformation_csys_combo.currentIndexChanged.connect(self._on_deformation_csys_changed)
        
        # Mode toggles (not mutually exclusive with output types)
        self.combination_history_checkbox.toggled.connect(self._toggle_combination_history_mode)
        self.plasticity_correction_checkbox.toggled.connect(self._toggle_plasticity_options)
        
        # Plasticity options
        self.material_profile_button.clicked.connect(self.open_material_profile_dialog)
        self.temperature_field_button.clicked.connect(self.file_handler.select_temperature_field_file)
        
        # Node entry
        self.node_line_edit.returnPressed.connect(self._on_node_entered)
        
        # Solve button
        self.solve_button.clicked.connect(self._on_solve_clicked)
    
    # ========== RST File Loading Callbacks ==========
    
    def on_base_rst_loaded(self, analysis_data: AnalysisData, filename: str):
        """Handle UI updates after base RST file is loaded."""
        self.analysis1_data = analysis_data
        self.base_rst_loaded = True
        
        # Update UI
        self.base_rst_path.setText(filename)
        self.base_info_label.setText(f"Load steps: {analysis_data.num_load_steps}")
        
        # Format time values for display
        time_info = ""
        if analysis_data.time_values:
            time_strs = [f"{t:.4g}s" for t in analysis_data.time_values]
            num_points = len(time_strs)
            if num_points >= 20:
                # Show first 5, dots, last 5, and count
                first_five = ', '.join(time_strs[:5])
                last_five = ', '.join(time_strs[-5:])
                time_info = f"  Time Points: {first_five} ......... {last_five}   ({num_points} Time Points)\n"
            elif num_points > 10:
                # Show first 3, dots, last 3
                time_info = f"  Time Points: {', '.join(time_strs[:3])} ... {', '.join(time_strs[-3:])}\n"
            else:
                # Show all
                time_info = f"  Time Points: {', '.join(time_strs)}\n"
        
        # Nodal forces availability
        nodal_forces_status = "Available" if analysis_data.nodal_forces_available else "Not Available"
        
        # Log with unit and time information
        self.console_textbox.append(
            f"\n{'='*60}\n"
            f"Loaded Base Analysis RST: {os.path.basename(filename)}\n"
            f"  Load Steps: {analysis_data.num_load_steps}\n"
            f"{time_info}"
            f"  Named Selections: {len(analysis_data.named_selections)}\n"
            f"  Unit System: {analysis_data.unit_system}\n"
            f"  Stress Unit: {analysis_data.stress_unit} (converting to MPa)\n"
            f"  Nodal Forces: {nodal_forces_status}\n"
            f"{'='*60}"
        )
        
        # Update named selections dropdown
        self._update_named_selections()
        
        # Update combination table columns
        self._update_combination_table_columns()
        
        # Enable controls
        self._update_solve_button_state()
        self._enable_output_checkboxes()
    
    def on_combine_rst_loaded(self, analysis_data: AnalysisData, filename: str):
        """Handle UI updates after combine RST file is loaded."""
        self.analysis2_data = analysis_data
        self.combine_rst_loaded = True
        
        # Update UI
        self.combine_rst_path.setText(filename)
        self.combine_info_label.setText(f"Load steps: {analysis_data.num_load_steps}")
        
        # Format time values for display
        time_info = ""
        if analysis_data.time_values:
            time_strs = [f"{t:.4g}s" for t in analysis_data.time_values]
            num_points = len(time_strs)
            if num_points >= 20:
                # Show first 5, dots, last 5, and count
                first_five = ', '.join(time_strs[:5])
                last_five = ', '.join(time_strs[-5:])
                time_info = f"  Time Points: {first_five} ......... {last_five}   ({num_points} Time Points)\n"
            elif num_points > 10:
                # Show first 3, dots, last 3
                time_info = f"  Time Points: {', '.join(time_strs[:3])} ... {', '.join(time_strs[-3:])}\n"
            else:
                # Show all
                time_info = f"  Time Points: {', '.join(time_strs)}\n"
        
        # Nodal forces availability
        nodal_forces_status = "Available" if analysis_data.nodal_forces_available else "Not Available"
        
        # Log with unit and time information
        self.console_textbox.append(
            f"\n{'='*60}\n"
            f"Loaded Analysis to Combine RST: {os.path.basename(filename)}\n"
            f"  Load Steps: {analysis_data.num_load_steps}\n"
            f"{time_info}"
            f"  Named Selections: {len(analysis_data.named_selections)}\n"
            f"  Unit System: {analysis_data.unit_system}\n"
            f"  Stress Unit: {analysis_data.stress_unit} (converting to MPa)\n"
            f"  Nodal Forces: {nodal_forces_status}\n"
            f"{'='*60}"
        )
        
        # Update named selections dropdown
        self._update_named_selections()
        
        # Update combination table columns
        self._update_combination_table_columns()
        
        # Enable controls
        self._update_solve_button_state()
        self._enable_output_checkboxes()
    
    def _on_named_selection_source_changed(self, index: int):
        """Refresh list when the named selection source filter changes."""
        self.named_selection_handler.on_named_selection_source_changed(index)

    def _get_named_selection_source_mode(self) -> str:
        """Return active source mode for named selection list."""
        return self.named_selection_handler.get_named_selection_source_mode()

    def _get_named_selection_sets(self):
        """Get named selection name sets for both analyses."""
        return self.named_selection_handler.get_named_selection_sets()

    def _update_named_selections(self):
        """Update named selections dropdown based on current source filter."""
        self.named_selection_handler.update_named_selections()
    
    def _update_combination_table_columns(self):
        """Update combination-table columns from the current analysis metadata."""
        self.combination_table_handler.update_combination_table_columns()
    
    def _enable_output_checkboxes(self):
        """Enable output checkboxes when both RST files are loaded."""
        self.output_state_handler.enable_output_checkboxes()
    
    # ========== Combination Table Methods ==========
    
    def _setup_table_delegates(self):
        """Setup delegates for combination-table editing and validation."""
        self.combination_table_handler.setup_table_delegates()
    
    def _apply_numeric_delegate_to_columns(self):
        """Apply numeric delegate to all coefficient columns (column 2 onwards)."""
        self.combination_table_handler.apply_numeric_delegate_to_columns()

    def _update_coefficient_cell_highlight(self, item: QTableWidgetItem):
        """
        Update the background color of a coefficient cell based on its value.
        
        Non-zero values get a light green background to visually distinguish them
        from zero/empty values.
        
        Args:
            item: The QTableWidgetItem to update.
        """
        self.combination_table_handler.update_coefficient_cell_highlight(item)

    def _on_coefficient_cell_changed(self, row: int, column: int):
        """
        Handle cell changes in the combination table to update highlighting.
        
        This is called whenever a user manually edits a cell in the table.
        Only coefficient columns (column 2 onwards) are highlighted.
        
        Args:
            row: The row index of the changed cell.
            column: The column index of the changed cell.
        """
        self.combination_table_handler.on_coefficient_cell_changed(row, column)

    def _add_table_row(self):
        """Add a new row to the combination table."""
        self.combination_table_handler.add_table_row()
    
    def _delete_table_row(self):
        """Delete the selected row from the combination table."""
        self.combination_table_handler.delete_table_row()
    
    def get_combination_table_data(self) -> Optional[CombinationTableData]:
        """
        Extract combination table data from the UI table widget.
        
        Returns:
            CombinationTableData or None if invalid.
        """
        return self.combination_table_handler.get_combination_table_data()
    
    def set_combination_table_data(self, data: CombinationTableData):
        """
        Populate the UI table widget from CombinationTableData.
        
        Args:
            data: CombinationTableData to display.
        """
        self.combination_table_handler.set_combination_table_data(data)
    
    # ========== UI State Methods ==========
    
    def _update_solve_button_state(self):
        """Update solve button enabled state."""
        self.output_state_handler.update_solve_button_state()
    
    def _toggle_combination_history_mode(self, checked: bool):
        """Toggle combination history mode (single node analysis)."""
        self.output_state_handler.toggle_combination_history_mode(checked)
    
    def _toggle_plasticity_options(self, checked: bool):
        """Toggle plasticity correction options visibility."""
        self.output_state_handler.toggle_plasticity_options(checked)
    
    def _toggle_nodal_forces_csys_combo(self, checked: bool):
        """Toggle nodal forces coordinate system combo visibility."""
        self.output_state_handler.toggle_nodal_forces_csys_combo(checked)
    
    def _toggle_deformation_csys_options(self, checked: bool):
        """Toggle deformation coordinate system options visibility."""
        self.output_state_handler.toggle_deformation_csys_options(checked)
    
    def _on_deformation_csys_changed(self, index: int):
        """Handle deformation coordinate system selection change."""
        self.output_state_handler.on_deformation_csys_changed(index)
    
    def _on_output_checkbox_toggled(self, source_checkbox, checked: bool):
        """
        Handle output type checkbox toggle with mutual exclusivity.
        
        Only one output type (von_mises, max_principal, min_principal, nodal_forces)
        can be selected at a time. When a checkbox is checked, all other output
        type checkboxes are automatically unchecked.
        
        Note: combination_history_checkbox and plasticity_correction_checkbox are
        NOT mutually exclusive - they can be combined with any output type.
        
        Args:
            source_checkbox: The checkbox that was toggled.
            checked: Whether the checkbox was checked or unchecked.
        """
        self.output_state_handler.on_output_checkbox_toggled(source_checkbox, checked)
    
    # ========== Analysis Methods ==========
    
    def _on_solve_clicked(self):
        """Handle solve button click."""
        # Cache current combination table data.
        self.combination_table = self.get_combination_table_data()
        
        # Build solver config
        config = self._build_solver_config()
        
        # Run analysis
        self.solve_run_controller.solve(config)
    
    def _build_solver_config(self) -> SolverConfig:
        """Build solver configuration from UI state."""
        # Determine if nodal forces should be in global coordinates
        # "Global" = index 0 = rotate_to_global=True
        # "Local (Element)" = index 1 = rotate_to_global=False
        nodal_forces_rotate_to_global = self.nodal_forces_csys_combo.currentIndex() == 0
        
        # Get cylindrical CS ID for deformation (None if Cartesian selected or empty)
        deformation_cylindrical_cs_id = None
        if self.deformation_csys_combo.currentIndex() == 1:  # Cylindrical selected
            deformation_cs_text = self.deformation_cs_input.text().strip()
            if deformation_cs_text:
                deformation_cylindrical_cs_id = int(deformation_cs_text)
        
        config = SolverConfig(
            calculate_von_mises=self.von_mises_checkbox.isChecked(),
            calculate_max_principal_stress=self.max_principal_stress_checkbox.isChecked(),
            calculate_min_principal_stress=self.min_principal_stress_checkbox.isChecked(),
            calculate_nodal_forces=self.nodal_forces_checkbox.isChecked(),
            calculate_deformation=self.deformation_checkbox.isChecked(),
            nodal_forces_rotate_to_global=nodal_forces_rotate_to_global,
            deformation_cylindrical_cs_id=deformation_cylindrical_cs_id,
            combination_history_mode=self.combination_history_checkbox.isChecked(),
            output_directory=self.project_directory,
        )
        
        if config.combination_history_mode:
            node_text = self.node_line_edit.text().strip()
            try:
                node_id = int(node_text)
                config.selected_node_id = node_id if node_id > 0 else None
            except ValueError:
                config.selected_node_id = None
        
        # Plasticity config
        if self.plasticity_correction_checkbox.isChecked():
            from core.data_models import PlasticityConfig
            config.plasticity = PlasticityConfig(
                enabled=True,
                method=self.plasticity_method_combo.currentText().lower().replace(" ", "_"),
                max_iterations=int(self.plasticity_max_iter_input.text() or 60),
                tolerance=float(self.plasticity_tolerance_input.text() or 1e-10),
                material_profile=self.material_profile_data,
                temperature_field=self.temperature_field_data,
                extrapolation_mode=self.plasticity_extrapolation_combo.currentText().lower(),
            )
        
        return config
    
    def get_selected_named_selection(self) -> Optional[str]:
        """Get the currently selected named selection."""
        return self.named_selection_handler.get_selected_named_selection()

    def get_scoping_reader_for_named_selection(self, ns_name: str):
        """
        Resolve which analysis reader should provide named-selection node scoping.

        If the same named selection exists in both analyses, Analysis 1 takes
        precedence to avoid mismatched node content.
        """
        return self.named_selection_handler.get_scoping_reader_for_named_selection(ns_name)

    def get_nodal_scoping_for_selected_named_selection(self):
        """Get nodal scoping for current selection based on the active source mode."""
        return self.named_selection_handler.get_nodal_scoping_for_selected_named_selection()
    
    # ========== Result Handling ==========
    
    def on_analysis_complete(self, result: CombinationResult):
        """Handle stress analysis completion."""
        self.result_payload_handler.on_analysis_complete(result)
    
    def on_forces_analysis_complete(self, result: NodalForcesResult):
        """Handle nodal forces analysis completion."""
        self.result_payload_handler.on_forces_analysis_complete(result)
    
    def on_deformation_analysis_complete(self, result: DeformationResult, is_standalone: bool = False):
        """
        Handle deformation analysis completion.
        
        Args:
            result: DeformationResult with displacement data.
            is_standalone: If True, deformation is the only output type - create and emit mesh.
        """
        self.result_payload_handler.on_deformation_analysis_complete(
            result,
            is_standalone=is_standalone,
        )
    
    def _on_node_entered(self):
        """Handle Enter key press in node ID field."""
        node_text = self.node_line_edit.text()
        if node_text.isdigit():
            self.console_textbox.append(f"Node ID entered: {node_text}")
    
    def plot_combination_history_for_node(self, node_id: int):
        """
        Trigger combination history analysis for a specific node.
        
        This method is called when a node is picked from the Display tab's
        right-click context menu "Plot Combination History for Selected Node".
        Sets the selected node and starts the solve path.
        
        Args:
            node_id: The node ID to compute combination history for.
        """
        # Update node ID in the UI
        self.node_line_edit.setText(str(node_id))
        
        # Enable combination history mode
        if not self.combination_history_checkbox.isChecked():
            self.combination_history_checkbox.setChecked(True)
        
        # Log and trigger solve
        self.console_textbox.append(
            f"\n{'='*60}\n"
            f"Computing Combination History for Node {node_id}\n"
            f"{'='*60}"
        )
        
        # Trigger the solve (this calls _on_solve_clicked internally)
        self._on_solve_clicked()
    
    # ========== Plasticity Dialog ==========
    
    def open_material_profile_dialog(self):
        """Launch the material profile editor dialog."""
        dialog = MaterialProfileDialog(self, self.material_profile_data)
        if dialog.exec_() == QDialog.Accepted:
            self.material_profile_data = dialog.get_data()
            self.console_textbox.append("Material profile updated.")
    
    def on_temperature_field_loaded(self, temperature_data, filename):
        """Handle UI updates after temperature field file is loaded."""
        self.temperature_field_data = temperature_data
        self.console_textbox.append(f"Loaded temperature field: {os.path.basename(filename)}")
    
    # ========== Progress Updates ==========
    
    @pyqtSlot(int)
    def update_progress_bar(self, value):
        """Update progress bar value."""
        self.progress_bar.setValue(value)
        self.progress_bar.setFormat(f"Progress: {value}%")
    
    # ========== Drag and Drop Support ==========

    def dragEnterEvent(self, event):
        """Handle drag enter event."""
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()
    
    def dropEvent(self, event):
        """Handle drop event."""
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        if files:
            file_path = files[0]
            # Determine file type by extension
            if file_path.lower().endswith('.rst'):
                # Ask user which analysis
                from PyQt5.QtWidgets import QInputDialog
                choice, ok = QInputDialog.getItem(
                    self, "Select Analysis",
                    "Load RST file as:",
                    ["Base Analysis (Analysis 1)", "Analysis to Combine (Analysis 2)"],
                    0, False
                )
                if ok:
                    if "Base" in choice:
                        self.file_handler._load_rst_file(file_path, is_base=True)
                    else:
                        self.file_handler._load_rst_file(file_path, is_base=False)
            elif file_path.lower().endswith('.csv'):
                self.file_handler._load_combination_csv(file_path)
