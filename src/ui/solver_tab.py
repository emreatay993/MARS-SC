"""
Solver tab implementation for MARS-SC (Solution Combination).

Provides the main solver interface for combining stress results from two
static analysis RST files using linear combination coefficients.
"""

import os
import sys
from typing import Optional, List

import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QPalette, QColor, QDoubleValidator, QBrush
from PyQt5.QtWidgets import (
    QMessageBox, QWidget, QDialog, QVBoxLayout,
    QTableWidgetItem, QFileDialog, QStyledItemDelegate, QLineEdit
)

from ui.styles.style_constants import NONZERO_COEFFICIENT_BG_COLOR


class NumericDelegate(QStyledItemDelegate):
    """
    Custom delegate that enforces numeric (float/integer) input for table cells.
    
    This delegate is applied to coefficient columns in the combination table
    to ensure only valid numeric values can be entered.
    """
    
    def createEditor(self, parent, option, index):
        """Create a line edit with numeric validation."""
        editor = QLineEdit(parent)
        validator = QDoubleValidator()
        validator.setNotation(QDoubleValidator.StandardNotation)
        editor.setValidator(validator)
        return editor
    
    def setEditorData(self, editor, index):
        """Set the editor's initial value from the model."""
        value = index.model().data(index, Qt.EditRole)
        if value:
            editor.setText(str(value))
        else:
            editor.setText("0.0")
    
    def setModelData(self, editor, model, index):
        """Validate and set the model data from the editor."""
        text = editor.text()
        try:
            # Try to parse as float
            value = float(text) if text else 0.0
            model.setData(index, str(value), Qt.EditRole)
        except ValueError:
            # If invalid, set to 0.0
            model.setData(index, "0.0", Qt.EditRole)


class ReadOnlyDelegate(QStyledItemDelegate):
    """
    Custom delegate that makes table cells read-only.
    
    This delegate is applied to the Type column since only "Linear" is supported.
    """
    
    def createEditor(self, parent, option, index):
        """Return None to prevent editing."""
        return None

# Import builders and managers
from ui.builders.solver_ui import SolverTabUIBuilder
from ui.handlers.file_handler import SolverFileHandler
from ui.handlers.ui_state_handler import SolverUIHandler
from ui.handlers.analysis_handler import SolverAnalysisHandler
from ui.handlers.log_handler import SolverLogHandler
from ui.dialogs.material_profile_dialog import MaterialProfileDialog
from ui.widgets.console import Logger
from ui.widgets.plotting import MatplotlibWidget
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
        combination_result_ready: Emitted when combination results are ready
    """
    
    # Signals
    initial_data_loaded = pyqtSignal(object)
    combination_result_ready = pyqtSignal(object, str, float, float)
    
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
        self.ui_handler = SolverUIHandler(self)
        self.analysis_handler = SolverAnalysisHandler(self)
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
        
        # Export controls
        self.single_combo_dropdown = self.components['single_combo_dropdown']
        self.export_single_btn = self.components['export_single_btn']
        self.export_group = self.components['export_group']
        
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
        
        # Export - DEPRECATED: Export functionality moved to Display tab's Export Output CSV button
        # self.export_single_btn.clicked.connect(self._export_single_combination)
    
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
    
    def _update_named_selections(self):
        """Update the named selections dropdown with common selections."""
        self.named_selection_combo.clear()
        
        if self.analysis1_data and self.analysis2_data:
            # Find common named selections
            ns1 = set(self.analysis1_data.named_selections)
            ns2 = set(self.analysis2_data.named_selections)
            common_ns = sorted(ns1.intersection(ns2))
            
            if common_ns:
                self.named_selection_combo.addItems(common_ns)
                self.named_selection_combo.setEnabled(True)
                self.refresh_ns_button.setEnabled(True)
            else:
                self.named_selection_combo.addItem("(No common named selections)")
                self.named_selection_combo.setEnabled(False)
        elif self.analysis1_data:
            # Only base loaded
            if self.analysis1_data.named_selections:
                self.named_selection_combo.addItems(self.analysis1_data.named_selections)
            else:
                self.named_selection_combo.addItem("(No named selections)")
            self.named_selection_combo.setEnabled(False)
        else:
            self.named_selection_combo.addItem("(Load RST files first)")
            self.named_selection_combo.setEnabled(False)
    
    def _update_combination_table_columns(self):
        """Update the combination table columns based on loaded RST files."""
        # Build column headers
        columns = ["Combination Name", "Type"]
        
        # Add Analysis 1 columns with time values
        if self.analysis1_data:
            for step_id in self.analysis1_data.load_step_ids:
                # Use time-based label if available
                label = self.analysis1_data.format_time_label(step_id, prefix="A1")
                columns.append(label)
        
        # Add Analysis 2 columns with time values
        if self.analysis2_data:
            for step_id in self.analysis2_data.load_step_ids:
                # Use time-based label if available
                label = self.analysis2_data.format_time_label(step_id, prefix="A2")
                columns.append(label)
        
        # Update table
        current_rows = self.combo_table.rowCount()
        self.combo_table.setColumnCount(len(columns))
        self.combo_table.setHorizontalHeaderLabels(columns)
        
        # Initialize cells with default values
        for row in range(current_rows):
            # Make Type column read-only
            type_item = self.combo_table.item(row, 1)
            if type_item:
                type_item.setFlags(type_item.flags() & ~Qt.ItemIsEditable)
                type_item.setToolTip("Only 'Linear' combination type is currently supported")
            
            for col in range(2, len(columns)):
                if self.combo_table.item(row, col) is None:
                    self.combo_table.setItem(row, col, QTableWidgetItem("0.0"))
        
        # Reapply numeric delegate to new columns
        self._apply_numeric_delegate_to_columns()
    
    def _enable_output_checkboxes(self):
        """Enable output checkboxes when both RST files are loaded."""
        enabled = self.base_rst_loaded and self.combine_rst_loaded
        
        self.von_mises_checkbox.setEnabled(enabled)
        self.max_principal_stress_checkbox.setEnabled(enabled)
        self.min_principal_stress_checkbox.setEnabled(enabled)
        self.combination_history_checkbox.setEnabled(enabled)
        self.plasticity_correction_checkbox.setEnabled(enabled)
        
        # Enable nodal forces checkbox only if both files have nodal forces
        nodal_forces_available = False
        if enabled and self.analysis1_data and self.analysis2_data:
            nodal_forces_available = (
                self.analysis1_data.nodal_forces_available and 
                self.analysis2_data.nodal_forces_available
            )
        
        self.nodal_forces_checkbox.setEnabled(nodal_forces_available)
        if not nodal_forces_available and enabled:
            self.nodal_forces_checkbox.setToolTip(
                "Nodal forces not available.\n"
                "At least one RST file does not contain nodal forces.\n"
                "Ensure 'Write element nodal forces' is enabled in ANSYS Output Controls."
            )
        else:
            self.nodal_forces_checkbox.setToolTip(
                "Combine nodal forces from both analyses.\n"
                "Requires 'Write element nodal forces' to be enabled in ANSYS Output Controls."
            )
        
        # Enable deformation checkbox only if both files have displacement results
        displacement_available = False
        if enabled and self.analysis1_data and self.analysis2_data:
            displacement_available = (
                self.analysis1_data.displacement_available and 
                self.analysis2_data.displacement_available
            )
        
        self.deformation_checkbox.setEnabled(displacement_available)
        if not displacement_available and enabled:
            self.deformation_checkbox.setToolTip(
                "Displacement results not available.\n"
                "At least one RST file does not contain displacement results."
            )
        else:
            self.deformation_checkbox.setToolTip(
                "Calculate combined displacement/deformation (UX, UY, UZ, U_mag).\n"
                "Can be selected alongside stress outputs.\n"
                "Enables deformed mesh visualization with scale control."
            )
    
    # ========== Combination Table Methods ==========
    
    def _setup_table_delegates(self):
        """
        Setup delegates for the combination table.
        
        - Column 0 (Name): Default delegate (editable text)
        - Column 1 (Type): ReadOnlyDelegate (non-editable, "Linear" only)
        - Columns 2+: NumericDelegate (only allow float/integer values)
        """
        # Make Type column (column 1) read-only
        self._readonly_delegate = ReadOnlyDelegate(self.combo_table)
        self.combo_table.setItemDelegateForColumn(1, self._readonly_delegate)
        
        # Apply numeric delegate to coefficient columns (will be updated when columns change)
        self._numeric_delegate = NumericDelegate(self.combo_table)
        self._apply_numeric_delegate_to_columns()
        
        # Connect to column count changes
        self.combo_table.model().columnsInserted.connect(self._apply_numeric_delegate_to_columns)
    
    def _apply_numeric_delegate_to_columns(self):
        """Apply numeric delegate to all coefficient columns (column 2 onwards)."""
        for col in range(2, self.combo_table.columnCount()):
            self.combo_table.setItemDelegateForColumn(col, self._numeric_delegate)

    def _update_coefficient_cell_highlight(self, item: QTableWidgetItem):
        """
        Update the background color of a coefficient cell based on its value.
        
        Non-zero values get a light green background to visually distinguish them
        from zero/empty values.
        
        Args:
            item: The QTableWidgetItem to update.
        """
        if item is None:
            return
        
        try:
            value = float(item.text())
            is_nonzero = value != 0.0
        except (ValueError, TypeError):
            is_nonzero = False
        
        if is_nonzero:
            item.setBackground(QBrush(QColor(NONZERO_COEFFICIENT_BG_COLOR)))
        else:
            # Reset to default (no background)
            item.setBackground(QBrush())

    def _on_coefficient_cell_changed(self, row: int, column: int):
        """
        Handle cell changes in the combination table to update highlighting.
        
        This is called whenever a user manually edits a cell in the table.
        Only coefficient columns (column 2 onwards) are highlighted.
        
        Args:
            row: The row index of the changed cell.
            column: The column index of the changed cell.
        """
        # Only apply highlighting to coefficient columns (column 2 onwards)
        if column < 2:
            return
        
        item = self.combo_table.item(row, column)
        self._update_coefficient_cell_highlight(item)

    def _add_table_row(self):
        """Add a new row to the combination table."""
        row_count = self.combo_table.rowCount()
        self.combo_table.insertRow(row_count)
        
        # Set default values
        self.combo_table.setItem(row_count, 0, QTableWidgetItem(f"Combination {row_count + 1}"))
        
        # Type column is read-only (only "Linear" supported)
        type_item = QTableWidgetItem("Linear")
        type_item.setFlags(type_item.flags() & ~Qt.ItemIsEditable)  # Make non-editable
        type_item.setToolTip("Only 'Linear' combination type is currently supported")
        self.combo_table.setItem(row_count, 1, type_item)
        
        for col in range(2, self.combo_table.columnCount()):
            self.combo_table.setItem(row_count, col, QTableWidgetItem("0.0"))
    
    def _delete_table_row(self):
        """Delete the selected row from the combination table."""
        selected_rows = self.combo_table.selectionModel().selectedRows()
        if selected_rows:
            self.combo_table.removeRow(selected_rows[0].row())
        elif self.combo_table.rowCount() > 1:
            self.combo_table.removeRow(self.combo_table.rowCount() - 1)
    
    def get_combination_table_data(self) -> Optional[CombinationTableData]:
        """
        Extract combination table data from the UI table widget.
        
        Returns:
            CombinationTableData or None if invalid.
        """
        if not self.analysis1_data or not self.analysis2_data:
            return None
        
        row_count = self.combo_table.rowCount()
        if row_count == 0:
            return None
        
        names = []
        types = []
        a1_coeffs = []
        a2_coeffs = []
        
        n_a1 = len(self.analysis1_data.load_step_ids)
        n_a2 = len(self.analysis2_data.load_step_ids)
        
        for row in range(row_count):
            # Get name and type
            name_item = self.combo_table.item(row, 0)
            type_item = self.combo_table.item(row, 1)
            
            names.append(name_item.text() if name_item else f"Combination {row+1}")
            types.append(type_item.text() if type_item else "Linear")
            
            # Get A1 coefficients (columns 2 to 2+n_a1)
            a1_row = []
            for col in range(2, 2 + n_a1):
                item = self.combo_table.item(row, col)
                try:
                    a1_row.append(float(item.text()) if item else 0.0)
                except ValueError:
                    a1_row.append(0.0)
            a1_coeffs.append(a1_row)
            
            # Get A2 coefficients (columns 2+n_a1 onwards)
            a2_row = []
            for col in range(2 + n_a1, 2 + n_a1 + n_a2):
                item = self.combo_table.item(row, col)
                try:
                    a2_row.append(float(item.text()) if item else 0.0)
                except ValueError:
                    a2_row.append(0.0)
            a2_coeffs.append(a2_row)
        
        return CombinationTableData(
            combination_names=names,
            combination_types=types,
            analysis1_coeffs=np.array(a1_coeffs),
            analysis2_coeffs=np.array(a2_coeffs),
            analysis1_step_ids=list(self.analysis1_data.load_step_ids),
            analysis2_step_ids=list(self.analysis2_data.load_step_ids),
        )
    
    def set_combination_table_data(self, data: CombinationTableData):
        """
        Populate the UI table widget from CombinationTableData.
        
        Args:
            data: CombinationTableData to display.
        """
        self.combination_table = data
        
        # Update columns first
        self._update_combination_table_columns()
        
        # Set row count
        self.combo_table.setRowCount(data.num_combinations)
        
        # Populate data
        for row in range(data.num_combinations):
            self.combo_table.setItem(row, 0, QTableWidgetItem(data.combination_names[row]))
            
            # Type column is read-only
            type_item = QTableWidgetItem(data.combination_types[row])
            type_item.setFlags(type_item.flags() & ~Qt.ItemIsEditable)
            type_item.setToolTip("Only 'Linear' combination type is currently supported")
            self.combo_table.setItem(row, 1, type_item)
            
            # A1 coefficients
            for i, coeff in enumerate(data.analysis1_coeffs[row]):
                item = QTableWidgetItem(str(coeff))
                self._update_coefficient_cell_highlight(item)
                self.combo_table.setItem(row, 2 + i, item)
            
            # A2 coefficients
            offset = 2 + data.num_analysis1_steps
            for i, coeff in enumerate(data.analysis2_coeffs[row]):
                item = QTableWidgetItem(str(coeff))
                self._update_coefficient_cell_highlight(item)
                self.combo_table.setItem(row, offset + i, item)
    
    # ========== UI State Methods ==========
    
    def _update_solve_button_state(self):
        """Update solve button enabled state."""
        can_solve = (
            self.base_rst_loaded and 
            self.combine_rst_loaded and
            self.combo_table.rowCount() > 0 and
            any([
                self.von_mises_checkbox.isChecked(),
                self.max_principal_stress_checkbox.isChecked(),
                self.min_principal_stress_checkbox.isChecked(),
                self.nodal_forces_checkbox.isChecked(),
                self.deformation_checkbox.isChecked()
            ])
        )
        self.solve_button.setEnabled(can_solve)
    
    def _toggle_combination_history_mode(self, checked: bool):
        """Toggle combination history mode (single node analysis)."""
        self.single_node_group.setVisible(checked)
        self._update_solve_button_state()
    
    def _toggle_plasticity_options(self, checked: bool):
        """Toggle plasticity correction options visibility."""
        self.plasticity_options_group.setVisible(checked)
    
    def _toggle_nodal_forces_csys_combo(self, checked: bool):
        """Toggle nodal forces coordinate system combo visibility."""
        self.nodal_forces_csys_combo.setVisible(checked)
    
    def _toggle_deformation_csys_options(self, checked: bool):
        """Toggle deformation coordinate system options visibility."""
        self.deformation_csys_combo.setVisible(checked)
        if checked:
            # Show CS ID input only if Cylindrical is selected
            is_cylindrical = self.deformation_csys_combo.currentIndex() == 1
            self.deformation_cs_id_label.setVisible(is_cylindrical)
            self.deformation_cs_input.setVisible(is_cylindrical)
        else:
            # Hide all CS options when deformation is unchecked
            self.deformation_cs_id_label.setVisible(False)
            self.deformation_cs_input.setVisible(False)
            # Reset to Cartesian and clear CS ID
            self.deformation_csys_combo.setCurrentIndex(0)
            self.deformation_cs_input.clear()
    
    def _on_deformation_csys_changed(self, index: int):
        """Handle deformation coordinate system selection change."""
        # index 0 = Cartesian (Global), index 1 = Cylindrical
        is_cylindrical = (index == 1)
        self.deformation_cs_id_label.setVisible(is_cylindrical)
        self.deformation_cs_input.setVisible(is_cylindrical)
        if not is_cylindrical:
            # Clear CS ID when switching back to Cartesian
            self.deformation_cs_input.clear()
    
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
        if not checked:
            # Allow unchecking without side effects
            self._update_solve_button_state()
            return
        
        # List of mutually exclusive output type checkboxes
        output_checkboxes = [
            self.von_mises_checkbox,
            self.max_principal_stress_checkbox,
            self.min_principal_stress_checkbox,
            self.nodal_forces_checkbox,
        ]
        
        # Block signals to prevent recursive calls
        for checkbox in output_checkboxes:
            checkbox.blockSignals(True)
        
        try:
            # Uncheck all other output type checkboxes
            for checkbox in output_checkboxes:
                if checkbox is not source_checkbox:
                    checkbox.setChecked(False)
        finally:
            # Re-enable signals
            for checkbox in output_checkboxes:
                checkbox.blockSignals(False)
        
        self._update_solve_button_state()
    
    # ========== Analysis Methods ==========
    
    def _on_solve_clicked(self):
        """Handle solve button click."""
        # Validate inputs
        if not self._validate_inputs():
            return
        
        # Get combination table data
        self.combination_table = self.get_combination_table_data()
        if not self.combination_table:
            QMessageBox.warning(self, "Invalid Table", "Please enter at least one combination.")
            return
        
        # Build solver config
        config = self._build_solver_config()
        
        # Run analysis
        self.analysis_handler.solve(config)
    
    def _validate_inputs(self) -> bool:
        """Validate inputs before solving."""
        if not self.base_rst_loaded:
            QMessageBox.warning(self, "Missing Input", "Please load a Base Analysis RST file.")
            return False
        
        if not self.combine_rst_loaded:
            QMessageBox.warning(self, "Missing Input", "Please load an Analysis to Combine RST file.")
            return False
        
        if self.named_selection_combo.currentText().startswith("("):
            QMessageBox.warning(self, "Missing Input", "Please select a valid Named Selection.")
            return False
        
        if not any([
            self.von_mises_checkbox.isChecked(),
            self.max_principal_stress_checkbox.isChecked(),
            self.min_principal_stress_checkbox.isChecked(),
            self.nodal_forces_checkbox.isChecked(),
            self.deformation_checkbox.isChecked()
        ]):
            QMessageBox.warning(
                self,
                "No Output Selected",
                "Please select at least one output type (stress, deformation, or nodal forces)."
            )
            return False
        
        # Validate nodal forces availability if selected
        if self.nodal_forces_checkbox.isChecked():
            if not (self.analysis1_data.nodal_forces_available and 
                    self.analysis2_data.nodal_forces_available):
                QMessageBox.warning(
                    self, "Nodal Forces Not Available",
                    "Nodal forces are not available in one or both RST files.\n\n"
                    "To enable nodal forces output, ensure 'Write element nodal forces' "
                    "is enabled in ANSYS Output Controls before running the analysis."
                )
                return False
        
        # Validate stress outputs are not selected for beam element named selections
        stress_outputs_selected = any([
            self.von_mises_checkbox.isChecked(),
            self.max_principal_stress_checkbox.isChecked(),
            self.min_principal_stress_checkbox.isChecked()
        ])
        
        if stress_outputs_selected:
            ns_name = self.get_selected_named_selection()
            if ns_name and self.file_handler.base_reader is not None:
                try:
                    has_beams = self.file_handler.base_reader.check_named_selection_has_beam_elements(ns_name)
                    if has_beams:
                        QMessageBox.warning(
                            self, "Stress Output Not Supported",
                            f"The selected Named Selection '{ns_name}' contains beam elements.\n\n"
                            "Stress tensor output (Von Mises, Max Principal, Min Principal) is not "
                            "supported for beam elements.\n\n"
                            "Please either:\n"
                            "  • Select a Named Selection that contains only solid/shell elements, or\n"
                            "  • Use 'Nodal Forces' output instead (if available)"
                        )
                        return False
                except Exception as e:
                    # Log the error but don't block - let the engine handle it
                    self.console_textbox.append(
                        f"[Warning] Could not verify element types for named selection: {e}"
                    )
        
        if self.combination_history_checkbox.isChecked():
            node_text = self.node_line_edit.text().strip()
            
            # Check for empty input
            if not node_text:
                QMessageBox.warning(
                    self, "Missing Node ID",
                    "Please enter a Node ID for combination history mode."
                )
                return False
            
            # Check for valid positive integer
            try:
                node_id = int(node_text)
                if node_id <= 0:
                    raise ValueError("Node ID must be positive")
            except ValueError:
                QMessageBox.warning(
                    self, "Invalid Node ID",
                    f"'{node_text}' is not a valid Node ID.\n\n"
                    "Please enter a positive integer."
                )
                return False
            
            # Check if node exists in current scoping (early validation)
            if self.file_handler.base_reader is not None:
                ns_name = self.get_selected_named_selection()
                if ns_name:
                    try:
                        scoping = self.file_handler.base_reader.get_nodal_scoping_from_named_selection(ns_name)
                        scoping_ids = list(scoping.ids)
                        if node_id not in scoping_ids:
                            QMessageBox.warning(
                                self, "Node Not Found",
                                f"Node ID {node_id} was not found in Named Selection '{ns_name}'.\n\n"
                                f"The selected Named Selection contains {len(scoping_ids):,} nodes.\n"
                                f"Please enter a valid Node ID from this selection."
                            )
                            return False
                    except Exception as e:
                        # Log but don't block - let the engine handle validation
                        self.console_textbox.append(
                            f"[Warning] Could not validate node ID: {e}"
                        )
        
        return True
    
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
            config.selected_node_id = int(self.node_line_edit.text())
        
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
        text = self.named_selection_combo.currentText()
        if text.startswith("("):
            return None
        return text
    
    # ========== Result Handling ==========
    
    def on_analysis_complete(self, result: CombinationResult):
        """Handle stress analysis completion."""
        self.combination_result = result
        
        # Update export controls - DEPRECATED: Export functionality moved to Display tab
        # self.single_combo_dropdown.clear()
        # if result.all_combo_results is not None:
        #     for name in self.combination_table.combination_names:
        #         self.single_combo_dropdown.addItem(name)
        #     self.single_combo_dropdown.setEnabled(True)
        #     self.export_single_btn.setEnabled(True)
        #     self.export_group.setVisible(True)
        
        # Create a PyVista mesh for display tab
        mesh = self._create_mesh_from_result(result)
        
        # Determine scalar bar title based on result type (stresses are in MPa)
        scalar_bar_titles = {
            'von_mises': 'Von Mises Stress [MPa]',
            'max_principal': 'Max Principal Stress (S1) [MPa]',
            'min_principal': 'Min Principal Stress (S3) [MPa]'
        }
        scalar_bar_title = scalar_bar_titles.get(result.result_type, 'Stress [MPa]')
        
        # Get data range from max envelope
        data_min = float(np.min(result.max_over_combo)) if result.max_over_combo is not None else 0.0
        data_max = float(np.max(result.max_over_combo)) if result.max_over_combo is not None else 0.0
        
        # Emit signal for display tab with mesh
        self.combination_result_ready.emit(mesh, scalar_bar_title, data_min, data_max)
    
    def on_forces_analysis_complete(self, result: NodalForcesResult):
        """Handle nodal forces analysis completion."""
        self.nodal_forces_result = result
        
        # Update export controls for forces - DEPRECATED: Export functionality moved to Display tab
        # self.single_combo_dropdown.clear()
        # if result.all_combo_fx is not None:
        #     for name in self.combination_table.combination_names:
        #         self.single_combo_dropdown.addItem(name)
        #     self.single_combo_dropdown.setEnabled(True)
        #     self.export_single_btn.setEnabled(True)
        #     self.export_group.setVisible(True)
        
        # Create a PyVista mesh for display tab with force magnitude
        mesh = self._create_mesh_from_forces_result(result)
        
        # Determine scalar bar title
        scalar_bar_title = f'Force Magnitude [{result.force_unit}]'
        
        # Get data range from max envelope
        data_min = float(np.min(result.max_magnitude_over_combo)) if result.max_magnitude_over_combo is not None else 0.0
        data_max = float(np.max(result.max_magnitude_over_combo)) if result.max_magnitude_over_combo is not None else 0.0
        
        # Emit signal for display tab with mesh
        self.combination_result_ready.emit(mesh, scalar_bar_title, data_min, data_max)
    
    def on_deformation_analysis_complete(self, result: DeformationResult, is_standalone: bool = False):
        """
        Handle deformation analysis completion.
        
        Args:
            result: DeformationResult with displacement data.
            is_standalone: If True, deformation is the only output type - create and emit mesh.
        """
        self.deformation_result = result
        
        # Log summary
        self.console_textbox.append(
            f"\nDeformation analysis complete\n"
            f"  Nodes: {result.num_nodes}\n"
            f"  Combinations: {result.num_combinations}\n"
            f"  Displacement Unit: {result.displacement_unit}\n"
        )
        
        if result.max_magnitude_over_combo is not None:
            max_val = np.max(result.max_magnitude_over_combo)
            max_node_idx = np.argmax(result.max_magnitude_over_combo)
            max_node_id = result.node_ids[max_node_idx]
            
            self.console_textbox.append(
                f"  Maximum Displacement: {max_val:.6f} {result.displacement_unit} at Node {max_node_id}\n"
            )
        
        # If deformation is the only output, create and emit mesh for display
        if is_standalone:
            mesh = self._create_mesh_from_deformation_result(result)
            
            # Determine scalar bar title
            scalar_bar_title = f'Displacement Magnitude [{result.displacement_unit}]'
            
            # Get data range from max envelope
            data_min = float(np.min(result.max_magnitude_over_combo)) if result.max_magnitude_over_combo is not None else 0.0
            data_max = float(np.max(result.max_magnitude_over_combo)) if result.max_magnitude_over_combo is not None else 0.0
            
            # Emit signal for display tab with mesh
            self.combination_result_ready.emit(mesh, scalar_bar_title, data_min, data_max)
    
    def _create_mesh_from_forces_result(self, result: NodalForcesResult):
        """
        Create a PyVista mesh from NodalForcesResult for visualization.
        
        Args:
            result: NodalForcesResult with node coordinates and force values.
            
        Returns:
            PyVista PolyData mesh with force magnitude as scalars.
        """
        import pyvista as pv
        
        # Create point cloud mesh from node coordinates
        mesh = pv.PolyData(result.node_coords)
        
        # Add node IDs (must be 'NodeID' for hover functionality to work)
        mesh['NodeID'] = result.node_ids
        
        # Add max force magnitude as the primary scalar
        if result.max_magnitude_over_combo is not None:
            mesh['Max_Force_Magnitude'] = result.max_magnitude_over_combo
            mesh['Force_Magnitude'] = result.max_magnitude_over_combo  # Alias for component switching
            mesh.set_active_scalars('Max_Force_Magnitude')
        
        # Add min force magnitude
        if result.min_magnitude_over_combo is not None:
            mesh['Min_Force_Magnitude'] = result.min_magnitude_over_combo
        
        # Add combination indices
        if result.combo_of_max is not None:
            mesh['Combo_of_Max'] = result.combo_of_max
        if result.combo_of_min is not None:
            mesh['Combo_of_Min'] = result.combo_of_min
        
        # Add max values for each force component (for envelope view component selection)
        # Compute max absolute value over combinations for FX, FY, FZ
        if result.all_combo_fx is not None:
            # Take the value from the combination that had max magnitude (for consistency)
            num_nodes = result.num_nodes
            fx_at_max = np.zeros(num_nodes)
            fy_at_max = np.zeros(num_nodes)
            fz_at_max = np.zeros(num_nodes)
            
            if result.combo_of_max is not None:
                for node_idx in range(num_nodes):
                    combo_idx = int(result.combo_of_max[node_idx])
                    fx_at_max[node_idx] = result.all_combo_fx[combo_idx, node_idx]
                    fy_at_max[node_idx] = result.all_combo_fy[combo_idx, node_idx]
                    fz_at_max[node_idx] = result.all_combo_fz[combo_idx, node_idx]
            
            mesh['FX'] = fx_at_max
            mesh['FY'] = fy_at_max
            mesh['FZ'] = fz_at_max
            
            # Add shear force if beam nodes present
            if result.has_beam_nodes:
                mesh['Shear_Force'] = np.sqrt(fy_at_max**2 + fz_at_max**2)
        
        return mesh
    
    def _create_mesh_from_deformation_result(self, result: DeformationResult):
        """
        Create a PyVista mesh from DeformationResult for visualization.
        
        Args:
            result: DeformationResult with node coordinates and displacement values.
            
        Returns:
            PyVista PolyData mesh with displacement magnitude as scalars.
        """
        import pyvista as pv
        
        # Create point cloud mesh from node coordinates
        mesh = pv.PolyData(result.node_coords)
        
        # Add node IDs (must be 'NodeID' for hover functionality to work)
        mesh['NodeID'] = result.node_ids
        
        # Add max displacement magnitude as the primary scalar (for envelope view)
        if result.max_magnitude_over_combo is not None:
            mesh['Max_U_mag'] = result.max_magnitude_over_combo
            mesh['U_mag'] = result.max_magnitude_over_combo  # Alias for component switching
            mesh.set_active_scalars('Max_U_mag')
        
        # Add min displacement magnitude
        if result.min_magnitude_over_combo is not None:
            mesh['Min_U_mag'] = result.min_magnitude_over_combo
        
        # Add combination indices
        if result.combo_of_max is not None:
            mesh['Combo_of_Max'] = result.combo_of_max
        if result.combo_of_min is not None:
            mesh['Combo_of_Min'] = result.combo_of_min
        
        # Add displacement components at the combination of max magnitude (for envelope view component selection)
        if result.all_combo_ux is not None:
            num_nodes = result.num_nodes
            ux_at_max = np.zeros(num_nodes)
            uy_at_max = np.zeros(num_nodes)
            uz_at_max = np.zeros(num_nodes)
            
            if result.combo_of_max is not None:
                for node_idx in range(num_nodes):
                    combo_idx = int(result.combo_of_max[node_idx])
                    ux_at_max[node_idx] = result.all_combo_ux[combo_idx, node_idx]
                    uy_at_max[node_idx] = result.all_combo_uy[combo_idx, node_idx]
                    uz_at_max[node_idx] = result.all_combo_uz[combo_idx, node_idx]
            
            mesh['UX'] = ux_at_max
            mesh['UY'] = uy_at_max
            mesh['UZ'] = uz_at_max
        
        return mesh
    
    def _create_mesh_from_result(self, result: CombinationResult):
        """
        Create a PyVista mesh from CombinationResult for visualization.
        
        Args:
            result: CombinationResult with node coordinates and stress values.
            
        Returns:
            PyVista PolyData mesh with stress values as scalars.
        """
        import pyvista as pv
        
        # Create point cloud mesh from node coordinates
        mesh = pv.PolyData(result.node_coords)
        
        # Add node IDs (must be 'NodeID' for hover functionality to work)
        mesh['NodeID'] = result.node_ids
        
        # Add max stress values as the primary scalar
        if result.max_over_combo is not None:
            mesh['Max_Stress'] = result.max_over_combo
            mesh.set_active_scalars('Max_Stress')
        
        # Add min stress values
        if result.min_over_combo is not None:
            mesh['Min_Stress'] = result.min_over_combo
        
        # Add combination indices
        if result.combo_of_max is not None:
            mesh['Combo_of_Max'] = result.combo_of_max
        if result.combo_of_min is not None:
            mesh['Combo_of_Min'] = result.combo_of_min
        
        # Store result type as field data (metadata) for display tab
        mesh.field_data['result_type'] = [result.result_type]
        
        return mesh
    
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
        It validates inputs and triggers the combination history solve.
        
        Args:
            node_id: The node ID to compute combination history for.
        """
        # Validate that data is loaded
        if not self.base_rst_loaded or not self.combine_rst_loaded:
            QMessageBox.warning(
                self, "Missing Data",
                "Please load both RST files before plotting combination history."
            )
            return
        
        # Validate that at least one output is selected (stress, deformation, or nodal forces)
        has_stress_output = any([
            self.von_mises_checkbox.isChecked(),
            self.max_principal_stress_checkbox.isChecked(),
            self.min_principal_stress_checkbox.isChecked()
        ])
        has_deformation_output = self.deformation_checkbox.isChecked()
        has_nodal_forces_output = self.nodal_forces_checkbox.isChecked()
        has_output = has_stress_output or has_deformation_output or has_nodal_forces_output
        
        if not has_output:
            QMessageBox.warning(
                self,
                "No Output Selected",
                "Please select at least one output type (stress, deformation, or nodal forces)."
            )
            return
        
        # Validate combination table
        if self.combo_table.rowCount() == 0:
            QMessageBox.warning(
                self, "Missing Combinations",
                "Please define at least one combination in the table."
            )
            return
        
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
    
    def _export_single_combination(self):
        """Export a single combination result to CSV (stress or forces)."""
        # Check if we have either stress or force results
        if self.combination_result is None and self.nodal_forces_result is None:
            QMessageBox.warning(self, "No Results", "Run analysis first before exporting.")
            return
        
        combo_idx = self.single_combo_dropdown.currentIndex()
        combo_name = self.single_combo_dropdown.currentText()
        
        # Determine if we're exporting stress or force results
        if self.combination_result is not None:
            # Stress results
            default_filename = f"{combo_name.replace(' ', '_')}_stress.csv"
            filename, _ = QFileDialog.getSaveFileName(
                self, f"Export Stress: {combo_name}",
                default_filename,
                "CSV Files (*.csv)"
            )
            
            if filename:
                self.file_handler.export_single_combination_result(
                    self.combination_result, combo_idx, filename
                )
                self.console_textbox.append(f"Exported stress for {combo_name} to {filename}")
        
        elif self.nodal_forces_result is not None:
            # Force results
            from file_io.exporters import export_nodal_forces_single_combination
            
            result = self.nodal_forces_result
            default_filename = f"{combo_name.replace(' ', '_')}_forces.csv"
            filename, _ = QFileDialog.getSaveFileName(
                self, f"Export Forces: {combo_name}",
                default_filename,
                "CSV Files (*.csv)"
            )
            
            if filename:
                # Get force components for this combination
                fx = result.all_combo_fx[combo_idx, :]
                fy = result.all_combo_fy[combo_idx, :]
                fz = result.all_combo_fz[combo_idx, :]
                
                export_nodal_forces_single_combination(
                    filename=filename,
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
                self.console_textbox.append(f"Exported forces for {combo_name} to {filename}")
    
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
