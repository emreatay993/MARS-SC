"""
Builds the Solver tab UI: RST inputs, combination table, output options, plasticity, console.
"""

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator, QFont, QPalette, QColor, QIntValidator
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QGridLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QProgressBar, QPushButton, QSizePolicy,
    QTabWidget, QVBoxLayout, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QSpinBox, QWidget
)


class ClearableSelectionTableWidget(QTableWidget):
    """Table that clears selection when you click empty space (no cell under cursor)."""

    def mousePressEvent(self, event):
        item = self.itemAt(event.pos())
        if item is None:
            self.clearSelection()
            self.setCurrentItem(None)
        super().mousePressEvent(event)

from utils.constants import (
    WINDOW_BACKGROUND_COLOR
)
from utils.tooltips import (
    TOOLTIP_VON_MISES, TOOLTIP_MAX_PRINCIPAL, TOOLTIP_MIN_PRINCIPAL,
    TOOLTIP_NODAL_FORCES, TOOLTIP_NODAL_FORCES_CSYS, TOOLTIP_COMBINATION_HISTORY,
    TOOLTIP_PLASTICITY_CORRECTION, TOOLTIP_OUTPUT_OPTIONS, TOOLTIP_PLASTICITY_METHOD, TOOLTIP_MAX_ITERATIONS,
    TOOLTIP_TOLERANCE, TOOLTIP_EXTRAPOLATION, TOOLTIP_IMPORT_CSV, TOOLTIP_EXPORT_CSV,
    TOOLTIP_MATERIAL_PROFILE, TOOLTIP_TEMPERATURE_FIELD_FILE, TOOLTIP_ITERATION_CONTROLS,
    TOOLTIP_DEFORMATION, TOOLTIP_DEFORMATION_CYLINDRICAL_CS,
    TOOLTIP_NAMED_SELECTION, TOOLTIP_NAMED_SELECTION_SOURCE, TOOLTIP_NAMED_SELECTION_REFRESH,
    TOOLTIP_BASE_RST, TOOLTIP_COMBINE_RST, TOOLTIP_COMBINATION_TABLE,
    TOOLTIP_ADD_ROW, TOOLTIP_DELETE_ROW
)
from ui.styles.style_constants import (
    BUTTON_STYLE, GROUP_BOX_STYLE, TAB_STYLE, READONLY_INPUT_STYLE,
    CHECKBOX_STYLE, CONSOLE_STYLE, PROGRESS_BAR_STYLE
)
from ui.widgets.plotting import MatplotlibWidget, PlotlyMaxWidget
from ui.widgets.collapsible_group import CollapsibleGroupBoxStyled


class SolverTabUIBuilder:
    """Assembles the Solver tab: RST inputs, named selection, combo table, outputs, plasticity."""

    def __init__(self):
        """Initialize the builder."""
        self.components = {}
    
    def build_file_input_section(self):
        """
        Build the RST file input section with buttons and path displays.
        
        Returns:
            CollapsibleGroupBoxStyled: Collapsible group box containing RST file input controls.
        """
        # Base Analysis RST (Analysis 1)
        base_rst_button = QPushButton('Select Base Analysis RST')
        base_rst_button.setStyleSheet(BUTTON_STYLE)
        base_rst_button.setFont(QFont('Arial', 8))
        base_rst_button.setToolTip(TOOLTIP_BASE_RST)
        base_rst_path = QLineEdit()
        base_rst_path.setReadOnly(True)
        base_rst_path.setStyleSheet(READONLY_INPUT_STYLE)
        base_rst_path.setPlaceholderText("No file selected")
        base_rst_path.setToolTip(TOOLTIP_BASE_RST)
        
        base_info_label = QLabel("Load steps: -")
        base_info_label.setStyleSheet("color: #666; font-size: 10px;")
        base_info_label.setToolTip(TOOLTIP_BASE_RST)

        # Analysis to Combine RST (Analysis 2)
        combine_rst_button = QPushButton('Select Analysis to Combine RST')
        combine_rst_button.setStyleSheet(BUTTON_STYLE)
        combine_rst_button.setFont(QFont('Arial', 8))
        combine_rst_button.setToolTip(TOOLTIP_COMBINE_RST)
        combine_rst_path = QLineEdit()
        combine_rst_path.setReadOnly(True)
        combine_rst_path.setStyleSheet(READONLY_INPUT_STYLE)
        combine_rst_path.setPlaceholderText("No file selected")
        combine_rst_path.setToolTip(TOOLTIP_COMBINE_RST)
        
        combine_info_label = QLabel("Load steps: -")
        combine_info_label.setStyleSheet("color: #666; font-size: 10px;")
        combine_info_label.setToolTip(TOOLTIP_COMBINE_RST)
        
        # Named Selection source and dropdown
        named_selection_source_label = QLabel("Named Selection Source:")
        named_selection_source_combo = QComboBox()
        named_selection_source_combo.addItem("Common (A1 & A2)", "common")
        named_selection_source_combo.addItem("Analysis 1 (Base)", "analysis1")
        named_selection_source_combo.addItem("Analysis 2 (Combine)", "analysis2")
        named_selection_source_combo.setMinimumWidth(170)
        named_selection_source_combo.setEnabled(False)
        named_selection_source_combo.setToolTip(TOOLTIP_NAMED_SELECTION_SOURCE)
        named_selection_source_label.setToolTip(TOOLTIP_NAMED_SELECTION_SOURCE)
        named_selection_source_combo.setItemData(0, "Show only names available in both analyses.", Qt.ToolTipRole)
        named_selection_source_combo.setItemData(1, "Show names from Analysis 1 only.", Qt.ToolTipRole)
        named_selection_source_combo.setItemData(2, "Show names from Analysis 2 only.", Qt.ToolTipRole)

        named_selection_label = QLabel("Named Selection:")
        named_selection_combo = QComboBox()
        named_selection_combo.setMinimumWidth(420)
        named_selection_combo.setMinimumContentsLength(48)
        named_selection_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        named_selection_combo.view().setMinimumWidth(520)
        named_selection_combo.setEnabled(False)
        named_selection_combo.addItem("(Load RST files first)")
        named_selection_combo.setToolTip(TOOLTIP_NAMED_SELECTION)
        named_selection_label.setToolTip(TOOLTIP_NAMED_SELECTION)
        
        refresh_ns_button = QPushButton("Refresh")
        refresh_ns_button.setStyleSheet(BUTTON_STYLE)
        refresh_ns_button.setFont(QFont('Arial', 8))
        refresh_ns_button.setMaximumWidth(80)
        refresh_ns_button.setEnabled(False)
        refresh_ns_button.setToolTip(TOOLTIP_NAMED_SELECTION_REFRESH)
        
        # Skip substeps checkbox
        skip_substeps_checkbox = QCheckBox("Skip Substeps (use last substep of each load step only)")
        skip_substeps_checkbox.setStyleSheet(CHECKBOX_STYLE)
        skip_substeps_checkbox.setToolTip(
            "When enabled, only the final substep of each load step will be loaded.\n"
            "This reduces the number of coefficient columns when RST files contain\n"
            "many substeps (e.g., from nonlinear analysis with auto time stepping).\n\n"
            "Note: This option must be set BEFORE loading RST files."
        )
        
        # Layout
        file_layout = QGridLayout()
        file_layout.addWidget(base_rst_button, 0, 0)
        file_layout.addWidget(base_rst_path, 0, 1)
        file_layout.addWidget(base_info_label, 0, 2)
        file_layout.addWidget(combine_rst_button, 1, 0)
        file_layout.addWidget(combine_rst_path, 1, 1)
        file_layout.addWidget(combine_info_label, 1, 2)
        
        # Named selection row
        ns_row = QHBoxLayout()
        ns_row.addWidget(named_selection_source_label)
        ns_row.addWidget(named_selection_source_combo)
        ns_row.addWidget(named_selection_label)
        ns_row.addWidget(named_selection_combo)
        ns_row.addWidget(refresh_ns_button)
        ns_row.addStretch()
        
        file_layout.addLayout(ns_row, 2, 0, 1, 3)
        
        # Skip substeps row
        file_layout.addWidget(skip_substeps_checkbox, 3, 0, 1, 3)
        
        # Use collapsible group box instead of QGroupBox
        file_group = CollapsibleGroupBoxStyled("Input Files", initially_expanded=True)
        file_group.setContentLayout(file_layout)
        file_group.setContentMargins(8, 4, 8, 4)
        
        # Store components for external access
        self.components['base_rst_button'] = base_rst_button
        self.components['base_rst_path'] = base_rst_path
        self.components['base_info_label'] = base_info_label
        self.components['combine_rst_button'] = combine_rst_button
        self.components['combine_rst_path'] = combine_rst_path
        self.components['combine_info_label'] = combine_info_label
        self.components['named_selection_source_combo'] = named_selection_source_combo
        self.components['named_selection_combo'] = named_selection_combo
        self.components['refresh_ns_button'] = refresh_ns_button
        self.components['skip_substeps_checkbox'] = skip_substeps_checkbox
        self.components['file_input_group'] = file_group
        
        return file_group
    
    def build_combination_table_section(self):
        """
        Build the combination coefficients table section.
        
        Returns:
            CollapsibleGroupBoxStyled: Collapsible group box containing the combination table.
        """
        # Table widget - use custom class that clears selection on empty space click
        combo_table = ClearableSelectionTableWidget()
        combo_table.setMinimumHeight(120)
        combo_table.setAlternatingRowColors(True)
        combo_table.setToolTip(TOOLTIP_COMBINATION_TABLE)
        combo_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        combo_table.setSelectionMode(QAbstractItemView.SingleSelection)
        combo_table.verticalHeader().setVisible(False)
        
        # Enable horizontal scrolling for tables with many coefficient columns
        # ResizeToContents auto-sizes columns based on header/content width
        # stretchLastSection=False ensures scrollbar appears when columns overflow
        combo_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        combo_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        combo_table.horizontalHeader().setStretchLastSection(False)
        combo_table.horizontalHeader().setMinimumSectionSize(50)
        
        # Initialize with placeholder columns
        combo_table.setColumnCount(3)
        combo_table.setHorizontalHeaderLabels(["Combination Name", "Type", "(Load RST files)"])
        combo_table.setRowCount(1)
        combo_table.setItem(0, 0, QTableWidgetItem("Combination 1"))
        
        # Type column is read-only (only "Linear" supported)
        type_item = QTableWidgetItem("Linear")
        type_item.setFlags(type_item.flags() & ~Qt.ItemIsEditable)
        type_item.setToolTip("Only 'Linear' combination type is currently supported")
        combo_table.setItem(0, 1, type_item)
        
        combo_table.setItem(0, 2, QTableWidgetItem("0.0"))
        
        # Table control buttons
        import_csv_btn = QPushButton("Import CSV")
        import_csv_btn.setStyleSheet(BUTTON_STYLE)
        import_csv_btn.setFont(QFont('Arial', 8))
        import_csv_btn.setToolTip(TOOLTIP_IMPORT_CSV)
        
        export_csv_btn = QPushButton("Export CSV")
        export_csv_btn.setStyleSheet(BUTTON_STYLE)
        export_csv_btn.setFont(QFont('Arial', 8))
        export_csv_btn.setToolTip(TOOLTIP_EXPORT_CSV)
        
        add_row_btn = QPushButton("Add Row")
        add_row_btn.setStyleSheet(BUTTON_STYLE)
        add_row_btn.setFont(QFont('Arial', 8))
        add_row_btn.setToolTip(TOOLTIP_ADD_ROW)
        
        delete_row_btn = QPushButton("Delete Row")
        delete_row_btn.setStyleSheet(BUTTON_STYLE)
        delete_row_btn.setFont(QFont('Arial', 8))
        delete_row_btn.setToolTip(TOOLTIP_DELETE_ROW)
        
        # Button layout
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(import_csv_btn)
        btn_layout.addWidget(export_csv_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(add_row_btn)
        btn_layout.addWidget(delete_row_btn)
        
        # Main layout
        table_layout = QVBoxLayout()
        table_layout.addWidget(combo_table)
        table_layout.addLayout(btn_layout)
        
        # Use collapsible group box instead of QGroupBox
        table_group = CollapsibleGroupBoxStyled("Combination Coefficients Table", initially_expanded=True)
        table_group.setContentLayout(table_layout)
        table_group.setContentMargins(8, 4, 8, 4)
        table_group.setToolTip(TOOLTIP_COMBINATION_TABLE)
        
        # Store components
        self.components['combo_table'] = combo_table
        self.components['import_csv_btn'] = import_csv_btn
        self.components['export_csv_btn'] = export_csv_btn
        self.components['add_row_btn'] = add_row_btn
        self.components['delete_row_btn'] = delete_row_btn
        self.components['combo_table_group'] = table_group
        
        return table_group
    
    def build_output_selection_section(self):
        """
        Build the output selection section with checkboxes.
        
        Returns:
            CollapsibleGroupBoxStyled: Collapsible group box containing output selection checkboxes.
        """
        # Create checkboxes for combination analysis outputs
        combination_history_checkbox = QCheckBox('Enable Combination History Mode (Single Node)')
        combination_history_checkbox.setStyleSheet(CHECKBOX_STYLE)
        combination_history_checkbox.setToolTip(TOOLTIP_COMBINATION_HISTORY)
        
        von_mises_checkbox = QCheckBox('Von-Mises Stress')
        von_mises_checkbox.setStyleSheet(CHECKBOX_STYLE)
        von_mises_checkbox.setChecked(True)  # Default selection
        von_mises_checkbox.setToolTip(TOOLTIP_VON_MISES)
        
        max_principal_stress_checkbox = QCheckBox('Max Principal Stress (S1)')
        max_principal_stress_checkbox.setStyleSheet(CHECKBOX_STYLE)
        max_principal_stress_checkbox.setToolTip(TOOLTIP_MAX_PRINCIPAL)
        
        min_principal_stress_checkbox = QCheckBox('Min Principal Stress (S3)')
        min_principal_stress_checkbox.setStyleSheet(CHECKBOX_STYLE)
        min_principal_stress_checkbox.setToolTip(TOOLTIP_MIN_PRINCIPAL)
        
        nodal_forces_checkbox = QCheckBox('Nodal Forces')
        nodal_forces_checkbox.setStyleSheet(CHECKBOX_STYLE)
        nodal_forces_checkbox.setToolTip(TOOLTIP_NODAL_FORCES)
        
        # Coordinate system selection for nodal forces (hidden by default)
        nodal_forces_csys_combo = QComboBox()
        nodal_forces_csys_combo.addItems(["Global", "Local (Element)"])
        nodal_forces_csys_combo.setToolTip(TOOLTIP_NODAL_FORCES_CSYS)
        nodal_forces_csys_combo.setMaximumWidth(120)
        nodal_forces_csys_combo.setVisible(False)  # Hidden until nodal forces is checked
        # Disable "Global" for now (known issues) and default to Local
        model = nodal_forces_csys_combo.model()
        if model is not None and model.item(0) is not None:
            model.item(0).setEnabled(False)
            model.item(0).setToolTip("Global option temporarily disabled")
        nodal_forces_csys_combo.setCurrentIndex(1)
        
        # Deformation (displacement) checkbox - can be selected with stress outputs
        deformation_checkbox = QCheckBox('Deformation (Displacement)')
        deformation_checkbox.setStyleSheet(CHECKBOX_STYLE)
        deformation_checkbox.setToolTip(TOOLTIP_DEFORMATION)
        
        # Coordinate system dropdown for deformation output (hidden by default)
        deformation_csys_combo = QComboBox()
        deformation_csys_combo.addItems(["Cartesian (Global)", "Cylindrical"])
        deformation_csys_combo.setToolTip(TOOLTIP_DEFORMATION_CYLINDRICAL_CS)
        deformation_csys_combo.setMaximumWidth(140)
        deformation_csys_combo.setVisible(False)
        
        # CS ID input for cylindrical option (hidden by default)
        deformation_cs_id_label = QLabel("CS ID:")
        deformation_cs_id_label.setToolTip("Enter the cylindrical coordinate system ID from your ANSYS model")
        deformation_cs_id_label.setVisible(False)
        
        deformation_cs_input = QLineEdit()
        deformation_cs_input.setPlaceholderText("e.g. 5")
        deformation_cs_input.setMaximumWidth(60)
        deformation_cs_input.setToolTip("Enter the cylindrical coordinate system ID from your ANSYS model")
        deformation_cs_input.setVisible(False)
        # Only allow integer input
        deformation_cs_input.setValidator(QIntValidator(0, 999999, deformation_cs_input))
        
        plasticity_correction_checkbox = QCheckBox('Enable Plasticity Correction')
        plasticity_correction_checkbox.setStyleSheet(CHECKBOX_STYLE)
        plasticity_correction_checkbox.setToolTip(TOOLTIP_PLASTICITY_CORRECTION)
        
        # Layout
        output_layout = QVBoxLayout()
        output_layout.addWidget(von_mises_checkbox)
        output_layout.addWidget(max_principal_stress_checkbox)
        output_layout.addWidget(min_principal_stress_checkbox)
        
        # Nodal forces row with checkbox and coordinate system dropdown
        nodal_forces_row = QHBoxLayout()
        nodal_forces_row.addWidget(nodal_forces_checkbox)
        nodal_forces_row.addWidget(nodal_forces_csys_combo)
        nodal_forces_row.addStretch()
        output_layout.addLayout(nodal_forces_row)
        
        # Deformation row with checkbox and coordinate system options
        deformation_row = QHBoxLayout()
        deformation_row.addWidget(deformation_checkbox)
        deformation_row.addWidget(deformation_csys_combo)
        deformation_row.addWidget(deformation_cs_id_label)
        deformation_row.addWidget(deformation_cs_input)
        deformation_row.addStretch()
        output_layout.addLayout(deformation_row)
        
        output_layout.addWidget(combination_history_checkbox)
        output_layout.addWidget(plasticity_correction_checkbox)
        
        # Use collapsible group box instead of QGroupBox
        output_group = CollapsibleGroupBoxStyled("Output Options", initially_expanded=True)
        output_group.setContentLayout(output_layout)
        output_group.setContentMargins(8, 4, 8, 4)
        output_group.setToolTip(TOOLTIP_OUTPUT_OPTIONS)
        
        # Store components
        self.components['combination_history_checkbox'] = combination_history_checkbox
        self.components['von_mises_checkbox'] = von_mises_checkbox
        self.components['max_principal_stress_checkbox'] = max_principal_stress_checkbox
        self.components['min_principal_stress_checkbox'] = min_principal_stress_checkbox
        self.components['nodal_forces_checkbox'] = nodal_forces_checkbox
        self.components['nodal_forces_csys_combo'] = nodal_forces_csys_combo
        self.components['deformation_checkbox'] = deformation_checkbox
        self.components['deformation_csys_combo'] = deformation_csys_combo
        self.components['deformation_cs_id_label'] = deformation_cs_id_label
        self.components['deformation_cs_input'] = deformation_cs_input
        self.components['plasticity_correction_checkbox'] = plasticity_correction_checkbox
        self.components['output_options_group'] = output_group
        
        # Initially disable checkboxes until files are loaded
        for key in ['von_mises_checkbox', 'max_principal_stress_checkbox',
                    'min_principal_stress_checkbox', 'nodal_forces_checkbox',
                    'deformation_checkbox', 'plasticity_correction_checkbox', 
                    'combination_history_checkbox']:
            self.components[key].setEnabled(False)
        
        return output_group
    
    def build_single_node_section(self):
        """
        Build the single node selection section for combination history mode.
        
        Returns:
            QGroupBox: Group box containing node ID input.
        """
        single_node_label = QLabel("Select a node:")
        single_node_label.setFont(QFont('Arial', 8))
        
        node_line_edit = QLineEdit()
        node_line_edit.setPlaceholderText("Enter Node ID")
        node_line_edit.setStyleSheet(BUTTON_STYLE)
        node_line_edit.setMaximumWidth(150)
        node_line_edit.setMinimumWidth(100)

        single_node_layout = QHBoxLayout()
        single_node_layout.addWidget(single_node_label)
        single_node_layout.addWidget(node_line_edit)

        single_node_group = QGroupBox("Scoping")
        single_node_group.setStyleSheet(GROUP_BOX_STYLE)
        single_node_group.setLayout(single_node_layout)
        single_node_group.setVisible(False)
        single_node_group.setMaximumWidth(250)
        
        # Store components
        self.components['single_node_label'] = single_node_label
        self.components['node_line_edit'] = node_line_edit
        self.components['single_node_group'] = single_node_group
        
        return single_node_group

    def build_plasticity_options_section(self):
        """
        Build the plasticity correction options section.

        Returns:
            QGroupBox: Group box for plasticity options (initially hidden).
        """
        plasticity_options_group = QGroupBox("Plasticity Correction Options")
        plasticity_options_group.setStyleSheet(GROUP_BOX_STYLE)

        layout = QVBoxLayout()

        material_profile_button = QPushButton('Enter Material Profile')
        material_profile_button.setStyleSheet(BUTTON_STYLE)
        material_profile_button.setFont(QFont('Arial', 8))
        material_profile_button.setToolTip(TOOLTIP_MATERIAL_PROFILE)

        temperature_field_button = QPushButton('Read Temperature Field File (.txt)')
        temperature_field_button.setStyleSheet(BUTTON_STYLE)
        temperature_field_button.setFont(QFont('Arial', 8))
        temperature_field_button.setToolTip(TOOLTIP_TEMPERATURE_FIELD_FILE)
        temperature_field_path = QLineEdit()
        temperature_field_path.setReadOnly(True)
        temperature_field_path.setStyleSheet(READONLY_INPUT_STYLE)
        temperature_field_path.setToolTip(TOOLTIP_TEMPERATURE_FIELD_FILE)

        file_row = QHBoxLayout()
        file_row.addWidget(temperature_field_button)
        file_row.addWidget(temperature_field_path)
        file_row.setStretch(0, 0)
        file_row.setStretch(1, 1)

        method_row = QHBoxLayout()
        method_label = QLabel("Select Method:")
        method_label.setToolTip(TOOLTIP_PLASTICITY_METHOD)
        method_combo = QComboBox()
        method_combo.addItems(["Neuber", "Glinka", "Incremental Buczynski-Glinka (IBG)"])
        method_combo.setToolTip(TOOLTIP_PLASTICITY_METHOD)
        
        # Disable IBG option
        model = method_combo.model()
        ibg_item = model.item(2)
        ibg_item.setEnabled(False)
        
        method_row.addWidget(method_label)
        method_row.addWidget(method_combo)
        method_row.addStretch()

        iteration_row = QHBoxLayout()
        iteration_label = QLabel("Iteration Controls:")
        iteration_label.setMinimumWidth(120)
        iteration_label.setToolTip(TOOLTIP_ITERATION_CONTROLS)
        max_iter_label = QLabel("Max Iterations")
        max_iter_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        max_iter_label.setToolTip(TOOLTIP_MAX_ITERATIONS)
        max_iter_input = QLineEdit("60")
        max_iter_input.setMaximumWidth(70)
        max_iter_validator = QIntValidator(1, 10000, max_iter_input)
        max_iter_input.setValidator(max_iter_validator)
        max_iter_input.setToolTip(TOOLTIP_MAX_ITERATIONS)
        tolerance_label = QLabel("Tolerance")
        tolerance_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        tolerance_label.setToolTip(TOOLTIP_TOLERANCE)
        tolerance_input = QLineEdit("1e-10")
        tolerance_input.setMaximumWidth(100)
        tolerance_validator = QDoubleValidator(0.0, 1.0, 12, tolerance_input)
        tolerance_validator.setNotation(QDoubleValidator.ScientificNotation)
        tolerance_input.setValidator(tolerance_validator)
        tolerance_input.setToolTip(TOOLTIP_TOLERANCE)
        iteration_row.addWidget(iteration_label)
        iteration_row.addWidget(max_iter_label)
        iteration_row.addWidget(max_iter_input)
        iteration_row.addWidget(tolerance_label)
        iteration_row.addWidget(tolerance_input)
        iteration_row.addStretch()

        warning_label = QLabel("Warning: Relaxed iteration settings may impact accuracy.")
        warning_label.setStyleSheet("color: #b36b00; font-style: italic;")
        warning_label.setToolTip(TOOLTIP_ITERATION_CONTROLS)
        warning_label.setVisible(False)

        # Extrapolation mode
        extrap_row = QHBoxLayout()
        extrap_label = QLabel("Extrapolation:")
        extrap_label.setToolTip(TOOLTIP_EXTRAPOLATION)
        extrap_combo = QComboBox()
        extrap_combo.addItems(["Linear", "Plateau"])
        extrap_combo.setToolTip(TOOLTIP_EXTRAPOLATION)
        extrap_row.addWidget(extrap_label)
        extrap_row.addWidget(extrap_combo)
        extrap_row.addStretch()

        layout.addWidget(material_profile_button)
        layout.addLayout(file_row)
        layout.addLayout(method_row)
        layout.addLayout(iteration_row)
        layout.addWidget(warning_label)
        layout.addLayout(extrap_row)
        plasticity_options_group.setLayout(layout)
        plasticity_options_group.setVisible(False)

        self.components['plasticity_options_group'] = plasticity_options_group
        self.components['material_profile_button'] = material_profile_button
        self.components['temperature_field_button'] = temperature_field_button
        self.components['temperature_field_path'] = temperature_field_path
        self.components['plasticity_method_combo'] = method_combo
        self.components['plasticity_max_iter_input'] = max_iter_input
        self.components['plasticity_tolerance_input'] = tolerance_input
        self.components['plasticity_warning_label'] = warning_label
        self.components['plasticity_extrapolation_combo'] = extrap_combo

        return plasticity_options_group
    
    def build_console_tabs_section(self):
        """
        Build the console and plotting tabs section.
        
        Returns:
            QTabWidget: Tab widget containing console and plots.
        """
        from PyQt5.QtWidgets import QTextEdit
        from PyQt5.QtGui import QFont
        
        # Console
        console_textbox = QTextEdit()
        console_textbox.setReadOnly(True)
        console_textbox.setStyleSheet(CONSOLE_STYLE)
        console_textbox.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        console_textbox.setText('Console Output:\n')

        terminal_font = QFont("Consolas", 8)
        terminal_font.setStyleHint(QFont.Monospace)
        console_textbox.setFont(terminal_font)

        # Create tab widget
        show_output_tab_widget = QTabWidget()
        show_output_tab_widget.setStyleSheet(TAB_STYLE)
        show_output_tab_widget.addTab(console_textbox, "Console")
        
        # Plot tabs are created up-front but added lazily when data is available.
        # This avoids reliance on setTabVisible behavior across Qt builds.
        plot_combo_history_tab = MatplotlibWidget()
        plot_combo_history_tab.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Plotly plot for max over combinations (uses PlotlyMaxWidget with update_envelope_plot)
        plot_max_combo_tab = PlotlyMaxWidget()
        
        # Plotly plot for min over combinations (uses PlotlyMaxWidget with update_envelope_plot)
        plot_min_combo_tab = PlotlyMaxWidget()
        
        # Store components
        self.components['console_textbox'] = console_textbox
        self.components['show_output_tab_widget'] = show_output_tab_widget
        self.components['plot_combo_history_tab'] = plot_combo_history_tab
        self.components['plot_max_combo_tab'] = plot_max_combo_tab
        self.components['plot_min_combo_tab'] = plot_min_combo_tab
        
        return show_output_tab_widget
    
    def build_progress_section(self):
        """
        Build the progress bar.
        
        Returns:
            QProgressBar: Progress bar widget.
        """
        progress_bar = QProgressBar()
        progress_bar.setStyleSheet(PROGRESS_BAR_STYLE)
        progress_bar.setValue(0)
        progress_bar.setAlignment(Qt.AlignCenter)
        progress_bar.setTextVisible(True)
        progress_bar.setVisible(False)
        
        self.components['progress_bar'] = progress_bar
        
        return progress_bar
    
    def build_solve_button(self):
        """
        Build the solve button.
        
        Returns:
            QPushButton: Solve button.
        """
        solve_button = QPushButton('SOLVE')
        solve_button.setStyleSheet(BUTTON_STYLE)
        solve_button.setFont(QFont('Arial', 9, QFont.Bold))
        solve_button.setEnabled(False)
        
        self.components['solve_button'] = solve_button
        
        return solve_button
    
    def set_window_palette(self, widget):
        """
        Set the window background color palette.
        
        Args:
            widget: The widget to apply the palette to.
        """
        palette = widget.palette()
        palette.setColor(QPalette.Window, QColor(*WINDOW_BACKGROUND_COLOR))
        widget.setPalette(palette)
    
    def build_complete_layout(self):
        """Build the full layout; console/tabs get stretch so they grow when sections collapse."""
        main_layout = QVBoxLayout()
        main_layout.setSpacing(6)
        
        # Build all sections
        file_group = self.build_file_input_section()
        combination_table_group = self.build_combination_table_section()
        output_group = self.build_output_selection_section()
        single_node_group = self.build_single_node_section()
        plasticity_options_group = self.build_plasticity_options_section()
        solve_button = self.build_solve_button()
        console_tabs = self.build_console_tabs_section()
        progress_bar = self.build_progress_section()
        
        # Combine options sections horizontally
        hbox_user_inputs = QHBoxLayout()
        hbox_user_inputs.addWidget(output_group)
        hbox_user_inputs.addWidget(single_node_group)
        hbox_user_inputs.addWidget(plasticity_options_group)
        hbox_user_inputs.addStretch()
        
        # Add collapsible sections with stretch factor 0 (no stretching)
        # These will only take the space they need
        main_layout.addWidget(file_group, 0)
        main_layout.addWidget(combination_table_group, 0)
        main_layout.addLayout(hbox_user_inputs, 0)
        main_layout.addWidget(solve_button, 0)
        
        # Console/tabs area gets stretch factor 1 (takes all remaining space)
        # This ensures it expands when collapsible groups are collapsed
        console_tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout.addWidget(console_tabs, 1)
        
        main_layout.addWidget(progress_bar, 0)
        
        return main_layout, self.components
