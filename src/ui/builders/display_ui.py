"""
Builds the Display tab UI: 3D view controls, result dropdowns, export buttons, etc.
"""

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import (
    QComboBox, QDoubleSpinBox, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QSpinBox, QVBoxLayout
)
from pyvistaqt import QtInteractor

from ui.styles.style_constants import (
    BUTTON_STYLE, GROUP_BOX_STYLE, READONLY_INPUT_STYLE
)

from utils.constants import (
    DEFAULT_POINT_SIZE, DEFAULT_BACKGROUND_COLOR
)
from utils.tooltips import (
    TOOLTIP_DISPLAY_LOAD_FILE, TOOLTIP_DISPLAY_FILE_PATH,
    TOOLTIP_DISPLAY_VISUALIZATION_CONTROLS, TOOLTIP_DISPLAY_POINT_SIZE,
    TOOLTIP_DISPLAY_LEGEND_RANGE, TOOLTIP_DISPLAY_SCALAR,
    TOOLTIP_DISPLAY_VIEW_COMBINATION, TOOLTIP_DISPLAY_FORCE_COMPONENT,
    TOOLTIP_DISPLAY_EXPORT_FORCES, TOOLTIP_DISPLAY_DISPLACEMENT_COMPONENT,
    TOOLTIP_DISPLAY_EXPORT_OUTPUT, TOOLTIP_DISPLAY_DEFORMATION_SCALE,
    TOOLTIP_DISPLAY_COMBINATION_POINT_CONTROLS,
    TOOLTIP_DISPLAY_COMBINATION_SELECTOR, TOOLTIP_DISPLAY_UPDATE_BUTTON,
    TOOLTIP_DISPLAY_SAVE_BUTTON, TOOLTIP_DISPLAY_EXTRACT_IC
)


class DisplayTabUIBuilder:
    """Assembles the Display tab in sections: file/graphics, results, export."""
    
    def __init__(self):
        """Initialize the builder."""
        self.components = {}
    
    def build_file_controls(self):
        """
        Build the file loading controls.
        
        Returns:
            QHBoxLayout: Layout containing file controls.
        """
        file_button = QPushButton('Load Visualization File')
        file_button.setStyleSheet(BUTTON_STYLE)
        file_button.setToolTip(TOOLTIP_DISPLAY_LOAD_FILE)

        file_path = QLineEdit()
        file_path.setReadOnly(True)
        file_path.setStyleSheet(READONLY_INPUT_STYLE)
        file_path.setToolTip(TOOLTIP_DISPLAY_FILE_PATH)
        
        file_layout = QHBoxLayout()
        file_layout.addWidget(file_button)
        file_layout.addWidget(file_path)
        
        # Store components
        self.components['file_button'] = file_button
        self.components['file_path'] = file_path
        
        return file_layout
    
    def build_visualization_controls(self):
        """
        Build the visualization control widgets.
        
        Returns:
            QGroupBox: Group box containing visualization controls.
        """
        # Point size control
        point_size = QSpinBox()
        point_size.setRange(1, 100)
        point_size.setValue(DEFAULT_POINT_SIZE)
        point_size.setPrefix("Size: ")
        point_size.setToolTip(TOOLTIP_DISPLAY_POINT_SIZE)
        
        # Scalar range controls
        scalar_min_spin = QDoubleSpinBox()
        scalar_max_spin = QDoubleSpinBox()
        scalar_min_spin.setPrefix("Min: ")
        scalar_max_spin.setPrefix("Max: ")
        scalar_min_spin.setDecimals(3)
        scalar_max_spin.setDecimals(3)
        scalar_min_spin.setToolTip(TOOLTIP_DISPLAY_LEGEND_RANGE)
        scalar_max_spin.setToolTip(TOOLTIP_DISPLAY_LEGEND_RANGE)
        
        # Deformation scale factor
        deformation_scale_label = QLabel("Deformation Scale Factor:")
        deformation_scale_edit = QLineEdit("1")
        validator = QDoubleValidator()
        validator.setNotation(QDoubleValidator.StandardNotation)
        deformation_scale_edit.setValidator(validator)
        deformation_scale_label.setToolTip(TOOLTIP_DISPLAY_DEFORMATION_SCALE)
        deformation_scale_edit.setToolTip(TOOLTIP_DISPLAY_DEFORMATION_SCALE)
        deformation_scale_label.setVisible(False)
        deformation_scale_edit.setVisible(False)
        # Scalar display selection combo box (for batch solve results)
        scalar_display_label = QLabel("Display:")
        scalar_display_combo = QComboBox()
        scalar_display_combo.setMinimumWidth(180)
        scalar_display_label.setToolTip(TOOLTIP_DISPLAY_SCALAR)
        scalar_display_combo.setToolTip(TOOLTIP_DISPLAY_SCALAR)
        # Default items - will be populated dynamically based on result type
        scalar_display_combo.addItem("Max Value")
        scalar_display_combo.addItem("Combo # of Max")
        scalar_display_label.setVisible(False)  # Hidden until batch results are loaded
        scalar_display_combo.setVisible(False)
        
        # View specific combination controls (for batch solve results)
        view_combination_label = QLabel("View Combination:")
        view_combination_combo = QComboBox()
        view_combination_combo.setMinimumWidth(200)
        view_combination_label.setToolTip(TOOLTIP_DISPLAY_VIEW_COMBINATION)
        view_combination_combo.setToolTip(TOOLTIP_DISPLAY_VIEW_COMBINATION)
        view_combination_combo.addItem("(Envelope View)")  # Default option
        view_combination_label.setVisible(False)  # Hidden until batch results are loaded
        view_combination_combo.setVisible(False)
        
        # Force component selection controls (for nodal forces results)
        force_component_label = QLabel("Force Component:")
        force_component_combo = QComboBox()
        force_component_combo.setMinimumWidth(150)
        force_component_combo.addItems([
            "Magnitude", "FX", "FY", "FZ",
            "Shear XY (FX^2+FY^2)^1/2",
            "Shear XZ (FX^2+FZ^2)^1/2",
            "Shear YZ (FY^2+FZ^2)^1/2",
        ])
        force_component_label.setToolTip(TOOLTIP_DISPLAY_FORCE_COMPONENT)
        force_component_combo.setToolTip(TOOLTIP_DISPLAY_FORCE_COMPONENT)
        force_component_label.setVisible(False)  # Hidden until nodal forces are loaded
        force_component_combo.setVisible(False)
        
        # Export forces CSV button
        export_forces_button = QPushButton("Export Forces CSV")
        export_forces_button.setStyleSheet(BUTTON_STYLE)
        export_forces_button.setToolTip(TOOLTIP_DISPLAY_EXPORT_FORCES)
        export_forces_button.setVisible(False)  # Hidden until nodal forces are loaded
        
        # Displacement component selection controls (for deformation results)
        displacement_component_label = QLabel("Displacement:")
        displacement_component_combo = QComboBox()
        displacement_component_combo.setMinimumWidth(120)
        displacement_component_combo.addItems(["U_mag", "UX", "UY", "UZ"])
        displacement_component_label.setToolTip(TOOLTIP_DISPLAY_DISPLACEMENT_COMPONENT)
        displacement_component_combo.setToolTip(TOOLTIP_DISPLAY_DISPLACEMENT_COMPONENT)
        displacement_component_label.setVisible(False)  # Hidden until deformation is available
        displacement_component_combo.setVisible(False)
        
        # Export output CSV (stress, forces, deformation)
        export_output_button = QPushButton("Export Output CSV")
        export_output_button.setStyleSheet(BUTTON_STYLE)
        export_output_button.setToolTip(TOOLTIP_DISPLAY_EXPORT_OUTPUT)
        export_output_button.setVisible(False)  # Hidden until results are available
        
        # Layout
        node_point_size_label = QLabel("Node Point Size:")
        node_point_size_label.setToolTip(TOOLTIP_DISPLAY_POINT_SIZE)
        legend_range_label = QLabel("Legend Range:")
        legend_range_label.setToolTip(TOOLTIP_DISPLAY_LEGEND_RANGE)

        graphics_control_layout = QHBoxLayout()
        graphics_control_layout.addWidget(node_point_size_label)
        graphics_control_layout.addWidget(point_size)
        graphics_control_layout.addWidget(legend_range_label)
        graphics_control_layout.addWidget(scalar_min_spin)
        graphics_control_layout.addWidget(scalar_max_spin)
        graphics_control_layout.addWidget(scalar_display_label)
        graphics_control_layout.addWidget(scalar_display_combo)
        graphics_control_layout.addWidget(view_combination_label)
        graphics_control_layout.addWidget(view_combination_combo)
        graphics_control_layout.addWidget(force_component_label)
        graphics_control_layout.addWidget(force_component_combo)
        graphics_control_layout.addWidget(export_forces_button)
        graphics_control_layout.addWidget(displacement_component_label)
        graphics_control_layout.addWidget(displacement_component_combo)
        graphics_control_layout.addWidget(export_output_button)
        graphics_control_layout.addWidget(deformation_scale_label)
        graphics_control_layout.addWidget(deformation_scale_edit)
        graphics_control_layout.addStretch()
        
        graphics_control_group = QGroupBox("Visualization Controls")
        graphics_control_group.setStyleSheet(GROUP_BOX_STYLE)
        graphics_control_group.setLayout(graphics_control_layout)
        graphics_control_group.setToolTip(TOOLTIP_DISPLAY_VISUALIZATION_CONTROLS)
        
        # Store components
        self.components['point_size'] = point_size
        self.components['scalar_min_spin'] = scalar_min_spin
        self.components['scalar_max_spin'] = scalar_max_spin
        self.components['scalar_display_label'] = scalar_display_label
        self.components['scalar_display_combo'] = scalar_display_combo
        self.components['view_combination_label'] = view_combination_label
        self.components['view_combination_combo'] = view_combination_combo
        self.components['force_component_label'] = force_component_label
        self.components['force_component_combo'] = force_component_combo
        self.components['export_forces_button'] = export_forces_button
        self.components['displacement_component_label'] = displacement_component_label
        self.components['displacement_component_combo'] = displacement_component_combo
        self.components['export_output_button'] = export_output_button
        self.components['deformation_scale_label'] = deformation_scale_label
        self.components['deformation_scale_edit'] = deformation_scale_edit
        self.components['graphics_control_layout'] = graphics_control_layout
        self.components['graphics_control_group'] = graphics_control_group
        
        return graphics_control_group
    
    def build_time_point_controls(self):
        """
        Build the combination point selection controls.
        
        For MARS-SC, this allows selecting which combination to display results for.
        
        Returns:
            QGroupBox: Group box containing combination point controls.
        """
        selected_time_label = QLabel("Display results for Combination:")
        selected_time_label.setToolTip(TOOLTIP_DISPLAY_COMBINATION_SELECTOR)
        
        # Combination dropdown instead of time spinbox for MARS-SC
        combination_combo = QComboBox()
        combination_combo.setMinimumWidth(200)
        combination_combo.setToolTip(TOOLTIP_DISPLAY_COMBINATION_SELECTOR)
        
        # Keep the spinbox for backwards compatibility but hidden by default
        time_point_spinbox = QDoubleSpinBox()
        time_point_spinbox.setDecimals(5)
        time_point_spinbox.setPrefix("Time (seconds): ")
        time_point_spinbox.setRange(0, 0)
        time_point_spinbox.setVisible(False)  # Hidden for MARS-SC
        
        update_time_button = QPushButton("Update")
        update_time_button.setToolTip(TOOLTIP_DISPLAY_UPDATE_BUTTON)
        save_time_button = QPushButton("Save Combination as CSV")
        save_time_button.setToolTip(TOOLTIP_DISPLAY_SAVE_BUTTON)
        
        extract_ic_button = QPushButton("Export Velocity as Initial Condition in APDL")
        extract_ic_button.setStyleSheet(BUTTON_STYLE)
        extract_ic_button.setToolTip(TOOLTIP_DISPLAY_EXTRACT_IC)
        extract_ic_button.setVisible(False)  # Not used in MARS-SC
        
        # Layout
        time_point_layout = QHBoxLayout()
        time_point_layout.addWidget(selected_time_label)
        time_point_layout.addWidget(combination_combo)
        time_point_layout.addWidget(time_point_spinbox)
        time_point_layout.addWidget(update_time_button)
        time_point_layout.addWidget(save_time_button)
        time_point_layout.addWidget(extract_ic_button)
        time_point_layout.addStretch()
        
        time_point_group = QGroupBox("Combination Point Controls")
        time_point_group.setStyleSheet(GROUP_BOX_STYLE)
        time_point_group.setLayout(time_point_layout)
        time_point_group.setToolTip(TOOLTIP_DISPLAY_COMBINATION_POINT_CONTROLS)
        time_point_group.setVisible(False)
        
        # Store components
        self.components['selected_time_label'] = selected_time_label
        self.components['combination_combo'] = combination_combo
        self.components['time_point_spinbox'] = time_point_spinbox
        self.components['update_time_button'] = update_time_button
        self.components['save_time_button'] = save_time_button
        self.components['extract_ic_button'] = extract_ic_button
        self.components['time_point_layout'] = time_point_layout
        self.components['time_point_group'] = time_point_group
        
        return time_point_group

    def build_plotter(self, parent):
        """
        Build the PyVista plotter widget.
        
        Args:
            parent: Parent widget for the plotter.
        
        Returns:
            QtInteractor: PyVista plotter widget.
        """
        plotter = QtInteractor(parent=parent)
        plotter.set_background(DEFAULT_BACKGROUND_COLOR)
        plotter.setContextMenuPolicy(Qt.CustomContextMenu)
        
        # Use isometric (parallel) projection instead of perspective
        # This maintains consistent object sizes regardless of distance
        plotter.enable_parallel_projection()
        
        self.components['plotter'] = plotter
        
        return plotter
    
    def build_complete_layout(self, parent):
        """
        Build the complete display tab layout with all sections.
        
        Args:
            parent: Parent widget for components that need it.
        
        Returns:
            tuple: (main_layout, components_dict)
        """
        main_layout = QVBoxLayout()
        
        # Build all sections
        file_layout = self.build_file_controls()
        graphics_control_group = self.build_visualization_controls()
        time_point_group = self.build_time_point_controls()
        plotter = self.build_plotter(parent)
        
        # Add all to main layout
        main_layout.addLayout(file_layout)
        main_layout.addWidget(graphics_control_group)
        main_layout.addWidget(time_point_group)
        main_layout.addWidget(plotter)
        
        return main_layout, self.components
