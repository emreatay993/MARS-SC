"""
Display tab: 3D visualization of FEA results (PyVista). UI is built via
DisplayTabUIBuilder; visualization/hotspots live in handler classes.
"""

import numpy as np
import pyvista as pv

from PyQt5.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QMessageBox, QWidget

# Import builders and managers
from ui.builders.display_ui import DisplayTabUIBuilder
from core.visualization import VisualizationManager, HotspotDetector
from ui.handlers.display_state import DisplayState
from ui.handlers.display_file_handler import DisplayFileHandler
from ui.handlers.display_visualization_handler import DisplayVisualizationHandler
from ui.handlers.display_interaction_handler import DisplayInteractionHandler
from ui.handlers.display_export_handler import DisplayExportHandler
from ui.handlers.display_results_handler import DisplayResultsHandler
from ui.handlers.display_contour_sync_handler import DisplayContourSyncHandler


class DisplayTab(QWidget):
    """
    3D FEA results view. Builders wire up the UI; VisualizationManager and HotspotDetector do the heavy lifting.

    Signals: node_picked_signal(int), time_point_update_requested(float, dict),
    combination_update_requested(int, dict).
    """
    
    # Signals
    node_picked_signal = pyqtSignal(int)
    time_point_update_requested = pyqtSignal(float, dict)
    combination_update_requested = pyqtSignal(int, dict)  # (combo_index, options)
    
    def __init__(self, parent=None):
        """Initialize the Display Tab."""
        super().__init__(parent)
        
        # Managers for complex logic
        self.viz_manager = VisualizationManager()
        self.hotspot_detector = HotspotDetector()

        # Shared state and handler scaffolding
        self.state = DisplayState()
        self.file_handler = DisplayFileHandler(self, self.state, self.viz_manager)
        self.visual_handler = DisplayVisualizationHandler(self, self.state, self.viz_manager)
        self.interaction_handler = DisplayInteractionHandler(self, self.state, self.hotspot_detector)
        self.export_handler = DisplayExportHandler(self, self.state)
        self.results_handler = DisplayResultsHandler(self, self.state, self.visual_handler)
        self.contour_sync_handler = DisplayContourSyncHandler(self, self.state)
        self.plotting_handler = None
        
        # Build UI using builder
        builder = DisplayTabUIBuilder()
        layout, self.components = builder.build_complete_layout(self)
        self.setLayout(layout)
        
        # Store commonly used components as direct attributes
        self._setup_component_references()
        
        # State tracking
        self.current_mesh = None
        self.current_actor = None
        self.camera_state = None
        self.camera_widget = None
        self.hover_annotation = None
        self.hover_observer = None
        self.last_hover_time = 0  # For frame rate throttling
        self.data_column = "Result"  # Track current data column name
        
        # Combination results metadata (for scalar display and hover annotations)
        self.combination_names = []  # List of combination names
        self.current_result_type = None  # "von_mises", "max_principal", or "min_principal"
        self.all_combo_results = None  # Full results array, shape (num_combinations, num_nodes)
        self.nodal_forces_result = None  # NodalForcesResult for force-based visualization
        self.deformation_result = None  # DeformationResult for displacement visualization
        self.current_contour_type = None  # "Stress", "Forces", or "Deformation"
        self.temp_solver = None
        self.time_values = None
        self.original_node_coords = None
        self.last_valid_deformation_scale = 1.0
        
        # Hotspot and picking state
        self.highlight_actor = None
        self.box_widget = None
        self.hotspot_dialog = None
        self.is_point_picking_active = False
        
        # Node tracking state
        self.target_node_index = None
        self.target_node_id = None
        self.target_node_label_actor = None
        self.label_point_data = None
        self.marker_poly = None
        self.target_node_marker_actor = None
        self.last_goto_node_id = None
        
        # Track if camera widget needs to be re-initialized on show
        self._camera_widget_pending = False
        self._first_show = True
        
        # Connect signals
        self._connect_signals()
        
        # Show initial welcome message on plotter
        self._show_welcome_message()
    
    def _show_welcome_message(self):
        """Display welcome message on empty plotter."""
        self.plotter.add_text(
            "No data loaded.\n\n"
            "To visualize mesh points:\n"
            "  1. Load Base Analysis and Combine Analysis RST files\n"
            "     in the Solver tab and run the analysis, OR\n"
            "  2. Use 'Load Visualization File' above to load\n"
            "     a CSV file with node coordinates and scalar data.",
            position=(0.22, 0.5),
            viewport=True,
            font_size=12,
            color="gray",
            name="welcome_message"
        )
    
    def _setup_component_references(self):
        """Create direct references to frequently used components."""
        # File controls
        self.file_button = self.components['file_button']
        self.file_path = self.components['file_path']
        
        # Visualization controls
        self.plotter = self.components['plotter']
        self.point_size = self.components['point_size']
        self.scalar_min_spin = self.components['scalar_min_spin']
        self.scalar_max_spin = self.components['scalar_max_spin']
        self.scalar_display_label = self.components['scalar_display_label']
        self.scalar_display_combo = self.components['scalar_display_combo']
        self.contour_type_label = self.components.get('contour_type_label')
        self.contour_type_combo = self.components.get('contour_type_combo')
        self.deformation_scale_label = self.components['deformation_scale_label']
        self.deformation_scale_edit = self.components['deformation_scale_edit']
        
        # View specific combination controls
        self.view_combination_label = self.components['view_combination_label']
        self.view_combination_combo = self.components['view_combination_combo']
        
        # Force component controls (for nodal forces visualization)
        self.force_component_label = self.components['force_component_label']
        self.force_component_combo = self.components['force_component_combo']
        self.export_forces_button = self.components['export_forces_button']
        
        # Displacement component controls (for deformation visualization)
        self.displacement_component_label = self.components.get('displacement_component_label')
        self.displacement_component_combo = self.components.get('displacement_component_combo')
        self.export_output_button = self.components.get('export_output_button')
        
        # Combination/Time point controls (MARS-SC: uses combination_combo)
        self.combination_combo = self.components.get('combination_combo')
        self.time_point_spinbox = self.components['time_point_spinbox']
        self.update_time_button = self.components['update_time_button']
        self.save_time_button = self.components['save_time_button']
        self.extract_ic_button = self.components['extract_ic_button']
        self.time_point_group = self.components['time_point_group']
    
    def set_plotting_handler(self, plotting_handler):
        """Set the plotting handler for this display tab."""
        self.plotting_handler = plotting_handler
        self.file_handler.set_plotting_handler(plotting_handler)


    def _connect_signals(self):
        """Connect UI signals to their handlers."""
        # File controls
        self.file_button.clicked.connect(self.load_file)
        
        # Visualization controls
        self.point_size.valueChanged.connect(self.update_point_size)
        self.scalar_min_spin.valueChanged.connect(self._update_scalar_range)
        self.scalar_max_spin.valueChanged.connect(self._update_scalar_range)
        self.scalar_min_spin.valueChanged.connect(
            lambda v: self.scalar_max_spin.setMinimum(v)
        )
        self.scalar_max_spin.valueChanged.connect(
            lambda v: self.scalar_min_spin.setMaximum(v)
        )
        self.scalar_display_combo.currentIndexChanged.connect(
            self._on_scalar_display_changed
        )
        if self.contour_type_combo is not None:
            self.contour_type_combo.currentIndexChanged.connect(
                self._on_contour_type_changed
            )
        self.view_combination_combo.currentIndexChanged.connect(
            self._on_view_combination_changed
        )
        self.force_component_combo.currentIndexChanged.connect(
            self._on_force_component_changed
        )
        self.export_forces_button.clicked.connect(
            self._on_export_forces_clicked
        )
        
        # Displacement component controls
        if self.displacement_component_combo is not None:
            self.displacement_component_combo.currentIndexChanged.connect(
                self._on_displacement_component_changed
            )
        if self.export_output_button is not None:
            self.export_output_button.clicked.connect(
                self._on_export_output_clicked
            )
        
        self.deformation_scale_edit.editingFinished.connect(
            self._validate_deformation_scale
        )
        
        # Combination/Time point controls
        self.update_time_button.clicked.connect(self.update_time_point_results)
        self.save_time_button.clicked.connect(self.save_time_point_results)
        self.extract_ic_button.clicked.connect(self.extract_initial_conditions)
        
        # MARS-SC: Connect combination combo if available
        if self.combination_combo is not None:
            self.combination_combo.currentIndexChanged.connect(
                self._on_combination_changed
            )
        
        # Context menu
        self.plotter.customContextMenuRequested.connect(self.show_context_menu)
    
    def _get_stress_type_label(self, result_type: str = None) -> str:
        """
        Get a descriptive label for the current stress result type.
        
        Args:
            result_type: Optional result type override. If None, uses self.current_result_type.
        
        Returns:
            Descriptive label like "S_vm", "S1", "S3", or "Force".
        """
        rt = result_type or self.current_result_type
        
        stress_type_labels = {
            "von_mises": "S_vm",
            "max_principal": "S1",
            "min_principal": "S3",
        }
        
        return stress_type_labels.get(rt, "Stress")
    
    def _get_descriptive_legend_title(self, base_name: str, result_type: str = None) -> str:
        """
        Generate a descriptive legend title that includes the stress type.
        
        Args:
            base_name: Base name like "Max", "Min", "Combo_0", etc.
            result_type: Optional result type override. If None, uses self.current_result_type.
        
        Returns:
            Descriptive title like "Max_S_vm [MPa]", "Combo_1_S1 [MPa]".
        """
        rt = result_type or self.current_result_type
        stress_label = self._get_stress_type_label(rt)
        
        # Check if we're dealing with force results
        if self.nodal_forces_result is not None and self.all_combo_results is None:
            return f"{base_name}_Force [N]"
        
        return f"{base_name}_{stress_label} [MPa]"
    
    @pyqtSlot(int)
    def _on_combination_changed(self, index):
        """
        Handle combination selection change.
        
        Args:
            index: New combination index.
        """
        if index >= 0:
            combo_name = self.combination_combo.currentText()
            print(f"DisplayTab: Combination changed to #{index}: {combo_name}")

    @pyqtSlot(int)
    def _on_contour_type_changed(self, index):
        """Handle contour family selection changes."""
        self.contour_sync_handler.on_contour_type_changed(index)
    
    @pyqtSlot(int)
    def _on_scalar_display_changed(self, index):
        """
        Handle scalar display selection change.
        """
        self.contour_sync_handler.on_scalar_display_changed(index)
    
    @pyqtSlot(int)
    def _on_view_combination_changed(self, index):
        """
        Handle view combination selection change.
        """
        self.contour_sync_handler.on_view_combination_changed(index)
    
    def _show_envelope_view(self):
        """
        Switch visualization back to envelope view (Max/Min across combinations).
        """
        if self.view_combination_combo is not None:
            self.view_combination_combo.setCurrentIndex(0)
        self.contour_sync_handler.sync_from_current_state()
    
    def _show_specific_combination(self, combo_idx: int):
        """
        Update visualization to show results for a specific combination.
        
        Args:
            combo_idx: Index of the combination to display (0-based).
        """
        if self.view_combination_combo is not None:
            target_index = combo_idx + 1  # index 0 is envelope
            if 0 <= target_index < self.view_combination_combo.count():
                self.view_combination_combo.setCurrentIndex(target_index)
        self.contour_sync_handler.sync_from_current_state()
    
    def populate_view_combination_options(self, combination_names: list):
        """
        Populate the view combination dropdown with available combinations.
        
        Args:
            combination_names: List of combination names from the combination table.
        """
        self.view_combination_combo.blockSignals(True)
        self.view_combination_combo.clear()
        
        # First item is always the envelope view option
        self.view_combination_combo.addItem("(Envelope View)")
        
        # Add each combination as an option
        for i, name in enumerate(combination_names):
            self.view_combination_combo.addItem(f"{i + 1}: {name}")
        
        self.view_combination_combo.setCurrentIndex(0)  # Default to envelope view
        self.view_combination_combo.blockSignals(False)
        
        # Show the view combination controls
        self.view_combination_label.setVisible(True)
        self.view_combination_combo.setVisible(True)
    
    def populate_scalar_display_options(self, result_type: str, has_min_data: bool = False):
        """
        Populate the scalar display dropdown based on result type.
        
        For min_principal stress, includes min value options since negative
        (compressive) stresses are significant for this type.
        
        Args:
            result_type: Type of stress result ("von_mises", "max_principal", "min_principal").
            has_min_data: Whether min stress data is available in the mesh.
        """
        self.scalar_display_combo.blockSignals(True)
        self.scalar_display_combo.clear()
        
        # Always add max value and combo of max options
        self.scalar_display_combo.addItem("Max Value")
        self.scalar_display_combo.addItem("Combo # of Max")
        
        # For min_principal stress, add min value options
        if result_type == "min_principal" and has_min_data:
            self.scalar_display_combo.addItem("Min Value")
            self.scalar_display_combo.addItem("Combo # of Min")
        
        self.scalar_display_combo.setCurrentIndex(0)  # Default to Max Value
        self.scalar_display_combo.blockSignals(False)
        
        # Store result type for hover annotation and legend title generation
        self.current_result_type = result_type
        
        # Show the scalar display controls
        self.scalar_display_label.setVisible(True)
        self.scalar_display_combo.setVisible(True)
        
        # Update the legend title to match the default selection with descriptive format
        # This ensures the legend shows "Max_S_vm [MPa]" instead of generic title
        self._on_scalar_display_changed(0)
    
    def _update_force_component_options(self, has_beam_nodes: bool):
        """
        Update force component dropdown options.
        
        Args:
            has_beam_nodes: True if selection contains nodes attached to beam elements.
        """
        self.force_component_combo.blockSignals(True)
        self.force_component_combo.clear()
        
        # Base options always available
        items = ["Magnitude", "FX", "FY", "FZ"]
        
        # Add shear options for different axis pairs
        items.extend([
            "Shear XY (FX^2+FY^2)^1/2",
            "Shear XZ (FX^2+FZ^2)^1/2",
            "Shear YZ (FY^2+FZ^2)^1/2",
        ])
        
        self.force_component_combo.addItems(items)
        self.force_component_combo.setCurrentIndex(0)  # Default to Magnitude
        self.force_component_combo.blockSignals(False)
    
    def _setup_force_component_arrays(self, combo_idx: int):
        """
        Add force component arrays to mesh for the selected combination.
        
        This sets up FX, FY, FZ, Magnitude, and shear arrays
        on the current mesh for the specified combination.
        
        Args:
            combo_idx: Index of the combination to set up arrays for.
        """
        from ui.handlers.display_mesh_arrays import attach_force_component_arrays

        attach_force_component_arrays(self.current_mesh, self.nodal_forces_result, combo_idx)
    
    def _setup_displacement_component_arrays(self, combo_idx: int):
        """
        Add displacement component arrays to mesh for the selected combination.
        
        This sets up UX, UY, UZ, and U_mag arrays on the current mesh for 
        the specified combination.
        
        Args:
            combo_idx: Index of the combination to set up arrays for.
        """
        from ui.handlers.display_mesh_arrays import attach_deformation_specific_arrays

        attach_deformation_specific_arrays(self.current_mesh, self.deformation_result, combo_idx)
    
    @pyqtSlot(int)
    def _on_force_component_changed(self, index: int):
        """
        Handle force component selection change.
        
        Args:
            index: New component index from the dropdown.
        """
        self.contour_sync_handler.on_force_component_changed(index)
    
    def _on_export_forces_clicked(self):
        """Handle click on Export Forces CSV button."""
        # Delegate to export handler
        self.export_handler.export_forces_csv()
    
    def _show_force_component_controls(self, show: bool, has_beam_nodes: bool = False):
        """
        Show or hide force component controls.
        
        Args:
            show: Whether to show the controls.
            has_beam_nodes: Whether beam nodes are present (affects Shear option).
        """
        self.force_component_label.setVisible(show)
        self.force_component_combo.setVisible(show)
        self.export_forces_button.setVisible(show)
        
        if show:
            self._update_force_component_options(has_beam_nodes)
    
    @pyqtSlot(int)
    def _on_displacement_component_changed(self, index: int):
        """
        Handle displacement component selection change.
        
        Args:
            index: New component index from the dropdown.
        """
        self.contour_sync_handler.on_displacement_component_changed(index)
    
    def _on_export_output_clicked(self):
        """Handle click on Export Output CSV button."""
        # Delegate to export handler
        self.export_handler.export_output_csv()
    
    def _show_displacement_component_controls(self, show: bool):
        """
        Show or hide displacement component controls.
        
        Args:
            show: Whether to show the controls.
        """
        if self.displacement_component_label is not None:
            self.displacement_component_label.setVisible(show)
        if self.displacement_component_combo is not None:
            self.displacement_component_combo.setVisible(show)
        # Note: export_output_button visibility is managed separately by _update_export_output_button_visibility()
    
    def _update_export_output_button_visibility(self):
        """
        Update visibility of Export Output CSV button.
        
        The button should be visible when ANY result type is available:
        - Stress results (combination_result from solver tab)
        - Nodal forces results
        - Deformation results
        """
        if self.export_output_button is None:
            return
        
        # Check for any available results
        has_stress = False
        has_forces = self.nodal_forces_result is not None
        has_deformation = self.deformation_result is not None
        
        # Check stress result from solver tab
        try:
            solver_tab = self.window().solver_tab
            if solver_tab and solver_tab.combination_result is not None:
                has_stress = True
        except (AttributeError, RuntimeError):
            pass
        
        # Show button if any result type is available
        show_button = has_stress or has_forces or has_deformation
        self.export_output_button.setVisible(show_button)
    
    @pyqtSlot(object)
    def _setup_initial_view(self, initial_data):
        """
        Setup initial view with loaded data.
        
        Args:
            initial_data: Tuple of (time_values, node_coords, node_ids, deformation_loaded)
        """
        time_values, node_coords, df_node_ids, deformation_is_loaded = initial_data
        
        # Store data
        self.time_values = time_values
        self.original_node_coords = node_coords
        self.state.time_values = time_values
        self.state.original_node_coords = node_coords
        
        # Update UI controls with time range
        self._update_time_controls(time_values)
        
        # Update deformation scale control
        self._update_deformation_controls(deformation_is_loaded)
        
        # Create and display initial mesh
        if node_coords is not None:
            mesh = self.viz_manager.create_mesh_from_coords(
                node_coords, df_node_ids
            )
            self.state.current_mesh = mesh
            self.current_mesh = mesh
            self.update_visualization()
            self.plotter.reset_camera()
    
    def _update_time_controls(self, time_values):
        """Update time-related UI controls with time range."""
        min_time, max_time = np.min(time_values), np.max(time_values)
        
        # Time point spinbox (kept for backwards compatibility)
        self.time_point_spinbox.setRange(min_time, max_time)
        self.time_point_spinbox.setValue(min_time)
        
        # Compute average sampling interval
        if len(time_values) > 1:
            avg_dt = np.mean(np.diff(time_values))
        else:
            avg_dt = 1.0
        self.time_point_spinbox.setSingleStep(avg_dt)
        self.time_point_group.setVisible(True)
        # Note: Deformation controls visibility is managed by _update_deformation_controls()
    
    def update_combination_controls(self, combination_names):
        """
        Update the combination dropdown with available combinations.
        
        Args:
            combination_names: List of combination names from the combination table.
        """
        if self.combination_combo is not None:
            self.combination_combo.clear()
            if combination_names:
                self.combination_combo.addItems(combination_names)
            self.time_point_group.setVisible(True)
    
    def get_selected_combination_index(self):
        """
        Get the currently selected combination index.
        
        Returns:
            int: Index of the selected combination, or -1 if none selected.
        """
        if self.combination_combo is not None:
            return self.combination_combo.currentIndex()
        return -1
    
    def get_selected_combination_name(self):
        """
        Get the currently selected combination name.
        
        Returns:
            str: Name of the selected combination, or empty string if none.
        """
        if self.combination_combo is not None:
            return self.combination_combo.currentText()
        return ""
    
    def _update_deformation_controls(self, deformation_loaded):
        """Update deformation scale controls based on availability."""
        if deformation_loaded:
            # Show and enable deformation controls when deformations are loaded
            self.deformation_scale_label.setVisible(True)
            self.deformation_scale_edit.setVisible(True)
            self.deformation_scale_edit.setEnabled(True)
            self.deformation_scale_edit.setText(
                str(self.last_valid_deformation_scale)
            )
        else:
            # Hide deformation controls when deformations are not loaded
            self.deformation_scale_label.setVisible(False)
            self.deformation_scale_edit.setVisible(False)
            self.deformation_scale_edit.setEnabled(False)
            self.deformation_scale_edit.setText("0")
    
    @pyqtSlot(bool)
    def load_file(self, checked=False):
        """Open file dialog and load visualization file."""
        self.file_handler.open_file_dialog()
    
    def update_visualization(self):
        """Update the 3D visualization with current mesh."""
        self.visual_handler.update_visualization()
    
    
    @pyqtSlot(int)
    def update_point_size(self, value):
        """Update the point size of the displayed mesh."""
        self.visual_handler.update_point_size()
    
    @pyqtSlot(float)
    def _update_scalar_range(self, value):
        """Update the scalar range of the color map."""
        self.visual_handler.update_scalar_range()
    
    @pyqtSlot()
    def _validate_deformation_scale(self):
        """Validate deformation scale factor input."""
        self.visual_handler.validate_deformation_scale()
    @pyqtSlot(bool)
    def update_time_point_results(self, checked=False):
        """
        Request combination point calculation and update visualization.
        
        For MARS-SC, this requests recalculation for the selected combination.
        """
        # Get main tab (solver tab) to check which outputs are selected
        main_tab = self.window().solver_tab
        if main_tab is None:
            QMessageBox.warning(
                self, "Not Ready",
                "Solver tab not initialized."
            )
            return
        
        # Gather options - updated for MARS-SC
        options = {
            'compute_von_mises': main_tab.von_mises_checkbox.isChecked(),
            'compute_max_principal': main_tab.max_principal_stress_checkbox.isChecked(),
            'compute_min_principal': main_tab.min_principal_stress_checkbox.isChecked(),
            # MARS-SC specific options
            'combination_index': self.get_selected_combination_index(),
            'combination_name': self.get_selected_combination_name(),
        }
        
        # For backwards compatibility, also pass as time if spinbox is visible/used
        if self.time_point_spinbox.isVisible():
            selected_time = self.time_point_spinbox.value()
        else:
            # Use combination index as a pseudo-time value
            selected_time = float(self.get_selected_combination_index())
        
        # Emit signal to request computation
        combo_idx = self.get_selected_combination_index()
        combo_name = self.get_selected_combination_name()
        print(f"DisplayTab: Requesting update for combination #{combo_idx}: {combo_name}")
        self.time_point_update_requested.emit(selected_time, options)
    
    @pyqtSlot(bool)
    def save_time_point_results(self, checked=False):
        """Save currently displayed results to CSV."""
        self.export_handler.save_time_point_results()
    
    @pyqtSlot(bool)
    def extract_initial_conditions(self, checked=False):
        """Extract velocity initial conditions and export to APDL format."""
        self.export_handler.extract_initial_conditions()
    @pyqtSlot('QPoint')
    def show_context_menu(self, position):
        """Create and display the right-click context menu."""
        self.interaction_handler.show_context_menu(position)
    
    @pyqtSlot(object, str, float, float)
    def update_view_with_results(self, mesh, scalar_bar_title, data_min, data_max):
        """
        Update visualization with computed results.
        
        Args:
            mesh: PyVista mesh with results.
            scalar_bar_title: Title for the scalar bar.
            data_min: Minimum data value.
            data_max: Maximum data value.
        """
        # TODO(emre): Manual validation after contour refactor:
        # - Verify whether the reported deformations are true on the contour.
        # - Verify whether legend updates properly, especially for deformation results.
        # - Verify whether hover annotation results are correct for deformations.
        # Update current mesh and track state
        self.current_mesh = mesh
        self.state.current_mesh = mesh
        
        # Use the passed scalar_bar_title initially (will be updated when scalar display changes)
        self.data_column = scalar_bar_title
        self.state.data_column = scalar_bar_title
        
        # Update scalar range spin boxes
        self.scalar_min_spin.blockSignals(True)
        self.scalar_max_spin.blockSignals(True)
        self.scalar_min_spin.setRange(data_min, data_max)
        self.scalar_max_spin.setRange(data_min, 1e30)
        self.scalar_min_spin.setValue(data_min)
        self.scalar_max_spin.setValue(data_max)
        self.scalar_min_spin.blockSignals(False)
        self.scalar_max_spin.blockSignals(False)
        
        # Extract result type and combination names from batch solve results
        result_type = None
        if 'result_type' in mesh.field_data:
            result_type = mesh.field_data['result_type'][0]
            self.current_result_type = result_type
        
        # Get combination names and all_combo_results from solver tab if available
        try:
            solver_tab = self.window().solver_tab
            if solver_tab and solver_tab.combination_table:
                self.combination_names = solver_tab.combination_table.combination_names
            else:
                self.combination_names = []
            
            # Get all_combo_results for viewing individual combinations (stress results)
            if solver_tab and solver_tab.combination_result:
                self.all_combo_results = solver_tab.combination_result.all_combo_results
            else:
                self.all_combo_results = None
            
            # Get nodal forces result for viewing individual force combinations
            if solver_tab and solver_tab.nodal_forces_result:
                self.nodal_forces_result = solver_tab.nodal_forces_result
            else:
                self.nodal_forces_result = None
            
            # Get deformation result for deformed mesh and displacement component selection
            if solver_tab and solver_tab.deformation_result:
                self.deformation_result = solver_tab.deformation_result
                # Store original coordinates for deformation scaling
                self.original_node_coords = mesh.points.copy()
            else:
                self.deformation_result = None
        except (AttributeError, RuntimeError):
            self.combination_names = []
            self.all_combo_results = None
            self.nodal_forces_result = None
            self.deformation_result = None
        
        # Deformation scale controls depend only on deformation availability
        has_deformation = (
            self.deformation_result is not None
            or "Max_U_mag" in mesh.array_names
            or "Min_U_mag" in mesh.array_names
            or "Def_Max_U_mag" in mesh.array_names
            or "U_mag" in mesh.array_names
        )
        self._update_deformation_controls(has_deformation)
        
        # Synchronize contour families/options and update visualization
        self.contour_sync_handler.sync_from_current_state()
        
        # Clear file path since this is computed data, not loaded from file
        self.file_path.clear()
        
        # Show IC export button if velocity components are present
        if all(key in mesh.array_names for key in ['vel_x', 'vel_y', 'vel_z']):
            self.extract_ic_button.setVisible(True)
        else:
            self.extract_ic_button.setVisible(False)
        
        # Update Export Output CSV button visibility
        self._update_export_output_button_visibility()
    def _clear_visualization(self):
        """Properly clear existing visualization."""
        self.interaction_handler.clear_goto_node_markers()
        
        # Clear hover elements
        self.visual_handler.clear_hover_elements()
        
        # Clear box widget
        if self.box_widget:
            self.box_widget.Off()
            self.box_widget = None
            self.state.box_widget = None
        
        # Clear camera widget
        if self.camera_widget:
            self.camera_widget.EnabledOff()
            self.camera_widget = None
            self.state.camera_widget = None
        
        self.plotter.clear()
        
        if self.current_mesh:
            self.current_mesh.clear_data()
            self.current_mesh = None
            self.state.current_mesh = None
        
        self.current_actor = None
        self.state.current_actor = None
        self.scalar_min_spin.clear()
        self.scalar_max_spin.clear()
        self.file_path.clear()
        
        # Clear combination data
        self.all_combo_results = None
        self.nodal_forces_result = None
        self.deformation_result = None
        self.combination_names = []
        
        # Reset and hide view combination controls
        self.view_combination_combo.blockSignals(True)
        self.view_combination_combo.clear()
        self.view_combination_combo.addItem("(Envelope View)")
        self.view_combination_combo.blockSignals(False)
        self.view_combination_label.setVisible(False)
        self.view_combination_combo.setVisible(False)
        
        # Hide force component controls
        self._show_force_component_controls(False)
        self._show_displacement_component_controls(False)

        # Reset contour selector
        self.state.current_contour_type = None
        self.current_contour_type = None
        if self.contour_type_combo is not None:
            self.contour_type_combo.blockSignals(True)
            self.contour_type_combo.clear()
            self.contour_type_combo.addItems(["Stress", "Forces", "Deformation"])
            self.contour_type_combo.blockSignals(False)
            self.contour_type_combo.setVisible(False)
        if self.contour_type_label is not None:
            self.contour_type_label.setVisible(False)
        
        # Hide export output button
        if self.export_output_button is not None:
            self.export_output_button.setVisible(False)
        
        # Re-enable scalar display controls
        self.scalar_display_combo.setEnabled(True)
        self.scalar_display_label.setEnabled(True)
    
    def showEvent(self, event):
        """Handle tab becoming visible - fix camera widget sizing."""
        super().showEvent(event)
        
        # On first show or when camera widget is pending, initialize it properly
        if self._first_show or self._camera_widget_pending:
            self._first_show = False
            self._camera_widget_pending = False
            # Short delay to allow Qt layout to settle
            QTimer.singleShot(30, self._reinitialize_camera_widget)
    
    def _reinitialize_camera_widget(self):
        """Reinitialize camera widget with proper sizing after tab is visible."""
        if hasattr(self, 'visual_handler') and self.visual_handler:
            self.visual_handler._clear_camera_widget()
            self.visual_handler._add_camera_widget()
    
    def __del__(self):
        """Cleanup when widget is destroyed."""
        if hasattr(self, 'plotter'):
            self.plotter.close()


