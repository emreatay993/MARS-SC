"""
Refactored Display Tab for 3D visualization.

This module provides the DisplayTab widget that handles 3D visualization of FEA
results using PyVista. The class has been refactored to use UI builders and
delegate complex logic to manager classes.
"""

import numpy as np
import pyvista as pv

from PyQt5.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QMessageBox, QWidget, QStyle

# Import builders and managers
from ui.builders.display_ui import DisplayTabUIBuilder
from core.visualization import VisualizationManager, AnimationManager, HotspotDetector
from ui.handlers.display_state import DisplayState
from ui.handlers.display_file_handler import DisplayFileHandler
from ui.handlers.display_visualization_handler import DisplayVisualizationHandler
from ui.handlers.display_animation_handler import DisplayAnimationHandler
from ui.handlers.display_interaction_handler import DisplayInteractionHandler
from ui.handlers.display_export_handler import DisplayExportHandler
from ui.handlers.display_results_handler import DisplayResultsHandler


class DisplayTab(QWidget):
    """
    Refactored Display Tab for 3D visualization of FEA results.
    
    This class uses UI builders for construction and delegates complex logic
    to manager classes (VisualizationManager, AnimationManager, HotspotDetector).
    
    Signals:
        node_picked_signal: Emitted when a node is picked (int: node_id)
        time_point_update_requested: Emitted when time point update is needed
        combination_update_requested: Emitted when combination update is needed (MARS-SC)
        animation_precomputation_requested: Emitted when animation precomputation is needed
    """
    
    # Signals
    node_picked_signal = pyqtSignal(int)
    time_point_update_requested = pyqtSignal(float, dict)
    combination_update_requested = pyqtSignal(int, dict)  # (combo_index, options)
    animation_precomputation_requested = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        """Initialize the Display Tab."""
        super().__init__(parent)
        
        # Managers for complex logic
        self.viz_manager = VisualizationManager()
        self.anim_manager = AnimationManager()
        self.hotspot_detector = HotspotDetector()

        # Shared state and handler scaffolding
        self.state = DisplayState()
        self.file_handler = DisplayFileHandler(self, self.state, self.viz_manager)
        self.visual_handler = DisplayVisualizationHandler(self, self.state, self.viz_manager)
        self.animation_handler = DisplayAnimationHandler(self, self.state, self.anim_manager)
        self.interaction_handler = DisplayInteractionHandler(self, self.state, self.hotspot_detector)
        self.export_handler = DisplayExportHandler(self, self.state)
        self.results_handler = DisplayResultsHandler(self, self.state, self.visual_handler)
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
        self.anim_timer = None
        self.time_text_actor = None
        self.current_anim_time = 0.0
        self.animation_paused = False
        self.temp_solver = None
        self.time_values = None
        self.original_node_coords = None
        self.last_valid_deformation_scale = 1.0
        self.current_anim_frame_index = 0
        self.is_deformation_included_in_anim = False
        self.state.current_anim_frame_index = 0
        self.state.is_deformation_included_in_anim = False
        
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
        self.freeze_tracked_node = False
        self.freeze_baseline = None
        
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
        self.deformation_scale_label = self.components['deformation_scale_label']
        self.deformation_scale_edit = self.components['deformation_scale_edit']
        self.absolute_deformation_checkbox = self.components['absolute_deformation_checkbox']
        
        # Combination/Time point controls (MARS-SC: uses combination_combo)
        self.combination_combo = self.components.get('combination_combo')
        self.time_point_spinbox = self.components['time_point_spinbox']
        self.update_time_button = self.components['update_time_button']
        self.save_time_button = self.components['save_time_button']
        self.extract_ic_button = self.components['extract_ic_button']
        self.time_point_group = self.components['time_point_group']
        
        # Animation controls (hidden in MARS-SC - static combination results)
        self.anim_interval_spin = self.components['anim_interval_spin']
        self.anim_start_spin = self.components['anim_start_spin']
        self.anim_end_spin = self.components['anim_end_spin']
        self.play_button = self.components['play_button']
        self.pause_button = self.components['pause_button']
        self.stop_button = self.components['stop_button']
        self.time_step_mode_combo = self.components['time_step_mode_combo']
        self.custom_step_spin = self.components['custom_step_spin']
        self.actual_interval_spin = self.components['actual_interval_spin']
        self.save_anim_button = self.components['save_anim_button']
        self.anim_group = self.components['anim_group']

        # Apply standard media icons so the playback controls are visually recognisable
        style = self.style()
        self.play_button.setIcon(style.standardIcon(QStyle.SP_MediaPlay))
        self.pause_button.setIcon(style.standardIcon(QStyle.SP_MediaPause))
        self.stop_button.setIcon(style.standardIcon(QStyle.SP_MediaStop))
    
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
        
        # Animation controls (kept for backwards compatibility, hidden in MARS-SC)
        self.anim_start_spin.valueChanged.connect(self._update_anim_range_min)
        self.anim_end_spin.valueChanged.connect(self._update_anim_range_max)
        self.play_button.clicked.connect(self.start_animation)
        self.pause_button.clicked.connect(self.pause_animation)
        self.stop_button.clicked.connect(self.stop_animation)
        self.time_step_mode_combo.currentTextChanged.connect(
            self._update_step_spinbox_state
        )
        self.save_anim_button.clicked.connect(self.save_animation)
        
        # Context menu
        self.plotter.customContextMenuRequested.connect(self.show_context_menu)
    
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
    def _on_scalar_display_changed(self, index):
        """
        Handle scalar display selection change.
        
        Switches the active scalar field displayed on the mesh based on user selection.
        Options are: Max Value, Min Value (min principal only), Combo # of Max, Combo # of Min.
        
        Args:
            index: Index of the selected item in the scalar display combo box.
        """
        if self.current_mesh is None:
            return
        
        selected_text = self.scalar_display_combo.currentText()
        
        # Map display text to mesh array names
        scalar_map = {
            "Max Value": "Max_Stress",
            "Min Value": "Min_Stress",
            "Combo # of Max": "Combo_of_Max",
            "Combo # of Min": "Combo_of_Min",
        }
        
        array_name = scalar_map.get(selected_text)
        if array_name and array_name in self.current_mesh.array_names:
            # Update active scalars
            self.current_mesh.set_active_scalars(array_name)
            self.data_column = array_name
            self.state.data_column = array_name
            
            # Update scalar range based on new data
            data = self.current_mesh[array_name]
            data_min = float(np.min(data))
            data_max = float(np.max(data))
            
            # Update spinboxes
            self.scalar_min_spin.blockSignals(True)
            self.scalar_max_spin.blockSignals(True)
            self.scalar_min_spin.setRange(data_min, data_max)
            self.scalar_max_spin.setRange(data_min, 1e30)
            self.scalar_min_spin.setValue(data_min)
            self.scalar_max_spin.setValue(data_max)
            self.scalar_min_spin.blockSignals(False)
            self.scalar_max_spin.blockSignals(False)
            
            # Refresh visualization
            self.update_visualization()
            
            print(f"DisplayTab: Scalar display changed to '{selected_text}' ({array_name})")
    
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
        
        # Store result type for hover annotation
        self.current_result_type = result_type
        
        # Show the scalar display controls
        self.scalar_display_label.setVisible(True)
        self.scalar_display_combo.setVisible(True)
    
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
        
        # Animation time range (kept for backwards compatibility, hidden in MARS-SC)
        self.anim_start_spin.setRange(min_time, max_time)
        self.anim_end_spin.setRange(min_time, max_time)
        self.anim_start_spin.setValue(min_time)
        self.anim_end_spin.setValue(max_time)
        self.actual_interval_spin.setMaximum(len(time_values))
        self.actual_interval_spin.setValue(1)
        
        # Show controls - Note: anim_group stays hidden in MARS-SC
        # self.anim_group.setVisible(True)  # Hidden for MARS-SC
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
            self.absolute_deformation_checkbox.setVisible(True)
        else:
            # Hide deformation controls when deformations are not loaded
            self.deformation_scale_label.setVisible(False)
            self.deformation_scale_edit.setVisible(False)
            self.deformation_scale_edit.setEnabled(False)
            self.deformation_scale_edit.setText("0")
            self.absolute_deformation_checkbox.setVisible(False)
    
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
    
    @pyqtSlot(float)
    def _update_anim_range_min(self, value):
        """Ensure animation end time is not less than start time."""
        self.anim_end_spin.setMinimum(value)
    
    @pyqtSlot(float)
    def _update_anim_range_max(self, value):
        """Ensure animation start time does not exceed end time."""
        self.anim_start_spin.setMaximum(value)
    
    @pyqtSlot(str)
    def _update_step_spinbox_state(self, text):
        """Toggle between custom and actual time step modes."""
        if text == "Custom Time Step":
            self.custom_step_spin.setVisible(True)
            self.actual_interval_spin.setVisible(False)
        else:
            self.custom_step_spin.setVisible(False)
            self.actual_interval_spin.setVisible(True)
    
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
    
    @pyqtSlot(bool)
    def start_animation(self, checked=False):
        """Start animation playback or resume if paused."""
        self.animation_handler.start_animation()
    
    def _estimate_animation_ram(self, num_nodes, num_anim_steps, include_deformation):
        """Estimate peak RAM needed for animation precomputation in GB."""
        return self.animation_handler.estimate_animation_ram(
            num_nodes, num_anim_steps, include_deformation
        )
    
    @pyqtSlot(bool)
    def pause_animation(self, checked=False):
        """Pause animation playback."""
        self.animation_handler.pause_animation()
    
    @pyqtSlot(bool)
    def stop_animation(self, checked=False):
        """Stop animation, release precomputed data, and reset state."""
        self.animation_handler.stop_animation()
    
    @pyqtSlot(bool)
    def save_animation(self, checked=False):
        """Save animation to file (MP4 or GIF)."""
        self.animation_handler.save_animation()
    
    def _get_save_path_and_format(self):
        """Delegate to animation handler for backwards compatibility."""
        return self.animation_handler.get_save_path_and_format()

    def _write_animation_to_file(self, file_path, file_format):
        """Delegate to animation handler for backwards compatibility."""
        return self.animation_handler.write_animation_to_file(file_path, file_format)
    
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
        # Update current mesh and track state
        self.current_mesh = mesh
        self.state.current_mesh = mesh
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
        
        # Get combination names from solver tab if available
        try:
            solver_tab = self.window().solver_tab
            if solver_tab and solver_tab.combination_table:
                self.combination_names = solver_tab.combination_table.combination_names
            else:
                self.combination_names = []
        except (AttributeError, RuntimeError):
            self.combination_names = []
        
        # Populate scalar display options if this is batch solve result
        has_min_data = "Min_Stress" in mesh.array_names
        if "Max_Stress" in mesh.array_names or has_min_data:
            self.populate_scalar_display_options(result_type or "von_mises", has_min_data)
        else:
            # Hide scalar display controls for non-batch results
            self.scalar_display_label.setVisible(False)
            self.scalar_display_combo.setVisible(False)
        
        # Update the visualization
        self.update_visualization()
        
        # Clear file path since this is computed data, not loaded from file
        self.file_path.clear()
        
        # Show IC export button if velocity components are present
        if all(key in mesh.array_names for key in ['vel_x', 'vel_y', 'vel_z']):
            self.extract_ic_button.setVisible(True)
        else:
            self.extract_ic_button.setVisible(False)
    
    @pyqtSlot(object)
    def on_animation_data_ready(self, precomputed_data):
        """Receive precomputed animation data and start playback."""
        from PyQt5.QtWidgets import QApplication
        QApplication.restoreOverrideCursor()
        
        if precomputed_data is None:
            print("Animation precomputation failed. See console for details.")
            self.stop_animation()
            return
        
        print("DisplayTab: Received precomputed animation data. Starting playback.")
        
        # Unpack data
        (precomputed_scalars, precomputed_coords, precomputed_anim_times, 
         data_column_name, is_deformation_included) = precomputed_data
        
        # Store in animation manager
        self.anim_manager.precomputed_scalars = precomputed_scalars
        self.anim_manager.precomputed_coords = precomputed_coords
        self.anim_manager.precomputed_anim_times = precomputed_anim_times
        self.anim_manager.data_column_name = data_column_name
        self.anim_manager.is_deformation_included = is_deformation_included
        self.animation_handler.set_state_attr(
            "is_deformation_included_in_anim", is_deformation_included
        )
        
        # Update data column for scalar bar title and hover annotation
        self.data_column = data_column_name
        self.state.data_column = data_column_name
        
        # Update scalar range spinboxes based on precomputed data
        data_min = np.min(precomputed_scalars)
        data_max = np.max(precomputed_scalars)
        self.scalar_min_spin.blockSignals(True)
        self.scalar_max_spin.blockSignals(True)
        self.scalar_min_spin.setRange(data_min, data_max)
        self.scalar_max_spin.setRange(data_min, 1e30)
        self.scalar_min_spin.setValue(data_min)
        self.scalar_max_spin.setValue(data_max)
        self.scalar_min_spin.blockSignals(False)
        self.scalar_max_spin.blockSignals(False)
        
        # Update the current mesh with first frame data and rebuild visualization
        # This ensures scalar bar title and range are updated
        self.animation_handler.set_state_attr("current_anim_frame_index", 0)
        self.animation_handler.set_state_attr("animation_paused", False)
        
        try:
            # Get first frame data
            scalars, coords, time_val = self.anim_manager.get_frame_data(0)
            
            # Update mesh scalars
            if self.current_mesh is not None:
                self.current_mesh[data_column_name] = scalars
                self.current_mesh.set_active_scalars(data_column_name)
                
                # Update coordinates if deformation is included
                if coords is not None:
                    self.current_mesh.points = coords.copy()
                    try:
                        self.current_mesh.points_modified()
                    except AttributeError:
                        self.current_mesh.GetPoints().Modified()
                
                # Rebuild visualization with new scalar bar title and range
                self.update_visualization()
            
            # Re-create tracked node markers AFTER update_visualization (which clears the plotter)
            if self.target_node_index is not None:
                try:
                    point_coords = (precomputed_coords[self.target_node_index, :, 0] 
                                   if precomputed_coords is not None 
                                   else self.current_mesh.points[self.target_node_index])
                    
                    self.marker_poly = pv.PolyData([point_coords])
                    self.interaction_handler.set_state_attr('marker_poly', self.marker_poly)
                    self.target_node_marker_actor = self.plotter.add_points(
                        self.marker_poly,
                        color='black',
                        point_size=self.point_size.value() * 2,
                        render_points_as_spheres=True,
                        opacity=0.3
                    )
                    self.interaction_handler.set_state_attr('target_node_marker_actor', self.target_node_marker_actor)
                    
                    self.label_point_data = pv.PolyData([point_coords])
                    self.interaction_handler.set_state_attr('label_point_data', self.label_point_data)
                    self.target_node_label_actor = self.plotter.add_point_labels(
                        self.label_point_data, [f"Node {self.target_node_id}"],
                        name="target_node_label",
                        font_size=16, text_color='red',
                        always_visible=True, show_points=False
                    )
                    self.interaction_handler.set_state_attr('target_node_label_actor', self.target_node_label_actor)
                    
                    # Hide if frozen
                    if self.freeze_tracked_node:
                        if self.target_node_marker_actor:
                            self.target_node_marker_actor.SetVisibility(False)
                        if self.target_node_label_actor:
                            self.target_node_label_actor.SetVisibility(False)
                        self.plotter.render()
                        
                except IndexError:
                    print("Warning: Could not re-create tracked node marker.")
                    self.interaction_handler.clear_goto_node_markers()
            
            # Now render the first frame with the time text
            self.animation_handler.animate_frame(update_index=False)
        except Exception as e:
            QMessageBox.critical(
                self, "Animation Error",
                f"Failed initial frame render: {str(e)}"
            )
            self.stop_animation()
            return
        
        self.anim_timer = QTimer(self)
        self.animation_handler.set_state_attr("anim_timer", self.anim_timer)
        self.anim_timer.timeout.connect(self.animation_handler.animate_frame)
        self.anim_timer.start(self.anim_interval_spin.value())
        
        # Update UI state
        self.deformation_scale_edit.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.stop_button.setEnabled(True)
        self.save_anim_button.setEnabled(True)
    
    def _clear_visualization(self):
        """Properly clear existing visualization."""
        self.stop_animation()
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
        if self.anim_timer is not None:
            self.anim_timer.stop()
        if hasattr(self, 'plotter'):
            self.plotter.close()
