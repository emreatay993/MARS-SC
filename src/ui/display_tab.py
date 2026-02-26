"""
Display tab: 3D visualization of FEA results (PyVista). UI is built via
DisplayTabUIBuilder; visualization/hotspots live in handler classes.
"""

import numpy as np

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
from ui.handlers.display_contour_sync_handler import DisplayContourSyncHandler
from ui.handlers.display_recompute_policy import (
    get_recompute_button_text,
    get_recompute_cached_note_text,
    get_recompute_note_text,
    is_stress_on_demand_recompute_available as stress_recompute_available,
)
from ui.visualization_data import VisualizationData, SolverOutputFlags


class DisplayTab(QWidget):
    """
    3D FEA results view. Builders wire up the UI; VisualizationManager and HotspotDetector do the heavy lifting.

    Signals: node_picked_signal(int), node_picked_for_history_popup(int),
    recompute_corrected_combination_requested(int).
    """
    
    # Signals
    node_picked_signal = pyqtSignal(int)
    node_picked_for_history_popup = pyqtSignal(int)
    recompute_corrected_combination_requested = pyqtSignal(int)
    
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
        self.stress_result = None  # CombinationResult for stress-based visualization/export
        self.all_combo_results = None  # Full results array, shape (num_combinations, num_nodes)
        self.nodal_forces_result = None  # NodalForcesResult for force-based visualization
        self.deformation_result = None  # DeformationResult for displacement visualization
        self.output_flags = SolverOutputFlags()
        self.current_contour_type = None  # "Stress", "Forces", or "Deformation"
        self.recomputed_stress_combo_cache = {}  # combo_idx -> corrected stress array
        self._recompute_pending_combo = None
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
        self.recompute_combo_note = self.components.get('recompute_combo_note')
        self.recompute_combo_button = self.components.get('recompute_combo_button')
        self._fit_combo_popup_to_contents(self.view_combination_combo)
        
        # Force component controls (for nodal forces visualization)
        self.force_component_label = self.components['force_component_label']
        self.force_component_combo = self.components['force_component_combo']
        self.export_forces_button = self.components['export_forces_button']
        
        # Displacement component controls (for deformation visualization)
        self.displacement_component_label = self.components.get('displacement_component_label')
        self.displacement_component_combo = self.components.get('displacement_component_combo')
        self.export_output_button = self.components.get('export_output_button')
        
        # Time point controls
        self.time_point_spinbox = self.components['time_point_spinbox']
        self.save_time_button = self.components['save_time_button']
        self.time_point_group = self.components['time_point_group']

    def _fit_combo_popup_to_contents(self, combo):
        """
        Ensure combo popup rows show full item text (no forced ellipsis).

        The combo widget itself can stay compact; the popup width is expanded
        to fit the longest entry and a horizontal scrollbar is enabled as a fallback.
        """
        if combo is None:
            return

        view = combo.view()
        if view is None:
            return

        view.setTextElideMode(Qt.ElideNone)
        view.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        if combo.count() == 0:
            return

        font_metrics = combo.fontMetrics()
        longest_item_width = max(
            font_metrics.horizontalAdvance(combo.itemText(i))
            for i in range(combo.count())
        )

        # Include popup paddings/checkmark area so full text is visible.
        popup_width = max(combo.width(), longest_item_width + 48)
        view.setMinimumWidth(popup_width)
    
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
        if self.recompute_combo_button is not None:
            self.recompute_combo_button.clicked.connect(
                self._on_recompute_combo_clicked
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
        self.save_time_button.clicked.connect(self.save_time_point_results)
        
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
    
    @pyqtSlot(int)
    def _on_contour_type_changed(self, index):
        """Handle contour family selection changes."""
        self.contour_sync_handler.on_contour_type_changed(index)
        self._update_recompute_combo_controls()
    
    @pyqtSlot(int)
    def _on_scalar_display_changed(self, index):
        """
        Handle scalar display selection change.
        """
        self.contour_sync_handler.on_scalar_display_changed(index)
        self._update_recompute_combo_controls()
    
    @pyqtSlot(int)
    def _on_view_combination_changed(self, index):
        """
        Handle view combination selection change.
        """
        self.contour_sync_handler.on_view_combination_changed(index)
        self._update_recompute_combo_controls()

    @pyqtSlot(bool)
    def _on_recompute_combo_clicked(self, checked=False):
        """Request an on-demand corrected recompute for the selected combination."""
        _ = checked
        if self.view_combination_combo is None:
            return

        view_idx = self.view_combination_combo.currentIndex()
        combo_idx = view_idx - 1
        if combo_idx < 0:
            return

        self._recompute_pending_combo = combo_idx
        if self.recompute_combo_button is not None:
            self.recompute_combo_button.setEnabled(False)
            self.recompute_combo_button.setText("Recomputing...")
        self.recompute_corrected_combination_requested.emit(combo_idx)

    def is_stress_on_demand_recompute_available(self) -> bool:
        """Whether current stress data supports on-demand combination recompute."""
        return stress_recompute_available(self.stress_result)

    def get_recomputed_stress_values(self, combo_idx: int):
        """Return cached on-demand corrected stress values for a combination if present."""
        return self.recomputed_stress_combo_cache.get(int(combo_idx))

    @pyqtSlot(object)
    def on_recomputed_combination_ready(self, payload):
        """Receive on-demand corrected combination data from Solver tab."""
        combo_idx = int(payload.get("combination_index", -1))
        success = bool(payload.get("success", False))

        if self.recompute_combo_button is not None:
            self.recompute_combo_button.setEnabled(True)
            self.recompute_combo_button.setText(get_recompute_button_text(self.stress_result))

        if success:
            values = np.asarray(payload.get("stress_values", []), dtype=float).reshape(-1)
            if self.current_mesh is not None and values.size == self.current_mesh.n_points:
                self.recomputed_stress_combo_cache[combo_idx] = values
                self.current_mesh[f"Combo_{combo_idx}_Stress"] = values
                self.contour_sync_handler.sync_from_current_state()
            elif self.recompute_combo_note is not None:
                self.recompute_combo_note.setVisible(True)
                self.recompute_combo_note.setText(
                    "Recompute completed but node count did not match current display mesh."
                )
        elif self.recompute_combo_note is not None:
            error_msg = payload.get("error", "On-demand recompute failed.")
            self.recompute_combo_note.setVisible(True)
            self.recompute_combo_note.setText(str(error_msg))

        self._recompute_pending_combo = None
        self._update_recompute_combo_controls()

    def _update_recompute_combo_controls(self):
        """Update visibility/state for on-demand corrected-combination controls."""
        note = self.recompute_combo_note
        button = self.recompute_combo_button
        if note is None or button is None:
            return

        available = self.is_stress_on_demand_recompute_available()
        if not available:
            note.setVisible(False)
            button.setVisible(False)
            return

        note.setVisible(True)
        note.setText(get_recompute_note_text(self.stress_result))

        view_idx = self.view_combination_combo.currentIndex() if self.view_combination_combo is not None else 0
        combo_idx = view_idx - 1
        if combo_idx < 0:
            button.setVisible(False)
            return

        button.setVisible(True)
        cached = combo_idx in self.recomputed_stress_combo_cache
        if cached:
            note.setText(get_recompute_cached_note_text(self.stress_result))
        if self._recompute_pending_combo == combo_idx:
            button.setEnabled(False)
            button.setText("Recomputing...")
        else:
            button.setEnabled(True)
            button.setText(get_recompute_button_text(self.stress_result))
    
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
        - Stress results
        - Nodal forces results
        - Deformation results
        """
        if self.export_output_button is None:
            return
        
        # Check for any available results
        has_stress = self.stress_result is not None
        has_forces = self.nodal_forces_result is not None
        has_deformation = self.deformation_result is not None
        
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

    @pyqtSlot(object)
    def update_view_with_payload(self, payload):
        """Update visualization state from typed solver data."""
        if payload is None:
            return
        if not isinstance(payload, VisualizationData):
            QMessageBox.warning(self, "Invalid Data", "Display data has an unexpected type.")
            return

        self.stress_result = payload.stress_result
        self.all_combo_results = (
            payload.stress_result.all_combo_results if payload.stress_result is not None else None
        )
        self.nodal_forces_result = payload.forces_result
        self.deformation_result = payload.deformation_result
        self.combination_names = list(payload.combination_names or [])
        self.output_flags = payload.output_flags or SolverOutputFlags()
        self.recomputed_stress_combo_cache = {}
        self._recompute_pending_combo = None

        self.update_view_with_results(
            payload.mesh,
            payload.scalar_bar_title,
            payload.data_min,
            payload.data_max,
        )
    
    @pyqtSlot(bool)
    def save_time_point_results(self, checked=False):
        """Save currently displayed results to CSV."""
        self.export_handler.save_time_point_results()

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
        
        # Keep all-combination stress arrays aligned with cached stress result.
        self.all_combo_results = (
            self.stress_result.all_combo_results if self.stress_result is not None else None
        )

        # Store original coordinates for deformation scaling.
        if self.deformation_result is not None:
            self.original_node_coords = mesh.points.copy()
        
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
        self._update_recompute_combo_controls()
        
        # Clear file path since this is computed data, not loaded from file
        self.file_path.clear()

        # Update Export Output CSV button visibility
        self._update_export_output_button_visibility()

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
        if self.visual_handler:
            self.visual_handler._clear_camera_widget()
            self.visual_handler._add_camera_widget()
    
    def __del__(self):
        """Cleanup when widget is destroyed."""
        try:
            if self.plotter:
                self.plotter.close()
        except Exception:
            pass
