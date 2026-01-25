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
        self.all_combo_results = None  # Full results array, shape (num_combinations, num_nodes)
        self.nodal_forces_result = None  # NodalForcesResult for force-based visualization
        self.deformation_result = None  # DeformationResult for displacement visualization
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
    def _on_scalar_display_changed(self, index):
        """
        Handle scalar display selection change.
        
        Switches the active scalar field displayed on the mesh based on user selection.
        Options are: Max Value, Min Value (min principal only), Combo # of Max, Combo # of Min.
        Supports both stress results and force results.
        
        Args:
            index: Index of the selected item in the scalar display combo box.
        """
        if self.current_mesh is None:
            return
        
        selected_text = self.scalar_display_combo.currentText()
        
        # Map display text to mesh array names - stress results
        scalar_map = {
            "Max Value": "Max_Stress",
            "Min Value": "Min_Stress",
            "Combo # of Max": "Combo_of_Max",
            "Combo # of Min": "Combo_of_Min",
        }
        
        # Map display text to mesh array names - force results (fallback)
        force_scalar_map = {
            "Max Value": "Max_Force_Magnitude",
            "Min Value": "Min_Force_Magnitude",
            "Combo # of Max": "Combo_of_Max",
            "Combo # of Min": "Combo_of_Min",
        }
        
        # Try stress array first, then force array
        array_name = scalar_map.get(selected_text)
        is_force_result = False
        if array_name not in self.current_mesh.array_names:
            array_name = force_scalar_map.get(selected_text)
            is_force_result = True
        
        if array_name and array_name in self.current_mesh.array_names:
            # Update active scalars
            self.current_mesh.set_active_scalars(array_name)
            
            # Generate descriptive legend title
            if "Combo_of_Max" in array_name or "Combo_of_Min" in array_name:
                # For combination index display, use simple name without units
                legend_title = array_name
            elif is_force_result:
                # Force results
                if "Max" in array_name:
                    legend_title = "Max_Force [N]"
                else:
                    legend_title = "Min_Force [N]"
            else:
                # Stress results - use descriptive title
                stress_label = self._get_stress_type_label()
                if "Max" in array_name:
                    legend_title = f"Max_{stress_label} [MPa]"
                else:
                    legend_title = f"Min_{stress_label} [MPa]"
            
            self.data_column = legend_title
            self.state.data_column = legend_title
            
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
            
            print(f"DisplayTab: Scalar display changed to '{selected_text}' (legend: {legend_title})")
    
    @pyqtSlot(int)
    def _on_view_combination_changed(self, index):
        """
        Handle view combination selection change.
        
        When a specific combination is selected, updates the mesh with that
        combination's results. When "(Envelope View)" is selected, reverts to
        showing the envelope data (Max/Min across all combinations).
        
        Args:
            index: Index of the selected item (0 = Envelope View, 1+ = specific combination).
        """
        if self.current_mesh is None:
            return
        
        if index == 0:
            # Envelope View - show the Max/Min envelope data
            self._show_envelope_view()
            # Re-enable the scalar display combo for envelope options
            self.scalar_display_combo.setEnabled(True)
            self.scalar_display_label.setEnabled(True)
        else:
            # Specific combination selected
            combo_idx = index - 1  # Account for "(Envelope View)" being at index 0
            self._show_specific_combination(combo_idx)
            # Disable the scalar display combo since we're showing specific combination
            self.scalar_display_combo.setEnabled(False)
            self.scalar_display_label.setEnabled(False)
    
    def _show_envelope_view(self):
        """
        Switch visualization back to envelope view (Max/Min across combinations).
        
        Restores the display to show envelope data based on current scalar_display_combo selection.
        """
        if self.current_mesh is None:
            return
        
        # For force results, keep force component controls visible in envelope view
        # For stress results, hide them (stress doesn't have component selection)
        if self.nodal_forces_result is not None and self.all_combo_results is None:
            # Force results - keep controls visible
            has_beam_nodes = self.nodal_forces_result.has_beam_nodes
            self._show_force_component_controls(True, has_beam_nodes)
        else:
            # Stress results or no results - hide force controls
            self._show_force_component_controls(False)
        
        # Show displacement controls if deformation result is available
        if self.deformation_result is not None:
            self._show_displacement_component_controls(True)
        else:
            self._show_displacement_component_controls(False)
        
        # Trigger the scalar display handler to refresh the view
        self._on_scalar_display_changed(self.scalar_display_combo.currentIndex())
        print("DisplayTab: Switched to Envelope View")
    
    def _show_specific_combination(self, combo_idx: int):
        """
        Update visualization to show results for a specific combination.
        
        Handles both stress results (from all_combo_results) and force results
        (from nodal_forces_result).
        
        Args:
            combo_idx: Index of the combination to display (0-based).
        """
        if self.current_mesh is None:
            print(f"DisplayTab: Cannot show combination {combo_idx} - no mesh available")
            return
        
        combo_name = self.combination_names[combo_idx] if combo_idx < len(self.combination_names) else f"Combination {combo_idx + 1}"
        is_force_result = False
        
        # Try stress results first
        if self.all_combo_results is not None:
            if combo_idx < 0 or combo_idx >= self.all_combo_results.shape[0]:
                print(f"DisplayTab: Invalid combination index {combo_idx}")
                return
            
            # Get the stress values for this specific combination
            # all_combo_results shape is (num_combinations, num_nodes)
            combo_values = self.all_combo_results[combo_idx, :]
            array_name = f"Combo_{combo_idx}_Stress"
            
        # Try force results
        elif self.nodal_forces_result is not None:
            try:
                # Set up all force component arrays for this combination
                self._setup_force_component_arrays(combo_idx)
                
                # Get force magnitude as default display
                combo_values = self.nodal_forces_result.get_force_magnitude(combo_idx)
                array_name = 'Force_Magnitude'  # Use consistent array name
                is_force_result = True
                
                # Show force component controls
                has_beam_nodes = self.nodal_forces_result.has_beam_nodes
                self._show_force_component_controls(True, has_beam_nodes)
            except (ValueError, IndexError) as e:
                print(f"DisplayTab: Cannot get force magnitude for combination {combo_idx}: {e}")
                return
        else:
            print(f"DisplayTab: Cannot show combination {combo_idx} - no data available")
            return
        
        # Add this data to the mesh (for stress results - force arrays already set up above)
        if not is_force_result:
            array_name = f"Combo_{combo_idx}_Stress"
            self.current_mesh[array_name] = combo_values
            # Hide force component controls for stress results
            self._show_force_component_controls(False)
        self.current_mesh.set_active_scalars(array_name)
        
        # Show displacement controls if deformation result is available
        if self.deformation_result is not None:
            self._show_displacement_component_controls(True)
            # Also set up displacement component arrays for this combination
            self._setup_displacement_component_arrays(combo_idx)
        else:
            self._show_displacement_component_controls(False)
        
        # Generate descriptive legend title
        if is_force_result:
            force_unit = self.nodal_forces_result.force_unit
            legend_title = f"Force_Magnitude [{force_unit}]"
        else:
            stress_label = self._get_stress_type_label()
            legend_title = f"Combo_{combo_idx + 1}_{stress_label} [MPa]"
        
        self.data_column = legend_title
        self.state.data_column = legend_title
        
        # Update scalar range based on this combination's data
        data_min = float(np.min(combo_values))
        data_max = float(np.max(combo_values))
        
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
        
        print(f"DisplayTab: Showing results for '{combo_name}' (index {combo_idx})")
    
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
        Update force component dropdown options based on beam presence.
        
        Args:
            has_beam_nodes: True if selection contains nodes attached to beam elements.
        """
        self.force_component_combo.blockSignals(True)
        self.force_component_combo.clear()
        
        # Base options always available
        items = ["Magnitude", "FX", "FY", "FZ"]
        
        # Add Shear option only if beam nodes are present
        if has_beam_nodes:
            items.append("Shear (FY²+FZ²)^½")
        
        self.force_component_combo.addItems(items)
        self.force_component_combo.setCurrentIndex(0)  # Default to Magnitude
        self.force_component_combo.blockSignals(False)
    
    def _setup_force_component_arrays(self, combo_idx: int):
        """
        Add force component arrays to mesh for the selected combination.
        
        This sets up FX, FY, FZ, Magnitude, and optionally Shear arrays
        on the current mesh for the specified combination.
        
        Args:
            combo_idx: Index of the combination to set up arrays for.
        """
        if self.nodal_forces_result is None or self.current_mesh is None:
            return
        
        result = self.nodal_forces_result
        
        # Get force components for this combination
        fx = result.all_combo_fx[combo_idx, :]
        fy = result.all_combo_fy[combo_idx, :]
        fz = result.all_combo_fz[combo_idx, :]
        
        # Add arrays to mesh
        self.current_mesh['FX'] = fx
        self.current_mesh['FY'] = fy
        self.current_mesh['FZ'] = fz
        self.current_mesh['Force_Magnitude'] = np.sqrt(fx**2 + fy**2 + fz**2)
        
        # Add shear force if beam nodes present
        if result.has_beam_nodes:
            self.current_mesh['Shear_Force'] = np.sqrt(fy**2 + fz**2)
    
    def _setup_displacement_component_arrays(self, combo_idx: int):
        """
        Add displacement component arrays to mesh for the selected combination.
        
        This sets up UX, UY, UZ, and U_mag arrays on the current mesh for 
        the specified combination.
        
        Args:
            combo_idx: Index of the combination to set up arrays for.
        """
        if self.deformation_result is None or self.current_mesh is None:
            return
        
        result = self.deformation_result
        
        # Validate combination index
        if combo_idx < 0 or combo_idx >= result.num_combinations:
            print(f"DisplayTab: Invalid combination index {combo_idx} for displacement")
            return
        
        # Get displacement components for this combination
        ux = result.all_combo_ux[combo_idx, :]
        uy = result.all_combo_uy[combo_idx, :]
        uz = result.all_combo_uz[combo_idx, :]
        
        # Add arrays to mesh
        self.current_mesh['UX'] = ux
        self.current_mesh['UY'] = uy
        self.current_mesh['UZ'] = uz
        self.current_mesh['U_mag'] = np.sqrt(ux**2 + uy**2 + uz**2)
    
    @pyqtSlot(int)
    def _on_force_component_changed(self, index: int):
        """
        Handle force component selection change.
        
        Switches the displayed force component (FX, FY, FZ, Magnitude, or Shear).
        
        Args:
            index: New component index from the dropdown.
        """
        if self.current_mesh is None:
            return
        
        # Map index to array name
        component_map = {
            0: 'Force_Magnitude',
            1: 'FX',
            2: 'FY',
            3: 'FZ',
            4: 'Shear_Force'  # Only present if beam nodes exist
        }
        
        array_name = component_map.get(index, 'Force_Magnitude')
        
        # For envelope view, Force_Magnitude might be stored as Max_Force_Magnitude
        if array_name == 'Force_Magnitude' and 'Force_Magnitude' not in self.current_mesh.array_names:
            if 'Max_Force_Magnitude' in self.current_mesh.array_names:
                array_name = 'Max_Force_Magnitude'
        
        # Check if array exists in mesh
        if array_name not in self.current_mesh.array_names:
            print(f"DisplayTab: Array '{array_name}' not found in mesh. Available: {self.current_mesh.array_names}")
            return
        
        # Set active scalars and update visualization
        self.current_mesh.set_active_scalars(array_name)
        
        # Update legend title
        force_unit = "N"
        if self.nodal_forces_result is not None:
            force_unit = self.nodal_forces_result.force_unit
        
        # Use display-friendly names for legend
        display_name = array_name
        if array_name == 'Max_Force_Magnitude':
            display_name = 'Force_Magnitude'
        
        component_labels = {
            'Force_Magnitude': f'Force_Magnitude [{force_unit}]',
            'Max_Force_Magnitude': f'Force_Magnitude [{force_unit}]',
            'FX': f'FX [{force_unit}]',
            'FY': f'FY [{force_unit}]',
            'FZ': f'FZ [{force_unit}]',
            'Shear_Force': f'Shear_Force [{force_unit}]'
        }
        
        self.data_column = component_labels.get(array_name, f'{array_name} [{force_unit}]')
        self.state.data_column = self.data_column
        
        # Update scalar range based on selected component's data
        data = self.current_mesh[array_name]
        data_min = float(np.min(data))
        data_max = float(np.max(data))
        
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
        
        print(f"DisplayTab: Switched to force component '{display_name}'")
    
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
    
    def _on_displacement_component_changed(self, index: int):
        """
        Handle displacement component selection change.
        
        Switches the displayed displacement component (UX, UY, UZ, or U_mag).
        
        Args:
            index: New component index from the dropdown.
        """
        if self.current_mesh is None or self.deformation_result is None:
            return
        
        # Map index to array name
        # Combo order: U_mag, UX, UY, UZ
        component_map = {
            0: 'U_mag',
            1: 'UX',
            2: 'UY',
            3: 'UZ'
        }
        
        array_name = component_map.get(index, 'U_mag')
        
        # Check if array exists in mesh
        if array_name not in self.current_mesh.array_names:
            print(f"DisplayTab: Array '{array_name}' not found in mesh. Available: {self.current_mesh.array_names}")
            return
        
        # Set active scalars and update visualization
        self.current_mesh.set_active_scalars(array_name)
        
        # Update legend title
        disp_unit = "mm"
        if self.deformation_result is not None:
            disp_unit = self.deformation_result.displacement_unit
        
        component_labels = {
            'U_mag': f'U_mag [{disp_unit}]',
            'UX': f'UX [{disp_unit}]',
            'UY': f'UY [{disp_unit}]',
            'UZ': f'UZ [{disp_unit}]'
        }
        
        self.data_column = component_labels.get(array_name, f'{array_name} [{disp_unit}]')
        self.state.data_column = self.data_column
        
        # Update scalar range based on selected component's data
        data = self.current_mesh[array_name]
        data_min = float(np.min(data))
        data_max = float(np.max(data))
        
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
        
        print(f"DisplayTab: Switched to displacement component '{array_name}'")
    
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
        
        # Populate scalar display options if this is batch solve result
        has_min_data = "Min_Stress" in mesh.array_names
        has_force_data = "Max_Force_Magnitude" in mesh.array_names
        
        if "Max_Stress" in mesh.array_names or has_min_data:
            # Stress results
            self.populate_scalar_display_options(result_type or "von_mises", has_min_data)
            
            # Populate view combination dropdown if we have combination names and all results
            if self.combination_names and self.all_combo_results is not None:
                self.populate_view_combination_options(self.combination_names)
            else:
                # Hide view combination controls if no individual results available
                self.view_combination_label.setVisible(False)
                self.view_combination_combo.setVisible(False)
        elif has_force_data:
            # Force results - use simplified options
            self.scalar_display_combo.blockSignals(True)
            self.scalar_display_combo.clear()
            self.scalar_display_combo.addItem("Max Value")
            self.scalar_display_combo.addItem("Combo # of Max")
            if "Min_Force_Magnitude" in mesh.array_names:
                self.scalar_display_combo.addItem("Min Value")
                self.scalar_display_combo.addItem("Combo # of Min")
            self.scalar_display_combo.setCurrentIndex(0)
            self.scalar_display_combo.blockSignals(False)
            self.scalar_display_label.setVisible(True)
            self.scalar_display_combo.setVisible(True)
            
            # Populate view combination dropdown for forces
            if self.combination_names and self.nodal_forces_result is not None:
                self.populate_view_combination_options(self.combination_names)
                
                # Show force component controls (with Shear option if beam nodes present)
                has_beam_nodes = self.nodal_forces_result.has_beam_nodes
                self._show_force_component_controls(True, has_beam_nodes)
            else:
                self.view_combination_label.setVisible(False)
                self.view_combination_combo.setVisible(False)
                self._show_force_component_controls(False)
        else:
            # Hide scalar display controls for non-batch results
            self.scalar_display_label.setVisible(False)
            self.scalar_display_combo.setVisible(False)
            self.view_combination_label.setVisible(False)
            self.view_combination_combo.setVisible(False)
        
        # Handle deformation controls visibility and deformation scale controls
        has_deformation = "Max_U_mag" in mesh.array_names or "U_mag" in mesh.array_names
        if self.deformation_result is not None or has_deformation:
            self._show_displacement_component_controls(True)
            # Show deformation scale controls
            self.deformation_scale_label.setVisible(True)
            self.deformation_scale_edit.setVisible(True)
            
            # If this is a deformation-only result, also populate view combination options
            if has_deformation and "Max_Stress" not in mesh.array_names and not has_force_data:
                # Deformation-only result - set up scalar display options for displacement
                self.scalar_display_combo.blockSignals(True)
                self.scalar_display_combo.clear()
                self.scalar_display_combo.addItem("Max Value")
                self.scalar_display_combo.addItem("Combo # of Max")
                if "Min_U_mag" in mesh.array_names:
                    self.scalar_display_combo.addItem("Min Value")
                    self.scalar_display_combo.addItem("Combo # of Min")
                self.scalar_display_combo.setCurrentIndex(0)
                self.scalar_display_combo.blockSignals(False)
                self.scalar_display_label.setVisible(True)
                self.scalar_display_combo.setVisible(True)
                
                # Populate view combination dropdown for deformation
                if self.combination_names and self.deformation_result is not None:
                    self.populate_view_combination_options(self.combination_names)
        else:
            self._show_displacement_component_controls(False)
            # Hide deformation scale controls if no deformation
            self.deformation_scale_label.setVisible(False)
            self.deformation_scale_edit.setVisible(False)
        
        # Update the visualization
        self.update_visualization()
        
        # Clear file path since this is computed data, not loaded from file
        self.file_path.clear()
        
        # Show IC export button if velocity components are present
        if all(key in mesh.array_names for key in ['vel_x', 'vel_y', 'vel_z']):
            self.extract_ic_button.setVisible(True)
        else:
            self.extract_ic_button.setVisible(False)
        
        # Update Export Output CSV button visibility
        self._update_export_output_button_visibility()
    
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
        if self.anim_timer is not None:
            self.anim_timer.stop()
        if hasattr(self, 'plotter'):
            self.plotter.close()
