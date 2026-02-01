"""
Visualization updates and rendering helpers for the Display tab.
"""

import time
from typing import Optional
import numpy as np
import vtk
from PyQt5.QtCore import QTimer

from ui.handlers.display_base_handler import DisplayBaseHandler
from core.visualization import VisualizationManager


class DisplayVisualizationHandler(DisplayBaseHandler):
    """Coordinates rendering operations on the PyVista plotter."""

    def __init__(self, tab, state, viz_manager: VisualizationManager):
        super().__init__(tab, state)
        self.viz_manager = viz_manager

    def apply_deformed_coordinates(self, combo_idx: Optional[int] = None) -> bool:
        """
        Apply deformed coordinates to the current mesh based on deformation scale.
        
        This modifies the mesh node positions to show the deformed shape. The scalar
        coloring (stress, force, or displacement component) is independent of this.
        
        Args:
            combo_idx: Combination index to use. If None, uses current view_combination_combo.
                       Index 0 in combo means envelope view (use max displacement).
        
        Returns:
            bool: True if deformation was applied successfully, False otherwise.
        """
        mesh = self.state.current_mesh or self.tab.current_mesh
        if mesh is None:
            return False
        
        # Check if we have deformation results
        deformation_result = getattr(self.tab, 'deformation_result', None)
        if deformation_result is None:
            return False
        
        # Get original coordinates (stored when mesh was created)
        original_coords = getattr(self.tab, 'original_node_coords', None)
        if original_coords is None:
            # Store original coordinates the first time
            self.tab.original_node_coords = mesh.points.copy()
            original_coords = self.tab.original_node_coords
        
        # Get deformation scale factor
        try:
            scale = float(self.tab.deformation_scale_edit.text())
        except (ValueError, AttributeError):
            scale = 1.0
        
        # Determine which combination to use
        if combo_idx is None:
            view_combo_idx = self.tab.view_combination_combo.currentIndex()
        else:
            view_combo_idx = combo_idx
        
        # Get displacement data
        try:
            if view_combo_idx == 0:
                # Envelope view - use displacement at max magnitude
                # For envelope, we can use the combo_of_max to get the displacement
                # at each node's maximum magnitude combination, or simply use all zeros
                # For simplicity, in envelope mode we'll show the shape at max overall
                if deformation_result.all_combo_ux is None:
                    return False
                
                # Find which combination has the maximum overall magnitude
                all_mag = np.sqrt(
                    deformation_result.all_combo_ux**2 + 
                    deformation_result.all_combo_uy**2 + 
                    deformation_result.all_combo_uz**2
                )
                max_per_combo = np.max(all_mag, axis=1)
                max_combo_idx = int(np.argmax(max_per_combo))
                
                ux = deformation_result.all_combo_ux[max_combo_idx, :]
                uy = deformation_result.all_combo_uy[max_combo_idx, :]
                uz = deformation_result.all_combo_uz[max_combo_idx, :]
            else:
                # Specific combination (subtract 1 to account for envelope at index 0)
                actual_combo_idx = view_combo_idx - 1
                if actual_combo_idx < 0 or actual_combo_idx >= deformation_result.num_combinations:
                    return False
                
                ux = deformation_result.all_combo_ux[actual_combo_idx, :]
                uy = deformation_result.all_combo_uy[actual_combo_idx, :]
                uz = deformation_result.all_combo_uz[actual_combo_idx, :]
        except (IndexError, TypeError, AttributeError) as e:
            print(f"DisplayVisualizationHandler: Error getting displacement data: {e}")
            return False
        
        # Validate array sizes match
        if len(ux) != mesh.n_points or len(original_coords) != mesh.n_points:
            print(f"DisplayVisualizationHandler: Size mismatch - mesh points: {mesh.n_points}, "
                  f"displacement: {len(ux)}, original coords: {len(original_coords)}")
            return False
        
        # Apply scaled deformation to coordinates
        # Both coordinates and displacement are in mm (MARS-SC uses mm-N-MPa unit system)
        deformed_coords = original_coords.copy()
        deformed_coords[:, 0] += scale * ux
        deformed_coords[:, 1] += scale * uy
        deformed_coords[:, 2] += scale * uz
        
        # Update mesh points
        mesh.points = deformed_coords
        
        return True
    
    def reset_to_original_coordinates(self) -> bool:
        """
        Reset mesh coordinates to original (undeformed) state.
        
        Returns:
            bool: True if reset was successful, False otherwise.
        """
        mesh = self.state.current_mesh or self.tab.current_mesh
        if mesh is None:
            return False
        
        original_coords = getattr(self.tab, 'original_node_coords', None)
        if original_coords is None:
            return False
        
        mesh.points = original_coords.copy()
        return True

    def update_visualization(self) -> None:
        """Refresh the 3D view with the current mesh."""
        mesh = self.state.current_mesh or self.tab.current_mesh
        if mesh is None:
            return

        plotter = self.tab.plotter
        plotter.clear()
        
        # Apply deformed coordinates if deformation results are available
        deformation_result = getattr(self.tab, 'deformation_result', None)
        if deformation_result is not None:
            self.apply_deformed_coordinates()

        # Use active scalars if set, otherwise fall back to first array (e.g., NodeID)
        active_scalars = mesh.active_scalars_name
        if not active_scalars and mesh.array_names:
            active_scalars = mesh.array_names[0]
        if active_scalars:
            # Preserve display labels with units when available
            if not self.tab.data_column or self.tab.data_column == "Result" or self.tab.data_column == active_scalars:
                self.state.data_column = active_scalars
                self.tab.data_column = active_scalars
            else:
                self.state.data_column = self.tab.data_column

        # Get scalar bar digit format from state
        digits = getattr(self.state, 'scalar_bar_digits', 4)
        scalar_bar_fmt = f"%.{digits}f"
        
        actor = plotter.add_mesh(
            mesh,
            scalars=active_scalars,
            point_size=self.tab.point_size.value(),
            render_points_as_spheres=True,
            show_scalar_bar=True,
            cmap="jet",
            below_color="gray",
            above_color="magenta",
            scalar_bar_args={
                "title": self.tab.data_column,
                "fmt": scalar_bar_fmt,
                "position_x": 0.04,
                "position_y": 0.35,
                "width": 0.05,
                "height": 0.5,
                "vertical": True,
                "title_font_size": 14,
                "label_font_size": 12,
                "shadow": True,
                "n_labels": 10,
                "interactive": False,
            },
        )

        self.state.current_actor = actor
        self.tab.current_actor = actor

        if self.tab.scalar_min_spin.value() != self.tab.scalar_max_spin.value():
            actor.mapper.scalar_range = (
                self.tab.scalar_min_spin.value(),
                self.tab.scalar_max_spin.value(),
            )

        self.setup_hover_annotation()

        plotter.reset_camera()
        
        # Clear old camera widget if it exists
        self._clear_camera_widget()
        
        # Force render to establish window size
        plotter.render()
        
        # Check if tab is visible - if so, add widget immediately
        # If not visible, set flag for showEvent to handle it
        if self.tab.isVisible():
            # Tab is visible, add widget with minimal delay
            QTimer.singleShot(10, self._add_camera_widget)
        else:
            # Tab not visible yet, mark as pending for showEvent
            self.tab._camera_widget_pending = True

    def _clear_camera_widget(self) -> None:
        """Remove existing camera orientation widget."""
        if self.state.camera_widget:
            try:
                self.state.camera_widget.EnabledOff()
                if hasattr(self.tab.plotter, 'remove_actor'):
                    try:
                        self.tab.plotter.remove_actor(self.state.camera_widget)
                    except Exception:
                        pass
            except Exception:
                pass
            self.state.camera_widget = None
            self.tab.camera_widget = None

    def _add_camera_widget(self) -> None:
        """Add camera orientation widget after Qt layout has settled."""
        try:
            # Render again to ensure proper sizing
            self.tab.plotter.render()
            
            # Add camera widget with correct size
            camera_widget = self.tab.plotter.add_camera_orientation_widget()
            camera_widget.EnabledOn()
            
            # Store reference
            self.state.camera_widget = camera_widget
            self.tab.camera_widget = camera_widget
        except Exception:
            pass  # Plotter may have been closed

    def setup_hover_annotation(self) -> None:
        """Set up hover callbacks to display node information with enhanced details."""
        mesh = self.state.current_mesh or self.tab.current_mesh
        if not mesh or "NodeID" not in mesh.array_names:
            return

        self.clear_hover_elements()

        annotation = self.tab.plotter.add_text(
            "",
            position="upper_right",
            font_size=8,
            color="black",
            name="hover_annotation",
        )
        self.state.hover_annotation = annotation
        self.tab.hover_annotation = annotation

        picker = vtk.vtkPointPicker()
        picker.SetTolerance(0.025)  # 2.5% of window diagonal for better zoom-in tolerance

        def hover_callback(obj, _event):
            now = time.time()
            if (now - self.state.last_hover_time) < 0.033:  # 30 FPS throttle
                return

            current_mesh = self.state.current_mesh or self.tab.current_mesh
            if current_mesh is None:
                return

            iren = obj
            pos = iren.GetEventPosition()
            picker.Pick(pos[0], pos[1], 0, self.tab.plotter.renderer)
            point_id = picker.GetPointId()

            if point_id != -1 and point_id < current_mesh.n_points:
                node_id = current_mesh["NodeID"][point_id]
                
                # Build annotation text with enhanced information
                lines = [f"Node ID: {int(node_id)}"]
                
                # Check if this is batch solve result with envelope data
                has_max_stress = "Max_Stress" in current_mesh.array_names
                has_min_stress = "Min_Stress" in current_mesh.array_names
                has_combo_of_max = "Combo_of_Max" in current_mesh.array_names
                has_combo_of_min = "Combo_of_Min" in current_mesh.array_names
                has_force_envelope = "Max_Force_Magnitude" in current_mesh.array_names
                
                if has_max_stress or has_min_stress:
                    # This is batch solve result - show enhanced information
                    
                    # Get combination names if available
                    combo_names = getattr(self.tab, 'combination_names', [])
                    result_type = getattr(self.tab, 'current_result_type', None)
                    
                    stress_unit = "MPa"
                    # Show max value with combination info
                    if has_max_stress:
                        max_val = current_mesh["Max_Stress"][point_id]
                        if has_combo_of_max and combo_names:
                            combo_idx = int(current_mesh["Combo_of_Max"][point_id])
                            combo_name = combo_names[combo_idx] if combo_idx < len(combo_names) else f"#{combo_idx + 1}"
                            lines.append(f"Max: {max_val:.5f} {stress_unit} ({combo_name})")
                        elif has_combo_of_max:
                            combo_idx = int(current_mesh["Combo_of_Max"][point_id])
                            lines.append(f"Max: {max_val:.5f} {stress_unit} (Combo #{combo_idx + 1})")
                        else:
                            lines.append(f"Max: {max_val:.5f} {stress_unit}")
                    
                    # Show min value with combination info (only for min_principal stress)
                    if has_min_stress and result_type == "min_principal":
                        min_val = current_mesh["Min_Stress"][point_id]
                        if has_combo_of_min and combo_names:
                            combo_idx = int(current_mesh["Combo_of_Min"][point_id])
                            combo_name = combo_names[combo_idx] if combo_idx < len(combo_names) else f"#{combo_idx + 1}"
                            lines.append(f"Min: {min_val:.5f} {stress_unit} ({combo_name})")
                        elif has_combo_of_min:
                            combo_idx = int(current_mesh["Combo_of_Min"][point_id])
                            lines.append(f"Min: {min_val:.5f} {stress_unit} (Combo #{combo_idx + 1})")
                        else:
                            lines.append(f"Min: {min_val:.5f} {stress_unit}")
                elif has_force_envelope:
                    # Force envelope visualization - show current scalar and combo info if available
                    combo_names = getattr(self.tab, 'combination_names', [])
                    force_unit = "N"
                    if getattr(self.tab, 'nodal_forces_result', None) is not None:
                        force_unit = self.tab.nodal_forces_result.force_unit
                    active_name = current_mesh.active_scalars_name or self.tab.data_column
                    if active_name in current_mesh.array_names:
                        value = current_mesh[active_name][point_id]
                        if active_name.startswith("Combo_of_"):
                            lines.append(f"{active_name}: Combo #{int(value) + 1}")
                        else:
                            lines.append(f"{active_name} [{force_unit}]: {value:.5f}")
                        # Map active scalar to combo index array and also show the opposite envelope combo
                        def _append_combo_line(label, field):
                            if field in current_mesh.array_names:
                                combo_idx = int(current_mesh[field][point_id])
                                if combo_names and 0 <= combo_idx < len(combo_names):
                                    lines.append(f"{label}: {combo_names[combo_idx]}")
                                else:
                                    lines.append(f"{label}: Combo #{combo_idx + 1}")

                        combo_field = None
                        if active_name == "Max_Force_Magnitude":
                            combo_field = "Combo_of_Max"
                        elif active_name == "Min_Force_Magnitude":
                            combo_field = "Combo_of_Min"
                        elif active_name.startswith("Max_") or active_name.startswith("Min_"):
                            combo_field = f"Combo_of_{active_name}"

                        # Show both max and min combos when available
                        def _append_value_line(label, field):
                            if field in current_mesh.array_names:
                                val = current_mesh[field][point_id]
                                lines.append(f"{label} [{force_unit}]: {val:.5f}")

                        if active_name in ("Combo_of_Max", "Combo_of_Min"):
                            _append_value_line("Max", "Max_Force_Magnitude")
                            _append_combo_line("Combo of Max", "Combo_of_Max")
                            _append_value_line("Min", "Min_Force_Magnitude")
                            _append_combo_line("Combo of Min", "Combo_of_Min")
                        elif active_name.startswith("Combo_of_Max_"):
                            suffix = active_name.replace("Combo_of_Max_", "", 1)
                            _append_value_line("Max", f"Max_{suffix}")
                            _append_combo_line("Combo of Max", active_name)
                            _append_value_line("Min", f"Min_{suffix}")
                            _append_combo_line("Combo of Min", f"Combo_of_Min_{suffix}")
                        elif active_name.startswith("Combo_of_Min_"):
                            suffix = active_name.replace("Combo_of_Min_", "", 1)
                            _append_value_line("Min", f"Min_{suffix}")
                            _append_combo_line("Combo of Min", active_name)
                            _append_value_line("Max", f"Max_{suffix}")
                            _append_combo_line("Combo of Max", f"Combo_of_Max_{suffix}")
                        elif active_name == "Max_Force_Magnitude":
                            _append_value_line("Max", "Max_Force_Magnitude")
                            _append_combo_line("Combo of Max", "Combo_of_Max")
                            _append_value_line("Min", "Min_Force_Magnitude")
                            _append_combo_line("Combo of Min", "Combo_of_Min")
                        elif active_name == "Min_Force_Magnitude":
                            _append_value_line("Min", "Min_Force_Magnitude")
                            _append_combo_line("Combo of Min", "Combo_of_Min")
                            _append_value_line("Max", "Max_Force_Magnitude")
                            _append_combo_line("Combo of Max", "Combo_of_Max")
                        elif active_name.startswith("Max_"):
                            _append_value_line("Max", active_name)
                            _append_combo_line("Combo of Max", combo_field)
                            if combo_field:
                                min_field = active_name.replace("Max_", "Min_", 1)
                                _append_value_line("Min", min_field)
                                _append_combo_line("Combo of Min", combo_field.replace("Combo_of_Max_", "Combo_of_Min_"))
                        elif active_name.startswith("Min_"):
                            _append_value_line("Min", active_name)
                            _append_combo_line("Combo of Min", combo_field)
                            if combo_field:
                                max_field = active_name.replace("Min_", "Max_", 1)
                                _append_value_line("Max", max_field)
                                _append_combo_line("Combo of Max", combo_field.replace("Combo_of_Min_", "Combo_of_Max_"))
                        else:
                            if combo_field:
                                _append_combo_line(combo_field, combo_field)
                    else:
                        # Fallback to standard display value if active array not found
                        value = current_mesh[self.tab.data_column][point_id]
                        lines.append(f"{self.tab.data_column}: {value:.5f}")
                else:
                    # Standard visualization - show current data column value with units when applicable
                    active_name = current_mesh.active_scalars_name or self.tab.data_column
                    value = current_mesh[active_name][point_id]
                    if active_name.startswith("Combo_of_"):
                        lines.append(f"{active_name}: Combo #{int(value) + 1}")
                    else:
                        unit = ""
                        if "Stress" in active_name or active_name.endswith("_Stress"):
                            unit = "MPa"
                        elif active_name in ("U_mag", "UX", "UY", "UZ", "Max_U_mag", "Min_U_mag"):
                            disp_unit = "mm"
                            if getattr(self.tab, 'deformation_result', None) is not None:
                                disp_unit = self.tab.deformation_result.displacement_unit
                            unit = disp_unit
                        elif active_name in (
                            "Force_Magnitude", "Max_Force_Magnitude", "Min_Force_Magnitude",
                            "FX", "FY", "FZ", "Shear_XY", "Shear_XZ", "Shear_YZ",
                            "Max_FX", "Min_FX", "Max_FY", "Min_FY", "Max_FZ", "Min_FZ",
                            "Max_Shear_XY", "Min_Shear_XY", "Max_Shear_XZ", "Min_Shear_XZ",
                            "Max_Shear_YZ", "Min_Shear_YZ", "Shear_Force"
                        ):
                            if getattr(self.tab, 'nodal_forces_result', None) is not None:
                                unit = self.tab.nodal_forces_result.force_unit
                        if unit:
                            lines.append(f"{active_name} [{unit}]: {value:.5f}")
                        else:
                            lines.append(f"{active_name}: {value:.5f}")
                
                annotation.SetText(2, "\n".join(lines))
            else:
                annotation.SetText(2, "")

            iren.GetRenderWindow().Render()
            self.state.last_hover_time = now
            self.tab.last_hover_time = now

        observer_id = self.tab.plotter.iren.add_observer(
            "MouseMoveEvent", hover_callback
        )
        self.state.hover_observer = observer_id
        self.tab.hover_observer = observer_id

    def clear_hover_elements(self) -> None:
        """Remove hover annotation text and observer callbacks."""
        if self.state.hover_annotation:
            try:
                self.tab.plotter.remove_actor(self.state.hover_annotation)
            except Exception:
                pass
            self.state.hover_annotation = None
            self.tab.hover_annotation = None

        if self.state.hover_observer:
            try:
                self.tab.plotter.iren.remove_observer(self.state.hover_observer)
            except Exception:
                pass
            self.state.hover_observer = None
            self.tab.hover_observer = None

    def update_point_size(self) -> None:
        """Adjust point size and refresh hover annotations."""
        actor = self.state.current_actor or self.tab.current_actor
        if actor is None:
            return

        self.clear_hover_elements()
        actor.GetProperty().SetPointSize(self.tab.point_size.value())
        self.setup_hover_annotation()
        self.tab.plotter.render()

    def update_scalar_range(self) -> None:
        """Update scalar range on the current actor."""
        actor = self.state.current_actor or self.tab.current_actor
        if actor is None:
            return

        actor.mapper.scalar_range = (
            self.tab.scalar_min_spin.value(),
            self.tab.scalar_max_spin.value(),
        )
        self.tab.plotter.render()

    def validate_deformation_scale(self) -> None:
        """Validate deformation scale factor input and update visualization."""
        text = self.tab.deformation_scale_edit.text()
        try:
            value = float(text)
        except ValueError:
            fallback = str(self.state.last_valid_deformation_scale)
            self.tab.deformation_scale_edit.setText(fallback)
            self.tab.last_valid_deformation_scale = self.state.last_valid_deformation_scale
            return

        self.state.last_valid_deformation_scale = value
        self.tab.last_valid_deformation_scale = value
        
        # If deformation results are available, update the visualization
        deformation_result = getattr(self.tab, 'deformation_result', None)
        if deformation_result is not None:
            self.update_visualization()

    def apply_scalar_field(self, field_name: str, values) -> bool:
        """
        Apply a scalar field to the current mesh and refresh the visualization.

        Args:
            field_name: Name of the scalar field to apply.
            values: Iterable of scalar values per node.

        Returns:
            bool: True if the field was applied successfully, False otherwise.
        """
        mesh = self.state.current_mesh or self.tab.current_mesh
        if mesh is None:
            return False

        array = np.asarray(values)
        if array.ndim > 1:
            array = array.reshape(-1)

        if mesh.n_points != array.shape[0]:
            raise ValueError(
                f"Scalar field '{field_name}' length {array.shape[0]} does not match mesh nodes {mesh.n_points}"
            )

        mesh[field_name] = array
        mesh.set_active_scalars(field_name)

        self.state.data_column = field_name
        self.tab.data_column = field_name

        self.update_visualization()
        return True
