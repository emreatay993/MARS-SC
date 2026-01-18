"""
Visualization updates and rendering helpers for the Display tab.
"""

import time
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

    def update_visualization(self) -> None:
        """Refresh the 3D view with the current mesh."""
        mesh = self.state.current_mesh or self.tab.current_mesh
        if mesh is None:
            return

        plotter = self.tab.plotter
        plotter.clear()

        # Use active scalars if set, otherwise fall back to first array (e.g., NodeID)
        active_scalars = mesh.active_scalars_name
        if not active_scalars and mesh.array_names:
            active_scalars = mesh.array_names[0]
        if active_scalars:
            self.state.data_column = active_scalars
            self.tab.data_column = active_scalars

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
                "fmt": "%.4f",
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
                
                if has_max_stress or has_min_stress:
                    # This is batch solve result - show enhanced information
                    
                    # Get combination names if available
                    combo_names = getattr(self.tab, 'combination_names', [])
                    result_type = getattr(self.tab, 'current_result_type', None)
                    
                    # Show max value with combination info
                    if has_max_stress:
                        max_val = current_mesh["Max_Stress"][point_id]
                        if has_combo_of_max and combo_names:
                            combo_idx = int(current_mesh["Combo_of_Max"][point_id])
                            combo_name = combo_names[combo_idx] if combo_idx < len(combo_names) else f"#{combo_idx + 1}"
                            lines.append(f"Max: {max_val:.5f} ({combo_name})")
                        elif has_combo_of_max:
                            combo_idx = int(current_mesh["Combo_of_Max"][point_id])
                            lines.append(f"Max: {max_val:.5f} (Combo #{combo_idx + 1})")
                        else:
                            lines.append(f"Max: {max_val:.5f}")
                    
                    # Show min value with combination info (only for min_principal stress)
                    if has_min_stress and result_type == "min_principal":
                        min_val = current_mesh["Min_Stress"][point_id]
                        if has_combo_of_min and combo_names:
                            combo_idx = int(current_mesh["Combo_of_Min"][point_id])
                            combo_name = combo_names[combo_idx] if combo_idx < len(combo_names) else f"#{combo_idx + 1}"
                            lines.append(f"Min: {min_val:.5f} ({combo_name})")
                        elif has_combo_of_min:
                            combo_idx = int(current_mesh["Combo_of_Min"][point_id])
                            lines.append(f"Min: {min_val:.5f} (Combo #{combo_idx + 1})")
                        else:
                            lines.append(f"Min: {min_val:.5f}")
                else:
                    # Standard visualization - show current data column value
                    value = current_mesh[self.tab.data_column][point_id]
                    lines.append(f"{self.tab.data_column}: {value:.5f}")
                
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
        """Validate deformation scale factor input."""
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
