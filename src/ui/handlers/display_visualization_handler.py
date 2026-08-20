"""
Visualization updates and rendering helpers for the Display tab.
"""

import time
from typing import Optional
import numpy as np
import pyvista as pv
import vtk
from PyQt5.QtCore import QThread, QTimer, pyqtSignal
from PyQt5.QtWidgets import QMessageBox

from ui.handlers.display_base_handler import DisplayBaseHandler
from core.data_models import MeshTopologyData
from core.visualization import VisualizationManager


class _MeshTopologyWorker(QThread):
    """Run one lazy topology build away from the Qt GUI thread."""

    completed = pyqtSignal(object)

    def __init__(self, provider, node_ids: np.ndarray, include_whole_model: bool):
        super().__init__()
        self.provider = provider
        self.node_ids = np.asarray(node_ids, dtype=np.int64).copy()
        self.include_whole_model = bool(include_whole_model)
        self.result = None
        self.error = None

    def run(self) -> None:
        try:
            self.result = self.provider.build_visualization_topology(
                self.node_ids,
                include_whole_model=self.include_whole_model,
            )
        except Exception as exc:
            self.error = str(exc)
        finally:
            self.completed.emit(self)


class DisplayVisualizationHandler(DisplayBaseHandler):
    """Coordinates rendering operations on the PyVista plotter."""

    def __init__(self, tab, state, viz_manager: VisualizationManager):
        super().__init__(tab, state)
        self.viz_manager = viz_manager
        self._topology_provider = None
        self._topology_data: Optional[MeshTopologyData] = None
        self._topology_node_ids = None
        self._topology_includes_whole = False
        self._result_surface_mesh = None
        self._whole_surface_mesh = None
        self._whole_context_mesh = None
        self._topology_worker = None
        self._worker_request = None
        self._pending_topology_request = False
        self._payload_generation = 0

    def set_topology_provider(self, provider) -> None:
        """Replace solver-backed topology source without invoking it."""
        self._payload_generation += 1
        self._topology_provider = provider
        self._topology_data = None
        self._topology_node_ids = None
        self._topology_includes_whole = False
        self._result_surface_mesh = None
        self._whole_surface_mesh = None
        self._whole_context_mesh = None
        self._pending_topology_request = False
        self.update_mesh_control_state()

    def _mesh_view(self) -> str:
        value = self.tab.mesh_view_combo.currentData()
        return value if value in {"points", "contour_mesh", "mesh_points"} else "points"

    def _mesh_scope(self) -> str:
        return "whole" if self.tab.mesh_scope_combo.currentData() == "whole" else "result"

    def _mesh_edges_visible(self) -> bool:
        return self.tab.mesh_edges_checkbox.isChecked()

    def _has_nonzero_deformation(self) -> bool:
        if self.tab.deformation_result is None:
            return False
        try:
            return float(self.tab.deformation_scale_edit.text()) != 0.0
        except (TypeError, ValueError):
            return self.state.last_valid_deformation_scale != 0.0

    def update_mesh_control_state(self) -> None:
        """Apply provider, view, and deformation availability to mesh controls."""
        if not hasattr(self.tab, "mesh_view_combo"):
            return

        available = self._topology_provider is not None
        if not available and self.tab.mesh_view_combo.currentIndex() != 0:
            self.tab.mesh_view_combo.blockSignals(True)
            self.tab.mesh_view_combo.setCurrentIndex(0)
            self.tab.mesh_view_combo.blockSignals(False)
        self.tab.mesh_view_combo.setEnabled(available)

        view = self._mesh_view()
        deformed = self._has_nonzero_deformation()
        if deformed and self._mesh_scope() == "whole":
            self.tab.mesh_scope_combo.blockSignals(True)
            self.tab.mesh_scope_combo.setCurrentIndex(0)
            self.tab.mesh_scope_combo.blockSignals(False)
        self.tab.mesh_scope_combo.setEnabled(available and view != "points" and not deformed)
        self.tab.mesh_edges_checkbox.setEnabled(available and view != "points")
        self.tab.point_size.setEnabled(view != "contour_mesh")

    def on_mesh_view_changed(self) -> None:
        self.update_mesh_control_state()
        if self._mesh_view() != "points":
            self._request_topology()
        self.update_visualization()

    def on_mesh_scope_changed(self) -> None:
        self.update_mesh_control_state()
        if self._mesh_view() != "points":
            self._request_topology()
        self.update_visualization()

    def on_mesh_edges_changed(self, visible: bool) -> None:
        """Update current mesh actors without rebuilding the view."""
        actors = []
        if self._mesh_view() == "contour_mesh":
            actors.append(self.state.current_actor or self.tab.current_actor)
        actors.extend(
            self.tab.plotter.actors.get(name)
            for name in ("mesh_context", "whole_mesh_context")
        )
        for actor in actors:
            if actor is not None:
                actor.GetProperty().SetEdgeVisibility(visible)
        self.tab.plotter.render()

    def _topology_matches_current_request(self) -> bool:
        mesh = self.state.current_mesh or self.tab.current_mesh
        if (
            self._topology_data is None
            or mesh is None
            or "NodeID" not in mesh.array_names
            or self._topology_node_ids is None
        ):
            return False
        if not np.array_equal(self._topology_node_ids, np.asarray(mesh["NodeID"])):
            return False
        return self._mesh_scope() != "whole" or self._topology_includes_whole

    def _request_topology(self) -> bool:
        if self._mesh_view() == "points" or self._topology_matches_current_request():
            return self._topology_matches_current_request()

        mesh = self.state.current_mesh or self.tab.current_mesh
        if self._topology_provider is None or mesh is None or "NodeID" not in mesh.array_names:
            return False

        if self._topology_worker is not None:
            self._pending_topology_request = True
            return False

        include_whole = self._mesh_scope() == "whole"
        worker = _MeshTopologyWorker(
            self._topology_provider,
            np.asarray(mesh["NodeID"]),
            include_whole,
        )
        self._worker_request = {
            "payload_generation": self._payload_generation,
            "provider": self._topology_provider,
            "node_ids": np.asarray(mesh["NodeID"]).copy(),
            "include_whole": include_whole,
        }
        self._topology_worker = worker
        worker.completed.connect(self.tab._on_mesh_topology_worker_completed)
        worker.start()
        return False

    def on_topology_worker_finished(self, worker) -> None:
        if worker is not self._topology_worker:
            return
        request = self._worker_request
        self._topology_worker = None
        self._worker_request = None
        if worker is None or request is None:
            return

        mesh = self.state.current_mesh or self.tab.current_mesh
        current_request = (
            request["payload_generation"] == self._payload_generation
            and request["provider"] is self._topology_provider
            and mesh is not None
            and "NodeID" in mesh.array_names
            and np.array_equal(request["node_ids"], np.asarray(mesh["NodeID"]))
        )

        if worker.error is None and current_request:
            self._topology_data = worker.result
            self._topology_node_ids = request["node_ids"]
            self._topology_includes_whole = request["include_whole"]
            self._build_topology_meshes()
        elif (
            worker.error
            and current_request
            and self._mesh_view() != "points"
            and (self._mesh_scope() != "whole" or request["include_whole"])
        ):
            self.tab.mesh_view_combo.blockSignals(True)
            self.tab.mesh_view_combo.setCurrentIndex(0)
            self.tab.mesh_view_combo.blockSignals(False)
            QMessageBox.warning(self.tab, "Mesh Unavailable", worker.error)

        worker.deleteLater()
        pending = self._pending_topology_request
        self._pending_topology_request = False
        self.update_mesh_control_state()
        if pending and self._mesh_view() != "points" and not self._topology_matches_current_request():
            self._request_topology()
        self.update_visualization()

    @staticmethod
    def _polydata(points, faces=None, lines=None):
        mesh = pv.PolyData()
        mesh.points = np.asarray(points, dtype=float)
        if faces is not None and np.asarray(faces).size:
            mesh.faces = np.asarray(faces, dtype=np.int64)
        if lines is not None and np.asarray(lines).size:
            mesh.lines = np.asarray(lines, dtype=np.int64)
        return mesh

    def _build_topology_meshes(self) -> None:
        data = self._topology_data
        point_mesh = self.state.current_mesh or self.tab.current_mesh
        if data is None or point_mesh is None:
            return

        reference_points = self.tab.original_node_coords
        if reference_points is None:
            reference_points = point_mesh.points
        self._result_surface_mesh = self._polydata(
            reference_points,
            data.result_faces,
            data.result_lines,
        )
        if data.whole_points_mm is not None:
            self._whole_surface_mesh = self._polydata(
                data.whole_points_mm,
                data.whole_faces,
                data.whole_lines,
            )
            self._whole_context_mesh = self._polydata(
                data.whole_points_mm,
                data.context_faces,
                data.context_lines,
            )

    def _sync_result_surface_mesh(self, point_mesh) -> None:
        if self._result_surface_mesh is None:
            return
        self._result_surface_mesh.SetPoints(point_mesh.GetPoints())
        self._result_surface_mesh.GetPointData().ShallowCopy(point_mesh.GetPointData())
        self._result_surface_mesh.GetFieldData().ShallowCopy(point_mesh.GetFieldData())

    def _add_scalar_actor(self, plotter, mesh, active_scalars, *, points: bool):
        digits = self.state.scalar_bar_digits
        kwargs = {
            "scalars": active_scalars,
            "show_scalar_bar": True,
            "cmap": "jet",
            "below_color": "gray",
            "above_color": "magenta",
            "pickable": True,
            "reset_camera": False,
            "scalar_bar_args": {
                "title": self.tab.data_column,
                "fmt": f"%.{digits}f",
                "position_x": 0.04,
                "position_y": 0.35,
                "width": 0.13,
                "height": 0.5,
                "vertical": True,
                "title_font_size": 14,
                "label_font_size": 12,
                "shadow": True,
                "n_labels": 10,
                "interactive": False,
            },
        }
        if points:
            kwargs.update(
                style="points",
                point_size=self.tab.point_size.value(),
                render_points_as_spheres=True,
            )
        else:
            kwargs.update(
                show_edges=self._mesh_edges_visible(),
                edge_color="#4d4d4d",
                line_width=1,
            )
        actor = plotter.add_mesh(mesh, **kwargs)
        scalar_bar = plotter.scalar_bars[self.tab.data_column]
        scalar_bar.SetBarRatio(0.145)
        background = scalar_bar.GetBackgroundProperty()
        background.SetColor(1.0, 1.0, 1.0)
        background.SetOpacity(0.65)
        scalar_bar.SetDrawBackground(self.state.legend_background_enabled)
        return actor

    def _add_context_actor(self, plotter, mesh, name: str):
        if mesh is None or mesh.n_cells == 0:
            return None
        return plotter.add_mesh(
            mesh,
            name=name,
            color="#d9d9d9",
            edge_color="#4d4d4d",
            show_edges=self._mesh_edges_visible(),
            line_width=1,
            opacity=1.0,
            show_scalar_bar=False,
            pickable=False,
            reset_camera=False,
        )

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
        deformation_result = self.tab.deformation_result
        if deformation_result is None:
            return False
        
        # Get original coordinates (stored when mesh was created)
        original_coords = self.tab.original_node_coords
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
        
        original_coords = self.tab.original_node_coords
        if original_coords is None:
            return False
        
        mesh.points = original_coords.copy()
        return True

    def update_visualization(self) -> None:
        """Refresh the 3D view with the current mesh."""
        mesh = self.state.current_mesh or self.tab.current_mesh
        if mesh is None:
            return

        if self.tab.deformation_result is not None:
            self.apply_deformed_coordinates()
        self.update_mesh_control_state()

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

        requested_view = self._mesh_view()
        loading_topology = False
        if requested_view != "points" and not self._topology_matches_current_request():
            loading_topology = not self._request_topology()

        render_view = requested_view if self._topology_matches_current_request() else "points"
        if render_view != "points" and self._result_surface_mesh is None:
            self._build_topology_meshes()
        self._sync_result_surface_mesh(mesh)

        plotter = self.tab.plotter
        plotter.clear()

        if render_view == "contour_mesh":
            if self._mesh_scope() == "whole":
                self._add_context_actor(plotter, self._whole_context_mesh, "whole_mesh_context")
            actor = self._add_scalar_actor(
                plotter,
                self._result_surface_mesh,
                active_scalars,
                points=False,
            )
        elif render_view == "mesh_points":
            context_mesh = (
                self._whole_surface_mesh
                if self._mesh_scope() == "whole"
                else self._result_surface_mesh
            )
            self._add_context_actor(plotter, context_mesh, "mesh_context")
            actor = self._add_scalar_actor(
                plotter,
                mesh,
                active_scalars,
                points=True,
            )
        else:
            actor = self._add_scalar_actor(
                plotter,
                mesh,
                active_scalars,
                points=True,
            )

        if loading_topology:
            plotter.add_text(
                "Building mesh...",
                position="upper_left",
                font_size=10,
                color="gray",
                name="mesh_loading",
            )

        self.state.current_actor = actor
        self.tab.current_actor = actor

        if self.tab.scalar_min_spin.value() != self.tab.scalar_max_spin.value():
            actor.mapper.scalar_range = (
                self.tab.scalar_min_spin.value(),
                self.tab.scalar_max_spin.value(),
            )

        self.setup_hover_annotation()

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
                self.tab.plotter.remove_actor(self.state.camera_widget)
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
            position="upper_left",
            font_size=8,
            color="black",
            name="hover_annotation",
        )
        text_property = annotation.GetTextProperty()
        text_property.SetBackgroundColor(1.0, 1.0, 1.0)
        text_property.SetBackgroundOpacity(
            0.65 if self.state.hover_background_enabled else 0.0
        )
        self.state.hover_annotation = annotation
        self.tab.hover_annotation = annotation

        picker = vtk.vtkPointPicker()
        picker.SetTolerance(0.025)  # 2.5% of window diagonal for better zoom-in tolerance

        def hover_callback(obj, _event):
            now = time.time()
            if (now - self.state.last_hover_time) < 0.033:  # 30 FPS throttle
                return

            iren = obj
            pos = iren.GetEventPosition()
            picker.Pick(pos[0], pos[1], 0, self.tab.plotter.renderer)
            point_id = picker.GetPointId()
            picked_dataset = picker.GetDataSet()
            if picked_dataset is None:
                return
            current_mesh = pv.wrap(picked_dataset)

            if (
                point_id != -1
                and point_id < current_mesh.n_points
                and "NodeID" in current_mesh.array_names
            ):
                node_id = current_mesh["NodeID"][point_id]
                
                # Build annotation text with enhanced information
                lines = [f"Node ID: {int(node_id)}"]
                
                # Check if this is batch solve result with envelope data
                has_max_stress = "Max_Stress" in current_mesh.array_names
                has_min_stress = "Min_Stress" in current_mesh.array_names
                has_combo_of_max = "Combo_of_Max" in current_mesh.array_names
                has_combo_of_min = "Combo_of_Min" in current_mesh.array_names
                has_force_envelope = "Max_Force_Magnitude" in current_mesh.array_names
                active_contour_type = (
                    self.state.current_contour_type
                    or self.tab.current_contour_type
                )

                handled_deformation = (
                    active_contour_type == "Deformation"
                    and self._append_deformation_hover_lines(lines, current_mesh, point_id)
                )

                if active_contour_type == "Stress":
                    show_stress_block = has_max_stress or has_min_stress
                    show_force_block = False
                elif active_contour_type == "Forces":
                    show_stress_block = False
                    show_force_block = has_force_envelope
                else:
                    show_stress_block = has_max_stress or has_min_stress
                    show_force_block = has_force_envelope and not show_stress_block

                if handled_deformation:
                    pass
                elif show_stress_block:
                    # This is batch solve result - show enhanced information
                    
                    # Get combination names if available
                    combo_names = self.tab.combination_names
                    result_type = self.tab.current_result_type
                    
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
                elif show_force_block:
                    # Force envelope visualization - show current scalar and combo info if available
                    combo_names = self.tab.combination_names
                    force_unit = "N"
                    if self.tab.nodal_forces_result is not None:
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
                        elif active_name in ("U_mag", "UX", "UY", "UZ", "Max_U_mag", "Min_U_mag") or active_name.startswith("Def_"):
                            disp_unit = "mm"
                            if self.tab.deformation_result is not None:
                                disp_unit = self.tab.deformation_result.displacement_unit
                            unit = disp_unit
                        elif active_name in (
                            "Force_Magnitude", "Max_Force_Magnitude", "Min_Force_Magnitude",
                            "FX", "FY", "FZ", "Shear_XY", "Shear_XZ", "Shear_YZ",
                            "Max_FX", "Min_FX", "Max_FY", "Min_FY", "Max_FZ", "Min_FZ",
                            "Max_Shear_XY", "Min_Shear_XY", "Max_Shear_XZ", "Min_Shear_XZ",
                            "Max_Shear_YZ", "Min_Shear_YZ", "Shear_Force"
                        ):
                            if self.tab.nodal_forces_result is not None:
                                unit = self.tab.nodal_forces_result.force_unit
                        if unit:
                            lines.append(f"{active_name} [{unit}]: {value:.5f}")
                        else:
                            lines.append(f"{active_name}: {value:.5f}")
                
                annotation.SetText(annotation.UpperLeft, "\n".join(lines))
            else:
                annotation.SetText(annotation.UpperLeft, "")

            iren.GetRenderWindow().Render()
            self.state.last_hover_time = now
            self.tab.last_hover_time = now

        observer_id = self.tab.plotter.iren.add_observer(
            "MouseMoveEvent", hover_callback
        )
        self.state.hover_observer = observer_id
        self.tab.hover_observer = observer_id

    def _append_deformation_hover_lines(self, lines, mesh, point_id: int) -> bool:
        """Append deformation-family hover text. Returns True when handled."""
        active_name = mesh.active_scalars_name or self.tab.data_column
        deformation_fields = {
            "U_mag", "UX", "UY", "UZ", "Max_U_mag", "Min_U_mag"
        }
        if not active_name:
            return False

        is_deformation_field = active_name.startswith("Def_") or active_name in deformation_fields
        if not is_deformation_field:
            return False

        disp_unit = "mm"
        if self.tab.deformation_result is not None:
            disp_unit = self.tab.deformation_result.displacement_unit

        combo_names = self.tab.combination_names

        if active_name in mesh.array_names:
            value = mesh[active_name][point_id]
            if active_name.startswith("Def_Combo_of_"):
                combo_idx = int(value)
                if combo_names and 0 <= combo_idx < len(combo_names):
                    lines.append(f"{active_name}: {combo_names[combo_idx]}")
                else:
                    lines.append(f"{active_name}: Combo #{combo_idx + 1}")
            else:
                lines.append(f"{active_name} [{disp_unit}]: {value:.5f}")

        # Envelope detail lines for Def_* arrays
        component = None
        for suffix in ("U_mag", "UX", "UY", "UZ"):
            if active_name.endswith(suffix):
                component = suffix
                break

        if component and active_name.startswith("Def_"):
            max_field = f"Def_Max_{component}"
            min_field = f"Def_Min_{component}"
            combo_max_field = f"Def_Combo_of_Max_{component}"
            combo_min_field = f"Def_Combo_of_Min_{component}"

            if max_field in mesh.array_names:
                lines.append(f"Max [{disp_unit}]: {mesh[max_field][point_id]:.5f}")
            if combo_max_field in mesh.array_names:
                combo_idx = int(mesh[combo_max_field][point_id])
                if combo_names and 0 <= combo_idx < len(combo_names):
                    lines.append(f"Combo of Max: {combo_names[combo_idx]}")
                else:
                    lines.append(f"Combo of Max: Combo #{combo_idx + 1}")

            if min_field in mesh.array_names:
                lines.append(f"Min [{disp_unit}]: {mesh[min_field][point_id]:.5f}")
            if combo_min_field in mesh.array_names:
                combo_idx = int(mesh[combo_min_field][point_id])
                if combo_names and 0 <= combo_idx < len(combo_names):
                    lines.append(f"Combo of Min: {combo_names[combo_idx]}")
                else:
                    lines.append(f"Combo of Min: Combo #{combo_idx + 1}")

        return True

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
        self.update_mesh_control_state()
        
        # If deformation results are available, update the visualization
        deformation_result = self.tab.deformation_result
        if deformation_result is not None:
            self.update_visualization()

    def shutdown(self) -> None:
        """Wait for an active topology worker before Qt destroys it."""
        worker = self._topology_worker
        if worker is not None and worker.isRunning():
            worker.requestInterruption()
            worker.wait()

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
