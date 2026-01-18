"""
Node interaction, picking, and hotspot analysis for the Display tab.
"""

from typing import Optional

import numpy as np
import pyvista as pv
import vtk
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtWidgets import (
    QAction,
    QLabel,
    QInputDialog,
    QMenu,
    QMessageBox,
    QWidgetAction,
)

from core.visualization import HotspotDetector
from ui.handlers.display_base_handler import DisplayBaseHandler
from ui.widgets.dialogs import HotspotDialog
from ui.styles.style_constants import CONTEXT_MENU_STYLE


class DisplayInteractionHandler(DisplayBaseHandler):
    """Manages point picking, hotspot workflows, and context actions."""

    def __init__(self, tab, state, hotspot_detector: HotspotDetector):
        super().__init__(tab, state)
        self.hotspot_detector = hotspot_detector

    # ------------------------------------------------------------------
    # Context menu handling
    # ------------------------------------------------------------------
    def show_context_menu(self, position: QPoint) -> None:
        """Create and display the right-click context menu."""
        if self.tab.current_mesh is None:
            return

        context_menu = QMenu(self.tab)
        context_menu.setStyleSheet(CONTEXT_MENU_STYLE)

        # Selection tools
        self._add_section_title(context_menu, "Selection Tools")

        toggle_box_action = QAction(
            "Remove Selection Box" if self.state.box_widget else "Add Selection Box",
            self.tab,
        )
        toggle_box_action.triggered.connect(self.toggle_selection_box)
        context_menu.addAction(toggle_box_action)

        pick_action = QAction("Pick Box Center", self.tab)
        pick_action.setCheckable(True)
        pick_action.setChecked(self.state.is_point_picking_active)
        pick_action.setEnabled(self.tab.current_mesh is not None)
        pick_action.triggered.connect(self.toggle_point_picking_mode)
        context_menu.addAction(pick_action)

        context_menu.addSeparator()

        # Hotspot analysis
        self._add_section_title(context_menu, "Hotspot Analysis")

        hotspot_action = QAction("Find Hotspots (on current view)", self.tab)
        hotspot_action.setEnabled(
            self.tab.current_mesh is not None
            and self.tab.current_mesh.active_scalars is not None
        )
        hotspot_action.triggered.connect(self.find_hotspots_on_view)
        context_menu.addAction(hotspot_action)

        find_in_box_action = QAction("Find Hotspots in Selection", self.tab)
        find_in_box_action.setEnabled(self.state.box_widget is not None)
        find_in_box_action.triggered.connect(self.find_hotspots_in_box)
        context_menu.addAction(find_in_box_action)

        context_menu.addSeparator()

        # Point-based analysis
        self._add_section_title(context_menu, "Point-Based Analysis")

        plot_point_history_action = QAction(
            "Plot Combination History for Selected Node", self.tab
        )
        plot_point_history_action.triggered.connect(self.enable_time_history_picking)
        context_menu.addAction(plot_point_history_action)

        context_menu.addSeparator()

        # View control
        self._add_section_title(context_menu, "View Control")

        go_to_node_action = QAction("Go To Node", self.tab)
        has_mesh_and_node_ids = (
            self.tab.current_mesh is not None
            and "NodeID" in self.tab.current_mesh.array_names
        )
        is_animation_running_and_frozen = (
            self.tab.anim_timer is not None
            and self.tab.anim_timer.isActive()
            and self.state.freeze_tracked_node
        )
        go_to_node_action.setEnabled(has_mesh_and_node_ids and not is_animation_running_and_frozen)
        go_to_node_action.triggered.connect(self.go_to_node)
        context_menu.addAction(go_to_node_action)

        freeze_action = QAction("Lock Camera for Animation (freeze node)", self.tab)
        freeze_action.setCheckable(True)
        freeze_action.setChecked(self.state.freeze_tracked_node)
        freeze_action.setEnabled(self.state.target_node_index is not None)
        freeze_action.triggered.connect(self.toggle_freeze_node)
        context_menu.addAction(freeze_action)

        reset_camera_action = QAction("Reset Camera", self.tab)
        reset_camera_action.triggered.connect(self.tab.plotter.reset_camera)
        context_menu.addAction(reset_camera_action)

        context_menu.exec_(self.tab.plotter.mapToGlobal(position))

    # ------------------------------------------------------------------
    # Selection utilities
    # ------------------------------------------------------------------
    def toggle_selection_box(self, checked: bool = False) -> None:
        """Add or remove the selection box widget."""
        if self.state.box_widget is None:
            widget = self.tab.plotter.add_box_widget(callback=self._dummy_callback)
            widget.SetPlaceFactor(0.75)

            handle_property = widget.GetHandleProperty()
            handle_property.SetColor(0.8, 0.4, 0.2)
            handle_property.SetPointSize(1)

            selected_handle_property = widget.GetSelectedHandleProperty()
            selected_handle_property.SetColor(1.0, 0.5, 0.0)

            self.set_state_attr("box_widget", widget)
        else:
            self.tab.plotter.clear_box_widgets()
            self.set_state_attr("box_widget", None)

        self.tab.plotter.render()

    def toggle_point_picking_mode(self, checked: bool) -> None:
        """Enable or disable point picking for selection box placement."""
        self.set_state_attr("is_point_picking_active", checked)

        if checked:
            self.tab.plotter.enable_point_picking(
                callback=self.on_point_picked_for_box,
                show_message=False,
                use_picker=True,
                left_clicking=True,
            )
            self.tab.plotter.setCursor(Qt.CrossCursor)
        else:
            self.tab.plotter.disable_picking()
            self.tab.plotter.setCursor(Qt.ArrowCursor)

    def on_point_picked_for_box(self, *args) -> None:
        """Callback when a point is picked to position the selection box."""
        if not args or args[0].size == 0:
            return

        center = args[0]

        if self.state.box_widget is None:
            widget = self.tab.plotter.add_box_widget(callback=self._dummy_callback)
            widget.GetHandleProperty().SetColor(0.8, 0.4, 0.2)
            widget.GetSelectedHandleProperty().SetColor(1.0, 0.5, 0.0)
            widget.GetHandleProperty().SetPointSize(10)
            widget.GetSelectedHandleProperty().SetPointSize(15)
            self.set_state_attr("box_widget", widget)

            size = self.tab.current_mesh.length * 0.1 if self.tab.current_mesh else 1.0
            bounds = [
                center[0] - size / 2.0, center[0] + size / 2.0,
                center[1] - size / 2.0, center[1] + size / 2.0,
                center[2] - size / 2.0, center[2] + size / 2.0,
            ]
        else:
            box_geometry = vtk.vtkPolyData()
            self.state.box_widget.GetPolyData(box_geometry)
            current_bounds = box_geometry.GetBounds()

            x_size = current_bounds[1] - current_bounds[0]
            y_size = current_bounds[3] - current_bounds[2]
            z_size = current_bounds[5] - current_bounds[4]
            bounds = [
                center[0] - x_size / 2.0, center[0] + x_size / 2.0,
                center[1] - y_size / 2.0, center[1] + y_size / 2.0,
                center[2] - z_size / 2.0, center[2] + z_size / 2.0,
            ]

        if self.state.box_widget:
            self.state.box_widget.PlaceWidget(bounds)
        self.tab.plotter.render()
        self.toggle_point_picking_mode(False)

    @staticmethod
    def _dummy_callback(*_args) -> None:
        """No-op callback required by VTK box widget."""
        return

    # ------------------------------------------------------------------
    # Hotspot analysis
    # ------------------------------------------------------------------
    def find_hotspots_on_view(self, checked: bool = False) -> None:
        """Identify hotspots among the currently visible nodes."""
        if not self.tab.current_mesh:
            QMessageBox.warning(
                self.tab, "No Data",
                "There is no mesh loaded to find hotspots on."
            )
            return

        # Ensure mesh has active scalars set
        if self.tab.current_mesh.active_scalars is None:
            # Try to find and set a suitable scalar array (excluding NodeID)
            scalar_names = [n for n in self.tab.current_mesh.array_names 
                          if n not in ('NodeID', 'Combo_of_Max', 'Combo_of_Min')]
            if scalar_names:
                self.tab.current_mesh.set_active_scalars(scalar_names[0])
            else:
                QMessageBox.warning(
                    self.tab, "No Scalar Data",
                    "No scalar data found in the mesh for hotspot analysis."
                )
                return

        # Try using vtkSelectVisiblePoints for visibility filtering
        selector = vtk.vtkSelectVisiblePoints()
        selector.SetInputData(self.tab.current_mesh)
        selector.SetRenderer(self.tab.plotter.renderer)
        selector.SetTolerance(0.01)  # Add tolerance for point selection
        selector.Update()

        visible_mesh = pv.wrap(selector.GetOutput())
        
        # If vtkSelectVisiblePoints returns empty but we have a mesh with points,
        # fall back to using the camera frustum to filter visible points
        if visible_mesh.n_points == 0 and self.tab.current_mesh.n_points > 0:
            # Fallback: Use all points in the current mesh
            # This happens when the mesh is a point cloud without cell topology
            # The user can still see contour points, so we use the full mesh
            visible_mesh = self.tab.current_mesh.copy()
            
            # Optional: Filter by camera frustum bounds for very large meshes
            camera = self.tab.plotter.camera
            if camera is not None:
                try:
                    # Get the clipping range and frustum - for now just use full mesh
                    # as frustum clipping is complex and the mesh is usually small enough
                    pass
                except Exception:
                    pass
        
        if visible_mesh.n_points == 0:
            QMessageBox.information(
                self.tab, "No Visible Points",
                "No points are available for hotspot analysis."
            )
            return

        self._find_and_show_hotspots(visible_mesh)

    def find_hotspots_in_box(self, checked: bool = False) -> None:
        """Clip mesh to box bounds and run hotspot analysis."""
        if self.state.box_widget is None or self.tab.current_mesh is None:
            return

        box_geometry = vtk.vtkPolyData()
        self.state.box_widget.GetPolyData(box_geometry)
        bounds = box_geometry.GetBounds()

        clipped_mesh = self.tab.current_mesh.clip_box(bounds, invert=False)
        self._find_and_show_hotspots(clipped_mesh)

    def _find_and_show_hotspots(self, mesh_to_analyze: Optional[pv.PolyData]) -> None:
        """Run hotspot detection on the provided mesh."""
        if not mesh_to_analyze or mesh_to_analyze.n_points == 0:
            QMessageBox.information(
                self.tab, "No Nodes Found",
                "No nodes were found in the selected area."
            )
            return

        if 'NodeID' not in mesh_to_analyze.array_names:
            QMessageBox.warning(
                self.tab, "Missing Node IDs",
                "Node IDs are required to perform hotspot analysis."
            )
            return

        num_hotspots, ok = QInputDialog.getInt(
            self.tab, "Number of Hotspots",
            "How many top nodes to find?", 10, 1, 1000
        )
        if not ok:
            return

        try:
            scalar_values = mesh_to_analyze.active_scalars
            
            # If no active scalars, try to find a suitable scalar array
            if scalar_values is None:
                # Look for scalar arrays (excluding metadata arrays)
                exclude_arrays = {'NodeID', 'Combo_of_Max', 'Combo_of_Min', 'vtkOriginalPointIds'}
                scalar_names = [n for n in mesh_to_analyze.array_names if n not in exclude_arrays]
                
                if scalar_names:
                    # Use the first available scalar array
                    mesh_to_analyze.set_active_scalars(scalar_names[0])
                    scalar_values = mesh_to_analyze.active_scalars
                else:
                    QMessageBox.warning(
                        self.tab, "No Scalar Data",
                        "No scalar data found for hotspot analysis.\n"
                        "Please run analysis first to compute stress values."
                    )
                    return

            node_ids = mesh_to_analyze['NodeID']
            node_coords = mesh_to_analyze.points

            df_hotspots = self.hotspot_detector.detect_hotspots(
                scalar_values,
                node_ids,
                node_coords=node_coords,
                top_n=num_hotspots,
                mode='max'
            )

            if df_hotspots.empty:
                QMessageBox.information(
                    self.tab, "No Hotspots Found",
                    "No hotspots matched the specified criteria."
                )
                return

            scalar_name = mesh_to_analyze.active_scalars_name or "Result"
            df_hotspots = df_hotspots.rename(columns={"Value": scalar_name})

            if self.state.hotspot_dialog is not None:
                self.state.hotspot_dialog.close()

            dialog = HotspotDialog(df_hotspots, self.tab)
            dialog.node_selected.connect(self.highlight_and_focus_on_node)
            dialog.finished.connect(self.cleanup_hotspot_analysis)

            if self.state.box_widget is not None:
                self.state.box_widget.Off()

            self.set_state_attr("hotspot_dialog", dialog)
            dialog.show()

        except Exception as exc:
            QMessageBox.critical(self.tab, "Error", f"Failed to find hotspots: {exc}")

    def highlight_and_focus_on_node(self, node_id: int) -> None:
        """Highlight and focus camera on a specific node."""
        if self.tab.current_mesh is None:
            QMessageBox.warning(self.tab, "No Mesh", "Cannot highlight node - no mesh loaded.")
            return

        if self.state.highlight_actor:
            try:
                self.tab.plotter.renderer.RemoveActor(self.state.highlight_actor)
            except Exception:
                pass
            self.set_state_attr("highlight_actor", None)

        try:
            node_indices = np.where(self.tab.current_mesh['NodeID'] == node_id)[0]
            if len(node_indices) == 0:
                print(f"Node ID {node_id} not found in current mesh.")
                return

            point_index = node_indices[0]
            point_coords = self.tab.current_mesh.points[point_index]
            label_text = f"Node {node_id}"

            highlight_actor = self.tab.plotter.add_point_labels(
                point_coords, [label_text],
                name="hotspot_label",
                font_size=16,
                point_color='black',
                shape_opacity=0.5,
                point_size=self.tab.point_size.value() * 2,
                text_color='purple',
                always_visible=True,
                reset_camera=False  # Preserve current camera view before fly_to
            )

            self.set_state_attr("highlight_actor", highlight_actor)
            self.tab.plotter.fly_to(point_coords)

        except Exception as exc:
            QMessageBox.critical(
                self.tab, "Visualization Error",
                f"Could not highlight node {node_id}: {exc}"
            )

    def cleanup_hotspot_analysis(self) -> None:
        """Remove highlight labels and re-enable box widget after analysis."""
        if self.state.highlight_actor:
            try:
                self.tab.plotter.remove_actor("hotspot_label", reset_camera=False)
            except Exception:
                pass
            self.set_state_attr("highlight_actor", None)

        self.clear_goto_node_markers()

        if self.state.box_widget:
            self.state.box_widget.On()

        self.set_state_attr("hotspot_dialog", None)
        self.tab.plotter.render()

    # ------------------------------------------------------------------
    # Time history picking
    # ------------------------------------------------------------------
    def enable_time_history_picking(self, checked: bool = False) -> None:
        """Activate point picking for time-history plotting."""
        if (
            self.state.target_node_index is not None
            and self.state.target_node_id is not None
        ):
            reply = QMessageBox.question(
                self.tab,
                "Use Tracked Node?",
                f"Plot time history for tracked node {self.state.target_node_id}?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            if reply == QMessageBox.Yes:
                self.tab.node_picked_signal.emit(self.state.target_node_id)
                return

        if not self.tab.current_mesh or 'NodeID' not in self.tab.current_mesh.array_names:
            QMessageBox.warning(
                self.tab, "No Data",
                "Cannot pick a point. Please load data with NodeIDs first."
            )
            return

        print("Picking mode enabled: Click on a node to plot its time history.")
        self.tab.plotter.enable_point_picking(
            callback=self.on_point_picked_for_history,
            show_message=False,
            use_picker=True,
            left_clicking=True
        )
        self.tab.plotter.setCursor(Qt.CrossCursor)

    def on_point_picked_for_history(self, *args) -> None:
        """Handle node selection when plotting time history."""
        self.tab.plotter.disable_picking()
        self.tab.plotter.setCursor(Qt.ArrowCursor)

        if not args or len(args) == 0 or not isinstance(args[0], (np.ndarray, list, tuple)):
            print("Picking cancelled or missed the mesh.")
            return

        picked_coords = args[0]
        if len(picked_coords) == 0:
            print("Picking cancelled or missed the mesh.")
            return

        picked_point_index = self.tab.current_mesh.find_closest_point(picked_coords)

        if picked_point_index != -1 and picked_point_index < self.tab.current_mesh.n_points:
            try:
                node_id = self.tab.current_mesh['NodeID'][picked_point_index]
                point_coords = self.tab.current_mesh.points[picked_point_index]
                
                # Add visual indicator for picked node
                self._show_pick_indicator(point_coords, node_id)
                
                print(f"Node {node_id} picked. Emitting signal...")
                self.tab.node_picked_signal.emit(node_id)
            except (KeyError, IndexError) as exc:
                print(f"Could not retrieve NodeID: {exc}")
        else:
            print("No valid point selected.")

    def _show_pick_indicator(self, point_coords: np.ndarray, node_id: int) -> None:
        """Display a visual indicator (labeled point) at the picked node."""
        # Remove previous pick indicator if it exists
        if hasattr(self.state, 'pick_indicator_actor') and self.state.pick_indicator_actor:
            try:
                self.tab.plotter.remove_actor(self.state.pick_indicator_actor)
            except Exception:
                pass
            self.state.pick_indicator_actor = None

        try:
            # Create a labeled point indicator matching legacy style
            label_text = f"Node {node_id}"
            pick_actor = self.tab.plotter.add_point_labels(
                point_coords,
                [label_text],
                name='pick_indicator',
                font_size=16,
                point_color='black',
                shape_opacity=0.5,
                point_size=self.tab.point_size.value() * 2,
                text_color='red',
                always_visible=True,
                reset_camera=False  # Preserve current camera view
            )
            self.state.pick_indicator_actor = pick_actor
            self.tab.plotter.render()
        except Exception as exc:
            print(f"Could not show pick indicator: {exc}")

    # ------------------------------------------------------------------
    # Node navigation & tracking
    # ------------------------------------------------------------------
    def go_to_node(self, checked: bool = False) -> None:
        """Prompt for a Node ID and move the camera to that node."""
        if not self.tab.current_mesh or 'NodeID' not in self.tab.current_mesh.array_names:
            QMessageBox.warning(
                self.tab, "Action Unavailable",
                "No mesh with NodeIDs is currently loaded."
            )
            return

        default_val = self.state.last_goto_node_id if self.state.last_goto_node_id is not None else 0
        node_id, ok = QInputDialog.getInt(
            self.tab, "Go To Node", "Enter Node ID:", value=default_val
        )
        if not ok:
            return

        try:
            self.clear_goto_node_markers()

            node_indices = np.where(self.tab.current_mesh['NodeID'] == node_id)[0]
            if not node_indices.size:
                QMessageBox.warning(self.tab, "Not Found", f"Node ID {node_id} was not found.")
                return

            point_index = node_indices[0]
            point_coords = self.tab.current_mesh.points[point_index]

            marker_poly = pv.PolyData([point_coords])
            marker_actor = self.tab.plotter.add_points(
                marker_poly,
                color='black',
                point_size=self.tab.point_size.value() * 2,
                render_points_as_spheres=True,
                opacity=0.3,
            )

            label_point_data = pv.PolyData([point_coords])
            label_actor = self.tab.plotter.add_point_labels(
                label_point_data, [f"Node {node_id}"],
                name="target_node_label",
                font_size=16, text_color='red',
                always_visible=True, show_points=False
            )

            self.set_state_attr("marker_poly", marker_poly)
            self.set_state_attr("target_node_marker_actor", marker_actor)
            self.set_state_attr("label_point_data", label_point_data)
            self.set_state_attr("target_node_label_actor", label_actor)
            self.set_state_attr("target_node_index", int(point_index))
            self.set_state_attr("target_node_id", int(node_id))
            self.set_state_attr("last_goto_node_id", int(node_id))

            self.tab.plotter.fly_to(point_coords)

        except Exception as exc:
            QMessageBox.critical(self.tab, "Error", f"Could not go to node {node_id}: {exc}")

    def toggle_freeze_node(self, checked: bool) -> None:
        """Freeze or unfreeze camera tracking for the selected node."""
        if self.state.target_node_index is None or self.tab.current_mesh is None:
            QMessageBox.warning(
                self.tab, "No Tracked Node",
                "Use 'Go To Node' first, then lock the camera."
            )
            return

        self.set_state_attr("freeze_tracked_node", checked)
        if checked:
            baseline = self.tab.current_mesh.points[self.state.target_node_index].copy()
            self.set_state_attr("freeze_baseline", baseline)

            if self.tab.anim_timer is not None and self.tab.anim_timer.isActive():
                if self.state.target_node_label_actor:
                    self.state.target_node_label_actor.SetVisibility(False)
                if self.state.target_node_marker_actor:
                    self.state.target_node_marker_actor.SetVisibility(False)
        else:
            self.set_state_attr("freeze_baseline", None)

            if self.state.target_node_label_actor:
                self.state.target_node_label_actor.SetVisibility(True)
            if self.state.target_node_marker_actor:
                self.state.target_node_marker_actor.SetVisibility(True)

        self.tab.plotter.render()

    def clear_goto_node_markers(self) -> None:
        """Remove node markers and reset tracking state."""
        if self.state.target_node_marker_actor:
            try:
                self.tab.plotter.remove_actor(self.state.target_node_marker_actor)
            except Exception:
                pass

        if self.state.target_node_label_actor:
            try:
                self.tab.plotter.remove_actor(self.state.target_node_label_actor)
            except Exception:
                pass

        self.set_state_attr("target_node_marker_actor", None)
        self.set_state_attr("target_node_label_actor", None)
        self.set_state_attr("marker_poly", None)
        self.set_state_attr("label_point_data", None)
        self.set_state_attr("target_node_index", None)
        self.set_state_attr("target_node_id", None)
        self.set_state_attr("freeze_tracked_node", False)
        self.set_state_attr("freeze_baseline", None)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _add_section_title(menu: QMenu, title: str) -> None:
        """Insert a styled section title into the context menu."""
        label = QLabel(title)
        label.setProperty("class", "titleLabel")
        action = QWidgetAction(menu)
        action.setDefaultWidget(label)
        menu.addAction(action)
