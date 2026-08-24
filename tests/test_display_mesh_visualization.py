import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyvista as pv
import vtk
from PyQt5.QtCore import QCoreApplication, QObject, QPoint, pyqtSlot
from PyQt5.QtWidgets import QApplication

from core.data_models import MeshTopologyData
from core.visualization import HotspotDetector
from file_io.dpf_reader import DPFAnalysisReader, MeshTopologyProvider, _SurfaceTopology
from ui.application_controller import ApplicationController
from ui.handlers.display_interaction_handler import DisplayInteractionHandler
from ui.handlers.display_state import DisplayState
from ui.handlers.display_visualization_handler import DisplayVisualizationHandler
from ui.solver_tab import SolverTab
from ui.widgets.dialogs import HotspotDialog


class _Combo:
    def __init__(self, values, index=0):
        self.values = values
        self.index = index
        self.enabled = True

    def currentData(self):
        return self.values[self.index]

    def currentIndex(self):
        return self.index

    def setCurrentIndex(self, index):
        self.index = index

    def blockSignals(self, _blocked):
        pass

    def setEnabled(self, enabled):
        self.enabled = bool(enabled)


class _ValueWidget:
    def __init__(self, value):
        self._value = value
        self.enabled = True

    def value(self):
        return self._value

    def setEnabled(self, enabled):
        self.enabled = bool(enabled)


class _CheckBox:
    def __init__(self, checked=True):
        self.checked = checked
        self.enabled = False

    def isChecked(self):
        return self.checked

    def setEnabled(self, enabled):
        self.enabled = bool(enabled)


class _TextWidget:
    def __init__(self, text):
        self._text = str(text)

    def text(self):
        return self._text


class _Actor:
    def __init__(self):
        self.mapper = SimpleNamespace(scalar_range=None)
        self.edge_visibility = None
        self.property = SimpleNamespace(
            SetPointSize=lambda _value: None,
            SetEdgeVisibility=lambda value: setattr(
                self, "edge_visibility", bool(value)
            ),
        )

    def GetProperty(self):
        return self.property


class _Plotter:
    def __init__(self):
        self.mesh_calls = []
        self.text_calls = []
        self.scalar_bars = {}
        self.actors = {}
        self.reset_camera_calls = 0
        self.renderer = vtk.vtkRenderer()
        render_window = SimpleNamespace(Render=lambda: None)
        interactor = SimpleNamespace(GetRenderWindow=lambda: render_window)
        self.iren = SimpleNamespace(
            interactor=interactor,
            add_observer=lambda *_args: 1,
            remove_observer=lambda *_args: None,
        )

    def clear(self):
        self.mesh_calls.clear()
        self.text_calls.clear()
        self.scalar_bars.clear()
        self.actors.clear()

    def add_mesh(self, mesh, **kwargs):
        self.mesh_calls.append((mesh, kwargs))
        scalar_bar_args = kwargs.get("scalar_bar_args", {})
        if kwargs.get("show_scalar_bar") and scalar_bar_args.get("title"):
            self.scalar_bars[scalar_bar_args["title"]] = vtk.vtkScalarBarActor()
        actor = _Actor()
        if kwargs.get("name"):
            self.actors[kwargs["name"]] = actor
        return actor

    def add_text(self, text, **kwargs):
        self.text_calls.append((text, kwargs))
        return pv.CornerAnnotation(kwargs.get("position", "upper_left"), text)

    def add_actor(self, actor, **_kwargs):
        self.actors[f"actor_{len(self.actors)}"] = actor

    def reset_camera(self):
        self.reset_camera_calls += 1

    def render(self):
        pass

    def width(self):
        return 800

    def height(self):
        return 600


class _TopologyReceiver(QObject):
    def __init__(self, handler):
        super().__init__()
        self.handler = handler

    @pyqtSlot(object)
    def completed(self, worker):
        self.handler.on_topology_worker_finished(worker)


def _handler(view="points", scope="result"):
    point_mesh = pv.PolyData(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    point_mesh["NodeID"] = np.array([10, 20, 30])
    point_mesh["Result"] = np.array([1.0, 2.0, 3.0])
    point_mesh.set_active_scalars("Result")
    state = DisplayState(current_mesh=point_mesh, data_column="Result")
    tab = SimpleNamespace(
        current_mesh=point_mesh,
        current_actor=None,
        current_contour_type=None,
        mesh_view_combo=_Combo(["points", "contour_mesh", "mesh_points"], ["points", "contour_mesh", "mesh_points"].index(view)),
        mesh_scope_combo=_Combo(["result", "whole"], ["result", "whole"].index(scope)),
        mesh_edges_checkbox=_CheckBox(),
        point_size=_ValueWidget(8),
        scalar_min_spin=_ValueWidget(1.0),
        scalar_max_spin=_ValueWidget(3.0),
        deformation_scale_edit=_TextWidget("0"),
        deformation_result=None,
        original_node_coords=point_mesh.points.copy(),
        view_combination_combo=_Combo([None]),
        data_column="Result",
        plotter=_Plotter(),
        camera_widget=None,
        hover_annotation=None,
        hover_observer=None,
        last_hover_time=0.0,
        _camera_widget_pending=False,
        isVisible=lambda: False,
    )
    handler = DisplayVisualizationHandler(tab, state, SimpleNamespace())
    handler.setup_hover_annotation = lambda: None
    handler._clear_camera_widget = lambda: None
    return handler, tab


def _topology(include_whole=True):
    return MeshTopologyData(
        result_faces=np.array([3, 0, 1, 2]),
        result_lines=np.empty(0, dtype=np.int64),
        whole_points_mm=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]) if include_whole else None,
        whole_faces=np.array([3, 0, 1, 2]) if include_whole else None,
        whole_lines=np.empty(0, dtype=np.int64) if include_whole else None,
        context_faces=np.empty(0, dtype=np.int64) if include_whole else None,
        context_lines=np.empty(0, dtype=np.int64) if include_whole else None,
    )


def test_overlay_backgrounds_are_independent_and_reapplied():
    visual_handler, tab = _handler()
    state = visual_handler.state
    state.hover_background_enabled = True
    state.legend_background_enabled = True

    DisplayVisualizationHandler.setup_hover_annotation(visual_handler)
    hover = state.hover_annotation
    assert tab.plotter.text_calls[-1][1]["position"] == "upper_left"
    assert hover.GetTextProperty().GetBackgroundOpacity() == 0.65

    visual_handler._add_scalar_actor(tab.plotter, tab.current_mesh, "Result", points=True)
    legend = tab.plotter.scalar_bars["Result"]
    assert legend.GetDrawBackground()
    assert legend.GetBarRatio() == 0.145
    assert legend.GetBackgroundProperty().GetOpacity() == 0.65
    assert legend.GetTitleTextProperty().GetBackgroundOpacity() == 0.65

    hover.SetText(hover.UpperLeft, "Node ID: 20")
    interaction = DisplayInteractionHandler(tab, state, SimpleNamespace())
    assert interaction._is_click_on_hover_values(QPoint(100, 50))
    assert not interaction._is_click_on_hover_values(QPoint(700, 50))

    interaction._set_hover_background(False)
    assert not state.hover_background_enabled
    assert state.legend_background_enabled
    assert hover.GetTextProperty().GetBackgroundOpacity() == 0.0
    assert legend.GetDrawBackground()

    interaction._set_legend_background(False)
    assert not state.legend_background_enabled
    assert not legend.GetDrawBackground()
    assert legend.GetTitleTextProperty().GetBackgroundOpacity() == 0.0


def test_hover_can_lock_to_goto_node_selection():
    visual_handler, tab = _handler()
    tab.visual_handler = visual_handler
    state = visual_handler.state
    state.target_node_id = 20
    DisplayVisualizationHandler.setup_hover_annotation(visual_handler)
    interaction = DisplayInteractionHandler(tab, state, SimpleNamespace())

    interaction._set_hover_node_lock(True)

    hover = state.hover_annotation
    assert state.locked_hover_node_id == 20
    assert hover.GetText(hover.UpperLeft) == "Node ID: 20\nResult: 2.00000"

    interaction._set_hover_node_lock(False)
    assert state.locked_hover_node_id is None
    assert hover.GetText(hover.UpperLeft) == ""


def test_stress_hover_follows_the_displayed_scalar():
    handler, tab = _handler()
    mesh = tab.current_mesh
    tab.combination_names = ["Landing", "Thermal", "Burst"]
    mesh["Max_Stress"] = np.array([10.0, 20.0, 30.0])
    mesh["Combo_of_Max"] = np.array([2, 0, 1])
    mesh["Combo_1_Stress"] = np.array([11.0, 21.0, 31.0])

    expected = {
        "Combo_of_Max": "Combo of Max: Combo #1 — Landing",
        "Max_Stress": "Max: 20.00000 MPa (Combo #1 — Landing)",
        "Combo_1_Stress": "Combo #2 — Thermal: 21.00000 MPa",
    }
    for active_name, expected_line in expected.items():
        mesh.set_active_scalars(active_name)
        lines = []
        handler._append_stress_hover_line(lines, mesh, 1)
        assert lines == [expected_line]


def test_mouse_pivot_orbit_keeps_the_picked_point_anchored():
    mesh = pv.Sphere(radius=1.0, theta_resolution=60, phi_resolution=60)
    mesh["Result"] = mesh.points[:, 2]
    plotter = pv.Plotter(off_screen=True, window_size=(800, 600))
    actor = plotter.add_mesh(mesh, scalars="Result", pickable=True)
    plotter.enable_parallel_projection()
    plotter.camera_position = [(0, 0, 5), (0, 0, 0), (0, 1, 0)]
    plotter.camera.parallel_scale = 2.0
    plotter.render()

    state = DisplayState(current_mesh=mesh, current_actor=actor, data_column="Result")
    tab = SimpleNamespace(plotter=plotter, current_mesh=mesh, current_actor=actor)
    handler = DisplayInteractionHandler(tab, state, SimpleNamespace())
    handler.enable_mouse_pivot_rotation()

    expected_pivot = np.array([0.5, 0.0, np.sqrt(0.75)])
    renderer = plotter.renderer
    renderer.SetWorldPoint(*expected_pivot, 1.0)
    renderer.WorldToDisplay()
    mouse_position = np.asarray(renderer.GetDisplayPoint())[:2]
    interactor = plotter.iren.interactor
    interactor.SetEventInformation(*mouse_position.astype(int))
    interactor.InvokeEvent("LeftButtonPressEvent")

    picked_pivot = np.asarray(handler._rotation_pivot)
    renderer.SetWorldPoint(*picked_pivot, 1.0)
    renderer.WorldToDisplay()
    display_before = np.asarray(renderer.GetDisplayPoint())[:2]

    interactor.SetEventInformation(
        int(mouse_position[0] + 30),
        int(mouse_position[1] - 15),
    )
    interactor.InvokeEvent("MouseMoveEvent")
    renderer.SetWorldPoint(*picked_pivot, 1.0)
    renderer.WorldToDisplay()
    display_after = np.asarray(renderer.GetDisplayPoint())[:2]
    camera_position = np.asarray(plotter.camera.position)
    interactor.InvokeEvent("LeftButtonReleaseEvent")

    np.testing.assert_allclose(picked_pivot, expected_pivot, atol=0.05)
    np.testing.assert_allclose(display_after, display_before, atol=1e-8)
    assert np.linalg.norm(camera_position - np.array([0.0, 0.0, 5.0])) > 0.1
    assert not handler._mouse_pivot_rotation_active
    assert plotter.iren.get_interactor_style().GetState() == 0

    position_before_pan = np.asarray(plotter.camera.position)
    focal_point_before_pan = np.asarray(plotter.camera.focal_point)
    interactor.SetEventInformation(*mouse_position.astype(int))
    interactor.InvokeEvent("MiddleButtonPressEvent")
    interactor.SetEventInformation(
        int(mouse_position[0] + 30),
        int(mouse_position[1] - 15),
    )
    interactor.InvokeEvent("MouseMoveEvent")
    interactor.InvokeEvent("MiddleButtonReleaseEvent")
    pan_translation = np.asarray(plotter.camera.position) - position_before_pan
    np.testing.assert_allclose(
        np.asarray(plotter.camera.focal_point) - focal_point_before_pan,
        pan_translation,
    )
    assert np.linalg.norm(pan_translation) > 0.0

    fallback_pivot = np.asarray(plotter.camera.focal_point)
    interactor.SetEventInformation(5, 5)
    interactor.InvokeEvent("LeftButtonPressEvent")
    np.testing.assert_allclose(handler._rotation_pivot, fallback_pivot)
    interactor.InvokeEvent("LeftButtonReleaseEvent")

    state.is_point_picking_active = True
    interactor.SetEventInformation(*mouse_position.astype(int))
    interactor.InvokeEvent("LeftButtonPressEvent")
    assert not handler._mouse_pivot_rotation_active
    interactor.InvokeEvent("LeftButtonReleaseEvent")

    state.is_point_picking_active = False
    interactor.SetEventInformation(*mouse_position.astype(int), 0, 1)
    interactor.InvokeEvent("LeftButtonPressEvent")
    assert not handler._mouse_pivot_rotation_active
    interactor.InvokeEvent("LeftButtonReleaseEvent")
    plotter.close()


def test_points_mode_never_invokes_topology_provider():
    handler, tab = _handler()
    provider = SimpleNamespace(build_visualization_topology=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("called")))
    handler.set_topology_provider(provider)

    handler.update_visualization()

    assert len(tab.plotter.mesh_calls) == 1
    assert tab.plotter.mesh_calls[0][1]["style"] == "points"
    assert handler._topology_worker is None


def test_result_scope_defers_whole_model_connectivity():
    provider = MeshTopologyProvider("unused.rst")
    provider._surface = _SurfaceTopology(
        node_ids=np.array([10, 20, 30]),
        points_mm=np.zeros((3, 3)),
        face_offsets=np.array([0, 3]),
        face_connectivity=np.array([0, 1, 2]),
        line_offsets=np.array([0]),
        line_connectivity=np.empty(0, dtype=np.int64),
    )

    result = provider.build_visualization_topology(
        np.array([10, 20, 30]),
        include_whole_model=False,
    )

    assert result.whole_faces is None
    assert provider._whole_cells is None
    assert provider._scope_context is None


def test_render_modes_keep_one_scalar_actor_and_non_pickable_context():
    handler, tab = _handler(view="mesh_points")
    handler._topology_provider = object()
    handler._topology_data = _topology()
    handler._topology_node_ids = np.asarray(tab.current_mesh["NodeID"]).copy()
    handler._topology_includes_whole = True
    handler._build_topology_meshes()

    handler.update_visualization()
    assert len(tab.plotter.mesh_calls) == 2
    assert tab.plotter.mesh_calls[0][1]["pickable"] is False
    assert tab.plotter.mesh_calls[0][1]["show_edges"] is True
    assert tab.plotter.mesh_calls[1][1]["pickable"] is True
    assert tab.plotter.mesh_calls[1][1]["style"] == "points"

    tab.mesh_edges_checkbox.checked = False
    handler.update_visualization()
    assert tab.plotter.mesh_calls[0][1]["show_edges"] is False

    tab.mesh_view_combo.index = 1
    handler.update_visualization()
    assert len(tab.plotter.mesh_calls) == 1
    assert tab.plotter.mesh_calls[0][1]["pickable"] is True
    assert tab.plotter.mesh_calls[0][1]["show_edges"] is False

    mesh_calls_before = len(tab.plotter.mesh_calls)
    handler.on_mesh_edges_changed(True)
    assert tab.current_actor.edge_visibility is True
    assert len(tab.plotter.mesh_calls) == mesh_calls_before


def test_view_mode_change_does_not_reset_camera():
    handler, tab = _handler(view="points")
    handler._topology_provider = object()
    handler._topology_data = _topology()
    handler._topology_node_ids = np.asarray(tab.current_mesh["NodeID"]).copy()
    handler._topology_includes_whole = True
    handler._build_topology_meshes()

    tab.mesh_view_combo.index = 1
    handler.on_mesh_view_changed()

    assert tab.plotter.reset_camera_calls == 0
    assert all(call[1]["reset_camera"] is False for call in tab.plotter.mesh_calls)


def test_view_mode_change_preserves_goto_node_selection():
    handler, tab = _handler(view="points")
    handler._topology_provider = object()
    handler._topology_data = _topology()
    handler._topology_node_ids = np.asarray(tab.current_mesh["NodeID"]).copy()
    handler._topology_includes_whole = True
    handler._build_topology_meshes()
    interaction = DisplayInteractionHandler(tab, handler.state, HotspotDetector())
    tab.interaction_handler = interaction
    marker_actor = _Actor()
    label_actor = _Actor()
    handler.state.target_node_id = 20
    handler.state.target_node_index = 1
    handler.state.target_node_marker_actor = marker_actor
    handler.state.target_node_label_actor = label_actor
    handler.state.marker_poly = pv.PolyData([[9.0, 9.0, 9.0]])
    handler.state.label_point_data = pv.PolyData([[9.0, 9.0, 9.0]])

    tab.mesh_view_combo.index = 1
    handler.on_mesh_view_changed()

    assert handler.state.target_node_id == 20
    assert marker_actor in tab.plotter.actors.values()
    assert label_actor in tab.plotter.actors.values()
    np.testing.assert_allclose(handler.state.marker_poly.points[0], [1.0, 0.0, 0.0])


def test_current_view_hotspots_exclude_occluded_and_offscreen_points():
    mesh = pv.PolyData(np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [10.0, 0.0, 0.0],
        [0.4, 0.4, 0.0],
    ]))
    mesh["NodeID"] = np.array([1, 2, 3, 4])
    mesh["Result"] = np.arange(4.0)
    mesh.set_active_scalars("Result")
    plotter = pv.Plotter(off_screen=True, window_size=(400, 400))
    actor = plotter.add_mesh(
        mesh, style="points", point_size=20, render_points_as_spheres=True
    )
    plotter.camera_position = [(0, 0, 5), (0, 0, 0), (0, 1, 0)]
    plotter.show(auto_close=False)
    state = DisplayState(current_mesh=mesh, current_actor=actor)
    tab = SimpleNamespace(current_mesh=mesh, current_actor=actor, plotter=plotter)
    interaction = DisplayInteractionHandler(tab, state, HotspotDetector())
    analyzed = []
    interaction._find_and_show_hotspots = analyzed.append

    try:
        interaction.find_hotspots_on_view()
        assert analyzed[0]["NodeID"].tolist() == [1, 4]
    finally:
        plotter.close()


def test_hotspot_table_shows_the_combination_for_each_envelope_value():
    visual_handler, tab = _handler()
    tab.combination_names = ["Landing", "Thermal", "Burst"]
    mesh = tab.current_mesh
    mesh["Max_Stress"] = np.array([30.0, 10.0, 20.0])
    mesh["Combo_of_Max"] = np.array([1, 0, 2])
    mesh.set_active_scalars("Max_Stress")

    interaction = DisplayInteractionHandler(tab, visual_handler.state, HotspotDetector())
    hotspots = HotspotDetector.detect_hotspots(
        mesh["Max_Stress"],
        mesh["NodeID"],
        node_coords=mesh.points,
        top_n=3,
    ).rename(columns={"Value": "Max_Stress"})

    interaction._add_hotspot_combination_column(hotspots, mesh, "Max_Stress")

    assert list(hotspots.columns[:4]) == [
        "Rank", "NodeID", "Max_Stress", "Combo of Max"
    ]
    assert hotspots["Combo of Max"].tolist() == [
        "#2 — Thermal", "#3 — Burst", "#1 — Landing"
    ]

    app = QApplication.instance() or QApplication([])
    dialog = HotspotDialog(hotspots)
    assert dialog.model.item(0, 1).text() == "10"
    assert dialog.model.item(0, 2).text() == "30.0000"
    assert dialog.model.item(0, 3).text() == "#2 — Thermal"
    selected_nodes = []
    dialog.node_selected.connect(selected_nodes.append)
    dialog._on_row_clicked(dialog.model.index(0, 3))
    assert selected_nodes == [10]
    dialog.close()
    assert app is not None


def test_hotspot_combo_field_tracks_the_active_envelope_family():
    expected = {
        "Max_Stress": ("Combo of Max", "Combo_of_Max"),
        "Min_Force_Magnitude": ("Combo of Min", "Combo_of_Min"),
        "Max_FX": ("Combo of Max", "Combo_of_Max_FX"),
        "Min_Shear_XY": ("Combo of Min", "Combo_of_Min_Shear_XY"),
        "Def_Max_U_mag": ("Combo of Max", "Def_Combo_of_Max_U_mag"),
        "Def_Min_UZ": ("Combo of Min", "Def_Combo_of_Min_UZ"),
        "Combo_2_Stress": (None, None),
    }

    for active_name, combo_column in expected.items():
        assert DisplayInteractionHandler._combo_column_for_scalar(active_name) == combo_column


def test_hotspot_combo_column_falls_back_to_number_and_skips_missing_metadata():
    visual_handler, tab = _handler()
    tab.combination_names = []
    mesh = tab.current_mesh
    mesh["Min_FX"] = np.array([3.0, 1.0, 2.0])
    mesh["Combo_of_Min_FX"] = np.array([2, 0, 1])
    interaction = DisplayInteractionHandler(tab, visual_handler.state, HotspotDetector())
    hotspots = HotspotDetector.detect_hotspots(
        mesh["Min_FX"], mesh["NodeID"], top_n=3
    ).rename(columns={"Value": "Min_FX"})

    interaction._add_hotspot_combination_column(hotspots, mesh, "Min_FX")
    assert hotspots["Combo of Min"].tolist() == ["#3", "#2", "#1"]

    without_metadata = hotspots.drop(columns="Combo of Min")
    interaction._add_hotspot_combination_column(without_metadata, mesh, "Max_FY")
    assert list(without_metadata.columns) == ["Rank", "NodeID", "Min_FX"]


def test_mesh_edge_control_is_enabled_only_for_mesh_views():
    handler, tab = _handler(view="points")
    handler._topology_provider = object()

    handler.update_mesh_control_state()
    assert tab.mesh_edges_checkbox.enabled is False

    tab.mesh_view_combo.index = 1
    handler.update_mesh_control_state()
    assert tab.mesh_edges_checkbox.enabled is True


def test_nonzero_deformation_forces_result_scope():
    handler, tab = _handler(view="mesh_points", scope="whole")
    handler._topology_provider = object()
    tab.deformation_result = object()
    tab.deformation_scale_edit = _TextWidget("1")

    handler.update_mesh_control_state()

    assert tab.mesh_scope_combo.currentData() == "result"
    assert tab.mesh_scope_combo.enabled is False


def test_slow_topology_provider_does_not_block_requesting_thread():
    app = QCoreApplication.instance() or QCoreApplication([])
    handler, tab = _handler(view="contour_mesh")
    receiver = _TopologyReceiver(handler)
    tab._on_mesh_topology_worker_completed = receiver.completed

    class _SlowProvider:
        def build_visualization_topology(self, _node_ids, include_whole_model=False):
            time.sleep(0.2)
            return _topology(include_whole_model)

    handler.set_topology_provider(_SlowProvider())
    started = time.perf_counter()
    handler._request_topology()
    request_elapsed = time.perf_counter() - started

    deadline = time.perf_counter() + 2.0
    while handler._topology_worker is not None and time.perf_counter() < deadline:
        app.processEvents()
        time.sleep(0.01)

    assert request_elapsed < 0.1
    assert handler._topology_data is not None


def test_shutdown_waits_for_active_topology_worker():
    app = QCoreApplication.instance() or QCoreApplication([])
    handler, tab = _handler(view="contour_mesh")
    receiver = _TopologyReceiver(handler)
    tab._on_mesh_topology_worker_completed = receiver.completed

    class _SlowProvider:
        def build_visualization_topology(self, _node_ids, include_whole_model=False):
            time.sleep(0.15)
            return _topology(include_whole_model)

    handler.set_topology_provider(_SlowProvider())
    handler._request_topology()
    worker = handler._topology_worker
    handler.shutdown()

    assert worker.isRunning() is False
    app.processEvents()


def test_application_close_explicitly_shuts_down_topology_worker():
    calls = []
    controller = SimpleNamespace(
        display_tab=SimpleNamespace(
            visual_handler=SimpleNamespace(shutdown=lambda: calls.append("shutdown"))
        ),
        plotting_handler=SimpleNamespace(
            cleanup_temp_files=lambda: calls.append("cleanup")
        ),
        _tooltip_filter=QObject(),
    )
    event = SimpleNamespace(accept=lambda: calls.append("accept"))

    ApplicationController.closeEvent(controller, event)

    assert calls == ["shutdown", "cleanup", "accept"]


def test_history_picker_uses_picked_dataset_node_id_not_nearest_coordinate():
    current = pv.PolyData(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
    current["NodeID"] = np.array([10, 20])
    emitted = []
    tab = SimpleNamespace(
        current_mesh=current,
        plotter=SimpleNamespace(disable_picking=lambda: None, setCursor=lambda _cursor: None),
        node_picked_for_history_popup=SimpleNamespace(emit=emitted.append),
    )
    handler = DisplayInteractionHandler(tab, DisplayState(current_mesh=current), SimpleNamespace())
    handler._show_pick_indicator = lambda _coords, _node_id: None
    picker = SimpleNamespace(GetDataSet=lambda: current, GetPointId=lambda: 1)

    handler.on_point_picked_for_history(np.array([0.0, 0.0, 0.0]), picker)

    assert emitted == [20]


def test_controller_routes_history_to_active_display_result():
    source = object()
    calls = []
    controller = SimpleNamespace(
        display_tab=SimpleNamespace(
            current_contour_type="Forces",
            stress_result=None,
            nodal_forces_result=source,
            deformation_result=None,
        ),
        solver_tab=SimpleNamespace(
            plot_combination_history_for_node=lambda *args, **kwargs: calls.append((args, kwargs))
        ),
    )

    ApplicationController._trigger_node_history(controller, 42, popup=True)

    assert calls == [
        ((42,), {"open_popup": True, "result_family": "Forces", "source_result": source})
    ]


def test_solver_tab_cache_hit_does_not_start_solve_or_toggle_history_mode():
    class _Controller:
        def show_cached_history(self, *_args):
            return True

        def solve(self, _config):
            raise AssertionError("cache hit must not start Solve")

    line_values = []
    console_values = []
    tab = SimpleNamespace(
        node_line_edit=SimpleNamespace(setText=line_values.append),
        _history_popup_requested=False,
        solve_run_controller=_Controller(),
        console_textbox=SimpleNamespace(append=console_values.append),
    )

    SolverTab.plot_combination_history_for_node(
        tab,
        42,
        open_popup=True,
        result_family="Stress",
        source_result=object(),
    )

    assert line_values == ["42"]
    assert tab._history_popup_requested is False
    assert any("already-computed" in message for message in console_values)


def test_real_rst_topology_is_nonempty_and_cached():
    rst_path = Path(__file__).resolve().parents[1] / "example_dataset" / "file_analysis1_.rst"
    reader = DPFAnalysisReader(str(rst_path))
    node_ids = np.asarray(reader.model.metadata.named_selection("MYBODY_EXT_NODES").ids)
    provider = MeshTopologyProvider(str(rst_path))

    started = time.perf_counter()
    first = provider.build_visualization_topology(node_ids, include_whole_model=True)
    first_elapsed = time.perf_counter() - started
    surface = provider._surface
    second = provider.build_visualization_topology(node_ids, include_whole_model=True)

    assert first.has_result_cells
    assert first.whole_points_mm.shape[1] == 3
    assert provider._surface is surface
    assert second.result_faces is first.result_faces
    assert first_elapsed >= 0.0
