import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyvista as pv
import vtk
from PyQt5.QtCore import QCoreApplication, QObject, QPoint, pyqtSlot

from core.data_models import MeshTopologyData
from file_io.dpf_reader import DPFAnalysisReader, MeshTopologyProvider, _SurfaceTopology
from ui.application_controller import ApplicationController
from ui.handlers.display_interaction_handler import DisplayInteractionHandler
from ui.handlers.display_state import DisplayState
from ui.handlers.display_visualization_handler import DisplayVisualizationHandler


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
        self.renderer = vtk.vtkRenderer()
        self.iren = SimpleNamespace(
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

    def reset_camera(self):
        pass

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
    assert hover.GetTextProperty().GetBackgroundOpacity() == 0.65

    visual_handler._add_scalar_actor(tab.plotter, tab.current_mesh, "Result", points=True)
    legend = tab.plotter.scalar_bars["Result"]
    assert legend.GetDrawBackground()
    assert legend.GetBarRatio() == 0.145
    assert legend.GetBackgroundProperty().GetOpacity() == 0.65

    hover.SetText(hover.UpperRight, "Node ID: 20")
    interaction = DisplayInteractionHandler(tab, state, SimpleNamespace())
    assert interaction._is_click_on_hover_values(QPoint(700, 50))
    assert not interaction._is_click_on_hover_values(QPoint(200, 300))

    interaction._set_hover_background(False)
    assert not state.hover_background_enabled
    assert state.legend_background_enabled
    assert hover.GetTextProperty().GetBackgroundOpacity() == 0.0
    assert legend.GetDrawBackground()

    interaction._set_legend_background(False)
    assert not state.legend_background_enabled
    assert not legend.GetDrawBackground()


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
