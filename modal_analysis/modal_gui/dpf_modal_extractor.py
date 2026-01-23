"""DPF-backed modal extraction helpers."""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple
import gc

import numpy as np

try:
    from ansys.dpf import core as dpf
    DPF_AVAILABLE = True
except ImportError:
    dpf = None
    DPF_AVAILABLE = False

from . import csv_writer, memory_policy


class DPFNotAvailableError(RuntimeError):
    """Raised when ansys-dpf-core is not installed."""


class ModalExtractionError(RuntimeError):
    """Raised when modal extraction fails."""


class ModalExtractionCanceled(RuntimeError):
    """Raised when modal extraction is canceled."""


def _require_dpf() -> None:
    if not DPF_AVAILABLE:
        raise DPFNotAvailableError("ansys-dpf-core is not installed.")


def list_named_selections(rst_path: str) -> List[str]:
    _require_dpf()
    model = dpf.Model(rst_path)
    names = model.metadata.available_named_selections
    return list(names) if names else []


def get_model_info(rst_path: str) -> dict:
    _require_dpf()
    model = dpf.Model(rst_path)
    info = {
        "num_nodes": 0,
        "num_elements": 0,
        "n_sets": 0,
        "unit_system": "Unknown",
        "named_selections": [],
    }
    try:
        mesh = model.metadata.meshed_region
        info["num_nodes"] = mesh.nodes.n_nodes
        info["num_elements"] = mesh.elements.n_elements
    except Exception:
        pass
    try:
        info["n_sets"] = model.metadata.time_freq_support.n_sets
    except Exception:
        pass
    try:
        info["unit_system"] = str(model.metadata.result_info.unit_system)
    except Exception:
        pass
    try:
        names = model.metadata.available_named_selections
        info["named_selections"] = list(names) if names else []
    except Exception:
        pass
    return info


def get_n_sets(rst_path: str) -> int:
    _require_dpf()
    model = dpf.Model(rst_path)
    try:
        return model.metadata.time_freq_support.n_sets
    except Exception:
        return 0


def get_nodal_scoping_ids(rst_path: str, named_selection: Optional[str]) -> List[int]:
    _require_dpf()
    model = dpf.Model(rst_path)
    scoping = _sorted_scoping(_resolve_nodal_scoping(model, named_selection))
    return list(scoping.ids)


def _get_length_conversion_factor(source_unit: str) -> float:
    conversion_to_mm = {
        "m": 1000.0,
        "meter": 1000.0,
        "meters": 1000.0,
        "mm": 1.0,
        "millimeter": 1.0,
        "millimeters": 1.0,
        "cm": 10.0,
        "in": 25.4,
        "inch": 25.4,
        "inches": 25.4,
        "ft": 304.8,
        "um": 0.001,
    }
    if not source_unit:
        return 1.0
    return conversion_to_mm.get(str(source_unit).lower(), 1.0)


def _resolve_nodal_scoping(model: "dpf.Model", ns_name: Optional[str]) -> "dpf.Scoping":
    mesh = model.metadata.meshed_region

    if ns_name is None or ns_name == "All Nodes":
        scoping = dpf.Scoping(location=dpf.locations.nodal)
        scoping.ids = list(mesh.nodes.scoping.ids)
        return scoping

    ns_scoping = model.metadata.named_selection(ns_name)
    if ns_scoping.location == dpf.locations.nodal:
        return ns_scoping

    try:
        transpose_op = dpf.operators.scoping.transpose()
        transpose_op.inputs.mesh_scoping.connect(ns_scoping)
        transpose_op.inputs.meshed_region.connect(mesh)
        transpose_op.inputs.inclusive.connect(1)
        if hasattr(transpose_op.outputs, "mesh_scoping"):
            return transpose_op.outputs.mesh_scoping()
        if hasattr(transpose_op.outputs, "mesh_scoping_as_scoping"):
            return transpose_op.outputs.mesh_scoping_as_scoping()
        result = transpose_op.eval()
        if isinstance(result, dpf.Scoping):
            return result
    except Exception:
        pass

    if ns_scoping.location == dpf.locations.elemental:
        node_ids = set()
        for elem_id in ns_scoping.ids:
            try:
                elem_idx = mesh.elements.scoping.index(elem_id)
                connectivity = mesh.elements.element_by_index(elem_idx).connectivity
                node_ids.update(connectivity)
            except Exception:
                continue
        nodal_scoping = dpf.Scoping(location=dpf.locations.nodal)
        nodal_scoping.ids = list(node_ids)
        return nodal_scoping

    raise ModalExtractionError(f"Failed to convert named selection '{ns_name}' to nodal scoping.")


def _sorted_scoping(scoping: "dpf.Scoping") -> "dpf.Scoping":
    node_ids = sorted(scoping.ids)
    sorted_scoping = dpf.Scoping(location=dpf.locations.nodal)
    sorted_scoping.ids = node_ids
    return sorted_scoping


def _get_node_coordinates(mesh: "dpf.MeshedRegion", node_ids: Sequence[int]) -> np.ndarray:
    coords_field = mesh.nodes.coordinates_field
    try:
        unit = coords_field.unit or "mm"
    except Exception:
        unit = "mm"
    factor = _get_length_conversion_factor(unit)

    coords = np.zeros((len(node_ids), 3), dtype=float)
    for idx, node_id in enumerate(node_ids):
        try:
            node_idx = mesh.nodes.scoping.index(node_id)
            coords[idx, :] = coords_field.data[node_idx]
        except Exception:
            coords[idx, :] = 0.0
    return coords * factor


def _connect_if_present(op: "dpf.Operator", input_name: str, value) -> None:
    if hasattr(op.inputs, input_name):
        getattr(op.inputs, input_name).connect(value)


def _connect_mesh_scoping(op: "dpf.Operator", scoping: "dpf.Scoping") -> None:
    for name in ("mesh_scoping", "scoping", "mesh_scoping_as_scoping"):
        if hasattr(op.inputs, name):
            getattr(op.inputs, name).connect(scoping)
            return


def _connect_requested_location(op: "dpf.Operator", location) -> None:
    for name in ("requested_location", "location"):
        if hasattr(op.inputs, name):
            getattr(op.inputs, name).connect(location)
            return


def _connect_label(op: "dpf.Operator", label: str) -> None:
    for name in ("label1", "label"):
        if hasattr(op.inputs, name):
            getattr(op.inputs, name).connect(label)
            return


def _get_first_field(fields_container) -> "dpf.Field":
    if fields_container is None or len(fields_container) == 0:
        raise ModalExtractionError("DPF returned empty fields container.")
    return fields_container[0]


def _align_field_to_nodes(
    field: "dpf.Field",
    requested_node_ids: Sequence[int],
    expected_components: int,
) -> np.ndarray:
    data = np.array(field.data)
    if data.ndim == 1:
        if expected_components == 1:
            data = data.reshape(-1, 1)
        else:
            raise ModalExtractionError(
                f"Expected {expected_components} components but got 1D data."
            )
    if data.shape[1] < expected_components:
        raise ModalExtractionError(
            f"Expected {expected_components} components but got {data.shape[1]}."
        )
    data = data[:, :expected_components]

    dpf_node_ids = np.array(field.scoping.ids, dtype=int)
    requested = np.array(requested_node_ids, dtype=int)

    if len(dpf_node_ids) == len(requested) and np.array_equal(dpf_node_ids, requested):
        return data

    id_to_idx = {nid: idx for idx, nid in enumerate(dpf_node_ids)}
    aligned = np.zeros((len(requested), expected_components), dtype=float)
    for out_idx, node_id in enumerate(requested):
        src_idx = id_to_idx.get(node_id)
        if src_idx is not None:
            aligned[out_idx, :] = data[src_idx, :]
    return aligned


def _result_operator(model: "dpf.Model", kind: str) -> "dpf.Operator":
    if kind == "stress":
        return model.results.stress()
    if kind == "displacement":
        return model.results.displacement()
    if kind == "strain":
        for name in ("elastic_strain", "total_strain", "strain"):
            op_factory = getattr(model.results, name, None)
            if op_factory is None:
                continue
            try:
                return op_factory()
            except Exception:
                continue
        raise ModalExtractionError("No strain operator available in this DPF build.")
    raise ModalExtractionError(f"Unknown result kind '{kind}'.")


def _result_components(kind: str) -> Tuple[str, ...]:
    if kind == "stress":
        return ("sx", "sy", "sz", "sxy", "syz", "sxz")
    if kind == "strain":
        return ("ex", "ey", "ez", "exy", "eyz", "exz")
    if kind == "displacement":
        return ("ux", "uy", "uz")
    raise ModalExtractionError(f"Unknown result kind '{kind}'.")


def _average_to_nodal(
    field: "dpf.Field",
    mesh_scoping: "dpf.Scoping",
    mesh: Optional["dpf.MeshedRegion"] = None,
) -> "dpf.Field":
    try:
        op = dpf.operators.averaging.elemental_nodal_to_nodal()
    except Exception:
        op = dpf.operators.averaging.to_nodal()

    _connect_if_present(op, "field", field)
    _connect_if_present(op, "mesh_scoping", mesh_scoping)
    if mesh is not None:
        _connect_if_present(op, "mesh", mesh)
    try:
        _connect_if_present(op, "should_average", True)
    except Exception:
        pass
    return op.outputs.field()


def _nodal_to_elemental_scoping(
    mesh: "dpf.MeshedRegion",
    nodal_scoping: "dpf.Scoping",
) -> "dpf.Scoping":
    transpose_op = dpf.operators.scoping.transpose()
    _connect_if_present(transpose_op, "mesh_scoping", nodal_scoping)
    _connect_if_present(transpose_op, "meshed_region", mesh)
    _connect_if_present(transpose_op, "inclusive", 1)
    if hasattr(transpose_op.outputs, "mesh_scoping"):
        return transpose_op.outputs.mesh_scoping()
    if hasattr(transpose_op.outputs, "mesh_scoping_as_scoping"):
        return transpose_op.outputs.mesh_scoping_as_scoping()
    result = transpose_op.eval()
    if isinstance(result, dpf.Scoping):
        return result
    raise ModalExtractionError("Failed to transpose nodal scoping to elemental scoping.")


def _split_scopings_by_material(
    mesh: "dpf.MeshedRegion",
    element_scoping: Optional["dpf.Scoping"] = None,
) -> "dpf.ScopingsContainer":
    split_op = dpf.operators.scoping.split_on_property_type()
    _connect_if_present(split_op, "mesh", mesh)
    _connect_requested_location(split_op, dpf.locations.elemental)
    _connect_label(split_op, "mat")
    if element_scoping is not None:
        _connect_if_present(split_op, "mesh_scoping", element_scoping)
    return split_op.outputs.mesh_scoping()


def _average_across_bodies_field(
    model: "dpf.Model",
    kind: str,
    set_id: int,
    nodal_scoping: "dpf.Scoping",
    mesh: "dpf.MeshedRegion",
) -> "dpf.Field":
    element_scoping = _nodal_to_elemental_scoping(mesh, nodal_scoping)
    mat_scopings = _split_scopings_by_material(mesh, element_scoping)

    op = _result_operator(model, kind)
    time_scoping = dpf.Scoping()
    time_scoping.ids = [int(set_id)]
    _connect_if_present(op, "time_scoping", time_scoping)
    _connect_mesh_scoping(op, mat_scopings)
    if hasattr(dpf.locations, "elemental_nodal"):
        _connect_requested_location(op, dpf.locations.elemental_nodal)
    else:
        _connect_requested_location(op, dpf.locations.nodal)

    fc = op.outputs.fields_container()

    eln_to_n = dpf.operators.averaging.elemental_nodal_to_nodal_fc()
    _connect_if_present(eln_to_n, "fields_container", fc)
    _connect_if_present(eln_to_n, "mesh", mesh)
    _connect_if_present(eln_to_n, "scoping", nodal_scoping)
    _connect_if_present(eln_to_n, "extend_weights_to_mid_nodes", True)

    merge_op = dpf.operators.utility.weighted_merge_fields_by_label()
    _connect_if_present(merge_op, "fields_container", eln_to_n.outputs.fields_container())
    _connect_if_present(merge_op, "label", "mat")
    try:
        merge_op.connect(1000, eln_to_n, 1)
    except Exception:
        try:
            _connect_if_present(merge_op, "weights1", eln_to_n.outputs.weights())
        except Exception:
            pass

    merged_fc = merge_op.outputs.fields_container()
    return _get_first_field(merged_fc)


def _evaluate_result(
    model: "dpf.Model",
    kind: str,
    set_id: int,
    mesh_scoping: "dpf.Scoping",
    force_nodal_average: bool = False,
    mesh: Optional["dpf.MeshedRegion"] = None,
    average_across_bodies: bool = True,
) -> "dpf.Field":
    if force_nodal_average and kind in ("stress", "strain") and average_across_bodies and mesh is not None:
        try:
            return _average_across_bodies_field(model, kind, set_id, mesh_scoping, mesh)
        except Exception:
            pass

    op = _result_operator(model, kind)

    time_scoping = dpf.Scoping()
    time_scoping.ids = [int(set_id)]
    _connect_if_present(op, "time_scoping", time_scoping)
    _connect_mesh_scoping(op, mesh_scoping)

    if force_nodal_average and kind in ("stress", "strain") and hasattr(dpf.locations, "elemental_nodal"):
        _connect_requested_location(op, dpf.locations.elemental_nodal)
    else:
        _connect_requested_location(op, dpf.locations.nodal)

    fields_container = op.outputs.fields_container()
    field = _get_first_field(fields_container)

    if force_nodal_average and kind in ("stress", "strain"):
        try:
            return _average_to_nodal(field, mesh_scoping, mesh)
        except Exception:
            return field

    return field


def _mode_ids_from_count(n_sets: int, mode_count: Optional[int]) -> List[int]:
    if n_sets <= 0:
        return []
    if mode_count is None:
        return list(range(1, n_sets + 1))
    return list(range(1, min(n_sets, int(mode_count)) + 1))


def _ensure_callback(cb: Optional[Callable], default_return=None) -> Callable:
    if cb is None:
        def _noop(*_args, **_kwargs):
            return default_return
        return _noop
    return cb


def extract_modal_tensor_csv(
    rst_path: str,
    output_csv_path: str,
    result_kind: str,
    named_selection: str = "All Nodes",
    mode_ids: Optional[Sequence[int]] = None,
    mode_count: Optional[int] = None,
    chunk_size: Optional[int] = None,
    node_order: Optional[Sequence[int]] = None,
    nodal_averaging: bool = True,
    average_across_bodies: bool = True,
    log_cb: Optional[Callable[[str], None]] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> None:
    _require_dpf()
    log = _ensure_callback(log_cb)
    progress = _ensure_callback(progress_cb)
    cancel_check = _ensure_callback(should_cancel, False)

    model = dpf.Model(rst_path)
    mesh = model.metadata.meshed_region
    if node_order is not None:
        output_order = list(node_order)
        unique_ids = list(dict.fromkeys(output_order))
        node_ids = unique_ids
    else:
        scoping = _sorted_scoping(_resolve_nodal_scoping(model, named_selection))
        node_ids = list(scoping.ids)
        output_order = node_ids

    n_nodes = len(output_order)

    if mode_ids is None:
        n_sets = model.metadata.time_freq_support.n_sets
        mode_ids = _mode_ids_from_count(n_sets, mode_count)

    if not mode_ids:
        raise ModalExtractionError("No mode IDs available to extract.")

    components = _result_components(result_kind)
    n_components = len(components)

    if chunk_size is None:
        chunk_size = memory_policy.compute_chunk_size(len(node_ids), len(mode_ids), n_components)

    total_chunks = memory_policy.compute_chunk_count(n_nodes, chunk_size)
    log(
        f"Extracting {result_kind} for {n_nodes} nodes, {len(mode_ids)} modes, "
        f"chunk size {chunk_size}."
    )

    header = ["NodeID", "X", "Y", "Z"]
    for set_id in mode_ids:
        for comp in components:
            header.append(f"{comp}_Mode{set_id}")

    with open(output_csv_path, "w", newline="") as handle:
        csv_writer.write_header(handle, header)

        has_duplicates = len(set(output_order)) != len(output_order)

        for chunk_index in range(total_chunks):
            if cancel_check():
                raise ModalExtractionCanceled("Extraction canceled.")

            start = chunk_index * chunk_size
            end = min(start + chunk_size, n_nodes)
            chunk_output_ids = output_order[start:end]

            if has_duplicates or node_order is not None:
                chunk_unique_ids = list(dict.fromkeys(chunk_output_ids))
                chunk_scoping = dpf.Scoping(location=dpf.locations.nodal)
                chunk_scoping.ids = chunk_unique_ids
                coords = _get_node_coordinates(mesh, chunk_unique_ids)

                coord_map = {nid: coords[idx] for idx, nid in enumerate(chunk_unique_ids)}

                mode_maps = []
                for set_id in mode_ids:
                    if cancel_check():
                        raise ModalExtractionCanceled("Extraction canceled.")

                    field = _evaluate_result(
                        model,
                        result_kind,
                        set_id,
                        chunk_scoping,
                        force_nodal_average=nodal_averaging,
                        mesh=mesh,
                        average_across_bodies=average_across_bodies,
                    )
                    aligned = _align_field_to_nodes(field, chunk_unique_ids, n_components)
                    mode_maps.append({nid: aligned[idx] for idx, nid in enumerate(chunk_unique_ids)})
                    del field

                rows = []
                for nid in chunk_output_ids:
                    row = [nid, *coord_map.get(nid, (0.0, 0.0, 0.0))]
                    for mode_map in mode_maps:
                        row.extend(mode_map.get(nid, np.zeros(n_components)))
                    rows.append(row)

                csv_writer.write_chunk(handle, np.array(rows, dtype=float))
            else:
                chunk_scoping = dpf.Scoping(location=dpf.locations.nodal)
                chunk_scoping.ids = chunk_output_ids

                coords = _get_node_coordinates(mesh, chunk_output_ids)

                chunk_data = np.zeros((len(chunk_output_ids), 4 + len(mode_ids) * n_components), dtype=float)
                chunk_data[:, 0] = np.array(chunk_output_ids, dtype=float)
                chunk_data[:, 1:4] = coords

                col = 4
                for set_id in mode_ids:
                    if cancel_check():
                        raise ModalExtractionCanceled("Extraction canceled.")

                    field = _evaluate_result(
                        model,
                        result_kind,
                        set_id,
                        chunk_scoping,
                        force_nodal_average=nodal_averaging,
                        mesh=mesh,
                        average_across_bodies=average_across_bodies,
                    )
                    aligned = _align_field_to_nodes(field, chunk_output_ids, n_components)
                    chunk_data[:, col:col + n_components] = aligned
                    col += n_components

                    del field

                csv_writer.write_chunk(handle, chunk_data)

            progress(chunk_index + 1, total_chunks)
            if chunk_index % 5 == 0:
                gc.collect()

    gc.collect()


def extract_modal_stress_csv(**kwargs) -> None:
    extract_modal_tensor_csv(result_kind="stress", **kwargs)


def extract_modal_displacement_csv(**kwargs) -> None:
    extract_modal_tensor_csv(result_kind="displacement", **kwargs)


def extract_modal_strain_csv(**kwargs) -> None:
    extract_modal_tensor_csv(result_kind="strain", **kwargs)
