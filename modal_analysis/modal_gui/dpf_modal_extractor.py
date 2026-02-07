"""DPF-backed modal extraction helpers."""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple
import gc
import time
import os

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


def _normalize_protocol(protocol: Optional[str]) -> Optional[str]:
    if not protocol:
        return None
    protocol = protocol.strip().lower()
    if protocol in ("grpc", "gRPC".lower()):
        return "grpc"
    if protocol in ("inprocess", "in_process", "in-process"):
        return "inprocess"
    return None


def _normalize_grpc_mode(mode: Optional[str]) -> Optional[str]:
    if not mode:
        return None
    mode = mode.strip().lower()
    if mode in ("mtls", "mTLS".lower()):
        return "mtls"
    if mode in ("insecure", "none", "off"):
        return "insecure"
    return None


def _start_local_server(protocol: Optional[str]):
    protocol = _normalize_protocol(protocol)
    if not protocol:
        return None

    try:
        from ansys.dpf.core import server_factory
    except Exception:
        return None

    if protocol == "grpc":
        grpc_mode = _normalize_grpc_mode(
            os.getenv("MODAL_DPF_GRPC_MODE")
        ) or server_factory.GrpcMode.Insecure
        config = server_factory.ServerConfig(
            protocol=server_factory.CommunicationProtocols.gRPC,
            grpc_mode=grpc_mode,
        )
    else:
        config = server_factory.ServerConfig(
            protocol=server_factory.CommunicationProtocols.InProcess
        )

    return dpf.start_local_server(as_global=False, config=config)


def list_named_selections(rst_path: str) -> List[str]:
    """List named selections in RST file.
    
    Uses DPF.
    """
    _require_dpf()
    model = dpf.Model(rst_path)
    names = model.metadata.available_named_selections
    return list(names) if names else []


def get_model_info(rst_path: str) -> dict:
    """Get model information from RST file.
    
    Uses DPF.
    """
    _require_dpf()
    model = dpf.Model(rst_path)
    info = {
        "num_nodes": 0,
        "num_elements": 0,
        "n_sets": 0,
        "unit_system": "Unknown",
        "named_selections": [],
        "available_results": [],
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
    try:
        avail = model.metadata.result_info.available_results
        info["available_results"] = [res.name for res in avail] if avail else []
    except Exception:
        pass
    return info


def _available_result_names(model: "dpf.Model") -> set[str]:
    try:
        avail = model.metadata.result_info.available_results
        return {res.name.lower() for res in avail} if avail else set()
    except Exception:
        return set()


def check_element_nodal_moments_available(
    rst_path: str,
    mode_id: Optional[int] = None,
    server_protocol: Optional[str] = None,
) -> bool:
    """Check whether ENMOX/ENMOY/ENMOZ can be extracted from this RST."""
    _require_dpf()

    protocol = _normalize_protocol(server_protocol) or _normalize_protocol(
        os.getenv("MODAL_DPF_PROTOCOL")
    )
    server = _start_local_server(protocol)

    try:
        model = dpf.Model(rst_path, server=server)
        available = _available_result_names(model)
        has_enf = "element_nodal_forces" in available or any("enf" in name for name in available)
        if not has_enf:
            return False

        n_sets = int(model.metadata.time_freq_support.n_sets)
        if n_sets <= 0:
            return False

        if mode_id is None:
            set_id = 1
        else:
            set_id = int(mode_id)
        set_id = max(1, min(set_id, n_sets))

        data_sources = dpf.DataSources(rst_path, server=server)
        return _element_nodal_dofs_available(
            model,
            set_id=set_id,
            dofs_value=1,
            server=server,
            data_sources=data_sources,
        )
    except Exception:
        return False
    finally:
        if server is not None:
            try:
                server.shutdown()
            except Exception:
                pass


def get_n_sets(rst_path: str) -> int:
    _require_dpf()
    model = dpf.Model(rst_path)
    try:
        return model.metadata.time_freq_support.n_sets
    except Exception:
        return 0


def get_nodal_scoping_ids(rst_path: str, named_selection: Optional[str]) -> List[int]:
    """Get node IDs for a named selection.
    
    Uses DPF.
    """
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


def _resolve_nodal_scoping(
    model: "dpf.Model",
    ns_name: Optional[str],
    server=None,
) -> "dpf.Scoping":
    mesh = model.metadata.meshed_region

    if ns_name is None or ns_name == "All Nodes":
        scoping = dpf.Scoping(location=dpf.locations.nodal, server=server)
        scoping.ids = list(mesh.nodes.scoping.ids)
        return scoping

    ns_scoping = model.metadata.named_selection(ns_name)
    if ns_scoping.location == dpf.locations.nodal:
        return ns_scoping

    try:
        transpose_op = dpf.operators.scoping.transpose(server=server)
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
        nodal_scoping = dpf.Scoping(location=dpf.locations.nodal, server=server)
        nodal_scoping.ids = list(node_ids)
        return nodal_scoping

    raise ModalExtractionError(f"Failed to convert named selection '{ns_name}' to nodal scoping.")


def _sorted_scoping(scoping: "dpf.Scoping", server=None) -> "dpf.Scoping":
    node_ids = sorted(scoping.ids)
    sorted_scoping = dpf.Scoping(location=dpf.locations.nodal, server=server)
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


def _connect_if_present(obj, input_name: str, value) -> None:
    try:
        target = obj.inputs
    except Exception:
        target = obj
    if hasattr(target, input_name):
        getattr(target, input_name).connect(value)


def _connect_mesh_scoping(obj, scoping: "dpf.Scoping") -> None:
    try:
        target = obj.inputs
    except Exception:
        target = obj
    for name in ("mesh_scoping", "scoping", "mesh_scoping_as_scoping"):
        if hasattr(target, name):
            getattr(target, name).connect(scoping)
            return


def _connect_requested_location(obj, location) -> None:
    try:
        target = obj.inputs
    except Exception:
        target = obj
    for name in ("requested_location", "location"):
        if hasattr(target, name):
            getattr(target, name).connect(location)
            return


def _connect_label(obj, label: str) -> None:
    try:
        target = obj.inputs
    except Exception:
        target = obj
    for name in ("label1", "label"):
        if hasattr(target, name):
            getattr(target, name).connect(label)
            return


def _get_first_field(fields_container) -> "dpf.Field":
    if fields_container is None or len(fields_container) == 0:
        raise ModalExtractionError("DPF returned empty fields container.")
    for idx in range(len(fields_container)):
        field = fields_container[idx]
        try:
            if np.array(field.data).size > 0:
                return field
        except Exception:
            continue
    return fields_container[0]


def _is_force_result_kind(kind: str) -> bool:
    return kind in ("element_nodal_force", "element_nodal_moment", "element_nodal_force_moment")


def _field_data_size(field: "dpf.Field") -> int:
    try:
        return int(np.array(field.data).size)
    except Exception:
        return 0


def _field_label_space(fields_container, index: int) -> dict:
    try:
        return fields_container.get_label_space(index)
    except Exception:
        return {}


def _select_element_nodal_field(fields_container, dofs_value: int) -> Optional["dpf.Field"]:
    best_match = None
    best_size = -1
    fallback = None
    fallback_size = -1

    for idx in range(len(fields_container)):
        field = fields_container[idx]
        size = _field_data_size(field)
        if size <= 0:
            continue
        label_space = _field_label_space(fields_container, idx)
        dofs = label_space.get("dofs")
        if dofs is None:
            if dofs_value == 0 and size > fallback_size:
                fallback = field
                fallback_size = size
            continue
        if int(dofs) == int(dofs_value) and size > best_size:
            best_match = field
            best_size = size

    if best_match is not None:
        return best_match
    if fallback is not None:
        return fallback
    return None


def _align_field_to_nodes(
    field: "dpf.Field",
    requested_node_ids: Sequence[int],
    expected_components: int,
    mesh_node_ids: Optional[Sequence[int]] = None,
    log_cb: Optional[Callable[[str], None]] = None,
) -> np.ndarray:
    """Align DPF field data to the requested node order.
    
    Returns an array of shape (len(requested_node_ids), expected_components).
    Nodes not found in field.scoping will have zero values.
    """
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

    def _align_with_ids(source_ids: np.ndarray) -> tuple[np.ndarray, int]:
        id_to_idx = {nid: idx for idx, nid in enumerate(source_ids)}
        aligned_local = np.zeros((len(requested), expected_components), dtype=float)
        matched_local = 0
        for out_idx, node_id in enumerate(requested):
            src_idx = id_to_idx.get(node_id)
            if src_idx is not None:
                aligned_local[out_idx, :] = data[src_idx, :]
                matched_local += 1
        return aligned_local, matched_local

    aligned, matched_count = _align_with_ids(dpf_node_ids)

    # Fallback: some result providers may return 0-based or 1-based indices
    # instead of true node IDs. If overlap is very low, attempt remap.
    if mesh_node_ids is not None and matched_count < max(1, int(0.01 * len(requested))):
        mesh_node_ids_arr = np.array(mesh_node_ids, dtype=int)
        remapped = None
        if dpf_node_ids.min(initial=0) >= 0 and dpf_node_ids.max(initial=-1) < len(mesh_node_ids_arr):
            remapped = mesh_node_ids_arr[dpf_node_ids]
        elif dpf_node_ids.min(initial=0) >= 1 and dpf_node_ids.max(initial=0) <= len(mesh_node_ids_arr):
            remapped = mesh_node_ids_arr[dpf_node_ids - 1]
        if remapped is not None:
            remapped_aligned, remapped_matched = _align_with_ids(remapped)
            if remapped_matched > matched_count:
                aligned = remapped_aligned
                matched_count = remapped_matched
                if log_cb:
                    log_cb(
                        "    Detected index-based scoping ids; remapped to mesh node IDs "
                        f"({remapped_matched}/{len(requested)} matched)."
                    )
    
    # Warn if significant mismatch
    if matched_count < len(requested):
        mismatch_pct = 100 * (len(requested) - matched_count) / len(requested)
        if log_cb and mismatch_pct > 1.0:
            log_cb(
                f"    Warning: {mismatch_pct:.1f}% of requested nodes not found in DPF results "
                f"({len(requested) - matched_count}/{len(requested)} nodes)"
            )
    
    return aligned


def _result_operator(
    kind: str,
    server=None,
    model: Optional["dpf.Model"] = None,
):
    ops_result = dpf.operators.result
    if kind == "stress":
        try:
            return ops_result.stress(server=server)
        except Exception:
            if model is not None:
                return model.results.stress()
            raise
    if kind == "displacement":
        try:
            return ops_result.displacement(server=server)
        except Exception:
            if model is not None:
                return model.results.displacement()
            raise
    if kind == "strain":
        for name in ("elastic_strain", "total_strain", "strain"):
            try:
                op_factory = getattr(ops_result, name, None)
                if op_factory is not None:
                    return op_factory(server=server)
            except Exception:
                pass
        if model is not None:
            for name in ("elastic_strain", "total_strain", "strain"):
                op_factory = getattr(model.results, name, None)
                if op_factory is None:
                    continue
                try:
                    return op_factory()
                except Exception:
                    continue
        raise ModalExtractionError("No strain operator available in this DPF build.")
    if kind in ("element_nodal_force", "element_nodal_moment"):
        op_factory = getattr(ops_result, "element_nodal_forces", None)
        if op_factory is not None:
            try:
                return op_factory(server=server)
            except Exception:
                pass
        if model is not None:
            op_factory = getattr(model.results, "element_nodal_forces", None)
            if op_factory is not None:
                try:
                    return op_factory()
                except Exception:
                    pass
        raise ModalExtractionError("No element_nodal_forces operator available in this DPF build.")
    raise ModalExtractionError(f"Unknown result kind '{kind}'.")


def _result_components(kind: str) -> Tuple[str, ...]:
    if kind == "stress":
        return ("sx", "sy", "sz", "sxy", "syz", "sxz")
    if kind == "strain":
        return ("ex", "ey", "ez", "exy", "eyz", "exz")
    if kind == "displacement":
        return ("ux", "uy", "uz")
    if kind == "element_nodal_force":
        return ("enfox", "enfoy", "enfoz")
    if kind == "element_nodal_moment":
        return ("enmox", "enmoy", "enmoz")
    if kind == "element_nodal_force_moment":
        return ("enfox", "enfoy", "enfoz", "enmox", "enmoy", "enmoz")
    raise ModalExtractionError(f"Unknown result kind '{kind}'.")


def _average_to_nodal(
    field: "dpf.Field",
    mesh_scoping: "dpf.Scoping",
    mesh: Optional["dpf.MeshedRegion"] = None,
    server=None,
) -> "dpf.Field":
    try:
        op = dpf.operators.averaging.elemental_nodal_to_nodal(server=server)
    except Exception:
        op = dpf.operators.averaging.to_nodal(server=server)

    _connect_if_present(op, "field", field)
    _connect_if_present(op, "mesh_scoping", mesh_scoping)
    if mesh is not None:
        _connect_if_present(op, "mesh", mesh)
    _connect_if_present(op, "should_average", True)
    return op.outputs.field()


def _nodal_to_elemental_scoping(
    mesh: "dpf.MeshedRegion",
    nodal_scoping: "dpf.Scoping",
    server=None,
) -> "dpf.Scoping":
    transpose_op = dpf.operators.scoping.transpose(server=server)
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
    server=None,
) -> "dpf.ScopingsContainer":
    split_op = dpf.operators.scoping.split_on_property_type(server=server)
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
    server=None,
    data_sources=None,
) -> "dpf.Field":
    element_scoping = _nodal_to_elemental_scoping(mesh, nodal_scoping, server=server)
    mat_scopings = _split_scopings_by_material(mesh, element_scoping, server=server)

    op = _result_operator(kind, server=server, model=model)
    time_scoping = dpf.Scoping(server=server)
    time_scoping.ids = [int(set_id)]
    _connect_if_present(op, "time_scoping", time_scoping)
    _connect_mesh_scoping(op, mat_scopings)
    if data_sources is not None:
        _connect_if_present(op, "data_sources", data_sources)
    if hasattr(dpf.locations, "elemental_nodal"):
        _connect_requested_location(op, dpf.locations.elemental_nodal)
    else:
        _connect_requested_location(op, dpf.locations.nodal)

    try:
        outputs = op.outputs
    except Exception:
        outputs = op
    if not hasattr(outputs, "fields_container"):
        raise ModalExtractionError("DPF result object missing fields_container output.")
    fc = outputs.fields_container()

    eln_to_n = dpf.operators.averaging.elemental_nodal_to_nodal_fc(server=server)
    _connect_if_present(eln_to_n, "fields_container", fc)
    _connect_if_present(eln_to_n, "mesh", mesh)
    _connect_if_present(eln_to_n, "scoping", nodal_scoping)
    _connect_if_present(eln_to_n, "extend_weights_to_mid_nodes", True)

    merge_op = dpf.operators.utility.weighted_merge_fields_by_label(server=server)
    _connect_if_present(merge_op, "fields_container", eln_to_n.outputs.fields_container())
    _connect_if_present(merge_op, "label", "mat")
    try:
        merge_op.connect(1000, eln_to_n, 1)
    except Exception:
        _connect_if_present(merge_op, "weights1", eln_to_n.outputs.weights())

    merged_fc = merge_op.outputs.fields_container()
    return _get_first_field(merged_fc)


def _evaluate_element_nodal_result(
    model: "dpf.Model",
    kind: str,
    set_id: int,
    mesh_scoping: "dpf.Scoping",
    mesh: Optional["dpf.MeshedRegion"] = None,
    server=None,
    data_sources=None,
    log_cb: Optional[Callable[[str], None]] = None,
) -> Optional["dpf.Field"]:
    target_dofs = 0 if kind == "element_nodal_force" else 1

    last_error = None
    for rotate_to_global in (True, False):
        try:
            op = _result_operator(kind, server=server, model=model)

            time_scoping = dpf.Scoping(server=server)
            time_scoping.ids = [int(set_id)]
            _connect_if_present(op, "time_scoping", time_scoping)
            _connect_mesh_scoping(op, mesh_scoping)
            if data_sources is not None:
                _connect_if_present(op, "data_sources", data_sources)
            _connect_if_present(op, "bool_rotate_to_global", rotate_to_global)
            _connect_if_present(op, "split_force_components", True)
            _connect_if_present(op, "read_beams", True)
            if hasattr(dpf.locations, "elemental_nodal"):
                _connect_requested_location(op, dpf.locations.elemental_nodal)

            try:
                outputs = op.outputs
            except Exception:
                outputs = op
            if not hasattr(outputs, "fields_container"):
                raise ModalExtractionError("DPF result object missing fields_container output.")
            fields_container = outputs.fields_container()
            field = _select_element_nodal_field(fields_container, target_dofs)
            if field is None:
                return None

            if mesh is not None and hasattr(dpf.locations, "elemental_nodal"):
                try:
                    if field.location == dpf.locations.elemental_nodal:
                        return _average_to_nodal(field, mesh_scoping, mesh, server=server)
                except Exception:
                    pass

            return field
        except Exception as exc:
            last_error = exc
            if rotate_to_global and log_cb is not None:
                log_cb(
                    "    Element nodal force rotation to global failed; retrying in local coordinates."
                )
            continue

    if isinstance(last_error, ModalExtractionError):
        raise last_error
    raise ModalExtractionError(
        f"Failed to evaluate element nodal {'moments' if kind == 'element_nodal_moment' else 'forces'}: {last_error}"
    )


def _element_nodal_dofs_available(
    model: "dpf.Model",
    set_id: int,
    dofs_value: int,
    *,
    server=None,
    data_sources=None,
) -> bool:
    for rotate_to_global in (True, False):
        try:
            op = _result_operator("element_nodal_force", server=server, model=model)
            time_scoping = dpf.Scoping(server=server)
            time_scoping.ids = [int(set_id)]
            _connect_if_present(op, "time_scoping", time_scoping)
            if data_sources is not None:
                _connect_if_present(op, "data_sources", data_sources)
            _connect_if_present(op, "bool_rotate_to_global", rotate_to_global)
            _connect_if_present(op, "split_force_components", True)
            _connect_if_present(op, "read_beams", True)
            if hasattr(dpf.locations, "elemental_nodal"):
                _connect_requested_location(op, dpf.locations.elemental_nodal)

            outputs = op.outputs if hasattr(op, "outputs") else op
            if not hasattr(outputs, "fields_container"):
                continue
            fields_container = outputs.fields_container()
            field = _select_element_nodal_field(fields_container, dofs_value)
            if field is not None and _field_data_size(field) > 0:
                return True
        except Exception:
            continue
    return False


def _evaluate_result(
    model: "dpf.Model",
    kind: str,
    set_id: int,
    mesh_scoping: "dpf.Scoping",
    force_nodal_average: bool = False,
    mesh: Optional["dpf.MeshedRegion"] = None,
    average_across_bodies: bool = True,
    server=None,
    data_sources=None,
    log_cb: Optional[Callable[[str], None]] = None,
) -> Optional["dpf.Field"]:
    if _is_force_result_kind(kind):
        return _evaluate_element_nodal_result(
            model,
            kind,
            set_id,
            mesh_scoping,
            mesh=mesh,
            server=server,
            data_sources=data_sources,
            log_cb=log_cb,
        )

    if force_nodal_average and kind in ("stress", "strain") and average_across_bodies and mesh is not None:
        try:
            return _average_across_bodies_field(
                model,
                kind,
                set_id,
                mesh_scoping,
                mesh,
                server=server,
                data_sources=data_sources,
            )
        except Exception:
            pass

    op = _result_operator(kind, server=server, model=model)

    time_scoping = dpf.Scoping(server=server)
    time_scoping.ids = [int(set_id)]
    _connect_if_present(op, "time_scoping", time_scoping)
    _connect_mesh_scoping(op, mesh_scoping)
    if data_sources is not None:
        _connect_if_present(op, "data_sources", data_sources)

    if force_nodal_average and kind in ("stress", "strain") and hasattr(dpf.locations, "elemental_nodal"):
        _connect_requested_location(op, dpf.locations.elemental_nodal)
    else:
        _connect_requested_location(op, dpf.locations.nodal)

    try:
        outputs = op.outputs
    except Exception:
        outputs = op
    if not hasattr(outputs, "fields_container"):
        raise ModalExtractionError("DPF result object missing fields_container output.")
    fields_container = outputs.fields_container()
    field = _get_first_field(fields_container)

    if force_nodal_average and kind in ("stress", "strain"):
        try:
            return _average_to_nodal(field, mesh_scoping, mesh, server=server)
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


def _evaluate_aligned_result(
    kind: str,
    set_id: int,
    node_ids: Sequence[int],
    n_components: int,
    *,
    model: "dpf.Model",
    mesh: "dpf.MeshedRegion",
    data_sources,
    scoping: "dpf.Scoping",
    nodal_averaging: bool,
    average_across_bodies: bool,
    server=None,
    log_cb: Optional[Callable[[str], None]] = None,
) -> np.ndarray:
    field = _evaluate_result(
        model,
        kind,
        set_id,
        scoping,
        force_nodal_average=nodal_averaging,
        mesh=mesh,
        average_across_bodies=average_across_bodies,
        server=server,
        data_sources=data_sources,
        log_cb=log_cb,
    )
    if field is None:
        return np.zeros((len(node_ids), n_components), dtype=float)
    mesh_node_ids = None
    try:
        mesh_node_ids = mesh.nodes.scoping.ids
    except Exception:
        mesh_node_ids = None
    return _align_field_to_nodes(
        field,
        node_ids,
        n_components,
        mesh_node_ids=mesh_node_ids,
        log_cb=log_cb,
    )


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
    server_protocol: Optional[str] = None,
    log_cb: Optional[Callable[[str], None]] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> None:
    log = _ensure_callback(log_cb)
    progress = _ensure_callback(progress_cb)
    cancel_check = _ensure_callback(should_cancel, False)
    
    # Use DPF backend only.
    _require_dpf()

    protocol = _normalize_protocol(server_protocol) or _normalize_protocol(
        os.getenv("MODAL_DPF_PROTOCOL")
    )
    server = _start_local_server(protocol)

    try:
        model = dpf.Model(rst_path, server=server)
        data_sources = dpf.DataSources(rst_path, server=server)
        mesh = model.metadata.meshed_region
        available = _available_result_names(model)
        if result_kind == "strain":
            if not {"elastic_strain", "total_strain", "strain"} & available:
                raise ModalExtractionError(
                    "Strain results are not available in this RST file."
                )
        if _is_force_result_kind(result_kind):
            has_enf = "element_nodal_forces" in available or any(
                "enf" in name for name in available
            )
            if not has_enf:
                raise ModalExtractionError(
                    "Element nodal force results are not available in this RST file."
                )
        if node_order is not None:
            output_order = list(node_order)
            unique_ids = list(dict.fromkeys(output_order))
            node_ids = unique_ids
        else:
            scoping = _sorted_scoping(
                _resolve_nodal_scoping(model, named_selection, server=server),
                server=server,
            )
            node_ids = list(scoping.ids)
            output_order = node_ids
        n_nodes = len(output_order)

        if mode_ids is None:
            n_sets = model.metadata.time_freq_support.n_sets
            mode_ids = _mode_ids_from_count(n_sets, mode_count)

        if not mode_ids:
            raise ModalExtractionError("No mode IDs available to extract.")

        combined_moment_available = True
        if result_kind == "element_nodal_moment":
            if not _element_nodal_dofs_available(
                model,
                set_id=int(mode_ids[0]),
                dofs_value=1,
                server=server,
                data_sources=data_sources,
            ):
                raise ModalExtractionError(
                    "Element nodal moments (ENMOX/ENMOY/ENMOZ) are not available in this RST file."
                )
        if result_kind == "element_nodal_force_moment":
            combined_moment_available = _element_nodal_dofs_available(
                model,
                set_id=int(mode_ids[0]),
                dofs_value=1,
                server=server,
                data_sources=data_sources,
            )
            if not combined_moment_available:
                log(
                    "Element nodal moments are not available for this extraction; "
                    "ENMO columns will be written as zeros."
                )

        components = _result_components(result_kind)
        n_components = len(components)

        def _evaluate_mode_aligned(
            set_id_value: int,
            current_node_ids: Sequence[int],
            current_scoping: "dpf.Scoping",
        ) -> np.ndarray:
            if result_kind != "element_nodal_force_moment":
                return _evaluate_aligned_result(
                    result_kind,
                    set_id_value,
                    current_node_ids,
                    n_components,
                    model=model,
                    mesh=mesh,
                    data_sources=data_sources,
                    scoping=current_scoping,
                    nodal_averaging=nodal_averaging,
                    average_across_bodies=average_across_bodies,
                    server=server,
                    log_cb=log,
                )

            force_aligned = _evaluate_aligned_result(
                "element_nodal_force",
                set_id_value,
                current_node_ids,
                3,
                model=model,
                mesh=mesh,
                data_sources=data_sources,
                scoping=current_scoping,
                nodal_averaging=nodal_averaging,
                average_across_bodies=average_across_bodies,
                server=server,
                log_cb=log,
            )
            if combined_moment_available:
                moment_aligned = _evaluate_aligned_result(
                    "element_nodal_moment",
                    set_id_value,
                    current_node_ids,
                    3,
                    model=model,
                    mesh=mesh,
                    data_sources=data_sources,
                    scoping=current_scoping,
                    nodal_averaging=nodal_averaging,
                    average_across_bodies=average_across_bodies,
                    server=server,
                    log_cb=log,
                )
            else:
                moment_aligned = np.zeros((len(current_node_ids), 3), dtype=float)
            return np.concatenate((force_aligned, moment_aligned), axis=1)

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

                chunk_start_time = time.perf_counter()
                start = chunk_index * chunk_size
                end = min(start + chunk_size, n_nodes)
                chunk_output_ids = output_order[start:end]
                log(
                    f"Chunk {chunk_index + 1}/{total_chunks}: nodes {start + 1}-{end} "
                    f"({len(chunk_output_ids)} nodes), {len(mode_ids)} modes"
                )

                if has_duplicates or node_order is not None:
                    chunk_unique_ids = list(dict.fromkeys(chunk_output_ids))
                    chunk_scoping = dpf.Scoping(location=dpf.locations.nodal, server=server)
                    chunk_scoping.ids = chunk_unique_ids
                    coords = _get_node_coordinates(mesh, chunk_unique_ids)

                    coord_map = {nid: coords[idx] for idx, nid in enumerate(chunk_unique_ids)}

                    mode_maps = []
                    for mode_index, set_id in enumerate(mode_ids, start=1):
                        if cancel_check():
                            raise ModalExtractionCanceled("Extraction canceled.")

                        log(
                            f"  Chunk {chunk_index + 1}/{total_chunks}: mode {mode_index}/{len(mode_ids)} "
                            f"(set {set_id})"
                        )
                        
                        aligned = _evaluate_mode_aligned(
                            set_id,
                            chunk_unique_ids,
                            chunk_scoping,
                        )
                        mode_maps.append({nid: aligned[idx] for idx, nid in enumerate(chunk_unique_ids)})
                        del aligned
                        gc.collect()

                    rows = []
                    for nid in chunk_output_ids:
                        row = [nid, *coord_map.get(nid, (0.0, 0.0, 0.0))]
                        for mode_map in mode_maps:
                            row.extend(mode_map.get(nid, np.zeros(n_components)))
                        rows.append(row)

                    csv_writer.write_chunk(handle, np.array(rows, dtype=float))
                else:
                    chunk_scoping = dpf.Scoping(location=dpf.locations.nodal, server=server)
                    chunk_scoping.ids = chunk_output_ids

                    coords = _get_node_coordinates(mesh, chunk_output_ids)

                    chunk_data = np.zeros((len(chunk_output_ids), 4 + len(mode_ids) * n_components), dtype=float)
                    chunk_data[:, 0] = np.array(chunk_output_ids, dtype=float)
                    chunk_data[:, 1:4] = coords

                    col = 4
                    for mode_index, set_id in enumerate(mode_ids, start=1):
                        if cancel_check():
                            raise ModalExtractionCanceled("Extraction canceled.")

                        log(
                            f"  Chunk {chunk_index + 1}/{total_chunks}: mode {mode_index}/{len(mode_ids)} "
                            f"(set {set_id})"
                        )
                        
                        aligned = _evaluate_mode_aligned(
                            set_id,
                            chunk_output_ids,
                            chunk_scoping,
                        )
                        chunk_data[:, col:col + n_components] = aligned
                        col += n_components
                        del aligned
                        gc.collect()

                    csv_writer.write_chunk(handle, chunk_data)

                chunk_duration = time.perf_counter() - chunk_start_time
                log(f"Chunk {chunk_index + 1}/{total_chunks} completed in {chunk_duration:.1f}s")
                progress(chunk_index + 1, total_chunks)
                if chunk_index % 5 == 0:
                    gc.collect()

        gc.collect()
    finally:
        if server is not None:
            try:
                server.shutdown()
            except Exception:
                pass


def extract_modal_stress_csv(**kwargs) -> None:
    extract_modal_tensor_csv(result_kind="stress", **kwargs)


def extract_modal_displacement_csv(**kwargs) -> None:
    extract_modal_tensor_csv(result_kind="displacement", **kwargs)


def extract_modal_strain_csv(**kwargs) -> None:
    extract_modal_tensor_csv(result_kind="strain", **kwargs)


def extract_modal_element_nodal_forces_csv(**kwargs) -> None:
    extract_modal_tensor_csv(result_kind="element_nodal_force", **kwargs)


def extract_modal_element_nodal_moments_csv(**kwargs) -> None:
    extract_modal_tensor_csv(result_kind="element_nodal_moment", **kwargs)


def extract_modal_element_nodal_forces_moments_csv(**kwargs) -> None:
    extract_modal_tensor_csv(result_kind="element_nodal_force_moment", **kwargs)
