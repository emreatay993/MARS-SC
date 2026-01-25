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

try:
    from . import pymapdl_extractor
    PYMAPDL_READER_AVAILABLE = pymapdl_extractor.PYMAPDL_READER_AVAILABLE
except ImportError:
    pymapdl_extractor = None
    PYMAPDL_READER_AVAILABLE = False

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


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _env_str(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip()


def _is_access_violation(exc: Exception) -> bool:
    message = str(exc).lower()
    return "access violation" in message or "0x0000000000000030" in message


def list_named_selections(rst_path: str) -> List[str]:
    """List named selections in RST file.
    
    Uses DPF if available, otherwise falls back to PyMAPDL Reader.
    """
    if DPF_AVAILABLE:
        try:
            model = dpf.Model(rst_path)
            names = model.metadata.available_named_selections
            return list(names) if names else []
        except Exception:
            pass
    
    if PYMAPDL_READER_AVAILABLE:
        return pymapdl_extractor.list_named_selections(rst_path)
    
    raise DPFNotAvailableError(
        "Neither ansys-dpf-core nor ansys-mapdl-reader is installed."
    )


def get_model_info(rst_path: str) -> dict:
    """Get model information from RST file.
    
    Uses DPF if available, otherwise falls back to PyMAPDL Reader.
    """
    # Try DPF first
    if DPF_AVAILABLE:
        try:
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
        except Exception:
            pass  # Fall through to PyMAPDL Reader
    
    # Fallback to PyMAPDL Reader
    if PYMAPDL_READER_AVAILABLE:
        return pymapdl_extractor.get_model_info(rst_path)
    
    raise DPFNotAvailableError(
        "Neither ansys-dpf-core nor ansys-mapdl-reader is installed."
    )


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
) -> "dpf.Field":
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
    log: Callable[[str], None],
    server=None,
) -> np.ndarray:
    try:
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
        )
        return _align_field_to_nodes(field, node_ids, n_components)
    except Exception as exc:
        if _is_access_violation(exc):
            log(f"Mode {set_id} hit access violation - will trigger fallback")
        raise


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
    backend: Optional[str] = None,  # "dpf", "pymapdl", or "auto" (default)
) -> None:
    log = _ensure_callback(log_cb)
    progress = _ensure_callback(progress_cb)
    cancel_check = _ensure_callback(should_cancel, False)
    
    # Determine backend: "dpf", "pymapdl", or "auto"
    selected_backend = (backend or _env_str("MODAL_EXTRACTION_BACKEND", "auto")).lower()
    
    if selected_backend == "pymapdl":
        # Force PyMAPDL Reader backend
        if not PYMAPDL_READER_AVAILABLE:
            raise ModalExtractionError(
                "PyMAPDL Reader backend requested but ansys-mapdl-reader is not installed. "
                "Install with: pip install ansys-mapdl-reader"
            )
        log("Using PyMAPDL Reader backend (forced)")
        return pymapdl_extractor.extract_modal_tensor_csv(
            rst_path=rst_path,
            output_csv_path=output_csv_path,
            result_kind=result_kind,
            named_selection=named_selection,
            mode_ids=mode_ids,
            mode_count=mode_count,
            chunk_size=chunk_size,
            node_order=node_order,
            log_cb=log_cb,
            progress_cb=progress_cb,
            should_cancel=should_cancel,
        )
    
    if selected_backend == "auto" and not DPF_AVAILABLE:
        # Auto mode: fall back to PyMAPDL if DPF not available
        if PYMAPDL_READER_AVAILABLE:
            log("DPF not available, using PyMAPDL Reader backend")
            return pymapdl_extractor.extract_modal_tensor_csv(
                rst_path=rst_path,
                output_csv_path=output_csv_path,
                result_kind=result_kind,
                named_selection=named_selection,
                mode_ids=mode_ids,
                mode_count=mode_count,
                chunk_size=chunk_size,
                node_order=node_order,
                log_cb=log_cb,
                progress_cb=progress_cb,
                should_cancel=should_cancel,
            )
        else:
            raise DPFNotAvailableError(
                "Neither ansys-dpf-core nor ansys-mapdl-reader is installed."
            )
    
    # Use DPF backend
    _require_dpf()
    
    # Auto-fallback to PyMAPDL Reader on access violations (only in auto mode)
    fallback_on_access_violation = (
        selected_backend == "auto" 
        and PYMAPDL_READER_AVAILABLE 
        and _env_flag("MODAL_DPF_FALLBACK_TO_PYMAPDL", True)
    )

    protocol = _normalize_protocol(server_protocol) or _normalize_protocol(
        os.getenv("MODAL_DPF_PROTOCOL")
    )
    server = _start_local_server(protocol)
    _dpf_fatal_error = None

    try:
        model = dpf.Model(rst_path, server=server)
        data_sources = dpf.DataSources(rst_path, server=server)
        mesh = model.metadata.meshed_region
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
                        
                        aligned = _evaluate_aligned_result(
                            result_kind,
                            set_id,
                            chunk_unique_ids,
                            n_components,
                            model=model,
                            mesh=mesh,
                            data_sources=data_sources,
                            scoping=chunk_scoping,
                            nodal_averaging=nodal_averaging,
                            average_across_bodies=average_across_bodies,
                            log=log,
                            server=server,
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
                        
                        aligned = _evaluate_aligned_result(
                            result_kind,
                            set_id,
                            chunk_output_ids,
                            n_components,
                            model=model,
                            mesh=mesh,
                            data_sources=data_sources,
                            scoping=chunk_scoping,
                            nodal_averaging=nodal_averaging,
                            average_across_bodies=average_across_bodies,
                            log=log,
                            server=server,
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
        return  # Success - return without fallback
        
    except Exception as dpf_exc:
        # Check if this is an access violation that might benefit from PyMAPDL fallback
        if fallback_on_access_violation and _is_access_violation(dpf_exc):
            _dpf_fatal_error = dpf_exc
            log(f"DPF encountered access violation: {dpf_exc}")
            log("Attempting fallback to PyMAPDL Reader backend...")
        else:
            raise
    finally:
        if server is not None:
            try:
                server.shutdown()
            except Exception:
                pass
    
    # Fallback to PyMAPDL Reader if DPF failed with access violation
    if _dpf_fatal_error is not None and fallback_on_access_violation:
        try:
            return pymapdl_extractor.extract_modal_tensor_csv(
                rst_path=rst_path,
                output_csv_path=output_csv_path,
                result_kind=result_kind,
                named_selection=named_selection,
                mode_ids=mode_ids,
                mode_count=mode_count,
                chunk_size=chunk_size,
                node_order=node_order,
                log_cb=log_cb,
                progress_cb=progress_cb,
                should_cancel=should_cancel,
            )
        except Exception as pymapdl_exc:
            log(f"PyMAPDL Reader fallback also failed: {pymapdl_exc}")
            # Raise the original DPF error
            raise _dpf_fatal_error from pymapdl_exc


def extract_modal_stress_csv(**kwargs) -> None:
    extract_modal_tensor_csv(result_kind="stress", **kwargs)


def extract_modal_displacement_csv(**kwargs) -> None:
    extract_modal_tensor_csv(result_kind="displacement", **kwargs)


def extract_modal_strain_csv(**kwargs) -> None:
    extract_modal_tensor_csv(result_kind="strain", **kwargs)
