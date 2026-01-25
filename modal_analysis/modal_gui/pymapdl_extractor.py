"""PyMAPDL Reader-based modal extraction - fallback for DPF failures."""

from __future__ import annotations

import gc
import time
from typing import Callable, List, Optional, Sequence

import numpy as np

try:
    from ansys.mapdl import reader as pymapdl_reader
    PYMAPDL_READER_AVAILABLE = True
except ImportError:
    pymapdl_reader = None
    PYMAPDL_READER_AVAILABLE = False

from . import csv_writer, memory_policy


class PyMAPDLReaderNotAvailableError(RuntimeError):
    """Raised when ansys-mapdl-reader is not installed."""


class ModalExtractionError(RuntimeError):
    """Raised when modal extraction fails."""


class ModalExtractionCanceled(RuntimeError):
    """Raised when modal extraction is canceled."""


def _require_reader() -> None:
    if not PYMAPDL_READER_AVAILABLE:
        raise PyMAPDLReaderNotAvailableError(
            "ansys-mapdl-reader is not installed. Install with: pip install ansys-mapdl-reader"
        )


def get_model_info(rst_path: str) -> dict:
    """Get model information using PyMAPDL Reader."""
    _require_reader()
    rst = pymapdl_reader.read_binary(rst_path)
    
    info = {
        "num_nodes": 0,
        "num_elements": 0,
        "n_sets": 0,
        "unit_system": "Unknown",
        "named_selections": [],
    }
    
    try:
        info["num_nodes"] = rst.mesh.n_node
    except Exception:
        pass
    try:
        info["num_elements"] = rst.mesh.n_elem
    except Exception:
        pass
    try:
        info["n_sets"] = rst.n_results
    except Exception:
        pass
    try:
        node_comps = list(rst.node_components.keys()) if rst.node_components else []
        elem_comps = list(rst.element_components.keys()) if rst.element_components else []
        info["named_selections"] = node_comps + elem_comps
    except Exception:
        pass
    
    return info


def list_named_selections(rst_path: str) -> List[str]:
    info = get_model_info(rst_path)
    return info.get("named_selections", [])


def get_nodal_scoping_ids(rst_path: str, named_selection: Optional[str]) -> List[int]:
    """Get node IDs for a named selection using PyMAPDL Reader.
    
    Note: PyMAPDL Reader has limited named selection support.
    For 'All Nodes' or None, returns all mesh node IDs.
    For other named selections, attempts to use node_components.
    """
    _require_reader()
    rst = pymapdl_reader.read_binary(rst_path)
    
    if named_selection is None or named_selection == "All Nodes":
        return sorted(rst.mesh.nnum.tolist())
    
    # Try node components
    if rst.node_components and named_selection in rst.node_components:
        return sorted(rst.node_components[named_selection].tolist())
    
    # Try element components and convert to nodes
    if rst.element_components and named_selection in rst.element_components:
        elem_ids = rst.element_components[named_selection]
        # Get nodes from elements (simplified - returns all mesh nodes for those elements)
        # This is a limitation of pymapdl_reader
        node_set = set()
        mesh = rst.mesh
        for elem_id in elem_ids:
            try:
                elem_idx = np.where(mesh.enum == elem_id)[0]
                if len(elem_idx) > 0:
                    # Get element connectivity
                    nodes = mesh.elem[elem_idx[0]]
                    node_set.update(n for n in nodes if n > 0)
            except Exception:
                continue
        if node_set:
            return sorted(node_set)
    
    # Fallback: return all nodes with warning
    return sorted(rst.mesh.nnum.tolist())


def _get_node_coordinates(rst, node_ids: Sequence[int]) -> np.ndarray:
    mesh = rst.mesh
    all_nnum = mesh.nnum
    all_nodes = mesh.nodes
    nnum_to_idx = {nid: idx for idx, nid in enumerate(all_nnum)}
    
    coords = np.zeros((len(node_ids), 3), dtype=float)
    for idx, nid in enumerate(node_ids):
        src_idx = nnum_to_idx.get(nid)
        if src_idx is not None:
            coords[idx, :] = all_nodes[src_idx]
    return coords


def _align_result_to_nodes(
    result_nnum: np.ndarray,
    result_data: np.ndarray,
    requested_node_ids: Sequence[int],
    n_components: int,
) -> np.ndarray:
    nnum_to_idx = {nid: idx for idx, nid in enumerate(result_nnum)}
    
    aligned = np.zeros((len(requested_node_ids), n_components), dtype=float)
    for out_idx, node_id in enumerate(requested_node_ids):
        src_idx = nnum_to_idx.get(node_id)
        if src_idx is not None:
            data_row = result_data[src_idx]
            if data_row.ndim == 0:
                aligned[out_idx, 0] = float(data_row)
            else:
                aligned[out_idx, :min(len(data_row), n_components)] = data_row[:n_components]
    return aligned


def _result_components(kind: str) -> tuple:
    if kind == "stress":
        return ("sx", "sy", "sz", "sxy", "syz", "sxz")
    if kind == "strain":
        return ("ex", "ey", "ez", "exy", "eyz", "exz")
    if kind == "displacement":
        return ("ux", "uy", "uz")
    raise ModalExtractionError(f"Unknown result kind '{kind}'.")


def _extract_result_for_mode(rst, kind: str, mode_index: int) -> tuple:
    try:
        if kind == "stress":
            nnum, data = rst.nodal_stress(mode_index)
        elif kind == "displacement":
            nnum, data = rst.nodal_solution(mode_index)
        elif kind == "strain":
            nnum, data = rst.nodal_elastic_strain(mode_index)
            data = data[:, :6]
        else:
            raise ModalExtractionError(f"Unknown result kind '{kind}'.")
        return nnum, data
    except Exception as e:
        raise ModalExtractionError(f"Failed to extract {kind} for mode {mode_index + 1}: {e}")


def _ensure_callback(cb: Optional[Callable], default_return=None) -> Callable:
    if cb is None:
        def _noop(*_args, **_kwargs):
            return default_return
        return _noop
    return cb


def _mode_ids_from_count(n_sets: int, mode_count: Optional[int]) -> List[int]:
    if n_sets <= 0:
        return []
    if mode_count is None:
        return list(range(1, n_sets + 1))
    return list(range(1, min(n_sets, int(mode_count)) + 1))


def extract_modal_tensor_csv(
    rst_path: str,
    output_csv_path: str,
    result_kind: str,
    named_selection: str = "All Nodes",
    mode_ids: Optional[Sequence[int]] = None,
    mode_count: Optional[int] = None,
    chunk_size: Optional[int] = None,
    node_order: Optional[Sequence[int]] = None,
    log_cb: Optional[Callable[[str], None]] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> None:
    """Extract modal tensor results using PyMAPDL Reader."""
    _require_reader()
    log = _ensure_callback(log_cb)
    progress = _ensure_callback(progress_cb)
    cancel_check = _ensure_callback(should_cancel, False)
    
    log("Using PyMAPDL Reader backend")
    rst = pymapdl_reader.read_binary(rst_path)
    
    if node_order is not None:
        output_order = list(node_order)
        unique_ids = list(dict.fromkeys(output_order))
        node_ids = unique_ids
    else:
        mesh = rst.mesh
        if named_selection and named_selection != "All Nodes":
            if named_selection in rst.node_components:
                node_ids = list(rst.node_components[named_selection])
            elif named_selection in rst.element_components:
                log(f"Warning: Element component '{named_selection}' - using all nodes instead")
                node_ids = list(mesh.nnum)
            else:
                log(f"Warning: Named selection '{named_selection}' not found - using all nodes")
                node_ids = list(mesh.nnum)
        else:
            node_ids = list(mesh.nnum)
        node_ids = sorted(node_ids)
        output_order = node_ids
    
    n_nodes = len(output_order)
    
    if mode_ids is None:
        n_sets = rst.n_results
        mode_ids = _mode_ids_from_count(n_sets, mode_count)
    
    if not mode_ids:
        raise ModalExtractionError("No mode IDs available to extract.")
    
    components = _result_components(result_kind)
    n_components = len(components)
    
    if chunk_size is None:
        chunk_size = memory_policy.compute_chunk_size(len(node_ids), len(mode_ids), n_components)
    
    total_chunks = memory_policy.compute_chunk_count(n_nodes, chunk_size)
    log(f"Extracting {result_kind} for {n_nodes} nodes, {len(mode_ids)} modes, chunk size {chunk_size}")
    
    coords_all = _get_node_coordinates(rst, node_ids)
    coord_map = {nid: coords_all[idx] for idx, nid in enumerate(node_ids)}
    
    header = ["NodeID", "X", "Y", "Z"]
    for set_id in mode_ids:
        for comp in components:
            header.append(f"{comp}_Mode{set_id}")
    
    log("Loading mode data...")
    mode_data_cache = {}
    
    for mode_idx, set_id in enumerate(mode_ids):
        if cancel_check():
            raise ModalExtractionCanceled("Extraction canceled.")
        
        rnum = set_id - 1
        try:
            nnum, data = _extract_result_for_mode(rst, result_kind, rnum)
            mode_data_cache[set_id] = (nnum, data)
            log(f"  Loaded mode {mode_idx + 1}/{len(mode_ids)} (set {set_id})")
        except Exception as e:
            log(f"  Mode {set_id} failed: {e} - using zeros")
            mode_data_cache[set_id] = None
        
        if (mode_idx + 1) % 10 == 0:
            gc.collect()
    
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
                f"({len(chunk_output_ids)} nodes)"
            )
            
            if has_duplicates or node_order is not None:
                chunk_unique_ids = list(dict.fromkeys(chunk_output_ids))
                mode_maps = []
                for set_id in mode_ids:
                    cached = mode_data_cache.get(set_id)
                    if cached is None:
                        mode_map = {nid: np.zeros(n_components) for nid in chunk_unique_ids}
                    else:
                        nnum, data = cached
                        aligned = _align_result_to_nodes(nnum, data, chunk_unique_ids, n_components)
                        mode_map = {nid: aligned[idx] for idx, nid in enumerate(chunk_unique_ids)}
                    mode_maps.append(mode_map)
                
                rows = []
                for nid in chunk_output_ids:
                    row = [nid, *coord_map.get(nid, (0.0, 0.0, 0.0))]
                    for mode_map in mode_maps:
                        row.extend(mode_map.get(nid, np.zeros(n_components)))
                    rows.append(row)
                
                csv_writer.write_chunk(handle, np.array(rows, dtype=float))
            else:
                chunk_data = np.zeros((len(chunk_output_ids), 4 + len(mode_ids) * n_components), dtype=float)
                chunk_data[:, 0] = np.array(chunk_output_ids, dtype=float)
                for idx, nid in enumerate(chunk_output_ids):
                    chunk_data[idx, 1:4] = coord_map.get(nid, (0.0, 0.0, 0.0))
                
                col = 4
                for set_id in mode_ids:
                    cached = mode_data_cache.get(set_id)
                    if cached is not None:
                        nnum, data = cached
                        aligned = _align_result_to_nodes(nnum, data, chunk_output_ids, n_components)
                        chunk_data[:, col:col + n_components] = aligned
                    col += n_components
                
                csv_writer.write_chunk(handle, chunk_data)
            
            chunk_duration = time.perf_counter() - chunk_start_time
            log(f"Chunk {chunk_index + 1}/{total_chunks} completed in {chunk_duration:.1f}s")
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
