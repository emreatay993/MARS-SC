"""Debug script to investigate stress discrepancy between DPF and ANSYS reference."""

import json
import pathlib
import time
import numpy as np
import pandas as pd

from ansys.dpf import core as dpf

DBG_PATH = pathlib.Path(r"c:\Users\emre_\PycharmProjects\Solution_Combination_ANSYS\.cursor\debug.log")

def dbg_log(location, message, data, hypothesis_id):
    """Log debug info to NDJSON file."""
    entry = {
        "location": location,
        "message": message,
        "data": data,
        "hypothesisId": hypothesis_id,
        "timestamp": time.time(),
        "sessionId": "debug-session"
    }
    with DBG_PATH.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def investigate_node(model, mesh, node_id, mode_id, ref_values):
    """Investigate stress values at a specific node."""
    
    # Get connected elements
    connected_elements = []
    for elem_idx in range(mesh.elements.n_elements):
        elem = mesh.elements.element_by_index(elem_idx)
        if node_id in elem.connectivity:
            connected_elements.append(elem.id)
    
    dbg_log("debug_stress_discrepancy.py:investigate_node", 
            f"Node {node_id} connectivity",
            {"node_id": node_id, "connected_elements": connected_elements, "n_elements": len(connected_elements)},
            "H2")
    
    # Get elemental_nodal stress
    time_scoping = dpf.Scoping()
    time_scoping.ids = [mode_id]
    
    stress_op = model.results.stress()
    stress_op.inputs.time_scoping.connect(time_scoping)
    stress_op.inputs.requested_location.connect(dpf.locations.elemental_nodal)
    fields = stress_op.outputs.fields_container()
    field = fields[0]
    
    # Get contributions from each element
    element_contributions = []
    for elem_id in connected_elements:
        elem_idx = mesh.elements.scoping.index(elem_id)
        elem = mesh.elements.element_by_index(elem_idx)
        connectivity = list(elem.connectivity)
        local_node_idx = connectivity.index(node_id)
        
        try:
            entity_data = field.get_entity_data_by_id(elem_id)
            stress_at_node = entity_data[local_node_idx].tolist()
            element_contributions.append({
                "elem_id": elem_id,
                "local_idx": local_node_idx,
                "stress_components": stress_at_node
            })
        except Exception as e:
            dbg_log("debug_stress_discrepancy.py:investigate_node",
                    f"Error getting element data",
                    {"elem_id": elem_id, "error": str(e)},
                    "H1")
    
    dbg_log("debug_stress_discrepancy.py:investigate_node",
            f"Element contributions for node {node_id}",
            {"element_contributions": element_contributions},
            "H3")
    
    # Calculate simple average
    if element_contributions:
        all_stress = np.array([c["stress_components"] for c in element_contributions])
        simple_avg = np.mean(all_stress, axis=0)
        dbg_log("debug_stress_discrepancy.py:investigate_node",
                f"Simple average of element contributions",
                {"simple_avg": simple_avg.tolist()},
                "H3")
    
    # Get DPF nodal value
    node_scoping = dpf.Scoping(location=dpf.locations.nodal)
    node_scoping.ids = [node_id]
    
    stress_op_nodal = model.results.stress()
    stress_op_nodal.inputs.time_scoping.connect(time_scoping)
    stress_op_nodal.inputs.requested_location.connect(dpf.locations.nodal)
    stress_op_nodal.inputs.mesh_scoping.connect(node_scoping)
    
    fields_nodal = stress_op_nodal.outputs.fields_container()
    dpf_nodal = fields_nodal[0].data[0].tolist()
    
    dbg_log("debug_stress_discrepancy.py:investigate_node",
            f"DPF nodal stress for node {node_id}",
            {"dpf_nodal": dpf_nodal, "ref_values": ref_values},
            "H1")
    
    # Calculate differences
    ref_array = np.array(ref_values)
    dpf_array = np.array(dpf_nodal)
    diff = ref_array - dpf_array
    
    components = ["sx", "sy", "sz", "sxy", "syz", "sxz"]
    comp_diff = {comp: float(diff[i]) for i, comp in enumerate(components)}
    
    dbg_log("debug_stress_discrepancy.py:investigate_node",
            f"Component-wise differences for node {node_id}",
            {"differences": comp_diff, "max_diff_component": components[np.argmax(np.abs(diff))]},
            "H4")


def main():
    # Load model
    rst_path = "example_rst_file_modal/file.rst"
    model = dpf.Model(rst_path)
    mesh = model.metadata.meshed_region
    
    dbg_log("debug_stress_discrepancy.py:main", "Starting stress discrepancy analysis",
            {"rst_path": rst_path, "n_nodes": mesh.nodes.n_nodes, "n_elements": mesh.elements.n_elements},
            "H1")
    
    # Load reference data
    ref_df = pd.read_csv("example_tensor_outputs/modal_stress_tensor_w_coords_2.csv")
    
    # Identify nodes with highest discrepancy (from previous analysis)
    problem_nodes = [4750, 4752, 4718, 4751, 4719]
    
    for node_id in problem_nodes:
        row = ref_df[ref_df["NodeID"] == node_id]
        if len(row) == 0:
            continue
            
        # Get reference values for Mode 10
        ref_values = [
            row["sx_Mode10"].values[0],
            row["sy_Mode10"].values[0],
            row["sz_Mode10"].values[0],
            row["sxy_Mode10"].values[0],
            row["syz_Mode10"].values[0],
            row["sxz_Mode10"].values[0],
        ]
        
        investigate_node(model, mesh, node_id, 10, ref_values)
    
    # Also check a node with low discrepancy for comparison
    low_discrepancy_node = 4976
    row = ref_df[ref_df["NodeID"] == low_discrepancy_node]
    ref_values = [
        row["sx_Mode10"].values[0],
        row["sy_Mode10"].values[0],
        row["sz_Mode10"].values[0],
        row["sxy_Mode10"].values[0],
        row["syz_Mode10"].values[0],
        row["sxz_Mode10"].values[0],
    ]
    
    dbg_log("debug_stress_discrepancy.py:main", "Checking low discrepancy node for comparison",
            {"node_id": low_discrepancy_node},
            "H2")
    investigate_node(model, mesh, low_discrepancy_node, 10, ref_values)
    
    # Check if problem nodes are at body boundaries
    # Look for named selections containing these nodes
    ns_names = model.metadata.available_named_selections
    
    for node_id in problem_nodes:
        containing_ns = []
        for ns_name in ns_names:
            try:
                scoping = model.metadata.named_selection(ns_name)
                if scoping.location == dpf.locations.nodal:
                    if node_id in scoping.ids:
                        containing_ns.append(ns_name)
            except:
                pass
        
        dbg_log("debug_stress_discrepancy.py:main",
                f"Named selections for node {node_id}",
                {"node_id": node_id, "named_selections": containing_ns},
                "H2")
    
    dbg_log("debug_stress_discrepancy.py:main", "Analysis complete", {}, "H1")


if __name__ == "__main__":
    main()
