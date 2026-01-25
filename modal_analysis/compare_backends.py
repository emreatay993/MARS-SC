"""Compare DPF and PyMAPDL backend extraction results - detailed analysis."""

import os
import sys

sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modal_analysis.modal_gui import dpf_modal_extractor

RST_FILE = r"C:\Users\emre_\PycharmProjects\Solution_Combination_ANSYS\modal_analysis\debug_rst\file1.rst"
OUTPUT_DPF = r"C:\Users\emre_\PycharmProjects\Solution_Combination_ANSYS\modal_analysis\debug_rst\_compare_dpf.csv"
OUTPUT_PYMAPDL = r"C:\Users\emre_\PycharmProjects\Solution_Combination_ANSYS\modal_analysis\debug_rst\_compare_pymapdl.csv"

NUM_MODES = 3
NAMED_SELECTION = "NS_MODAL_EXPANSION"


def log(msg):
    print(msg, flush=True)


def extract_dpf():
    print(f"[DPF] Extracting {NUM_MODES} modes for {NAMED_SELECTION}...", flush=True)
    dpf_modal_extractor.extract_modal_stress_csv(
        rst_path=RST_FILE,
        output_csv_path=OUTPUT_DPF,
        mode_count=NUM_MODES,
        named_selection=NAMED_SELECTION,
        backend="dpf",
        log_cb=log,
    )
    print("[DPF] Done.", flush=True)


def extract_pymapdl():
    print(f"[PyMAPDL] Extracting {NUM_MODES} modes for {NAMED_SELECTION}...", flush=True)
    dpf_modal_extractor.extract_modal_stress_csv(
        rst_path=RST_FILE,
        output_csv_path=OUTPUT_PYMAPDL,
        mode_count=NUM_MODES,
        named_selection=NAMED_SELECTION,
        backend="pymapdl",
        log_cb=log,
    )
    print("[PyMAPDL] Done.", flush=True)


def compare_detailed():
    print("\n" + "=" * 70, flush=True)
    print("DETAILED COMPARISON", flush=True)
    print("=" * 70, flush=True)
    
    df_dpf = pd.read_csv(OUTPUT_DPF)
    df_pymapdl = pd.read_csv(OUTPUT_PYMAPDL)
    
    print(f"\nDPF shape: {df_dpf.shape}", flush=True)
    print(f"PyMAPDL shape: {df_pymapdl.shape}", flush=True)
    
    if df_dpf.shape != df_pymapdl.shape:
        print("ERROR: Different shapes!", flush=True)
        return
    
    # Check node IDs
    dpf_nodes = df_dpf["NodeID"].values
    pymapdl_nodes = df_pymapdl["NodeID"].values
    if not np.array_equal(dpf_nodes, pymapdl_nodes):
        print("ERROR: Node IDs don't match!", flush=True)
        return
    print(f"Node IDs match ({len(dpf_nodes)} nodes)", flush=True)
    
    # Coordinates comparison
    print("\n--- COORDINATES ---", flush=True)
    for coord in ["X", "Y", "Z"]:
        diff = np.abs(df_dpf[coord].values - df_pymapdl[coord].values)
        max_diff = diff.max()
        print(f"  {coord}: max diff = {max_diff:.6e}", flush=True)
    
    # Stress components for each mode
    stress_components = ["sx", "sy", "sz", "sxy", "syz", "sxz"]
    
    print("\n--- STRESS COMPONENTS BY MODE ---", flush=True)
    for mode in range(1, NUM_MODES + 1):
        print(f"\n  Mode {mode}:", flush=True)
        for comp in stress_components:
            col = f"{comp}_Mode{mode}"
            if col not in df_dpf.columns:
                print(f"    {comp}: column not found", flush=True)
                continue
                
            dpf_vals = df_dpf[col].values
            pymapdl_vals = df_pymapdl[col].values
            
            abs_diff = np.abs(dpf_vals - pymapdl_vals)
            max_abs = abs_diff.max()
            mean_abs = abs_diff.mean()
            
            # Relative difference where significant
            magnitude = np.maximum(np.abs(dpf_vals), np.abs(pymapdl_vals))
            mask = magnitude > 1e-10
            if mask.any():
                rel_diff = abs_diff[mask] / magnitude[mask]
                max_rel = rel_diff.max() * 100
                mean_rel = rel_diff.mean() * 100
            else:
                max_rel = 0.0
                mean_rel = 0.0
            
            # Find node with max difference
            max_idx = np.argmax(abs_diff)
            max_node = int(dpf_nodes[max_idx])
            
            print(f"    {comp:3s}: max_abs={max_abs:10.4e}, mean_abs={mean_abs:10.4e}, "
                  f"max_rel={max_rel:6.4f}%, node={max_node}", flush=True)
    
    # Overall statistics
    print("\n--- OVERALL STATISTICS ---", flush=True)
    stress_cols = [f"{c}_Mode{m}" for m in range(1, NUM_MODES + 1) for c in stress_components]
    stress_cols = [c for c in stress_cols if c in df_dpf.columns]
    
    all_dpf = df_dpf[stress_cols].values.flatten()
    all_pymapdl = df_pymapdl[stress_cols].values.flatten()
    all_diff = np.abs(all_dpf - all_pymapdl)
    
    print(f"  Total values compared: {len(all_diff)}", flush=True)
    print(f"  Max absolute difference: {all_diff.max():.6e}", flush=True)
    print(f"  Mean absolute difference: {all_diff.mean():.6e}", flush=True)
    print(f"  Std absolute difference: {all_diff.std():.6e}", flush=True)
    
    # How many values have non-zero difference
    nonzero_diff = np.sum(all_diff > 1e-15)
    print(f"  Values with diff > 1e-15: {nonzero_diff} ({100*nonzero_diff/len(all_diff):.2f}%)", flush=True)
    
    # Sample nodes with largest differences
    print("\n--- NODES WITH LARGEST DIFFERENCES ---", flush=True)
    
    # Calculate per-node max difference across all components
    node_max_diff = np.zeros(len(dpf_nodes))
    for col in stress_cols:
        diff = np.abs(df_dpf[col].values - df_pymapdl[col].values)
        node_max_diff = np.maximum(node_max_diff, diff)
    
    # Top 10 nodes
    top_indices = np.argsort(node_max_diff)[-10:][::-1]
    print(f"  {'NodeID':>10} {'MaxDiff':>12} {'X':>10} {'Y':>10} {'Z':>10}", flush=True)
    print("  " + "-" * 60, flush=True)
    for idx in top_indices:
        nid = int(dpf_nodes[idx])
        diff = node_max_diff[idx]
        x, y, z = df_dpf.iloc[idx][["X", "Y", "Z"]]
        print(f"  {nid:>10} {diff:>12.4e} {x:>10.2f} {y:>10.2f} {z:>10.2f}", flush=True)
    
    # Detailed comparison for top differing nodes
    print("\n--- DETAILED VALUES FOR TOP 3 DIFFERING NODES ---", flush=True)
    for rank, idx in enumerate(top_indices[:3], 1):
        nid = int(dpf_nodes[idx])
        print(f"\n  Node {nid} (rank {rank}):", flush=True)
        print(f"    Coords: X={df_dpf.iloc[idx]['X']:.4f}, Y={df_dpf.iloc[idx]['Y']:.4f}, Z={df_dpf.iloc[idx]['Z']:.4f}", flush=True)
        for mode in range(1, NUM_MODES + 1):
            print(f"    Mode {mode}:", flush=True)
            for comp in stress_components:
                col = f"{comp}_Mode{mode}"
                if col in df_dpf.columns:
                    dpf_v = df_dpf.iloc[idx][col]
                    pym_v = df_pymapdl.iloc[idx][col]
                    diff = abs(dpf_v - pym_v)
                    rel = 100 * diff / max(abs(dpf_v), abs(pym_v), 1e-20)
                    print(f"      {comp}: DPF={dpf_v:12.4e}, PyMAPDL={pym_v:12.4e}, diff={diff:.2e} ({rel:.4f}%)", flush=True)
    
    # Final verdict
    print("\n" + "=" * 70, flush=True)
    if all_diff.max() < 1e-10:
        print("VERDICT: Results are IDENTICAL (max diff < 1e-10)", flush=True)
    elif all_diff.max() < 1e-6:
        print("VERDICT: Results match within numerical precision (max diff < 1e-6)", flush=True)
    else:
        rel_mask = np.abs(all_dpf) > 1e-10
        if rel_mask.any():
            max_rel = (all_diff[rel_mask] / np.abs(all_dpf[rel_mask])).max() * 100
            print(f"VERDICT: Max relative difference = {max_rel:.4f}%", flush=True)
        else:
            print(f"VERDICT: Max absolute difference = {all_diff.max():.4e}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dpf", action="store_true", help="Extract with DPF only")
    parser.add_argument("--pymapdl", action="store_true", help="Extract with PyMAPDL only")
    parser.add_argument("--compare", action="store_true", help="Compare existing files")
    args = parser.parse_args()
    
    if args.dpf:
        extract_dpf()
    elif args.pymapdl:
        extract_pymapdl()
    elif args.compare:
        compare_detailed()
    else:
        extract_dpf()
        extract_pymapdl()
        compare_detailed()
