"""Compare DPF and PyMAPDL backend extraction results."""

import os
import sys

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modal_analysis.modal_gui import dpf_modal_extractor

RST_FILE = r"C:\Users\emre_\PycharmProjects\Solution_Combination_ANSYS\modal_analysis\debug_rst\file1.rst"
OUTPUT_DPF = r"C:\Users\emre_\PycharmProjects\Solution_Combination_ANSYS\modal_analysis\debug_rst\_compare_dpf.csv"
OUTPUT_PYMAPDL = r"C:\Users\emre_\PycharmProjects\Solution_Combination_ANSYS\modal_analysis\debug_rst\_compare_pymapdl.csv"

NUM_MODES = 10


def log(msg):
    print(msg, flush=True)


def extract_dpf():
    print(f"[DPF] Extracting {NUM_MODES} modes...", flush=True)
    dpf_modal_extractor.extract_modal_stress_csv(
        rst_path=RST_FILE,
        output_csv_path=OUTPUT_DPF,
        mode_count=NUM_MODES,
        backend="dpf",
        log_cb=log,
    )
    print("[DPF] Done.", flush=True)


def extract_pymapdl():
    print(f"[PyMAPDL] Extracting {NUM_MODES} modes...", flush=True)
    dpf_modal_extractor.extract_modal_stress_csv(
        rst_path=RST_FILE,
        output_csv_path=OUTPUT_PYMAPDL,
        mode_count=NUM_MODES,
        backend="pymapdl",
        log_cb=log,
    )
    print("[PyMAPDL] Done.", flush=True)


def compare():
    print("\n[Compare] Loading CSVs...", flush=True)
    df_dpf = pd.read_csv(OUTPUT_DPF)
    df_pymapdl = pd.read_csv(OUTPUT_PYMAPDL)
    
    print(f"DPF shape: {df_dpf.shape}", flush=True)
    print(f"PyMAPDL shape: {df_pymapdl.shape}", flush=True)
    
    if df_dpf.shape != df_pymapdl.shape:
        print("ERROR: Different shapes!", flush=True)
        return
    
    # Check node IDs match
    if not np.array_equal(df_dpf["NodeID"].values, df_pymapdl["NodeID"].values):
        print("ERROR: Node IDs don't match!", flush=True)
        return
    print("Node IDs match.", flush=True)
    
    # Compare coordinates
    coord_cols = ["X", "Y", "Z"]
    coord_diff = np.abs(df_dpf[coord_cols].values - df_pymapdl[coord_cols].values)
    max_coord_diff = coord_diff.max()
    print(f"Max coordinate difference: {max_coord_diff:.6e}", flush=True)
    
    # Compare stress values
    stress_cols = [c for c in df_dpf.columns if c.startswith("s")]
    dpf_stress = df_dpf[stress_cols].values
    pymapdl_stress = df_pymapdl[stress_cols].values
    
    abs_diff = np.abs(dpf_stress - pymapdl_stress)
    max_abs_diff = abs_diff.max()
    mean_abs_diff = abs_diff.mean()
    
    print(f"\nStress comparison ({len(stress_cols)} columns):", flush=True)
    print(f"  Max absolute difference: {max_abs_diff:.6e}", flush=True)
    print(f"  Mean absolute difference: {mean_abs_diff:.6e}", flush=True)
    
    # Relative difference
    dpf_magnitude = np.abs(dpf_stress)
    mask = dpf_magnitude > 1e-10
    if mask.any():
        rel_diff = abs_diff[mask] / dpf_magnitude[mask]
        max_rel_diff = rel_diff.max()
        mean_rel_diff = rel_diff.mean()
        print(f"  Max relative difference: {max_rel_diff:.6e} ({max_rel_diff*100:.4f}%)", flush=True)
        print(f"  Mean relative difference: {mean_rel_diff:.6e} ({mean_rel_diff*100:.4f}%)", flush=True)
    
    # Sample comparison
    print("\nSample (first 3 nodes, sx_Mode1):", flush=True)
    print("NodeID | DPF | PyMAPDL | Diff", flush=True)
    for i in range(min(3, len(df_dpf))):
        nid = df_dpf["NodeID"].iloc[i]
        dpf_val = df_dpf["sx_Mode1"].iloc[i]
        pymapdl_val = df_pymapdl["sx_Mode1"].iloc[i]
        diff = abs(dpf_val - pymapdl_val)
        print(f"{nid:6.0f} | {dpf_val:12.4e} | {pymapdl_val:12.4e} | {diff:.2e}", flush=True)


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
        compare()
    else:
        # Run all
        extract_dpf()
        extract_pymapdl()
        compare()
