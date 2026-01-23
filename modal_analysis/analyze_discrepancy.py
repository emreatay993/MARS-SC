"""Analyze discrepancy patterns between reference and generated stress."""

import pandas as pd
import numpy as np

# Read both files
ref = pd.read_csv('example_tensor_outputs/modal_stress_tensor_w_coords_2.csv')
gen = pd.read_csv('_compare_outputs/modal_stress_tensor_w_coords.csv')

# Check precision of values
print('Sample reference values (node 4750, sxz_Mode10):')
ref_val = ref[ref['NodeID'] == 4750]['sxz_Mode10'].values[0]
print(f'  As stored: {ref_val}')
print(f'  With 15 decimals: {ref_val:.15f}')

print()
print('Sample generated values (node 4750, sxz_Mode10):')
gen_val = gen[gen['NodeID'] == 4750]['sxz_Mode10'].values[0]
print(f'  As stored: {gen_val}')
print(f'  With 15 decimals: {gen_val:.15f}')

# Check if there's any systematic pattern in the differences
stress_cols = [c for c in ref.columns if c.startswith('s')]
diffs = []
for col in stress_cols:
    ref_vals = ref[col].values
    gen_vals = gen[col].values
    diff = np.abs(ref_vals - gen_vals)
    diffs.append({
        'column': col,
        'max_diff': np.max(diff),
        'mean_diff': np.mean(diff),
        'nodes_with_diff_over_0.1': np.sum(diff > 0.1)
    })

print()
print('Columns with highest discrepancies:')
diffs_sorted = sorted(diffs, key=lambda x: x['max_diff'], reverse=True)[:10]
for d in diffs_sorted:
    print(f"  {d['column']}: max={d['max_diff']:.4f}, mean={d['mean_diff']:.6f}, n_large={d['nodes_with_diff_over_0.1']}")

# Check which modes and components have issues
print()
print('Discrepancy by component type:')
by_comp = {}
for d in diffs:
    comp = d['column'].split('_')[0]  # sx, sy, sz, sxy, syz, sxz
    if comp not in by_comp:
        by_comp[comp] = []
    by_comp[comp].append(d['max_diff'])

for comp in ['sx', 'sy', 'sz', 'sxy', 'syz', 'sxz']:
    vals = by_comp.get(comp, [])
    if vals:
        print(f"  {comp}: max={max(vals):.4f}, mean={np.mean(vals):.6f}")
