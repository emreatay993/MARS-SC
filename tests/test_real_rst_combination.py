"""
Integration test for combination algorithm using real RST data.

Tests the combination algorithm with actual RST files from example_dataset folder,
using different named selections as scopings and computing von Mises and principal stresses.
"""

import pytest
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from file_io.dpf_reader import DPFAnalysisReader, DPF_AVAILABLE
from solver.combination_engine import CombinationEngine
from core.data_models import CombinationTableData, CombinationResult


# Skip all tests if DPF is not available
pytestmark = pytest.mark.skipif(not DPF_AVAILABLE, reason="DPF not installed")

# Paths to RST files
RST_FILE_1 = os.path.join(os.path.dirname(__file__), '..', 'example_dataset', 'file_analysis1.rst')
RST_FILE_2 = os.path.join(os.path.dirname(__file__), '..', 'example_dataset', 'file_analysis2.rst')


def create_simple_combination_table(a1_step: int, a2_step: int) -> CombinationTableData:
    """
    Create a simple combination table that adds two analyses.
    
    Uses coefficient of 1.0 for both analyses (simple addition).
    
    Args:
        a1_step: Load step ID from Analysis 1 (e.g., 2 for second time step)
        a2_step: Load step ID from Analysis 2 (e.g., 4 for fourth time step)
        
    Returns:
        CombinationTableData with single simple combination
    """
    return CombinationTableData(
        combination_names=["Simple_Add_A1_A2"],
        combination_types=["Linear"],
        analysis1_coeffs=np.array([[1.0]]),  # Coefficient 1.0 for A1
        analysis2_coeffs=np.array([[1.0]]),  # Coefficient 1.0 for A2
        analysis1_step_ids=[a1_step],
        analysis2_step_ids=[a2_step],
    )


def print_stress_statistics(name: str, values: np.ndarray, node_ids: np.ndarray, unit: str = "Pa"):
    """Print statistics for stress results."""
    print(f"\n  {name}:")
    print(f"    Min:  {np.min(values):>15.4f} {unit} at Node {node_ids[np.argmin(values)]}")
    print(f"    Max:  {np.max(values):>15.4f} {unit} at Node {node_ids[np.argmax(values)]}")
    print(f"    Mean: {np.mean(values):>15.4f} {unit}")
    print(f"    Std:  {np.std(values):>15.4f} {unit}")


class TestRealRSTCombination:
    """Test combination algorithm with real RST files."""
    
    @pytest.fixture(scope="class")
    def readers(self):
        """Create DPF readers for both RST files."""
        if not os.path.exists(RST_FILE_1):
            pytest.skip(f"RST file not found: {RST_FILE_1}")
        if not os.path.exists(RST_FILE_2):
            pytest.skip(f"RST file not found: {RST_FILE_2}")
            
        reader1 = DPFAnalysisReader(RST_FILE_1)
        reader2 = DPFAnalysisReader(RST_FILE_2)
        return reader1, reader2
    
    def test_list_rst_metadata(self, readers):
        """Test listing metadata from RST files."""
        reader1, reader2 = readers
        
        # Get model info
        info1 = reader1.get_model_info()
        info2 = reader2.get_model_info()
        
        print("\n" + "="*80)
        print("RST FILE METADATA")
        print("="*80)
        
        print(f"\nAnalysis 1: {info1['file_path']}")
        print(f"  Nodes: {info1['num_nodes']:,}")
        print(f"  Elements: {info1['num_elements']:,}")
        print(f"  Load Steps: {info1['num_load_steps']}")
        print(f"  Analysis Type: {info1['analysis_type']}")
        print(f"  Unit System: {info1['unit_system']}")
        print(f"  Named Selections ({len(info1['named_selections'])}):")
        for ns in info1['named_selections']:
            print(f"    - {ns}")
        
        print(f"\nAnalysis 2: {info2['file_path']}")
        print(f"  Nodes: {info2['num_nodes']:,}")
        print(f"  Elements: {info2['num_elements']:,}")
        print(f"  Load Steps: {info2['num_load_steps']}")
        print(f"  Analysis Type: {info2['analysis_type']}")
        print(f"  Unit System: {info2['unit_system']}")
        print(f"  Named Selections ({len(info2['named_selections'])}):")
        for ns in info2['named_selections']:
            print(f"    - {ns}")
        
        # Validate step availability
        a1_steps = reader1.get_load_step_ids()
        a2_steps = reader2.get_load_step_ids()
        print(f"\nAnalysis 1 available steps: {a1_steps}")
        print(f"Analysis 2 available steps: {a2_steps}")
        
        assert info1['num_nodes'] > 0, "Analysis 1 should have nodes"
        assert info2['num_nodes'] > 0, "Analysis 2 should have nodes"
    
    def test_combination_with_all_nodes(self, readers):
        """Test combination using all nodes (no scoping)."""
        reader1, reader2 = readers
        
        # Determine available steps
        a1_steps = reader1.get_load_step_ids()
        a2_steps = reader2.get_load_step_ids()
        
        # Use step 2 from A1 and step 4 from A2 (or last available)
        a1_step = 2 if 2 in a1_steps else a1_steps[-1]
        a2_step = 4 if 4 in a2_steps else a2_steps[-1]
        
        print("\n" + "="*80)
        print(f"TESTING COMBINATION WITH ALL NODES")
        print(f"Using A1 Step {a1_step} + A2 Step {a2_step}")
        print("="*80)
        
        # Create simple combination table
        combo_table = create_simple_combination_table(a1_step, a2_step)
        
        # Get all nodes scoping from reader1
        all_nodes_scoping = reader1.get_all_nodes_scoping()
        print(f"\nTotal nodes in scoping: {len(all_nodes_scoping.ids):,}")
        
        # Create combination engine
        engine = CombinationEngine(reader1, reader2, all_nodes_scoping, combo_table)
        
        # Preload stress data
        print("Loading stress data...")
        engine.preload_stress_data()
        print(f"Stress data loaded for {engine.num_nodes:,} nodes")
        
        # Compute von Mises
        print("\nComputing von Mises stress...")
        vm_result = engine.compute_full_analysis(stress_type="von_mises")
        print_stress_statistics("Von Mises Stress", vm_result.max_over_combo, vm_result.node_ids)
        
        # Compute max principal
        print("\nComputing max principal stress (S1)...")
        engine.clear_cache()
        engine.preload_stress_data()
        s1_result = engine.compute_full_analysis(stress_type="max_principal")
        print_stress_statistics("Max Principal (S1)", s1_result.max_over_combo, s1_result.node_ids)
        
        # Compute min principal
        print("\nComputing min principal stress (S3)...")
        engine.clear_cache()
        engine.preload_stress_data()
        s3_result = engine.compute_full_analysis(stress_type="min_principal")
        print_stress_statistics("Min Principal (S3)", s3_result.min_over_combo, s3_result.node_ids)
        
        # Basic assertions
        assert vm_result.num_nodes > 0
        assert np.all(vm_result.max_over_combo >= 0)  # Von Mises is non-negative
        assert np.all(s1_result.max_over_combo >= s3_result.min_over_combo)  # S1 >= S3
    
    def test_combination_with_named_selections(self, readers):
        """Test combination using each named selection as scoping."""
        reader1, reader2 = readers
        
        # Get named selections from reader1 (assumed to be the same in both)
        named_selections = reader1.get_named_selections()
        
        if not named_selections:
            pytest.skip("No named selections in RST files")
        
        # Determine available steps
        a1_steps = reader1.get_load_step_ids()
        a2_steps = reader2.get_load_step_ids()
        
        # Use step 2 from A1 and step 4 from A2 (or last available)
        a1_step = 2 if 2 in a1_steps else a1_steps[-1]
        a2_step = 4 if 4 in a2_steps else a2_steps[-1]
        
        print("\n" + "="*80)
        print(f"TESTING COMBINATION WITH NAMED SELECTIONS")
        print(f"Using A1 Step {a1_step} + A2 Step {a2_step}")
        print(f"Named Selections to test: {len(named_selections)}")
        print("="*80)
        
        # Create simple combination table
        combo_table = create_simple_combination_table(a1_step, a2_step)
        
        results_summary = []
        
        for ns_name in named_selections:
            print(f"\n{'-'*60}")
            print(f"Named Selection: {ns_name}")
            print(f"{'-'*60}")
            
            try:
                # Get nodal scoping from named selection
                nodal_scoping = reader1.get_nodal_scoping_from_named_selection(ns_name)
                num_nodes = len(nodal_scoping.ids)
                print(f"  Nodes in selection: {num_nodes:,}")
                
                if num_nodes == 0:
                    print(f"  SKIPPED: Named selection '{ns_name}' has no nodes")
                    continue
                
                # Create combination engine
                engine = CombinationEngine(reader1, reader2, nodal_scoping, combo_table)
                
                # Preload stress data
                engine.preload_stress_data()
                
                # Compute von Mises
                vm_result = engine.compute_full_analysis(stress_type="von_mises")
                print_stress_statistics("Von Mises Stress", vm_result.max_over_combo, vm_result.node_ids)
                
                # Compute max principal
                engine.clear_cache()
                engine.preload_stress_data()
                s1_result = engine.compute_full_analysis(stress_type="max_principal")
                print_stress_statistics("Max Principal (S1)", s1_result.max_over_combo, s1_result.node_ids)
                
                # Compute min principal
                engine.clear_cache()
                engine.preload_stress_data()
                s3_result = engine.compute_full_analysis(stress_type="min_principal")
                print_stress_statistics("Min Principal (S3)", s3_result.min_over_combo, s3_result.node_ids)
                
                # Store summary
                results_summary.append({
                    'named_selection': ns_name,
                    'num_nodes': num_nodes,
                    'vm_max': np.max(vm_result.max_over_combo),
                    'vm_min': np.min(vm_result.max_over_combo),
                    's1_max': np.max(s1_result.max_over_combo),
                    's3_min': np.min(s3_result.min_over_combo),
                })
                
                # Basic assertions for this named selection
                assert vm_result.num_nodes == num_nodes
                assert np.all(vm_result.max_over_combo >= 0)
                
            except Exception as e:
                print(f"  ERROR processing '{ns_name}': {e}")
                continue
        
        # Print summary table
        if results_summary:
            print("\n" + "="*80)
            print("RESULTS SUMMARY")
            print("="*80)
            print(f"{'Named Selection':<30} {'Nodes':>8} {'VM Max':>12} {'S1 Max':>12} {'S3 Min':>12}")
            print("-"*80)
            for r in results_summary:
                print(f"{r['named_selection']:<30} {r['num_nodes']:>8,} {r['vm_max']:>12.2f} {r['s1_max']:>12.2f} {r['s3_min']:>12.2f}")
            
        assert len(results_summary) > 0, "At least one named selection should succeed"
    
    def test_individual_analysis_stresses(self, readers):
        """Test reading stresses from individual analyses (before combination)."""
        reader1, reader2 = readers
        
        # Get available steps
        a1_steps = reader1.get_load_step_ids()
        a2_steps = reader2.get_load_step_ids()
        
        a1_step = 2 if 2 in a1_steps else a1_steps[-1]
        a2_step = 4 if 4 in a2_steps else a2_steps[-1]
        
        print("\n" + "="*80)
        print("INDIVIDUAL ANALYSIS STRESSES (Before Combination)")
        print("="*80)
        
        # Get all nodes scoping
        all_scoping = reader1.get_all_nodes_scoping()
        
        # Read stress from Analysis 1
        print(f"\nAnalysis 1 - Step {a1_step}:")
        node_ids, sx, sy, sz, sxy, syz, sxz = reader1.read_stress_tensor_for_loadstep(
            a1_step, all_scoping
        )
        vm_a1 = CombinationEngine.compute_von_mises(sx, sy, sz, sxy, syz, sxz)
        print_stress_statistics("Von Mises", vm_a1, node_ids)
        
        # Read stress from Analysis 2
        print(f"\nAnalysis 2 - Step {a2_step}:")
        node_ids, sx, sy, sz, sxy, syz, sxz = reader2.read_stress_tensor_for_loadstep(
            a2_step, all_scoping
        )
        vm_a2 = CombinationEngine.compute_von_mises(sx, sy, sz, sxy, syz, sxz)
        print_stress_statistics("Von Mises", vm_a2, node_ids)
        
        assert len(vm_a1) > 0
        assert len(vm_a2) > 0
    
    def test_combination_vs_manual_calculation(self, readers):
        """Verify combination engine produces correct results by manual calculation."""
        reader1, reader2 = readers
        
        # Use step 2 from A1 and step 4 from A2
        a1_steps = reader1.get_load_step_ids()
        a2_steps = reader2.get_load_step_ids()
        a1_step = 2 if 2 in a1_steps else a1_steps[-1]
        a2_step = 4 if 4 in a2_steps else a2_steps[-1]
        
        print("\n" + "="*80)
        print("VERIFICATION: Combination Engine vs Manual Calculation")
        print("="*80)
        
        # Use a subset of nodes for faster testing
        all_scoping = reader1.get_all_nodes_scoping()
        
        # Read individual stresses
        _, sx1, sy1, sz1, sxy1, syz1, sxz1 = reader1.read_stress_tensor_for_loadstep(a1_step, all_scoping)
        _, sx2, sy2, sz2, sxy2, syz2, sxz2 = reader2.read_stress_tensor_for_loadstep(a2_step, all_scoping)
        
        # Manual combination (simple addition with coefficients 1.0 and 1.0)
        sx_manual = sx1 + sx2
        sy_manual = sy1 + sy2
        sz_manual = sz1 + sz2
        sxy_manual = sxy1 + sxy2
        syz_manual = syz1 + syz2
        sxz_manual = sxz1 + sxz2
        
        # Compute von Mises manually
        vm_manual = CombinationEngine.compute_von_mises(
            sx_manual, sy_manual, sz_manual, sxy_manual, syz_manual, sxz_manual
        )
        
        # Use combination engine
        combo_table = create_simple_combination_table(a1_step, a2_step)
        engine = CombinationEngine(reader1, reader2, all_scoping, combo_table)
        engine.preload_stress_data()
        
        # Get combined stress from engine
        sx_eng, sy_eng, sz_eng, sxy_eng, syz_eng, sxz_eng = engine.compute_combination_numpy(0)
        vm_engine = CombinationEngine.compute_von_mises(
            sx_eng, sy_eng, sz_eng, sxy_eng, syz_eng, sxz_eng
        )
        
        # Compare results
        print(f"\nComparing {len(vm_manual):,} node results...")
        
        max_diff_sx = np.max(np.abs(sx_manual - sx_eng))
        max_diff_vm = np.max(np.abs(vm_manual - vm_engine))
        
        print(f"  Max difference in SX: {max_diff_sx:.10e}")
        print(f"  Max difference in Von Mises: {max_diff_vm:.10e}")
        
        # Should be very close (numerical precision)
        np.testing.assert_allclose(sx_manual, sx_eng, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(vm_manual, vm_engine, rtol=1e-10, atol=1e-10)
        
        print("  VERIFICATION PASSED: Engine results match manual calculation")


def run_tests_standalone():
    """Run tests as standalone script for easier debugging."""
    print("\n" + "#"*80)
    print("# REAL RST COMBINATION TESTS")
    print("# Using: file_analysis1.rst (step 2) + file_analysis2.rst (step 4)")
    print("#"*80)
    
    if not DPF_AVAILABLE:
        print("\nERROR: DPF is not available. Install with: pip install ansys-dpf-core")
        return
    
    if not os.path.exists(RST_FILE_1):
        print(f"\nERROR: RST file not found: {RST_FILE_1}")
        return
    if not os.path.exists(RST_FILE_2):
        print(f"\nERROR: RST file not found: {RST_FILE_2}")
        return
    
    # Create readers
    reader1 = DPFAnalysisReader(RST_FILE_1)
    reader2 = DPFAnalysisReader(RST_FILE_2)
    readers = (reader1, reader2)
    
    # Create test instance
    test = TestRealRSTCombination()
    
    # Run tests
    print("\n>>> Running: test_list_rst_metadata")
    test.test_list_rst_metadata(readers)
    
    print("\n>>> Running: test_individual_analysis_stresses")
    test.test_individual_analysis_stresses(readers)
    
    print("\n>>> Running: test_combination_with_all_nodes")
    test.test_combination_with_all_nodes(readers)
    
    print("\n>>> Running: test_combination_with_named_selections")
    test.test_combination_with_named_selections(readers)
    
    print("\n>>> Running: test_combination_vs_manual_calculation")
    test.test_combination_vs_manual_calculation(readers)
    
    print("\n" + "#"*80)
    print("# ALL TESTS COMPLETED SUCCESSFULLY")
    print("#"*80)


if __name__ == "__main__":
    run_tests_standalone()
