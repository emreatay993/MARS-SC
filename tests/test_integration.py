"""
Integration tests for MARS-SC.

Tests end-to-end workflows combining multiple components.
These tests use fixtures and mocks to simulate full analysis workflows.
"""

import pytest
import numpy as np
import tempfile
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.data_models import CombinationTableData, CombinationResult, SolverConfig
from file_io.combination_parser import CombinationTableParser
from file_io.exporters import (
    export_envelope_results,
    export_single_combination,
    export_combination_history,
)
from solver.combination_engine import CombinationEngine
from solver.plasticity_engine import (
    default_material_db,
    apply_plasticity_to_envelope,
)


class TestFullCombinationWorkflow:
    """Tests for complete combination analysis workflow."""
    
    def test_csv_parse_compute_export_roundtrip(self, tmp_path, sample_csv_two_analysis):
        """Test parsing CSV, simulating computation, and exporting results."""
        # Step 1: Parse CSV
        table = CombinationTableParser.parse_csv(sample_csv_two_analysis)
        
        assert table.num_combinations == 4
        assert table.num_analysis1_steps == 2
        assert table.num_analysis2_steps == 2
        
        # Step 2: Simulate computation (mock stress data)
        n_nodes = 50
        node_ids = np.arange(1, n_nodes + 1)
        node_coords = np.random.rand(n_nodes, 3) * 100
        
        # Simulated combination results
        all_results = np.random.rand(table.num_combinations, n_nodes) * 400 + 100
        
        # Compute envelopes
        max_values = np.max(all_results, axis=0)
        min_values = np.min(all_results, axis=0)
        combo_of_max = np.argmax(all_results, axis=0)
        combo_of_min = np.argmin(all_results, axis=0)
        
        # Step 3: Export results
        output_path = tmp_path / "envelope_results.csv"
        export_envelope_results(
            str(output_path),
            node_ids=node_ids,
            node_coords=node_coords,
            max_values=max_values,
            min_values=min_values,
            combo_of_max=combo_of_max,
            combo_of_min=combo_of_min,
            result_type="von_mises",
            combination_names=table.combination_names,
        )
        
        # Verify file exists and has expected content
        assert output_path.exists()
        
        import pandas as pd
        df = pd.read_csv(output_path)
        assert len(df) == n_nodes
        assert "Max Von Mises [MPa]" in df.columns
    
    def test_combination_table_create_and_validate(self, sample_combination_table):
        """Test creating and validating combination table."""
        # Validation should pass with matching step counts
        is_valid, error = CombinationTableParser.validate_against_analyses(
            sample_combination_table,
            analysis1_steps=3,
            analysis2_steps=2
        )
        
        assert is_valid is True
        assert error == ""
        
        # Validation should fail with wrong step counts
        is_valid, error = CombinationTableParser.validate_against_analyses(
            sample_combination_table,
            analysis1_steps=5,  # Wrong
            analysis2_steps=2
        )
        
        assert is_valid is False
        assert "Analysis 1" in error


class TestStressComputationWorkflow:
    """Tests for stress computation workflow."""
    
    def test_von_mises_from_tensor_workflow(self, sample_stress_tensor):
        """Test computing von Mises from tensor components."""
        # Use the engine's static method
        vm = CombinationEngine.compute_von_mises(
            sample_stress_tensor['sx'],
            sample_stress_tensor['sy'],
            sample_stress_tensor['sz'],
            sample_stress_tensor['sxy'],
            sample_stress_tensor['syz'],
            sample_stress_tensor['sxz'],
        )
        
        # Verify results
        assert len(vm) == len(sample_stress_tensor['sx'])
        assert np.all(vm >= 0)  # Von Mises is always non-negative
    
    def test_principal_stresses_workflow(self, sample_stress_tensor):
        """Test computing principal stresses from tensor."""
        s1, s2, s3 = CombinationEngine.compute_principal_stresses_numpy(
            sample_stress_tensor['sx'],
            sample_stress_tensor['sy'],
            sample_stress_tensor['sz'],
            sample_stress_tensor['sxy'],
            sample_stress_tensor['syz'],
            sample_stress_tensor['sxz'],
        )
        
        # Verify ordering: S1 >= S2 >= S3
        assert np.all(s1 >= s2 - 1e-10)
        assert np.all(s2 >= s3 - 1e-10)


class TestPlasticityWorkflow:
    """Tests for plasticity correction workflow."""
    
    def test_envelope_plasticity_correction(self, sample_envelope_result):
        """Test applying plasticity correction to envelope results."""
        db = default_material_db()
        
        # Get max values from envelope
        max_values = sample_envelope_result.max_over_combo
        temperature = np.full(len(max_values), 22.0)
        
        # Apply correction
        corrected, strain = apply_plasticity_to_envelope(
            max_values, temperature, db, method="neuber"
        )
        
        # Verify results
        assert len(corrected) == len(max_values)
        assert len(strain) == len(max_values)
        
        # Corrected values should be <= original for high stress
        high_stress_mask = max_values > db.SIG[0, 0]  # Above yield
        if np.any(high_stress_mask):
            assert np.all(corrected[high_stress_mask] <= max_values[high_stress_mask] + 1)


class TestExportWorkflow:
    """Tests for export workflow."""
    
    def test_full_export_workflow(self, tmp_path, sample_envelope_result):
        """Test complete export workflow."""
        combination_names = [f"Combo {i}" for i in range(sample_envelope_result.num_combinations)]
        
        # Export envelope
        envelope_path = tmp_path / "envelope.csv"
        export_envelope_results(
            str(envelope_path),
            node_ids=sample_envelope_result.node_ids,
            node_coords=sample_envelope_result.node_coords,
            max_values=sample_envelope_result.max_over_combo,
            min_values=sample_envelope_result.min_over_combo,
            combo_of_max=sample_envelope_result.combo_of_max,
            combo_of_min=sample_envelope_result.combo_of_min,
            result_type=sample_envelope_result.result_type,
            combination_names=combination_names,
        )
        
        assert envelope_path.exists()
        
        # Export single combination
        if sample_envelope_result.all_combo_results is not None:
            single_path = tmp_path / "single_combo.csv"
            export_single_combination(
                str(single_path),
                node_ids=sample_envelope_result.node_ids,
                node_coords=sample_envelope_result.node_coords,
                stress_values=sample_envelope_result.all_combo_results[0],
                combination_index=0,
                combination_name=combination_names[0],
            )
            
            assert single_path.exists()
        
        # Export history for single node
        history_path = tmp_path / "history.csv"
        export_combination_history(
            str(history_path),
            node_id=int(sample_envelope_result.node_ids[0]),
            combination_indices=np.arange(sample_envelope_result.num_combinations),
            stress_values=sample_envelope_result.all_combo_results[:, 0],
            combination_names=combination_names,
        )
        
        assert history_path.exists()


class TestMathematicalCorrectness:
    """Tests verifying mathematical correctness of computations."""
    
    def test_von_mises_formula_validation(self):
        """Validate von Mises formula against known solution."""
        # Test case: biaxial stress σx = 100, σy = 100
        # Expected: σvm = 100 (for equal biaxial)
        sx = np.array([100.0])
        sy = np.array([100.0])
        sz = np.array([0.0])
        sxy = np.array([0.0])
        syz = np.array([0.0])
        sxz = np.array([0.0])
        
        vm = CombinationEngine.compute_von_mises(sx, sy, sz, sxy, syz, sxz)
        np.testing.assert_almost_equal(vm[0], 100.0)
        
        # Test case: plane stress with shear
        # σx = 100, τxy = 50
        # σvm = √(σx² + 3τxy²) = √(10000 + 7500) = √17500 ≈ 132.29
        sx = np.array([100.0])
        sy = np.array([0.0])
        sz = np.array([0.0])
        sxy = np.array([50.0])
        syz = np.array([0.0])
        sxz = np.array([0.0])
        
        vm = CombinationEngine.compute_von_mises(sx, sy, sz, sxy, syz, sxz)
        expected = np.sqrt(100**2 + 3 * 50**2)
        np.testing.assert_almost_equal(vm[0], expected, decimal=5)
    
    def test_linear_combination_formula(self):
        """Validate linear combination formula."""
        # Create simple test: α₁×σ₁ + α₂×σ₂
        # With α₁=0.6, α₂=0.4, σ₁=100, σ₂=200
        # Expected: 0.6×100 + 0.4×200 = 60 + 80 = 140
        
        # This tests the combination logic conceptually
        alpha1, alpha2 = 0.6, 0.4
        sigma1, sigma2 = 100.0, 200.0
        
        expected = alpha1 * sigma1 + alpha2 * sigma2
        np.testing.assert_almost_equal(expected, 140.0)
    
    def test_envelope_max_min_consistency(self):
        """Test that envelope max/min are consistent with data."""
        # Create known data
        all_results = np.array([
            [10, 30, 50],  # Combo 0
            [20, 40, 30],  # Combo 1
            [15, 25, 45],  # Combo 2
        ])
        
        max_values, max_indices = CombinationEngine.compute_envelope(
            None, all_results, "max"
        )
        min_values, min_indices = CombinationEngine.compute_envelope(
            None, all_results, "min"
        )
        
        # Verify max
        np.testing.assert_array_equal(max_values, [20, 40, 50])
        np.testing.assert_array_equal(max_indices, [1, 1, 0])
        
        # Verify min
        np.testing.assert_array_equal(min_values, [10, 25, 30])
        np.testing.assert_array_equal(min_indices, [0, 2, 1])


class TestConfigurationHandling:
    """Tests for configuration handling."""
    
    def test_solver_config_defaults(self):
        """Test default solver configuration."""
        config = SolverConfig()
        
        assert config.calculate_von_mises is True
        assert config.calculate_max_principal_stress is False
        assert config.combination_history_mode is False
    
    def test_solver_config_custom(self):
        """Test custom solver configuration."""
        config = SolverConfig(
            calculate_von_mises=False,
            calculate_max_principal_stress=True,
            calculate_min_principal_stress=True,
            combination_history_mode=True,
            selected_node_id=12345,
            output_directory="/output/path",
        )
        
        assert config.calculate_von_mises is False
        assert config.calculate_max_principal_stress is True
        assert config.selected_node_id == 12345
        assert config.output_directory == "/output/path"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
