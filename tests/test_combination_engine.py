"""
Tests for the Combination Engine.

Tests stress combination computation, von Mises calculation, principal stresses,
and envelope computation. Uses mocked DPF readers for unit testing.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.data_models import CombinationTableData, CombinationResult
from solver.stress_engine import StressCombinationEngine


class TestVonMisesComputation:
    """Tests for von Mises stress computation."""
    
    def test_von_mises_uniaxial_tension(self):
        """Test von Mises for uniaxial tension (σx only)."""
        sx = np.array([100.0])
        sy = np.array([0.0])
        sz = np.array([0.0])
        sxy = np.array([0.0])
        syz = np.array([0.0])
        sxz = np.array([0.0])
        
        vm = StressCombinationEngine.compute_von_mises(sx, sy, sz, sxy, syz, sxz)
        
        # For uniaxial: σvm = σx
        np.testing.assert_almost_equal(vm[0], 100.0)
    
    def test_von_mises_biaxial_equal(self):
        """Test von Mises for equal biaxial tension (σx = σy)."""
        sx = np.array([100.0])
        sy = np.array([100.0])
        sz = np.array([0.0])
        sxy = np.array([0.0])
        syz = np.array([0.0])
        sxz = np.array([0.0])
        
        vm = StressCombinationEngine.compute_von_mises(sx, sy, sz, sxy, syz, sxz)
        
        # For σx = σy: σvm = σx
        np.testing.assert_almost_equal(vm[0], 100.0)
    
    def test_von_mises_pure_shear(self):
        """Test von Mises for pure shear."""
        sx = np.array([0.0])
        sy = np.array([0.0])
        sz = np.array([0.0])
        sxy = np.array([100.0])
        syz = np.array([0.0])
        sxz = np.array([0.0])
        
        vm = StressCombinationEngine.compute_von_mises(sx, sy, sz, sxy, syz, sxz)
        
        # For pure shear: σvm = √3 * τxy
        expected = np.sqrt(3) * 100.0
        np.testing.assert_almost_equal(vm[0], expected)
    
    def test_von_mises_hydrostatic(self):
        """Test von Mises for hydrostatic stress (should be zero)."""
        p = 100.0
        sx = np.array([p])
        sy = np.array([p])
        sz = np.array([p])
        sxy = np.array([0.0])
        syz = np.array([0.0])
        sxz = np.array([0.0])
        
        vm = StressCombinationEngine.compute_von_mises(sx, sy, sz, sxy, syz, sxz)
        
        # Hydrostatic stress has σvm = 0
        np.testing.assert_almost_equal(vm[0], 0.0)
    
    def test_von_mises_general(self):
        """Test von Mises with general stress state."""
        sx = np.array([100.0])
        sy = np.array([50.0])
        sz = np.array([25.0])
        sxy = np.array([30.0])
        syz = np.array([20.0])
        sxz = np.array([10.0])
        
        vm = StressCombinationEngine.compute_von_mises(sx, sy, sz, sxy, syz, sxz)
        
        # Manual calculation
        expected = np.sqrt(0.5 * (
            (100-50)**2 + (50-25)**2 + (25-100)**2 + 
            6 * (30**2 + 20**2 + 10**2)
        ))
        np.testing.assert_almost_equal(vm[0], expected)
    
    def test_von_mises_vectorized(self):
        """Test von Mises with multiple nodes."""
        n_nodes = 100
        sx = np.random.rand(n_nodes) * 100
        sy = np.random.rand(n_nodes) * 100
        sz = np.random.rand(n_nodes) * 100
        sxy = np.random.rand(n_nodes) * 50
        syz = np.random.rand(n_nodes) * 50
        sxz = np.random.rand(n_nodes) * 50
        
        vm = StressCombinationEngine.compute_von_mises(sx, sy, sz, sxy, syz, sxz)
        
        assert len(vm) == n_nodes
        assert np.all(vm >= 0)  # Von Mises should always be non-negative


class TestPrincipalStressComputation:
    """Tests for principal stress computation."""
    
    def test_principal_uniaxial(self):
        """Test principal stresses for uniaxial case."""
        sx = np.array([100.0])
        sy = np.array([0.0])
        sz = np.array([0.0])
        sxy = np.array([0.0])
        syz = np.array([0.0])
        sxz = np.array([0.0])
        
        s1, s2, s3 = StressCombinationEngine.compute_principal_stresses_numpy(
            sx, sy, sz, sxy, syz, sxz
        )
        
        np.testing.assert_almost_equal(s1[0], 100.0)
        np.testing.assert_almost_equal(s2[0], 0.0)
        np.testing.assert_almost_equal(s3[0], 0.0)
    
    def test_principal_ordering(self):
        """Test that principal stresses are ordered S1 >= S2 >= S3."""
        # Random stress states
        np.random.seed(42)
        for _ in range(10):
            sx = np.random.rand(1) * 200 - 100
            sy = np.random.rand(1) * 200 - 100
            sz = np.random.rand(1) * 200 - 100
            sxy = np.random.rand(1) * 100 - 50
            syz = np.random.rand(1) * 100 - 50
            sxz = np.random.rand(1) * 100 - 50
            
            s1, s2, s3 = StressCombinationEngine.compute_principal_stresses_numpy(
                sx, sy, sz, sxy, syz, sxz
            )
            
            assert s1[0] >= s2[0] - 1e-10
            assert s2[0] >= s3[0] - 1e-10
    
    def test_principal_pure_shear(self):
        """Test principal stresses for pure shear state."""
        tau = 100.0
        sx = np.array([0.0])
        sy = np.array([0.0])
        sz = np.array([0.0])
        sxy = np.array([tau])
        syz = np.array([0.0])
        sxz = np.array([0.0])
        
        s1, s2, s3 = StressCombinationEngine.compute_principal_stresses_numpy(
            sx, sy, sz, sxy, syz, sxz
        )
        
        # For pure shear: S1 = τ, S2 = 0, S3 = -τ
        np.testing.assert_almost_equal(s1[0], tau)
        np.testing.assert_almost_equal(s2[0], 0.0)
        np.testing.assert_almost_equal(s3[0], -tau)
    
    def test_principal_hydrostatic(self):
        """Test principal stresses for hydrostatic state."""
        p = 100.0
        sx = np.array([p])
        sy = np.array([p])
        sz = np.array([p])
        sxy = np.array([0.0])
        syz = np.array([0.0])
        sxz = np.array([0.0])
        
        s1, s2, s3 = StressCombinationEngine.compute_principal_stresses_numpy(
            sx, sy, sz, sxy, syz, sxz
        )
        
        # All principal stresses should be equal to p
        np.testing.assert_almost_equal(s1[0], p)
        np.testing.assert_almost_equal(s2[0], p)
        np.testing.assert_almost_equal(s3[0], p)


class TestEnvelopeComputation:
    """Tests for envelope (max/min over combinations) computation."""
    
    def create_mock_engine(self, num_nodes=5, num_combos=3):
        """Create a mock StressCombinationEngine for envelope testing."""
        engine = Mock(spec=StressCombinationEngine)
        engine._node_ids = np.arange(num_nodes)
        engine._node_coords = np.random.rand(num_nodes, 3)
        engine.node_ids = engine._node_ids
        engine.node_coords = engine._node_coords
        engine.num_nodes = num_nodes
        return engine
    
    def test_compute_envelope_max(self):
        """Test computing max envelope."""
        # Test data: 3 combinations, 5 nodes
        all_combo_results = np.array([
            [10, 20, 30, 40, 50],  # Combo 0
            [15, 25, 25, 35, 45],  # Combo 1
            [12, 22, 35, 38, 55],  # Combo 2
        ])
        
        max_values, combo_indices = StressCombinationEngine.compute_envelope(
            None, all_combo_results, "max"
        )
        
        # Expected max at each node
        expected_max = np.array([15, 25, 35, 40, 55])
        expected_combo = np.array([1, 1, 2, 0, 2])
        
        np.testing.assert_array_equal(max_values, expected_max)
        np.testing.assert_array_equal(combo_indices, expected_combo)
    
    def test_compute_envelope_min(self):
        """Test computing min envelope."""
        all_combo_results = np.array([
            [10, 20, 30, 40, 50],  # Combo 0
            [15, 25, 25, 35, 45],  # Combo 1
            [12, 22, 35, 38, 55],  # Combo 2
        ])
        
        min_values, combo_indices = StressCombinationEngine.compute_envelope(
            None, all_combo_results, "min"
        )
        
        expected_min = np.array([10, 20, 25, 35, 45])
        expected_combo = np.array([0, 0, 1, 1, 1])
        
        np.testing.assert_array_equal(min_values, expected_min)
        np.testing.assert_array_equal(combo_indices, expected_combo)
    
    def test_compute_envelope_invalid_type(self):
        """Test that invalid envelope type raises error."""
        all_combo_results = np.array([[1, 2, 3]])
        
        with pytest.raises(ValueError, match="Unknown envelope type"):
            StressCombinationEngine.compute_envelope(None, all_combo_results, "average")


class TestCombinationComputation:
    """Tests for linear combination computation."""
    
    def test_combination_formula_simple(self):
        """Test basic linear combination: σ = α₁×σA1₁ + α₂×σA1₂."""
        # Create mock with known stress values
        # We'll test the numpy-based combination
        
        # Stress cache: (analysis_idx, step_id) -> (node_ids, sx, sy, sz, sxy, syz, sxz)
        # Analysis 1, Step 1: all components = 100
        # Analysis 1, Step 2: all components = 200
        node_ids = np.array([1, 2, 3])
        
        stress_cache = {
            (1, 1): (node_ids, 
                     np.array([100, 100, 100]),   # sx
                     np.array([100, 100, 100]),   # sy
                     np.array([100, 100, 100]),   # sz
                     np.array([50, 50, 50]),      # sxy
                     np.array([50, 50, 50]),      # syz
                     np.array([50, 50, 50])),     # sxz
            (1, 2): (node_ids,
                     np.array([200, 200, 200]),
                     np.array([200, 200, 200]),
                     np.array([200, 200, 200]),
                     np.array([100, 100, 100]),
                     np.array([100, 100, 100]),
                     np.array([100, 100, 100])),
        }
        
        # Create table with α₁=0.5, α₂=0.5
        table = CombinationTableData(
            combination_names=["Average"],
            combination_types=["Linear"],
            analysis1_coeffs=np.array([[0.5, 0.5]]),
            analysis2_coeffs=np.zeros((1, 0)),
            analysis1_step_ids=[1, 2],
            analysis2_step_ids=[],
        )
        
        # Manual calculation: 0.5*100 + 0.5*200 = 150
        expected_sx = np.array([150, 150, 150])
        
        # Create mock engine
        engine = Mock()
        engine._stress_cache = stress_cache
        engine.table = table
        engine._node_ids = node_ids
        engine.num_nodes = 3
        
        # Use actual method
        engine.compute_combination_numpy = StressCombinationEngine.compute_combination_numpy.__get__(
            engine, StressCombinationEngine
        )
        
        sx, sy, sz, sxy, syz, sxz = engine.compute_combination_numpy(0)
        
        np.testing.assert_array_almost_equal(sx, expected_sx)
        np.testing.assert_array_almost_equal(sxy, np.array([75, 75, 75]))  # 0.5*50 + 0.5*100
    
    def test_combination_with_zero_coefficients(self):
        """Test that zero coefficients are skipped efficiently."""
        node_ids = np.array([1, 2])
        
        stress_cache = {
            (1, 1): (node_ids,
                     np.array([100, 100]),
                     np.array([100, 100]),
                     np.array([100, 100]),
                     np.array([50, 50]),
                     np.array([50, 50]),
                     np.array([50, 50])),
            (1, 2): (node_ids,
                     np.array([999, 999]),  # Should be ignored (coeff = 0)
                     np.array([999, 999]),
                     np.array([999, 999]),
                     np.array([999, 999]),
                     np.array([999, 999]),
                     np.array([999, 999])),
        }
        
        table = CombinationTableData(
            combination_names=["OnlyFirst"],
            combination_types=["Linear"],
            analysis1_coeffs=np.array([[1.0, 0.0]]),  # Second step has zero coeff
            analysis2_coeffs=np.zeros((1, 0)),
            analysis1_step_ids=[1, 2],
            analysis2_step_ids=[],
        )
        
        engine = Mock()
        engine._stress_cache = stress_cache
        engine.table = table
        engine._node_ids = node_ids
        engine.num_nodes = 2
        
        engine.compute_combination_numpy = StressCombinationEngine.compute_combination_numpy.__get__(
            engine, StressCombinationEngine
        )
        
        sx, sy, sz, sxy, syz, sxz = engine.compute_combination_numpy(0)
        
        # Should only include step 1 (coeff=1.0), step 2 (coeff=0.0) ignored
        np.testing.assert_array_almost_equal(sx, np.array([100, 100]))


class TestCombinationEngineHelpers:
    """Tests for helper methods."""
    
    def test_get_combination_names(self):
        """Test getting combination names."""
        table = CombinationTableData(
            combination_names=["Combo A", "Combo B", "Combo C"],
            combination_types=["Linear", "Linear", "Linear"],
            analysis1_coeffs=np.zeros((3, 1)),
            analysis2_coeffs=np.zeros((3, 1)),
            analysis1_step_ids=[1],
            analysis2_step_ids=[1],
        )
        
        engine = Mock()
        engine.table = table
        engine.get_combination_names = StressCombinationEngine.get_combination_names.__get__(
            engine, StressCombinationEngine
        )
        
        names = engine.get_combination_names()
        
        assert names == ["Combo A", "Combo B", "Combo C"]


class TestCombinationResultIntegration:
    """Integration tests for full analysis workflow."""
    
    def test_combination_result_structure(self):
        """Test that CombinationResult is correctly populated."""
        # Create a result manually to verify structure
        n_nodes = 100
        n_combos = 5
        
        all_results = np.random.rand(n_combos, n_nodes) * 500
        max_values = np.max(all_results, axis=0)
        min_values = np.min(all_results, axis=0)
        combo_of_max = np.argmax(all_results, axis=0)
        combo_of_min = np.argmin(all_results, axis=0)
        
        result = CombinationResult(
            node_ids=np.arange(n_nodes),
            node_coords=np.random.rand(n_nodes, 3),
            max_over_combo=max_values,
            min_over_combo=min_values,
            combo_of_max=combo_of_max,
            combo_of_min=combo_of_min,
            result_type="von_mises",
            all_combo_results=all_results,
        )
        
        assert result.num_nodes == n_nodes
        assert result.num_combinations == n_combos
        assert len(result.max_over_combo) == n_nodes
        assert len(result.combo_of_max) == n_nodes
        
        # Verify max/min are consistent with all_results
        for i in range(n_nodes):
            assert result.max_over_combo[i] == all_results[combo_of_max[i], i]
            assert result.min_over_combo[i] == all_results[combo_of_min[i], i]


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_single_node(self):
        """Test with single node."""
        sx = np.array([100.0])
        sy = np.array([50.0])
        sz = np.array([25.0])
        sxy = np.array([0.0])
        syz = np.array([0.0])
        sxz = np.array([0.0])
        
        vm = StressCombinationEngine.compute_von_mises(sx, sy, sz, sxy, syz, sxz)
        s1, s2, s3 = StressCombinationEngine.compute_principal_stresses_numpy(
            sx, sy, sz, sxy, syz, sxz
        )
        
        assert len(vm) == 1
        assert len(s1) == 1
    
    def test_single_combination(self):
        """Test envelope with single combination."""
        all_results = np.array([[10, 20, 30, 40, 50]])  # 1 combo, 5 nodes
        
        max_values, combo_indices = StressCombinationEngine.compute_envelope(
            None, all_results, "max"
        )
        
        np.testing.assert_array_equal(max_values, [10, 20, 30, 40, 50])
        np.testing.assert_array_equal(combo_indices, [0, 0, 0, 0, 0])
    
    def test_all_zero_stress(self):
        """Test with zero stress state."""
        n = 10
        zeros = np.zeros(n)
        
        vm = StressCombinationEngine.compute_von_mises(
            zeros, zeros, zeros, zeros, zeros, zeros
        )
        
        np.testing.assert_array_equal(vm, zeros)
    
    def test_negative_stress(self):
        """Test with negative stress (compression)."""
        sx = np.array([-100.0])
        sy = np.array([0.0])
        sz = np.array([0.0])
        sxy = np.array([0.0])
        syz = np.array([0.0])
        sxz = np.array([0.0])
        
        vm = StressCombinationEngine.compute_von_mises(sx, sy, sz, sxy, syz, sxz)
        
        # Von Mises should still be positive (100)
        np.testing.assert_almost_equal(vm[0], 100.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
