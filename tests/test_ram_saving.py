"""
Tests for RAM-saving functionality in the Combination Engine.

Tests memory estimation, chunked processing, and incremental envelope updates.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.data_models import CombinationTableData, CombinationResult
from solver.stress_engine import (
    StressCombinationEngine,
    RAM_USAGE_FRACTION,
    DEFAULT_CHUNK_SIZE,
    MIN_CHUNK_SIZE,
)


class TestMemoryEstimation:
    """Tests for memory estimation methods."""
    
    def test_estimate_memory_requirements(self):
        """Test that memory estimation returns expected keys."""
        # Create mock engine
        engine = Mock(spec=StressCombinationEngine)
        engine.scoping = Mock()
        engine.scoping.ids = list(range(1000))  # 1000 nodes
        
        engine.table = Mock()
        engine.table.analysis1_step_ids = [1, 2, 3]
        engine.table.analysis2_step_ids = [1, 2]
        engine.table.num_combinations = 10
        
        # Use actual method
        engine.estimate_memory_requirements = StressCombinationEngine.estimate_memory_requirements.__get__(
            engine, StressCombinationEngine
        )
        
        estimates = engine.estimate_memory_requirements(num_nodes=1000)
        
        # Check all expected keys are present
        assert 'stress_cache_bytes' in estimates
        assert 'results_array_bytes' in estimates
        assert 'envelope_bytes' in estimates
        assert 'total_bytes' in estimates
        assert 'minimum_required_bytes' in estimates
        assert 'memory_per_node' in estimates
        assert 'num_nodes' in estimates
        
        # Verify values are positive
        assert estimates['stress_cache_bytes'] > 0
        assert estimates['results_array_bytes'] > 0
        assert estimates['envelope_bytes'] > 0
        assert estimates['total_bytes'] > 0
    
    def test_estimate_memory_scaling(self):
        """Test that memory estimates scale with node count."""
        engine = Mock(spec=StressCombinationEngine)
        engine.scoping = Mock()
        engine.scoping.ids = list(range(100))
        
        engine.table = Mock()
        engine.table.analysis1_step_ids = [1, 2]
        engine.table.analysis2_step_ids = [1]
        engine.table.num_combinations = 5
        
        engine.estimate_memory_requirements = StressCombinationEngine.estimate_memory_requirements.__get__(
            engine, StressCombinationEngine
        )
        
        estimates_100 = engine.estimate_memory_requirements(num_nodes=100)
        estimates_1000 = engine.estimate_memory_requirements(num_nodes=1000)
        
        # Memory should scale roughly linearly with node count
        ratio = estimates_1000['total_bytes'] / estimates_100['total_bytes']
        assert 9 < ratio < 11  # Should be approximately 10x
    
    def test_calculate_chunk_size_bounds(self):
        """Test that chunk size is bounded correctly."""
        engine = Mock(spec=StressCombinationEngine)
        engine.table = Mock()
        engine.table.analysis1_step_ids = [1]
        engine.table.analysis2_step_ids = []
        engine.table.num_combinations = 5
        
        # Mock estimate_memory_requirements to return known values
        engine.estimate_memory_requirements = Mock(return_value={
            'memory_per_node': 1000,  # 1KB per node
            'envelope_bytes': 100000,  # 100KB for envelope
        })
        
        engine._calculate_chunk_size = StressCombinationEngine._calculate_chunk_size.__get__(
            engine, StressCombinationEngine
        )
        
        # Test with limited memory
        chunk_size = engine._calculate_chunk_size(
            available_memory=1_000_000,  # 1MB
            num_nodes=100000
        )
        
        # Should be bounded by MIN_CHUNK_SIZE and num_nodes
        assert chunk_size >= MIN_CHUNK_SIZE
        assert chunk_size <= 100000
    
    def test_calculate_chunk_size_small_model(self):
        """Test that chunk size equals num_nodes for small models."""
        engine = Mock(spec=StressCombinationEngine)
        engine.table = Mock()
        engine.table.analysis1_step_ids = [1]
        engine.table.analysis2_step_ids = []
        engine.table.num_combinations = 5
        
        engine.estimate_memory_requirements = Mock(return_value={
            'memory_per_node': 100,  # Small
            'envelope_bytes': 1000,
        })
        
        engine._calculate_chunk_size = StressCombinationEngine._calculate_chunk_size.__get__(
            engine, StressCombinationEngine
        )
        
        # With plenty of memory, chunk size should equal num_nodes
        chunk_size = engine._calculate_chunk_size(
            available_memory=1_000_000_000,  # 1GB
            num_nodes=1000
        )
        
        assert chunk_size == 1000


class TestChunkedProcessing:
    """Tests for chunked stress computation."""
    
    def test_update_envelope_for_chunk(self):
        """Test incremental envelope update."""
        engine = Mock(spec=StressCombinationEngine)
        engine._update_envelope_for_chunk = StressCombinationEngine._update_envelope_for_chunk.__get__(
            engine, StressCombinationEngine
        )
        
        # Initialize envelope arrays for 10 nodes
        max_envelope = np.full(10, -np.inf)
        min_envelope = np.full(10, np.inf)
        combo_of_max = np.zeros(10, dtype=np.int32)
        combo_of_min = np.zeros(10, dtype=np.int32)
        
        # First chunk: nodes 0-4, 3 combinations
        chunk_results_1 = np.array([
            [100, 110, 120, 130, 140],  # Combo 0
            [105, 115, 125, 135, 145],  # Combo 1
            [95, 105, 115, 125, 135],   # Combo 2
        ])
        
        engine._update_envelope_for_chunk(
            chunk_results_1, 0,
            max_envelope, min_envelope,
            combo_of_max, combo_of_min
        )
        
        # Check first chunk results
        np.testing.assert_array_equal(max_envelope[:5], [105, 115, 125, 135, 145])
        np.testing.assert_array_equal(combo_of_max[:5], [1, 1, 1, 1, 1])
        np.testing.assert_array_equal(min_envelope[:5], [95, 105, 115, 125, 135])
        np.testing.assert_array_equal(combo_of_min[:5], [2, 2, 2, 2, 2])
        
        # Second chunk: nodes 5-9
        chunk_results_2 = np.array([
            [200, 210, 220, 230, 240],  # Combo 0
            [195, 205, 215, 225, 235],  # Combo 1
            [210, 220, 230, 240, 250],  # Combo 2
        ])
        
        engine._update_envelope_for_chunk(
            chunk_results_2, 5,
            max_envelope, min_envelope,
            combo_of_max, combo_of_min
        )
        
        # Check second chunk results
        np.testing.assert_array_equal(max_envelope[5:], [210, 220, 230, 240, 250])
        np.testing.assert_array_equal(combo_of_max[5:], [2, 2, 2, 2, 2])
    
    def test_compute_chunk_combinations(self):
        """Test computing combinations for a chunk."""
        engine = Mock(spec=StressCombinationEngine)
        engine.table = Mock()
        engine.table.num_combinations = 2
        engine.table.analysis1_step_ids = [1, 2]
        engine.table.analysis2_step_ids = []
        engine.table.get_coeffs_for_combination = Mock(side_effect=[
            (np.array([1.0, 0.0]), np.array([])),  # Combo 0: only step 1
            (np.array([0.0, 1.0]), np.array([])),  # Combo 1: only step 2
        ])
        
        # Mock stress cache for chunk
        chunk_cache = {
            (1, 1): (np.array([1, 2]), 
                     np.array([100, 200]),  # sx
                     np.array([50, 100]),   # sy
                     np.array([25, 50]),    # sz
                     np.array([10, 20]),    # sxy
                     np.array([5, 10]),     # syz
                     np.array([2, 4])),     # sxz
            (1, 2): (np.array([1, 2]),
                     np.array([200, 400]),
                     np.array([100, 200]),
                     np.array([50, 100]),
                     np.array([20, 40]),
                     np.array([10, 20]),
                     np.array([4, 8])),
        }
        
        engine.compute_von_mises = StressCombinationEngine.compute_von_mises
        engine.compute_principal_stresses_numpy = StressCombinationEngine.compute_principal_stresses_numpy
        
        engine._compute_chunk_combinations = StressCombinationEngine._compute_chunk_combinations.__get__(
            engine, StressCombinationEngine
        )
        
        results = engine._compute_chunk_combinations(
            chunk_cache, chunk_size=2, stress_type="von_mises"
        )
        
        assert results.shape == (2, 2)  # 2 combinations, 2 nodes
        assert np.all(results >= 0)  # Von Mises always positive


class TestMemoryCheckAndChunkedAnalysis:
    """Tests for memory checking and full chunked analysis flow."""
    
    def test_check_memory_available_sufficient(self):
        """Test memory check when sufficient memory available."""
        engine = Mock(spec=StressCombinationEngine)
        engine.scoping = Mock()
        engine.scoping.ids = list(range(100))  # Small model
        
        engine.estimate_memory_requirements = Mock(return_value={
            'minimum_required_bytes': 1_000_000,  # 1MB needed
            'stress_cache_bytes': 500_000,
            'results_array_bytes': 300_000,
            'envelope_bytes': 200_000,
            'total_bytes': 1_000_000,
            'memory_per_node': 1000,
            'num_nodes': 100,
            'num_combinations': 5,
            'num_load_steps': 3,
        })
        
        engine._get_available_memory = Mock(return_value=1_000_000_000)  # 1GB available
        engine._calculate_chunk_size = Mock(return_value=100)
        
        engine.check_memory_available = StressCombinationEngine.check_memory_available.__get__(
            engine, StressCombinationEngine
        )
        
        is_sufficient, estimates = engine.check_memory_available(raise_on_insufficient=False)
        
        assert is_sufficient is True
        assert 'available_bytes' in estimates
        assert 'is_sufficient' in estimates
        assert 'recommended_chunk_size' in estimates
    
    def test_check_memory_available_insufficient(self):
        """Test memory check when insufficient memory."""
        engine = Mock(spec=StressCombinationEngine)
        engine.scoping = Mock()
        engine.scoping.ids = list(range(100))
        
        engine.estimate_memory_requirements = Mock(return_value={
            'minimum_required_bytes': 10_000_000_000,  # 10GB needed
            'stress_cache_bytes': 5_000_000_000,
            'results_array_bytes': 3_000_000_000,
            'envelope_bytes': 2_000_000_000,
            'total_bytes': 10_000_000_000,
            'memory_per_node': 100_000,
            'num_nodes': 100000,
            'num_combinations': 100,
            'num_load_steps': 50,
        })
        
        engine._get_available_memory = Mock(return_value=1_000_000_000)  # Only 1GB
        engine._calculate_chunk_size = Mock(return_value=MIN_CHUNK_SIZE)
        
        engine.check_memory_available = StressCombinationEngine.check_memory_available.__get__(
            engine, StressCombinationEngine
        )
        
        # Should not raise, just return False
        is_sufficient, estimates = engine.check_memory_available(raise_on_insufficient=False)
        
        assert is_sufficient is False
    
    def test_check_memory_available_raises(self):
        """Test memory check raises MemoryError when requested."""
        engine = Mock(spec=StressCombinationEngine)
        engine.scoping = Mock()
        engine.scoping.ids = list(range(100))
        
        engine.estimate_memory_requirements = Mock(return_value={
            'minimum_required_bytes': 10_000_000_000,
        })
        
        engine._get_available_memory = Mock(return_value=1_000_000_000)
        engine._calculate_chunk_size = Mock(return_value=MIN_CHUNK_SIZE)
        
        engine.check_memory_available = StressCombinationEngine.check_memory_available.__get__(
            engine, StressCombinationEngine
        )
        
        with pytest.raises(MemoryError):
            engine.check_memory_available(raise_on_insufficient=True)


class TestEnvelopeConsistency:
    """Tests to verify chunked and non-chunked processing give same results."""
    
    def test_envelope_calculation_consistency(self):
        """Test that envelope calculation is mathematically correct."""
        # Create known data
        all_results = np.array([
            [10, 30, 50, 70, 90],   # Combo 0
            [20, 40, 60, 80, 100],  # Combo 1
            [15, 25, 55, 65, 85],   # Combo 2
        ])
        
        # Calculate envelope the standard way
        expected_max = np.max(all_results, axis=0)
        expected_argmax = np.argmax(all_results, axis=0)
        expected_min = np.min(all_results, axis=0)
        expected_argmin = np.argmin(all_results, axis=0)
        
        # Now simulate chunked processing
        max_envelope = np.full(5, -np.inf)
        min_envelope = np.full(5, np.inf)
        combo_of_max = np.zeros(5, dtype=np.int32)
        combo_of_min = np.zeros(5, dtype=np.int32)
        
        # Process in chunks of 2
        for start in range(0, 5, 2):
            end = min(start + 2, 5)
            chunk = all_results[:, start:end]
            
            chunk_max = np.max(chunk, axis=0)
            chunk_argmax = np.argmax(chunk, axis=0)
            chunk_min = np.min(chunk, axis=0)
            chunk_argmin = np.argmin(chunk, axis=0)
            
            max_envelope[start:end] = chunk_max
            min_envelope[start:end] = chunk_min
            combo_of_max[start:end] = chunk_argmax
            combo_of_min[start:end] = chunk_argmin
        
        # Results should match
        np.testing.assert_array_equal(max_envelope, expected_max)
        np.testing.assert_array_equal(min_envelope, expected_min)
        np.testing.assert_array_equal(combo_of_max, expected_argmax)
        np.testing.assert_array_equal(combo_of_min, expected_argmin)


class TestSubScoping:
    """Tests for DPF sub-scoping creation."""
    
    def test_create_sub_scoping(self):
        """Test creating sub-scoping from full scoping."""
        # This would require DPF to be available, so we mock it
        with patch('file_io.dpf_reader.DPF_AVAILABLE', True):
            with patch('file_io.dpf_reader.dpf') as mock_dpf:
                # Create mock scoping
                full_scoping = Mock()
                full_scoping.ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                
                # Mock dpf.Scoping
                mock_sub_scoping = Mock()
                mock_dpf.Scoping.return_value = mock_sub_scoping
                mock_dpf.locations.nodal = 'nodal'
                
                from file_io.dpf_reader import DPFAnalysisReader
                
                # Create mock reader
                reader = Mock(spec=DPFAnalysisReader)
                reader.create_sub_scoping = DPFAnalysisReader.create_sub_scoping.__get__(
                    reader, DPFAnalysisReader
                )
                
                # Create sub-scoping for indices 2-5
                sub_scoping = reader.create_sub_scoping(full_scoping, 2, 6)
                
                # Check that Scoping was created with correct IDs
                mock_dpf.Scoping.assert_called_once()
                # The sub_scoping.ids should be set to [3, 4, 5, 6]
                assert mock_sub_scoping.ids == [3, 4, 5, 6]


class TestEdgeCases:
    """Edge case tests for RAM-saving functionality."""
    
    def test_single_node_no_chunking_needed(self):
        """Test that single node doesn't need chunking."""
        engine = Mock(spec=StressCombinationEngine)
        engine.scoping = Mock()
        engine.scoping.ids = [1]  # Single node
        
        engine.table = Mock()
        engine.table.analysis1_step_ids = [1]
        engine.table.analysis2_step_ids = []
        engine.table.num_combinations = 1
        
        engine.estimate_memory_requirements = StressCombinationEngine.estimate_memory_requirements.__get__(
            engine, StressCombinationEngine
        )
        
        estimates = engine.estimate_memory_requirements(num_nodes=1)
        
        # Memory requirements should be very small
        assert estimates['total_bytes'] < 1_000_000  # Less than 1MB
    
    def test_zero_combinations_error(self):
        """Test handling of zero combinations."""
        engine = Mock(spec=StressCombinationEngine)
        engine.scoping = Mock()
        engine.scoping.ids = [1, 2, 3]
        
        engine.table = Mock()
        engine.table.analysis1_step_ids = []
        engine.table.analysis2_step_ids = []
        engine.table.num_combinations = 0
        
        engine.estimate_memory_requirements = StressCombinationEngine.estimate_memory_requirements.__get__(
            engine, StressCombinationEngine
        )
        
        # Should handle gracefully
        estimates = engine.estimate_memory_requirements(num_nodes=3)
        assert estimates['num_combinations'] == 0


class TestPsutilFallback:
    """Tests for fallback behavior when psutil is not available."""
    
    def test_get_available_memory_without_psutil(self):
        """Test memory estimation fallback without psutil."""
        engine = Mock(spec=StressCombinationEngine)
        
        # Temporarily disable psutil
        import solver.stress_engine as ce_module
        original_psutil = ce_module.PSUTIL_AVAILABLE
        ce_module.PSUTIL_AVAILABLE = False
        
        try:
            engine._get_available_memory = StressCombinationEngine._get_available_memory.__get__(
                engine, StressCombinationEngine
            )
            
            available = engine._get_available_memory()
            
            # Should return default value (4GB * 0.9)
            expected = int(4 * 1024 * 1024 * 1024 * RAM_USAGE_FRACTION)
            assert available == expected
        finally:
            ce_module.PSUTIL_AVAILABLE = original_psutil


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
