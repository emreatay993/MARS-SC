"""
Tests for core data models.

Tests CombinationTableData, CombinationResult, AnalysisData, and related models.
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.data_models import (
    AnalysisData,
    CombinationTableData,
    CombinationResult,
    SolverConfig,
    PlasticityConfig,
    AnalysisResult,
    MaterialProfileData,
    TemperatureFieldData,
)


class TestAnalysisData:
    """Tests for AnalysisData dataclass."""
    
    def test_creation(self):
        """Test basic creation of AnalysisData."""
        data = AnalysisData(
            file_path="test.rst",
            num_load_steps=3,
            load_step_ids=[1, 2, 3],
            named_selections=["NS1", "NS2"]
        )
        
        assert data.file_path == "test.rst"
        assert data.num_load_steps == 3
        assert data.load_step_ids == [1, 2, 3]
        assert data.named_selections == ["NS1", "NS2"]
    
    def test_empty_named_selections(self):
        """Test AnalysisData with no named selections."""
        data = AnalysisData(
            file_path="test.rst",
            num_load_steps=1,
            load_step_ids=[1],
            named_selections=[]
        )
        
        assert data.named_selections == []


class TestCombinationTableData:
    """Tests for CombinationTableData dataclass."""
    
    def test_creation(self):
        """Test basic creation of CombinationTableData."""
        table = CombinationTableData(
            combination_names=["Combo1", "Combo2"],
            combination_types=["Linear", "Linear"],
            analysis1_coeffs=np.array([[1.0, 0.5], [0.0, 1.0]]),
            analysis2_coeffs=np.array([[0.5, 0.0], [1.0, 0.5]]),
            analysis1_step_ids=[1, 2],
            analysis2_step_ids=[1, 2],
        )
        
        assert table.num_combinations == 2
        assert table.num_analysis1_steps == 2
        assert table.num_analysis2_steps == 2
    
    def test_get_coeffs_for_combination(self):
        """Test retrieving coefficients for a specific combination."""
        table = CombinationTableData(
            combination_names=["Combo1", "Combo2", "Combo3"],
            combination_types=["Linear", "Linear", "Linear"],
            analysis1_coeffs=np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]),
            analysis2_coeffs=np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]]),
            analysis1_step_ids=[1, 2],
            analysis2_step_ids=[1, 2],
        )
        
        a1_coeffs, a2_coeffs = table.get_coeffs_for_combination(0)
        np.testing.assert_array_equal(a1_coeffs, [1.0, 0.0])
        np.testing.assert_array_equal(a2_coeffs, [0.0, 1.0])
        
        a1_coeffs, a2_coeffs = table.get_coeffs_for_combination(1)
        np.testing.assert_array_equal(a1_coeffs, [0.5, 0.5])
        np.testing.assert_array_equal(a2_coeffs, [0.5, 0.5])
    
    def test_single_analysis_mode(self):
        """Test table with only Analysis 1 coefficients."""
        table = CombinationTableData(
            combination_names=["Single1", "Single2"],
            combination_types=["Linear", "Linear"],
            analysis1_coeffs=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]]),
            analysis2_coeffs=np.zeros((2, 0)),  # Empty
            analysis1_step_ids=[1, 2, 3],
            analysis2_step_ids=[],
        )
        
        assert table.num_analysis1_steps == 3
        assert table.num_analysis2_steps == 0


class TestCombinationResult:
    """Tests for CombinationResult dataclass."""
    
    def test_creation(self):
        """Test basic creation of CombinationResult."""
        result = CombinationResult(
            node_ids=np.array([1, 2, 3, 4, 5]),
            node_coords=np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]]),
            max_over_combo=np.array([100, 200, 150, 300, 250]),
            min_over_combo=np.array([10, 20, 15, 30, 25]),
            combo_of_max=np.array([0, 1, 0, 2, 1]),
            combo_of_min=np.array([2, 0, 1, 0, 2]),
            result_type="von_mises",
        )
        
        assert result.num_nodes == 5
        assert result.result_type == "von_mises"
    
    def test_num_combinations_with_all_results(self):
        """Test num_combinations when full results available."""
        all_results = np.random.rand(10, 100)  # 10 combinations, 100 nodes
        
        result = CombinationResult(
            node_ids=np.arange(100),
            node_coords=np.random.rand(100, 3),
            all_combo_results=all_results,
        )
        
        assert result.num_combinations == 10
    
    def test_num_combinations_without_all_results(self):
        """Test num_combinations returns 0 when no full results."""
        result = CombinationResult(
            node_ids=np.arange(100),
            node_coords=np.random.rand(100, 3),
        )
        
        assert result.num_combinations == 0


class TestSolverConfig:
    """Tests for SolverConfig dataclass."""
    
    def test_default_values(self):
        """Test default values of SolverConfig."""
        config = SolverConfig()
        
        assert config.calculate_von_mises is True
        assert config.calculate_max_principal_stress is False
        assert config.calculate_min_principal_stress is False
        assert config.combination_history_mode is False
        assert config.selected_node_id is None
    
    def test_custom_values(self):
        """Test custom configuration."""
        config = SolverConfig(
            calculate_von_mises=False,
            calculate_max_principal_stress=True,
            combination_history_mode=True,
            selected_node_id=12345,
        )
        
        assert config.calculate_von_mises is False
        assert config.calculate_max_principal_stress is True
        assert config.selected_node_id == 12345


class TestPlasticityConfig:
    """Tests for PlasticityConfig dataclass."""
    
    def test_is_active_false_when_disabled(self):
        """Test is_active returns False when disabled."""
        config = PlasticityConfig(enabled=False)
        assert config.is_active is False
    
    def test_is_active_false_without_material(self):
        """Test is_active returns False without material profile."""
        config = PlasticityConfig(enabled=True, material_profile=None)
        assert config.is_active is False
    
    def test_is_active_true_with_material(self):
        """Test is_active returns True with enabled and material."""
        # Create minimal material profile
        material = MaterialProfileData(
            youngs_modulus=pd.DataFrame({'Temperature (°C)': [20], "Young's Modulus [MPa]": [200000]}),
            poisson_ratio=pd.DataFrame({'Temperature (°C)': [20], "Poisson's Ratio": [0.3]}),
            plastic_curves={}
        )
        
        config = PlasticityConfig(enabled=True, material_profile=material)
        assert config.is_active is True


class TestMaterialProfileData:
    """Tests for MaterialProfileData dataclass."""
    
    def test_empty_profile(self):
        """Test creating empty material profile."""
        profile = MaterialProfileData.empty()
        
        assert profile.youngs_modulus.empty
        assert profile.poisson_ratio.empty
        assert len(profile.plastic_curves) == 0
        assert profile.has_data is False
    
    def test_has_data_with_youngs(self):
        """Test has_data returns True with Young's modulus data."""
        profile = MaterialProfileData(
            youngs_modulus=pd.DataFrame({'Temperature (°C)': [20], "Young's Modulus [MPa]": [200000]}),
            poisson_ratio=pd.DataFrame(),
            plastic_curves={}
        )
        
        assert profile.has_data is True
    
    def test_curve_for_temperature(self):
        """Test retrieving plastic curve for temperature."""
        curve_20 = pd.DataFrame({'Strain': [0, 0.1], 'Stress': [0, 400]})
        curve_100 = pd.DataFrame({'Strain': [0, 0.15], 'Stress': [0, 350]})
        
        profile = MaterialProfileData(
            youngs_modulus=pd.DataFrame(),
            poisson_ratio=pd.DataFrame(),
            plastic_curves={20.0: curve_20, 100.0: curve_100}
        )
        
        assert profile.curve_for_temperature(20.0) is not None
        assert profile.curve_for_temperature(50.0) is None


class TestAnalysisResult:
    """Tests for AnalysisResult dataclass."""
    
    def test_default_values(self):
        """Test default values."""
        result = AnalysisResult()
        
        assert result.combination_indices is None
        assert result.stress_values is None
        assert result.result_type == "unknown"
        assert result.node_id is None
        assert result.metadata == {}
    
    def test_with_data(self):
        """Test with actual data."""
        result = AnalysisResult(
            combination_indices=np.arange(10),
            stress_values=np.random.rand(10) * 100,
            result_type="von_mises",
            node_id=12345,
            metadata={"max_value": 95.5}
        )
        
        assert len(result.combination_indices) == 10
        assert result.node_id == 12345
        assert result.metadata["max_value"] == 95.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
