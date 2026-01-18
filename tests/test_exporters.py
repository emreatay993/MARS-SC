"""
Tests for export functions.

Tests CSV export for envelope results, single combinations, and histories.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from file_io.exporters import (
    export_to_csv,
    export_time_point_results,
    export_results_with_headers,
    export_envelope_results,
    export_single_combination,
    export_combination_history,
    export_all_combinations_batch,
)


class TestExportEnvelopeResults:
    """Tests for envelope results export."""
    
    def test_export_with_max_min(self, tmp_path):
        """Test exporting both max and min envelopes."""
        csv_path = tmp_path / "envelope.csv"
        
        node_ids = np.array([1, 2, 3, 4, 5])
        node_coords = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [2, 0, 0],
            [3, 0, 0],
            [4, 0, 0],
        ])
        max_values = np.array([100, 200, 150, 300, 250])
        min_values = np.array([10, 20, 15, 30, 25])
        combo_of_max = np.array([0, 1, 0, 2, 1])
        combo_of_min = np.array([2, 0, 1, 0, 2])
        combination_names = ["Combo A", "Combo B", "Combo C"]
        
        export_envelope_results(
            str(csv_path),
            node_ids=node_ids,
            node_coords=node_coords,
            max_values=max_values,
            min_values=min_values,
            combo_of_max=combo_of_max,
            combo_of_min=combo_of_min,
            result_type="von_mises",
            combination_names=combination_names,
        )
        
        # Read and verify
        df = pd.read_csv(csv_path)
        
        assert len(df) == 5
        assert "NodeID" in df.columns
        assert "X" in df.columns
        assert "Y" in df.columns
        assert "Z" in df.columns
        assert "Max Von Mises [MPa]" in df.columns
        assert "Min Von Mises [MPa]" in df.columns
        assert "Combination of Max (Index)" in df.columns
        assert "Combination of Max (Name)" in df.columns
        
        # Verify values
        np.testing.assert_array_equal(df["NodeID"].values, node_ids)
        np.testing.assert_array_almost_equal(df["Max Von Mises [MPa]"].values, max_values)
        assert df.iloc[0]["Combination of Max (Name)"] == "Combo A"
    
    def test_export_max_only(self, tmp_path):
        """Test exporting only max envelope."""
        csv_path = tmp_path / "max_only.csv"
        
        node_ids = np.array([1, 2, 3])
        max_values = np.array([100, 200, 150])
        
        export_envelope_results(
            str(csv_path),
            node_ids=node_ids,
            node_coords=None,
            max_values=max_values,
            min_values=None,
            result_type="max_principal",
        )
        
        df = pd.read_csv(csv_path)
        
        assert "Max Max Principal [MPa]" in df.columns
        assert "Min Max Principal [MPa]" not in df.columns
    
    def test_export_without_coordinates(self, tmp_path):
        """Test exporting without node coordinates."""
        csv_path = tmp_path / "no_coords.csv"
        
        node_ids = np.array([10, 20, 30])
        max_values = np.array([100, 200, 150])
        
        export_envelope_results(
            str(csv_path),
            node_ids=node_ids,
            node_coords=None,
            max_values=max_values,
        )
        
        df = pd.read_csv(csv_path)
        
        assert "NodeID" in df.columns
        assert "X" not in df.columns


class TestExportSingleCombination:
    """Tests for single combination export."""
    
    def test_basic_export(self, tmp_path):
        """Test basic single combination export."""
        csv_path = tmp_path / "single.csv"
        
        node_ids = np.array([1, 2, 3, 4])
        node_coords = np.random.rand(4, 3)
        stress_values = np.array([100, 150, 200, 175])
        
        export_single_combination(
            str(csv_path),
            node_ids=node_ids,
            node_coords=node_coords,
            stress_values=stress_values,
            combination_index=5,
            combination_name="Test Combination",
            result_type="von_mises",
        )
        
        df = pd.read_csv(csv_path)
        
        assert len(df) == 4
        assert "Von Mises [MPa]" in df.columns
        assert df["Combination Index"].iloc[0] == 5
        assert df["Combination Name"].iloc[0] == "Test Combination"
    
    def test_export_with_tensor(self, tmp_path):
        """Test export including full stress tensor."""
        csv_path = tmp_path / "with_tensor.csv"
        
        node_ids = np.array([1, 2])
        stress_values = np.array([100, 150])
        stress_tensor = np.array([
            [100, 50, 25, 10, 5, 3],  # Node 1: Sxx, Syy, Szz, Sxy, Syz, Sxz
            [150, 75, 40, 15, 8, 4],  # Node 2
        ])
        
        export_single_combination(
            str(csv_path),
            node_ids=node_ids,
            node_coords=None,
            stress_values=stress_values,
            combination_index=0,
            combination_name="Full Tensor",
            include_tensor=True,
            stress_tensor=stress_tensor,
        )
        
        df = pd.read_csv(csv_path)
        
        assert "Sxx [MPa]" in df.columns
        assert "Syy [MPa]" in df.columns
        assert "Szz [MPa]" in df.columns
        assert "Sxy [MPa]" in df.columns
        assert "Syz [MPa]" in df.columns
        assert "Sxz [MPa]" in df.columns
        
        # Verify tensor values
        np.testing.assert_almost_equal(df["Sxx [MPa]"].values, [100, 150])


class TestExportCombinationHistory:
    """Tests for combination history export (single node)."""
    
    def test_basic_history_export(self, tmp_path):
        """Test basic combination history export."""
        csv_path = tmp_path / "history.csv"
        
        combination_indices = np.array([0, 1, 2, 3, 4])
        stress_values = np.array([100, 120, 95, 150, 130])
        combination_names = ["C1", "C2", "C3", "C4", "C5"]
        
        export_combination_history(
            str(csv_path),
            node_id=12345,
            combination_indices=combination_indices,
            stress_values=stress_values,
            combination_names=combination_names,
            result_type="von_mises",
        )
        
        df = pd.read_csv(csv_path)
        
        assert len(df) == 5
        assert "Combination Index" in df.columns
        assert "Combination Name" in df.columns
        assert "Von Mises [MPa]" in df.columns
        assert "Node ID" in df.columns
        
        assert df["Node ID"].iloc[0] == 12345
    
    def test_history_with_plasticity(self, tmp_path):
        """Test combination history with plasticity correction data."""
        csv_path = tmp_path / "history_plastic.csv"
        
        combination_indices = np.array([0, 1, 2])
        stress_values = np.array([400, 500, 450])
        corrected_values = np.array([380, 420, 400])
        plastic_strain = np.array([0.0, 0.005, 0.002])
        
        export_combination_history(
            str(csv_path),
            node_id=999,
            combination_indices=combination_indices,
            stress_values=stress_values,
            corrected_values=corrected_values,
            plastic_strain=plastic_strain,
            result_type="von_mises",
        )
        
        df = pd.read_csv(csv_path)
        
        assert "Von Mises [MPa]" in df.columns
        assert "Von Mises Corrected [MPa]" in df.columns
        assert "Plastic Strain" in df.columns


class TestExportAllCombinationsBatch:
    """Tests for batch export of all combinations."""
    
    def test_wide_format_export(self, tmp_path):
        """Test wide-format batch export."""
        csv_path = tmp_path / "batch.csv"
        
        node_ids = np.array([1, 2, 3, 4, 5])
        node_coords = np.random.rand(5, 3)
        
        # 3 combinations, 5 nodes
        all_combinations_data = np.array([
            [100, 110, 120, 130, 140],  # Combo 0
            [105, 115, 125, 135, 145],  # Combo 1
            [95, 105, 115, 125, 135],   # Combo 2
        ])
        combination_names = ["Alpha", "Beta", "Gamma"]
        
        export_all_combinations_batch(
            str(csv_path),
            node_ids=node_ids,
            node_coords=node_coords,
            all_combinations_data=all_combinations_data,
            combination_names=combination_names,
            result_type="von_mises",
        )
        
        df = pd.read_csv(csv_path)
        
        assert len(df) == 5
        assert "NodeID" in df.columns
        assert "Alpha [Von Mises]" in df.columns
        assert "Beta [Von Mises]" in df.columns
        assert "Gamma [Von Mises]" in df.columns
        
        # Verify data
        np.testing.assert_array_almost_equal(
            df["Alpha [Von Mises]"].values,
            all_combinations_data[0]
        )
    
    def test_batch_export_without_names(self, tmp_path):
        """Test batch export without combination names."""
        csv_path = tmp_path / "batch_no_names.csv"
        
        node_ids = np.array([1, 2, 3])
        all_combinations_data = np.array([
            [100, 110, 120],
            [105, 115, 125],
        ])
        
        export_all_combinations_batch(
            str(csv_path),
            node_ids=node_ids,
            node_coords=None,
            all_combinations_data=all_combinations_data,
            combination_names=None,
        )
        
        df = pd.read_csv(csv_path)
        
        # Should use default names
        assert "Combo_0 [Von Mises]" in df.columns
        assert "Combo_1 [Von Mises]" in df.columns


class TestExportTimePointResults:
    """Tests for legacy time point export."""
    
    def test_basic_export(self, tmp_path):
        """Test basic time point results export."""
        csv_path = tmp_path / "timepoint.csv"
        
        node_ids = np.array([1, 2, 3])
        node_coords = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        scalar_data = np.array([100, 200, 150])
        
        export_time_point_results(
            node_ids, node_coords, scalar_data,
            scalar_name="Stress",
            filename=str(csv_path)
        )
        
        df = pd.read_csv(csv_path)
        
        assert len(df) == 3
        assert "NodeID" in df.columns
        assert "Stress" in df.columns
        np.testing.assert_array_almost_equal(df["Stress"].values, scalar_data)


class TestExportResultsWithHeaders:
    """Tests for generic export with custom headers."""
    
    def test_with_coordinates(self, tmp_path):
        """Test export with coordinates."""
        csv_path = tmp_path / "with_header.csv"
        
        node_ids = np.array([100, 200, 300])
        node_coords = np.random.rand(3, 3)
        data = np.array([50, 60, 70])
        
        export_results_with_headers(
            str(csv_path),
            node_ids, node_coords, data,
            header="CustomField"
        )
        
        df = pd.read_csv(csv_path)
        
        assert "NodeID" in df.columns
        assert "CustomField" in df.columns
        assert "X" in df.columns
    
    def test_without_coordinates(self, tmp_path):
        """Test export without coordinates."""
        csv_path = tmp_path / "no_coords.csv"
        
        node_ids = np.array([100, 200])
        data = np.array([50, 60])
        
        export_results_with_headers(
            str(csv_path),
            node_ids, None, data,
            header="Value"
        )
        
        df = pd.read_csv(csv_path)
        
        assert "NodeID" in df.columns
        assert "Value" in df.columns
        assert "X" not in df.columns


class TestExportCSVGeneric:
    """Tests for generic DataFrame export."""
    
    def test_simple_dataframe(self, tmp_path):
        """Test exporting simple DataFrame."""
        csv_path = tmp_path / "simple.csv"
        
        df = pd.DataFrame({
            "A": [1, 2, 3],
            "B": [4, 5, 6],
        })
        
        export_to_csv(df, str(csv_path))
        
        df_read = pd.read_csv(csv_path)
        pd.testing.assert_frame_equal(df, df_read)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
