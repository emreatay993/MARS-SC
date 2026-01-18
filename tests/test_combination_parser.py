"""
Tests for the combination table parser.

Tests CSV parsing, validation, and export functionality.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from file_io.combination_parser import (
    CombinationTableParser,
    CombinationTableParseError,
)
from core.data_models import CombinationTableData


class TestCombinationTableParserBasic:
    """Basic parsing tests."""
    
    def test_parse_two_analysis_csv(self, tmp_path):
        """Test parsing CSV with two analyses."""
        csv_content = """Combination Name,Type,A1_Set_1,A1_Set_2,A2_Set_1,A2_Set_2
Combo1,Linear,1.0,0.0,0.5,0.0
Combo2,Linear,0.0,1.0,0.0,0.5
Combo3,Linear,0.5,0.5,0.5,0.5
"""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text(csv_content)
        
        table = CombinationTableParser.parse_csv(str(csv_path))
        
        assert table.num_combinations == 3
        assert table.num_analysis1_steps == 2
        assert table.num_analysis2_steps == 2
        assert table.combination_names == ["Combo1", "Combo2", "Combo3"]
        
        # Check coefficients
        np.testing.assert_array_almost_equal(
            table.analysis1_coeffs[0], [1.0, 0.0]
        )
        np.testing.assert_array_almost_equal(
            table.analysis2_coeffs[0], [0.5, 0.0]
        )
    
    def test_parse_single_analysis_csv(self, tmp_path):
        """Test parsing CSV with single analysis (backward compatible)."""
        csv_content = """Combination Name,Type,Load Case 1,Load Case 2,Load Case 3
Combo1,Linear,1,0,0
Combo2,Linear,0,1,0
Combo3,Linear,0,0,1
"""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text(csv_content)
        
        table = CombinationTableParser.parse_csv(str(csv_path))
        
        assert table.num_combinations == 3
        assert table.num_analysis1_steps == 3
        assert table.num_analysis2_steps == 0
    
    def test_parse_alternative_prefixes(self, tmp_path):
        """Test parsing with alternative column prefixes (Base_, Combine_)."""
        csv_content = """Combination Name,Type,Base_Step_1,Base_Step_2,Combine_Step_1
Mix1,Linear,1.0,0.5,0.25
Mix2,Linear,0.5,1.0,0.75
"""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text(csv_content)
        
        table = CombinationTableParser.parse_csv(str(csv_path))
        
        assert table.num_analysis1_steps == 2
        assert table.num_analysis2_steps == 1
    
    def test_parse_analysis1_analysis2_prefixes(self, tmp_path):
        """Test parsing with Analysis1_, Analysis2_ prefixes."""
        csv_content = """Combination Name,Type,Analysis1_Set_1,Analysis2_Set_1
Test,Linear,1.0,2.0
"""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text(csv_content)
        
        table = CombinationTableParser.parse_csv(str(csv_path))
        
        assert table.num_analysis1_steps == 1
        assert table.num_analysis2_steps == 1


class TestCombinationTableParserValidation:
    """Validation and error handling tests."""
    
    def test_empty_csv_raises_error(self, tmp_path):
        """Test that empty CSV raises error."""
        csv_path = tmp_path / "empty.csv"
        csv_path.write_text("")
        
        with pytest.raises(CombinationTableParseError):
            CombinationTableParser.parse_csv(str(csv_path))
    
    def test_missing_columns_raises_error(self, tmp_path):
        """Test that CSV with too few columns raises error."""
        csv_content = """Combination Name,Type
Combo1,Linear
"""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text(csv_content)
        
        with pytest.raises(CombinationTableParseError, match="at least 3 columns"):
            CombinationTableParser.parse_csv(str(csv_path))
    
    def test_invalid_combination_type_raises_error(self, tmp_path):
        """Test that unsupported combination type raises error."""
        csv_content = """Combination Name,Type,A1_Set_1
Combo1,Quadratic,1.0
"""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text(csv_content)
        
        with pytest.raises(CombinationTableParseError, match="Unsupported combination type"):
            CombinationTableParser.parse_csv(str(csv_path))
    
    def test_invalid_coefficient_raises_error(self, tmp_path):
        """Test that non-numeric coefficients raise error."""
        csv_content = """Combination Name,Type,A1_Set_1
Combo1,Linear,abc
"""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text(csv_content)
        
        with pytest.raises(CombinationTableParseError, match="Invalid coefficient"):
            CombinationTableParser.parse_csv(str(csv_path))
    
    def test_file_not_found(self):
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(CombinationTableParseError):
            CombinationTableParser.parse_csv("nonexistent.csv")


class TestCombinationTableParserStepIdExtraction:
    """Tests for step ID extraction from column names."""
    
    def test_extract_step_ids_from_set_format(self, tmp_path):
        """Test extracting step IDs from 'Set_N' format."""
        csv_content = """Combination Name,Type,A1_Set_3,A1_Set_7,A2_Set_2
Combo1,Linear,1.0,0.5,0.25
"""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text(csv_content)
        
        table = CombinationTableParser.parse_csv(str(csv_path))
        
        assert table.analysis1_step_ids == [3, 7]
        assert table.analysis2_step_ids == [2]
    
    def test_extract_step_ids_from_step_format(self, tmp_path):
        """Test extracting step IDs from 'Step_N' format."""
        csv_content = """Combination Name,Type,A1_Step_1,A1_Step_2,A2_Step_5
Combo1,Linear,1.0,0.5,0.25
"""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text(csv_content)
        
        table = CombinationTableParser.parse_csv(str(csv_path))
        
        assert table.analysis1_step_ids == [1, 2]
        assert table.analysis2_step_ids == [5]
    
    def test_extract_step_ids_fallback(self, tmp_path):
        """Test fallback to sequential IDs when no number in column name."""
        csv_content = """Combination Name,Type,A1_Load,A1_Other,A2_Case
Combo1,Linear,1.0,0.5,0.25
"""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text(csv_content)
        
        table = CombinationTableParser.parse_csv(str(csv_path))
        
        # Should use 1, 2, 3, ... as defaults
        assert table.analysis1_step_ids == [1, 2]
        assert table.analysis2_step_ids == [1]


class TestCombinationTableParserExport:
    """Tests for CSV export functionality."""
    
    def test_export_csv_roundtrip(self, tmp_path):
        """Test that export and re-import produces same data."""
        original = CombinationTableData(
            combination_names=["Combo1", "Combo2"],
            combination_types=["Linear", "Linear"],
            analysis1_coeffs=np.array([[1.0, 0.5], [0.25, 0.75]]),
            analysis2_coeffs=np.array([[0.5, 0.25], [1.0, 0.5]]),
            analysis1_step_ids=[1, 2],
            analysis2_step_ids=[3, 4],
        )
        
        csv_path = tmp_path / "export.csv"
        CombinationTableParser.export_csv(original, str(csv_path))
        
        # Re-import
        reimported = CombinationTableParser.parse_csv(str(csv_path))
        
        assert reimported.combination_names == original.combination_names
        assert reimported.num_combinations == original.num_combinations
        np.testing.assert_array_almost_equal(
            reimported.analysis1_coeffs, original.analysis1_coeffs
        )
        np.testing.assert_array_almost_equal(
            reimported.analysis2_coeffs, original.analysis2_coeffs
        )
    
    def test_to_dataframe(self):
        """Test conversion to DataFrame."""
        table = CombinationTableData(
            combination_names=["Test1", "Test2"],
            combination_types=["Linear", "Linear"],
            analysis1_coeffs=np.array([[1.0], [0.5]]),
            analysis2_coeffs=np.array([[0.5], [1.0]]),
            analysis1_step_ids=[1],
            analysis2_step_ids=[1],
        )
        
        df = CombinationTableParser.to_dataframe(table)
        
        assert len(df) == 2
        assert "Combination Name" in df.columns
        assert "Type" in df.columns
        assert "A1_Set_1" in df.columns
        assert "A2_Set_1" in df.columns


class TestCombinationTableParserValidateAgainstAnalyses:
    """Tests for validation against analysis data."""
    
    def test_valid_table(self):
        """Test validation passes for matching step counts."""
        table = CombinationTableData(
            combination_names=["Test"],
            combination_types=["Linear"],
            analysis1_coeffs=np.array([[1.0, 0.5, 0.25]]),
            analysis2_coeffs=np.array([[0.5, 0.5]]),
            analysis1_step_ids=[1, 2, 3],
            analysis2_step_ids=[1, 2],
        )
        
        is_valid, error = CombinationTableParser.validate_against_analyses(
            table, analysis1_steps=3, analysis2_steps=2
        )
        
        assert is_valid is True
        assert error == ""
    
    def test_invalid_analysis1_count(self):
        """Test validation fails for wrong Analysis 1 step count."""
        table = CombinationTableData(
            combination_names=["Test"],
            combination_types=["Linear"],
            analysis1_coeffs=np.array([[1.0, 0.5]]),
            analysis2_coeffs=np.array([[0.5]]),
            analysis1_step_ids=[1, 2],
            analysis2_step_ids=[1],
        )
        
        is_valid, error = CombinationTableParser.validate_against_analyses(
            table, analysis1_steps=5, analysis2_steps=1
        )
        
        assert is_valid is False
        assert "Analysis 1" in error
    
    def test_invalid_analysis2_count(self):
        """Test validation fails for wrong Analysis 2 step count."""
        table = CombinationTableData(
            combination_names=["Test"],
            combination_types=["Linear"],
            analysis1_coeffs=np.array([[1.0]]),
            analysis2_coeffs=np.array([[0.5, 0.5, 0.5]]),
            analysis1_step_ids=[1],
            analysis2_step_ids=[1, 2, 3],
        )
        
        is_valid, error = CombinationTableParser.validate_against_analyses(
            table, analysis1_steps=1, analysis2_steps=5
        )
        
        assert is_valid is False
        assert "Analysis 2" in error


class TestCombinationTableParserCreateTables:
    """Tests for table creation helpers."""
    
    def test_create_empty_table(self):
        """Test creating empty table."""
        table = CombinationTableParser.create_empty_table(
            analysis1_step_ids=[1, 2, 3],
            analysis2_step_ids=[1, 2],
            num_combinations=2
        )
        
        assert table.num_combinations == 2
        assert table.num_analysis1_steps == 3
        assert table.num_analysis2_steps == 2
        
        # All coefficients should be zero
        np.testing.assert_array_equal(table.analysis1_coeffs, np.zeros((2, 3)))
        np.testing.assert_array_equal(table.analysis2_coeffs, np.zeros((2, 2)))
    
    def test_create_identity_table(self):
        """Test creating identity table."""
        table = CombinationTableParser.create_identity_table(
            analysis1_step_ids=[1, 2],
            analysis2_step_ids=[1]
        )
        
        # Should have 3 combinations: A1_Step_1, A1_Step_2, A2_Step_1
        assert table.num_combinations == 3
        
        # First combination: [1, 0] for A1, [0] for A2
        np.testing.assert_array_equal(table.analysis1_coeffs[0], [1.0, 0.0])
        np.testing.assert_array_equal(table.analysis2_coeffs[0], [0.0])
        
        # Second combination: [0, 1] for A1, [0] for A2
        np.testing.assert_array_equal(table.analysis1_coeffs[1], [0.0, 1.0])
        np.testing.assert_array_equal(table.analysis2_coeffs[1], [0.0])
        
        # Third combination: [0, 0] for A1, [1] for A2
        np.testing.assert_array_equal(table.analysis1_coeffs[2], [0.0, 0.0])
        np.testing.assert_array_equal(table.analysis2_coeffs[2], [1.0])


class TestRealWorldCSV:
    """Tests using real-world CSV format from example file."""
    
    def test_parse_example_csv(self, tmp_path):
        """Test parsing format matching combination_table_example.csv."""
        csv_content = """Combination Name,Type,SS - CDP Condition - Disp,SS - Maneuver - Limit Load 1,SS - Maneuver - Limit Load 2,SS - Maneuver - Limit Load 3,SS - Maneuver - Limit Load 4,SS - Maneuver - Limit Load 5,SS - Maneuver - Limit Load 6,SS - Maneuver - Limit Load 7
CDP + Man Limit 1,Linear,1,1,0,0,0,0,0,0
CDP + Man Limit 2,Linear,1,0,1,0,0,0,0,0
CDP + Man Limit 3,Linear,1,0,0,1,0,0,0,0
CDP + Man Limit 4,Linear,1,0,0,0,1,0,0,0
CDP + Man Limit 5,Linear,1,0,0,0,0,1,0,0
CDP + Man Limit 6,Linear,1,0,0,0,0,0,1,0
CDP + Man Limit 7,Linear,1,0,0,0,0,0,0,1
"""
        csv_path = tmp_path / "example.csv"
        csv_path.write_text(csv_content)
        
        table = CombinationTableParser.parse_csv(str(csv_path))
        
        assert table.num_combinations == 7
        # All columns assigned to Analysis 1 (single analysis mode)
        assert table.num_analysis1_steps == 8
        assert table.num_analysis2_steps == 0
        
        # Check first combination
        np.testing.assert_array_equal(
            table.analysis1_coeffs[0], 
            [1, 1, 0, 0, 0, 0, 0, 0]
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
