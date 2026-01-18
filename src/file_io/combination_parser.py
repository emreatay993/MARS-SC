"""
Combination Table Parser for MARS-SC (Solution Combination).

Parses and validates combination CSV files that define how to linearly combine
stress results from two analyses.

CSV Format:
-----------
Combination Name,Type,A1_Set_1,A1_Set_2,...,A2_Set_1,A2_Set_2,...
Combo1,Linear,1.0,0.5,...,0.0,1.0,...
Combo2,Linear,0.0,1.0,...,1.0,0.0,...

Columns:
- Combination Name: User-defined name for the combination
- Type: Combination type (currently only "Linear" is supported)
- A1_Set_N or Analysis1_Set_N: Coefficient for Analysis 1, Load Step N
- A2_Set_N or Analysis2_Set_N: Coefficient for Analysis 2, Load Step N

Alternative format (single analysis, backward compatible):
---------------------------------------------------------
Combination Name,Type,Load Case 1,Load Case 2,...
"""

import re
from typing import Tuple, List, Optional
import pandas as pd
import numpy as np

from core.data_models import CombinationTableData


class CombinationTableParseError(Exception):
    """Raised when parsing a combination table fails."""
    pass


class CombinationTableParser:
    """
    Parses and validates combination CSV files.
    
    Handles both single-analysis and two-analysis combination tables,
    with automatic detection of the format based on column headers.
    """
    
    # Patterns for recognizing analysis columns
    A1_PATTERNS = [
        re.compile(r'^A1[_\s-]', re.IGNORECASE),
        re.compile(r'^Analysis\s*1[_\s-]', re.IGNORECASE),
        re.compile(r'^Base[_\s-]', re.IGNORECASE),
    ]
    
    A2_PATTERNS = [
        re.compile(r'^A2[_\s-]', re.IGNORECASE),
        re.compile(r'^Analysis\s*2[_\s-]', re.IGNORECASE),
        re.compile(r'^Combine[_\s-]', re.IGNORECASE),
    ]
    
    @staticmethod
    def parse_csv(file_path: str) -> CombinationTableData:
        """
        Parse CSV file into CombinationTableData.
        
        Automatically detects whether the CSV contains coefficients for one
        or two analyses based on column header patterns.
        
        Args:
            file_path: Path to the CSV file.
            
        Returns:
            CombinationTableData object with parsed coefficients.
            
        Raises:
            CombinationTableParseError: If parsing fails.
            FileNotFoundError: If file does not exist.
        """
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            raise CombinationTableParseError(f"Failed to read CSV file: {e}")
        
        return CombinationTableParser._parse_dataframe(df)
    
    @staticmethod
    def parse_dataframe(df: pd.DataFrame) -> CombinationTableData:
        """
        Parse a pandas DataFrame into CombinationTableData.
        
        Args:
            df: DataFrame with combination table data.
            
        Returns:
            CombinationTableData object with parsed coefficients.
        """
        return CombinationTableParser._parse_dataframe(df)
    
    @staticmethod
    def _parse_dataframe(df: pd.DataFrame) -> CombinationTableData:
        """Internal parsing logic."""
        if df.empty:
            raise CombinationTableParseError("CSV file is empty")
        
        # Get column names
        columns = list(df.columns)
        
        # Validate required columns
        if len(columns) < 3:
            raise CombinationTableParseError(
                "CSV must have at least 3 columns: Combination Name, Type, and at least one coefficient"
            )
        
        # First column should be combination name
        name_col = columns[0]
        type_col = columns[1]
        coeff_cols = columns[2:]
        
        # Extract combination names and types
        combination_names = df[name_col].astype(str).tolist()
        combination_types = df[type_col].astype(str).tolist()
        
        # Validate combination types
        for i, ctype in enumerate(combination_types):
            if ctype.lower() not in ['linear']:
                raise CombinationTableParseError(
                    f"Row {i+2}: Unsupported combination type '{ctype}'. Only 'Linear' is supported."
                )
        
        # Categorize coefficient columns into Analysis 1 and Analysis 2
        a1_cols = []
        a2_cols = []
        unassigned_cols = []
        
        for col in coeff_cols:
            if CombinationTableParser._matches_patterns(col, CombinationTableParser.A1_PATTERNS):
                a1_cols.append(col)
            elif CombinationTableParser._matches_patterns(col, CombinationTableParser.A2_PATTERNS):
                a2_cols.append(col)
            else:
                unassigned_cols.append(col)
        
        # If no columns match patterns, assign all to Analysis 1 (single-analysis mode)
        if not a1_cols and not a2_cols:
            a1_cols = unassigned_cols
            unassigned_cols = []
        # If only A1 columns found, treat unassigned as A1 as well
        elif a1_cols and not a2_cols:
            a1_cols.extend(unassigned_cols)
            unassigned_cols = []
        # If both found but there are unassigned, raise error
        elif unassigned_cols:
            raise CombinationTableParseError(
                f"Unrecognized coefficient columns: {unassigned_cols}. "
                "Column names should start with 'A1_', 'A2_', 'Analysis1_', 'Analysis2_', 'Base_', or 'Combine_'."
            )
        
        # Extract step IDs from column names
        a1_step_ids = [CombinationTableParser._extract_step_id(col, i+1) for i, col in enumerate(a1_cols)]
        a2_step_ids = [CombinationTableParser._extract_step_id(col, i+1) for i, col in enumerate(a2_cols)]
        
        # Extract coefficient values
        try:
            a1_coeffs = df[a1_cols].values.astype(float) if a1_cols else np.zeros((len(df), 0))
            a2_coeffs = df[a2_cols].values.astype(float) if a2_cols else np.zeros((len(df), 0))
        except ValueError as e:
            raise CombinationTableParseError(f"Invalid coefficient value: {e}")
        
        return CombinationTableData(
            combination_names=combination_names,
            combination_types=combination_types,
            analysis1_coeffs=a1_coeffs,
            analysis2_coeffs=a2_coeffs,
            analysis1_step_ids=a1_step_ids,
            analysis2_step_ids=a2_step_ids,
        )
    
    @staticmethod
    def _matches_patterns(text: str, patterns: List[re.Pattern]) -> bool:
        """Check if text matches any of the given patterns."""
        return any(pattern.match(text) for pattern in patterns)
    
    @staticmethod
    def _extract_step_id(column_name: str, default: int) -> int:
        """
        Extract step ID from column name.
        
        Looks for patterns like "_Set_1", "_Step_2", "_1", etc.
        Falls back to default if no number found.
        """
        # Try to find a number at the end of the column name
        patterns = [
            re.compile(r'[_\s-]Set[_\s-]?(\d+)', re.IGNORECASE),
            re.compile(r'[_\s-]Step[_\s-]?(\d+)', re.IGNORECASE),
            re.compile(r'[_\s-](\d+)$'),
            re.compile(r'(\d+)$'),
        ]
        
        for pattern in patterns:
            match = pattern.search(column_name)
            if match:
                return int(match.group(1))
        
        return default
    
    @staticmethod
    def export_csv(data: CombinationTableData, file_path: str) -> None:
        """
        Export combination table to CSV.
        
        Args:
            data: CombinationTableData to export.
            file_path: Path to output CSV file.
        """
        df = CombinationTableParser.to_dataframe(data)
        df.to_csv(file_path, index=False)
    
    @staticmethod
    def to_dataframe(data: CombinationTableData) -> pd.DataFrame:
        """
        Convert CombinationTableData to pandas DataFrame.
        
        Args:
            data: CombinationTableData to convert.
            
        Returns:
            DataFrame with combination table data.
        """
        # Build column names
        columns = ['Combination Name', 'Type']
        
        # Add Analysis 1 columns
        for step_id in data.analysis1_step_ids:
            columns.append(f'A1_Set_{step_id}')
        
        # Add Analysis 2 columns
        for step_id in data.analysis2_step_ids:
            columns.append(f'A2_Set_{step_id}')
        
        # Build data rows
        rows = []
        for i in range(data.num_combinations):
            row = [data.combination_names[i], data.combination_types[i]]
            
            # Add Analysis 1 coefficients
            if data.num_analysis1_steps > 0:
                row.extend(data.analysis1_coeffs[i, :].tolist())
            
            # Add Analysis 2 coefficients
            if data.num_analysis2_steps > 0:
                row.extend(data.analysis2_coeffs[i, :].tolist())
            
            rows.append(row)
        
        return pd.DataFrame(rows, columns=columns)
    
    @staticmethod
    def validate_against_analyses(
        table: CombinationTableData,
        analysis1_steps: int,
        analysis2_steps: int
    ) -> Tuple[bool, str]:
        """
        Validate that table references valid load steps.
        
        Checks that the number of coefficient columns matches the number of
        load steps in each analysis.
        
        Args:
            table: CombinationTableData to validate.
            analysis1_steps: Number of load steps in Analysis 1.
            analysis2_steps: Number of load steps in Analysis 2.
            
        Returns:
            Tuple of (is_valid, error_message).
            If valid, error_message is empty.
        """
        errors = []
        
        # Check Analysis 1
        if table.num_analysis1_steps != analysis1_steps:
            errors.append(
                f"Combination table has {table.num_analysis1_steps} Analysis 1 columns, "
                f"but Analysis 1 RST has {analysis1_steps} load steps."
            )
        
        # Check Analysis 2
        if table.num_analysis2_steps != analysis2_steps:
            errors.append(
                f"Combination table has {table.num_analysis2_steps} Analysis 2 columns, "
                f"but Analysis 2 RST has {analysis2_steps} load steps."
            )
        
        # Check step IDs are within range
        for step_id in table.analysis1_step_ids:
            if step_id < 1 or step_id > analysis1_steps:
                errors.append(
                    f"Analysis 1 step ID {step_id} is out of range (1-{analysis1_steps})."
                )
        
        for step_id in table.analysis2_step_ids:
            if step_id < 1 or step_id > analysis2_steps:
                errors.append(
                    f"Analysis 2 step ID {step_id} is out of range (1-{analysis2_steps})."
                )
        
        if errors:
            return (False, " ".join(errors))
        
        return (True, "")
    
    @staticmethod
    def create_empty_table(
        analysis1_step_ids: List[int],
        analysis2_step_ids: List[int],
        num_combinations: int = 1
    ) -> CombinationTableData:
        """
        Create an empty combination table with the specified structure.
        
        Useful for initializing a new table in the UI.
        
        Args:
            analysis1_step_ids: Load step IDs from Analysis 1.
            analysis2_step_ids: Load step IDs from Analysis 2.
            num_combinations: Initial number of combination rows.
            
        Returns:
            CombinationTableData with zero coefficients.
        """
        n_a1 = len(analysis1_step_ids)
        n_a2 = len(analysis2_step_ids)
        
        return CombinationTableData(
            combination_names=[f"Combination {i+1}" for i in range(num_combinations)],
            combination_types=["Linear"] * num_combinations,
            analysis1_coeffs=np.zeros((num_combinations, n_a1)),
            analysis2_coeffs=np.zeros((num_combinations, n_a2)),
            analysis1_step_ids=list(analysis1_step_ids),
            analysis2_step_ids=list(analysis2_step_ids),
        )
    
    @staticmethod
    def create_identity_table(
        analysis1_step_ids: List[int],
        analysis2_step_ids: List[int]
    ) -> CombinationTableData:
        """
        Create a combination table with identity combinations.
        
        Creates one combination per load step where only that step has
        coefficient 1.0 and all others are 0.0. Useful for testing.
        
        Args:
            analysis1_step_ids: Load step IDs from Analysis 1.
            analysis2_step_ids: Load step IDs from Analysis 2.
            
        Returns:
            CombinationTableData with identity combinations.
        """
        n_a1 = len(analysis1_step_ids)
        n_a2 = len(analysis2_step_ids)
        total_steps = n_a1 + n_a2
        
        names = []
        types = []
        a1_coeffs = []
        a2_coeffs = []
        
        # Create identity combinations for Analysis 1
        for i, step_id in enumerate(analysis1_step_ids):
            names.append(f"A1_Step_{step_id}")
            types.append("Linear")
            a1_row = np.zeros(n_a1)
            a1_row[i] = 1.0
            a1_coeffs.append(a1_row)
            a2_coeffs.append(np.zeros(n_a2))
        
        # Create identity combinations for Analysis 2
        for i, step_id in enumerate(analysis2_step_ids):
            names.append(f"A2_Step_{step_id}")
            types.append("Linear")
            a1_coeffs.append(np.zeros(n_a1))
            a2_row = np.zeros(n_a2)
            a2_row[i] = 1.0
            a2_coeffs.append(a2_row)
        
        return CombinationTableData(
            combination_names=names,
            combination_types=types,
            analysis1_coeffs=np.array(a1_coeffs) if a1_coeffs else np.zeros((0, n_a1)),
            analysis2_coeffs=np.array(a2_coeffs) if a2_coeffs else np.zeros((0, n_a2)),
            analysis1_step_ids=list(analysis1_step_ids),
            analysis2_step_ids=list(analysis2_step_ids),
        )
