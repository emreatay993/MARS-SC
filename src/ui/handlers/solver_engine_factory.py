"""
Engine creation helpers for solver analysis.

This module centralizes engine construction so the main analysis handler can
focus on orchestration.
"""

from typing import Optional

from core.data_models import CombinationTableData
from solver.deformation_engine import DeformationCombinationEngine
from solver.nodal_forces_engine import NodalForcesCombinationEngine
from solver.stress_engine import StressCombinationEngine


class SolverEngineFactory:
    """Build stress, nodal-forces, and deformation engines."""

    @staticmethod
    def create_stress_engine(
        reader1,
        reader2,
        nodal_scoping,
        combo_table: Optional[CombinationTableData],
    ) -> StressCombinationEngine:
        """Create a stress-combination engine from shared run inputs."""
        SolverEngineFactory._validate_common_inputs(reader1, reader2, combo_table)
        return StressCombinationEngine(
            reader1=reader1,
            reader2=reader2,
            nodal_scoping=nodal_scoping,
            combination_table=combo_table,
        )

    @staticmethod
    def create_nodal_forces_engine(
        reader1,
        reader2,
        nodal_scoping,
        combo_table: Optional[CombinationTableData],
        rotate_to_global: bool,
    ) -> NodalForcesCombinationEngine:
        """Create a nodal-forces-combination engine from shared run inputs."""
        SolverEngineFactory._validate_common_inputs(reader1, reader2, combo_table)
        return NodalForcesCombinationEngine(
            reader1=reader1,
            reader2=reader2,
            nodal_scoping=nodal_scoping,
            combination_table=combo_table,
            rotate_to_global=rotate_to_global,
        )

    @staticmethod
    def create_deformation_engine(
        reader1,
        reader2,
        nodal_scoping,
        combo_table: Optional[CombinationTableData],
        cylindrical_cs_id: Optional[int],
    ) -> DeformationCombinationEngine:
        """Create a deformation-combination engine from shared run inputs."""
        SolverEngineFactory._validate_common_inputs(reader1, reader2, combo_table)
        return DeformationCombinationEngine(
            reader1=reader1,
            reader2=reader2,
            nodal_scoping=nodal_scoping,
            combination_table=combo_table,
            cylindrical_cs_id=cylindrical_cs_id,
        )

    @staticmethod
    def _validate_common_inputs(reader1, reader2, combo_table: Optional[CombinationTableData]) -> None:
        if reader1 is None or reader2 is None:
            raise ValueError("DPF readers not available. Please reload RST files.")
        if combo_table is None:
            raise ValueError("Combination table is empty or invalid.")
