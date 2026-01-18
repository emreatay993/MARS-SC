"""
Legacy analysis engine module - NOT USED in MARS-SC.

This module previously contained the AnalysisEngine class for MARS modal
transient analysis using PyTorch. It has been deprecated in MARS-SC.

For MARS-SC (Solution Combination), use:
    - solver.combination_engine.CombinationEngine - Linear combination of stress results
    - solver.plasticity_engine - Plasticity corrections for combination results
"""

# This module is intentionally empty to avoid importing torch dependencies.
# The legacy AnalysisEngine class has been removed.

__all__ = []


class AnalysisEngine:
    """
    DEPRECATED: Legacy modal transient analysis engine.
    
    This class is not used in MARS-SC. For static stress combination analysis,
    use solver.combination_engine.CombinationEngine instead.
    
    Example usage for MARS-SC:
        from solver.combination_engine import CombinationEngine
        
        engine = CombinationEngine(base_reader, combine_reader, combination_table)
        engine.preload_stress_data(named_selection, stress_type='von_mises')
        result = engine.compute_full_analysis()
    """
    
    def __init__(self):
        raise NotImplementedError(
            "AnalysisEngine is deprecated in MARS-SC. "
            "Use solver.combination_engine.CombinationEngine for static stress combination."
        )
