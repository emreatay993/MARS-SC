"""
File I/O operations for loading and exporting data.

Note: This package is named 'file_io' (not 'io') to avoid conflicts
with Python's built-in io module.
"""

from file_io.dpf_reader import (
    DPFAnalysisReader,
    DPFNotAvailableError,
    NodalForcesNotAvailableError,
    BeamElementNotSupportedError,
    scale_stress_field,
    add_stress_fields,
    compute_principal_stresses,
    compute_von_mises_from_field,
    scale_force_field,
    add_force_fields,
    compute_force_magnitude,
)

from file_io.combination_parser import (
    CombinationTableParser,
    CombinationTableParseError,
)

__all__ = [
    # DPF Reader
    'DPFAnalysisReader',
    'DPFNotAvailableError',
    'NodalForcesNotAvailableError',
    'BeamElementNotSupportedError',
    'scale_stress_field',
    'add_stress_fields',
    'compute_principal_stresses',
    'compute_von_mises_from_field',
    'scale_force_field',
    'add_force_fields',
    'compute_force_magnitude',
    # Combination Parser
    'CombinationTableParser',
    'CombinationTableParseError',
]
