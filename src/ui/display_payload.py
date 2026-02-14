"""
Typed payloads for solver-to-display communication.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from core.data_models import CombinationResult, DeformationResult, NodalForcesResult


@dataclass
class SolverOutputFlags:
    """Boolean flags that describe which solver outputs are active for a run."""

    compute_von_mises: bool = False
    compute_max_principal: bool = False
    compute_min_principal: bool = False
    compute_nodal_forces: bool = False
    compute_deformation: bool = False


@dataclass
class DisplayResultPayload:
    """Full display update payload emitted by SolverTab."""

    mesh: object
    scalar_bar_title: str
    data_min: float
    data_max: float
    combination_names: List[str] = field(default_factory=list)
    stress_result: Optional[CombinationResult] = None
    forces_result: Optional[NodalForcesResult] = None
    deformation_result: Optional[DeformationResult] = None
    output_flags: SolverOutputFlags = field(default_factory=SolverOutputFlags)
