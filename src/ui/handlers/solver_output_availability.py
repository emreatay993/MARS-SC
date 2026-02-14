"""
Shared output-availability policy for solver-tab analyses.

Centralizing these checks avoids drift between UI enablement and run-time validation.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SolverOutputAvailability:
    """Availability flags for optional solver outputs."""

    nodal_forces: bool
    displacement: bool


def evaluate_output_availability(analysis1_data, analysis2_data) -> SolverOutputAvailability:
    """
    Evaluate output availability from currently loaded analysis metadata.

    Args:
        analysis1_data: Base analysis metadata object (or None).
        analysis2_data: Combined analysis metadata object (or None).

    Returns:
        SolverOutputAvailability with nodal-force and displacement availability.
    """
    if analysis1_data is None or analysis2_data is None:
        return SolverOutputAvailability(
            nodal_forces=False,
            displacement=False,
        )

    nodal_forces_available = bool(
        getattr(analysis1_data, "nodal_forces_available", False)
        and getattr(analysis2_data, "nodal_forces_available", False)
    )
    displacement_available = bool(
        getattr(analysis1_data, "displacement_available", False)
        and getattr(analysis2_data, "displacement_available", False)
    )

    return SolverOutputAvailability(
        nodal_forces=nodal_forces_available,
        displacement=displacement_available,
    )
