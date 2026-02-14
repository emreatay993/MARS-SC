from enum import Enum


class ContourType(str, Enum):
    """Supported contour families for the Display tab."""

    STRESS = "Stress"
    FORCES = "Forces"
    DEFORMATION = "Deformation"
