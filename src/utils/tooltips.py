"""
Tooltip text constants for MARS-SC (Solution Combination).

Centralizes all tooltip messages used throughout the application UI.
This makes tooltips easier to maintain, update, and ensures consistency.

Usage:
    from utils.tooltips import TOOLTIP_NODAL_FORCES, TOOLTIP_NODAL_FORCES_CSYS
    widget.setToolTip(TOOLTIP_NODAL_FORCES)
"""

# =============================================================================
# Output Options Tooltips
# =============================================================================

TOOLTIP_VON_MISES = (
    "Calculate von Mises equivalent stress.\n\n"
    "Formula: σ_vm = √(0.5 × [(σx-σy)² + (σy-σz)² + (σz-σx)² + 6×(τxy² + τyz² + τxz²)])"
)

TOOLTIP_MAX_PRINCIPAL = (
    "Calculate Maximum Principal Stress (S1).\n\n"
    "The largest eigenvalue of the stress tensor.\n"
    "Represents the maximum normal stress at each point."
)

TOOLTIP_MIN_PRINCIPAL = (
    "Calculate Minimum Principal Stress (S3).\n\n"
    "The smallest eigenvalue of the stress tensor.\n"
    "Represents the minimum (most compressive) normal stress at each point."
)

TOOLTIP_NODAL_FORCES = (
    "Combine nodal forces from both analyses.\n\n"
    "Requires 'Nodal Forces' to be enabled in ANSYS Output Controls."
)

TOOLTIP_DEFORMATION = (
    "Calculate combined displacement/deformation (UX, UY, UZ, U_mag).\n\n"
    "Can be selected alongside stress outputs.\n"
    "Enables deformed mesh visualization with scale control."
)

TOOLTIP_NODAL_FORCES_CSYS = (
    "Coordinate System for Nodal Forces Output\n"
    "─────────────────────────────────────────\n\n"
    "• Global (Default):\n"
    "   Forces are rotated to the global Cartesian coordinate system.\n"
    "   FX, FY, FZ represent forces in global X, Y, Z directions.\n"
    "   Best for: Summing forces, comparing across elements, reactions.\n\n"
    "• Local (Element):\n"
    "   Forces remain in each element's local coordinate system.\n"
    "   Best for: Shell elements where in-plane vs. out-of-plane matters.\n\n"
    "Note: Beam and pipe element forces are always in element-local\n"
    "coordinates regardless of this setting (ANSYS behavior)."
)

TOOLTIP_COMBINATION_HISTORY = (
    "Enable Combination History Mode (Single Node).\n\n"
    "When enabled, compute stress history for a single node\n"
    "across all combinations instead of computing the envelope\n"
    "for all nodes."
)

TOOLTIP_PLASTICITY_CORRECTION = (
    "Enable plasticity correction for elastic-plastic analysis.\n\n"
    "Applies Neuber or Glinka correction to convert elastic\n"
    "stresses to elastic-plastic stresses using material curves."
)

# =============================================================================
# File Input Tooltips
# =============================================================================

TOOLTIP_BASE_RST = (
    "Select the base analysis RST file (Analysis 1).\n\n"
    "This is typically the primary load case or the reference analysis."
)

TOOLTIP_COMBINE_RST = (
    "Select the analysis to combine RST file (Analysis 2).\n\n"
    "This analysis will be combined with the base analysis\n"
    "using the coefficients defined in the combination table."
)

TOOLTIP_NAMED_SELECTION = (
    "Select a Named Selection to scope the analysis.\n\n"
    "Only nodes within the selected Named Selection will be processed.\n"
    "The Named Selection must exist in both RST files."
)

# =============================================================================
# Combination Table Tooltips
# =============================================================================

TOOLTIP_COMBINATION_TABLE = (
    "Define combination coefficients for linear superposition.\n\n"
    "Each row represents a load combination.\n"
    "σ_combined = Σ(α_i × σ_A1_i) + Σ(β_j × σ_A2_j)\n\n"
    "Where α and β are the coefficients for each load step."
)

TOOLTIP_IMPORT_CSV = (
    "Import combination coefficients from a CSV file.\n\n"
    "CSV format should match the table columns."
)

TOOLTIP_EXPORT_CSV = (
    "Export the current combination table to a CSV file."
)

# =============================================================================
# Plasticity Options Tooltips
# =============================================================================

TOOLTIP_PLASTICITY_METHOD = (
    "Select the plasticity correction method:\n\n"
    "• Neuber: Uses Neuber's rule (σ × ε = σ_e² / E)\n"
    "• Glinka: Uses Glinka's ESED method (energy-based)\n"
    "• IBG: Incremental Buczynski-Glinka (not yet implemented)"
)

TOOLTIP_MAX_ITERATIONS = (
    "Maximum number of iterations for the plasticity solver.\n\n"
    "Default: 60 iterations. Increase if convergence issues occur."
)

TOOLTIP_TOLERANCE = (
    "Convergence tolerance for the plasticity solver.\n\n"
    "Default: 1e-10. Smaller values give more accurate results\n"
    "but may require more iterations."
)

TOOLTIP_EXTRAPOLATION = (
    "Extrapolation mode for stress-strain curve:\n\n"
    "• Linear: Linearly extrapolate beyond the curve data\n"
    "• Plateau: Hold the last value constant (safer)"
)
