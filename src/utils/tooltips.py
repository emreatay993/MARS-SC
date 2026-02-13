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

TOOLTIP_DEFORMATION_CYLINDRICAL_CS = (
    "Cylindrical Coordinate System for Deformation Output\n"
    "────────────────────────────────────────────────────\n\n"
    "Enter a coordinate system ID to transform displacement results\n"
    "from global Cartesian (X, Y, Z) to cylindrical coordinates (R, Theta, Z).\n\n"
    "• Leave empty: Results remain in global Cartesian coordinates (default)\n"
    "• Enter CS ID: Results are rotated to the specified cylindrical CS\n\n"
    "The coordinate system must exist in the RST file.\n"
    "Use ANSYS coordinate system IDs (e.g., from LOCAL or CS commands).\n\n"
    "Output components when cylindrical CS is used:\n"
    "  UX -> Radial displacement (UR)\n"
    "  UY -> Tangential displacement (U_theta)\n"
    "  UZ -> Axial displacement (UZ)"
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

TOOLTIP_OUTPUT_OPTIONS = (
    "<html>"
    "<b>Output Options</b><br><br>"
    "Choose what to compute from the selected Named Selection.<br><br>"
    "<b>Mutual exclusivity:</b><br>"
    "- Only one stress type can be selected at a time: Von Mises, Max Principal, or Min Principal.<br>"
    "- Nodal Forces is mutually exclusive with stress types.<br>"
    "- Deformation can be selected together with stress outputs.<br>"
    "- Combination History Mode changes output to single-node history across combinations."
    "</html>"
)

# =============================================================================
# File Input Tooltips
# =============================================================================

TOOLTIP_BASE_RST = (
    "Select the base analysis RST file (Analysis 1).\n\n"
    "This is typically the primary load case or the reference analysis, "
    "for example a steady-state analysis."
)

TOOLTIP_COMBINE_RST = (
    "Select the analysis to RST file to be combined (Analysis 2), "
    "for example, a maneuver analysis result for all possible load "
    "directions and scenarios.\n\n"
    "This analysis will be combined with the base analysis\n"
    "using the coefficients defined in the combination table."
)

TOOLTIP_NAMED_SELECTION = (
    "Select the Named Selection used to scope the analysis.\n\n"
    "Only nodes within the selected Named Selection are processed.\n"
    "The available list depends on 'Named Selection Source'.\n\n"
    "If a name exists in both analyses, node content is always taken\n"
    "from Analysis 1 (Base) to avoid mismatched node sets."
)

TOOLTIP_NAMED_SELECTION_SOURCE = (
    "Choose which Named Selection names are shown in the list.\n\n"
    "Common (A1 & A2): Only names that exist in both analyses.\n"
    "Analysis 1 (Base): Show names from base analysis only.\n"
    "Analysis 2 (Combine): Show names from combine analysis only.\n\n"
    "For same-name selections in both analyses, Analysis 1 node content\n"
    "is used as the scoping source."
)

TOOLTIP_NAMED_SELECTION_REFRESH = (
    "Refresh the Named Selection list using the current source filter."
)

# =============================================================================
# Display Tab Tooltips
# =============================================================================

TOOLTIP_DISPLAY_LOAD_FILE = (
    "<html>"
    "<b>Load Visualization CSV</b><br><br>"
    "Expected CSV format:<br>"
    "- Required columns: <b>X</b>, <b>Y</b>, <b>Z</b> (exact names)<br>"
    "- Optional column: <b>NodeID</b><br>"
    "- Optional scalar/result columns: one or more numeric columns<br><br>"
    "<b>How it is interpreted:</b><br>"
    "- Mesh points are created from X/Y/Z coordinates.<br>"
    "- If NodeID exists, it is used for node labels/export mapping.<br>"
    "- For standard CSVs, the first non-coordinate column is used as active contour data.<br><br>"
    "<b>Example header:</b><br>"
    "<code>NodeID,X,Y,Z,Von Mises [MPa]</code><br><br>"
    "Envelope CSVs exported by MARS-SC are also supported."
    "</html>"
)

TOOLTIP_DISPLAY_FILE_PATH = (
    "Path of the currently loaded visualization CSV file."
)

TOOLTIP_DISPLAY_VISUALIZATION_CONTROLS = (
    "<html>"
    "<b>Visualization Controls</b><br><br>"
    "Adjust how results are rendered and exported in the 3D view.<br>"
    "Some controls appear only when relevant result data is available "
    "(stress, forces, or deformation)."
    "</html>"
)

TOOLTIP_DISPLAY_POINT_SIZE = (
    "Adjust rendered node point size in the 3D view."
)

TOOLTIP_DISPLAY_LEGEND_RANGE = (
    "Set custom legend limits for contour visualization.\n\n"
    "Min and Max define the displayed color range."
)

TOOLTIP_DISPLAY_SCALAR = (
    "Select which data to display as the contour:\n"
    "- Max Value: Maximum stress/force across all combinations\n"
    "- Min Value: Minimum stress/force (when available)\n"
    "- Combo # of Max: Combination index producing the maximum\n"
    "- Combo # of Min: Combination index producing the minimum"
)

TOOLTIP_DISPLAY_VIEW_COMBINATION = (
    "Select a specific combination to visualize its results.\n"
    "Choose '(Envelope View)' to return to Max/Min envelope view."
)

TOOLTIP_DISPLAY_FORCE_COMPONENT = (
    "Select which force component to display:\n"
    "- Magnitude: sqrt(FX^2 + FY^2 + FZ^2)\n"
    "- FX, FY, FZ: Directional components\n"
    "- Shear XY/XZ/YZ: In-plane component magnitudes"
)

TOOLTIP_DISPLAY_EXPORT_FORCES = (
    "Export nodal forces for the currently selected combination to CSV.\n"
    "Includes directional components, magnitude, shear values, and metadata."
)

TOOLTIP_DISPLAY_DISPLACEMENT_COMPONENT = (
    "Select which displacement component to display:\n"
    "- U_mag: Total displacement magnitude\n"
    "- UX, UY, UZ: Directional displacement components"
)

TOOLTIP_DISPLAY_EXPORT_OUTPUT = (
    "Export current output data to CSV files.\n\n"
    "Envelope View exports envelope fields based on current display selection.\n"
    "Single Combination exports results for the selected combination.\n"
    "Available stress, force, and deformation outputs are exported."
)

TOOLTIP_DISPLAY_DEFORMATION_SCALE = (
    "Scale factor for displaying deformed shape.\n"
    "1.0 = physical scale, >1 exaggerates deformation, <1 reduces it."
)

TOOLTIP_DISPLAY_COMBINATION_POINT_CONTROLS = (
    "<html>"
    "<b>Combination Point Controls</b><br><br>"
    "Choose and export a single combination result state for display workflows."
    "</html>"
)

TOOLTIP_DISPLAY_COMBINATION_SELECTOR = (
    "Select which combination index/name to display in this control section."
)

TOOLTIP_DISPLAY_UPDATE_BUTTON = (
    "Apply the selected combination/time control to update displayed results."
)

TOOLTIP_DISPLAY_SAVE_BUTTON = (
    "Save the currently selected combination results to CSV."
)

TOOLTIP_DISPLAY_EXTRACT_IC = (
    "Export velocity as APDL initial condition commands.\n"
    "Used for workflows that transfer results to transient setups."
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
    "<html>"
    "<b>Import Combination CSV</b><br><br>"
    "Expected minimum headers:<br>"
    "- <b>Combination Name</b>, <b>Type</b>, coefficient columns...<br><br>"
    "Recommended coefficient headers:<br>"
    "- <b>A1_Set_1, A1_Set_2, ...</b><br>"
    "- <b>A2_Set_1, A2_Set_2, ...</b><br><br>"
    "Also accepted prefixes for coefficient columns:<br>"
    "- Analysis1_, Base_ for Analysis 1<br>"
    "- Analysis2_, Combine_ for Analysis 2<br><br>"
    "Notes:<br>"
    "- Type must be <b>Linear</b>.<br>"
    "- If no A1/A2-style prefixes are found, all coefficient columns are treated as Analysis 1."
    "</html>"
)

TOOLTIP_EXPORT_CSV = (
    "Export the current combination table to a CSV file."
)

TOOLTIP_ADD_ROW = (
    "Add a new combination row with default zero coefficients."
)

TOOLTIP_DELETE_ROW = (
    "Delete the currently selected combination row."
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
