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
    "Enable elastic-plastic correction (Neuber/Glinka) for local yielding.\n\n"
    "Valid when: linear-elastic FE stress is representative, yielding is\n"
    "localized (small plastic zone), material curve is monotonic/stabilized,\n"
    "and loading is mainly proportional or monotonic.\n\n"
    "Use caution for large plastic zones, strong non-proportional multiaxial\n"
    "or cyclic path-dependent loading (ratcheting/hysteresis), or major\n"
    "redistribution/global collapse behavior."
)

TOOLTIP_MATERIAL_PROFILE = (
    "<html>"
    "<b>Material Profile</b><br><br>"
    "Opens the dedicated <b>Material Profile</b> editor where you define the<br>"
    "temperature-dependent material inputs used by plasticity correction.<br><br>"
    "In that editor you can fill tables directly, or import/export a JSON profile.<br><br>"
    "<b>Expected import format (JSON):</b><br>"
    "- Root keys: <b>youngs_modulus</b>, <b>poisson_ratio</b>, <b>plastic_curves</b><br>"
    "- Table sections use: <b>columns</b> and <b>data</b><br><br>"
    "<b>Required columns:</b><br>"
    "- Young's modulus: <code>Temperature (°C)</code>, <code>Young's Modulus [MPa]</code><br>"
    "- Poisson ratio: <code>Temperature (°C)</code>, <code>Poisson's Ratio</code><br>"
    "- Plastic curves (per temperature): <code>Plastic Strain</code>, <code>True Stress [MPa]</code><br><br>"
    "<b>Minimal example:</b><br>"
    "<code>{</code><br>"
    "<code>&nbsp;&nbsp;\"youngs_modulus\": {\"columns\": [\"Temperature (°C)\", \"Young's Modulus [MPa]\"], \"data\": [[20, 210000]]},</code><br>"
    "<code>&nbsp;&nbsp;\"poisson_ratio\": {\"columns\": [\"Temperature (°C)\", \"Poisson's Ratio\"], \"data\": [[20, 0.3]]},</code><br>"
    "<code>&nbsp;&nbsp;\"plastic_curves\": [{\"temperature\": 20, \"columns\": [\"Plastic Strain\", \"True Stress [MPa]\"], \"data\": [[0.0, 250], [0.01, 300]]}]</code><br>"
    "<code>}</code>"
    "</html>"
)

TOOLTIP_MATERIAL_DIALOG_OVERVIEW = (
    "<html>"
    "<b>Material Profile Editor</b><br><br>"
    "Define temperature-dependent Young's modulus, Poisson ratio, and plastic strain curves.<br>"
    "These datasets are used when plasticity correction is enabled."
    "</html>"
)

TOOLTIP_MATERIAL_DIALOG_IMPORT_PROFILE = (
    "<html>"
    "<b>Import Material Profile (.json)</b><br><br>"
    "<b>Required root keys:</b><br>"
    "- <code>youngs_modulus</code><br>"
    "- <code>poisson_ratio</code><br>"
    "- <code>plastic_curves</code><br><br>"
    "<b>Section structure:</b><br>"
    "- Objects with <code>columns</code> and <code>data</code><br>"
    "- Each plastic curve entry also includes <code>temperature</code><br><br>"
    "<b>Required columns:</b><br>"
    "- Young's modulus: <code>Temperature (°C)</code>, <code>Young's Modulus [MPa]</code><br>"
    "- Poisson ratio: <code>Temperature (°C)</code>, <code>Poisson's Ratio</code><br>"
    "- Plastic curve: <code>Plastic Strain</code>, <code>True Stress [MPa]</code>"
    "</html>"
)

TOOLTIP_MATERIAL_DIALOG_EXPORT_PROFILE = (
    "Export the full material profile (all tabs) to a JSON file."
)

TOOLTIP_MATERIAL_DIALOG_YOUNGS_TABLE = (
    "Young's modulus table. Expected columns: Temperature (°C), Young's Modulus [MPa]."
)

TOOLTIP_MATERIAL_DIALOG_POISSON_TABLE = (
    "Poisson's ratio table. Expected columns: Temperature (°C), Poisson's Ratio."
)

TOOLTIP_MATERIAL_DIALOG_PLASTIC_TABLE = (
    "Plastic curve table for the selected temperature. Expected columns: Plastic Strain, True Stress [MPa]."
)

TOOLTIP_MATERIAL_DIALOG_ADD_ROW = (
    "Add an empty row to the current table."
)

TOOLTIP_MATERIAL_DIALOG_REMOVE_SELECTED = (
    "Remove selected row(s) from the current table."
)

TOOLTIP_MATERIAL_DIALOG_IMPORT_YOUNGS = (
    "<html>"
    "<b>Import Young's Modulus CSV/TXT</b><br><br>"
    "<b>Required columns (exact):</b><br>"
    "- <code>Temperature (°C)</code><br>"
    "- <code>Young's Modulus [MPa]</code><br><br>"
    "<b>Example header:</b><br>"
    "<code>Temperature (°C),Young's Modulus [MPa]</code>"
    "</html>"
)

TOOLTIP_MATERIAL_DIALOG_IMPORT_POISSON = (
    "<html>"
    "<b>Import Poisson Ratio CSV/TXT</b><br><br>"
    "<b>Required columns (exact):</b><br>"
    "- <code>Temperature (°C)</code><br>"
    "- <code>Poisson's Ratio</code><br><br>"
    "<b>Example header:</b><br>"
    "<code>Temperature (°C),Poisson's Ratio</code>"
    "</html>"
)

TOOLTIP_MATERIAL_DIALOG_EXPORT_TABLE = (
    "Export the current table to CSV."
)

TOOLTIP_MATERIAL_DIALOG_TEMPERATURE_LIST = (
    "Temperature sets for plastic strain curves. Select one to edit its curve table."
)

TOOLTIP_MATERIAL_DIALOG_ADD_TEMPERATURE = (
    "Add a new temperature entry (°C) for a plastic strain curve."
)

TOOLTIP_MATERIAL_DIALOG_REMOVE_TEMPERATURE = (
    "Remove the selected temperature and its associated plastic curve."
)

TOOLTIP_MATERIAL_DIALOG_IMPORT_PLASTIC_CURVE = (
    "<html>"
    "<b>Import Plastic Curve CSV/TXT</b><br><br>"
    "Imports curve data for the currently selected temperature.<br><br>"
    "<b>Required columns (exact):</b><br>"
    "- <code>Plastic Strain</code><br>"
    "- <code>True Stress [MPa]</code><br><br>"
    "<b>Example header:</b><br>"
    "<code>Plastic Strain,True Stress [MPa]</code>"
    "</html>"
)

TOOLTIP_MATERIAL_DIALOG_EXPORT_PLASTIC_CURVE = (
    "Export the plastic curve for the selected temperature to CSV."
)

TOOLTIP_MATERIAL_DIALOG_SAVE = (
    "Save changes and apply this material profile to the solver."
)

TOOLTIP_MATERIAL_DIALOG_CANCEL = (
    "Close without applying changes."
)

TOOLTIP_TEMPERATURE_FIELD_FILE = (
    "<html>"
    "<b>Temperature Field File</b><br><br>"
    "Optional input for temperature-dependent plasticity.<br><br>"
    "<b>Expected table format:</b><br>"
    "- Text table with <b>tab</b> or <b>whitespace</b> separators<br>"
    "- Must include a node-ID column named <code>Node Number</code><br>"
    "- Include one or more numeric temperature columns (first non-node column is used by default)<br><br>"
    "<b>Example header:</b><br>"
    "<code>Node Number    Temperature (°C)</code><br>"
    "<code>101    350.0</code><br><br>"
    "<b>Multiple temperature columns are allowed:</b><br>"
    "<code>Node Number    Temp_Load1    Temp_Load2</code>"
    "</html>"
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
# Navigator Tooltips
# =============================================================================

TOOLTIP_NAVIGATOR = (
    "<html>"
    "<b>Navigator</b><br><br>"
    "Browse project files used in MARS-SC workflows.<br><br>"
    "<b>Capabilities</b><br>"
    "&#8226; Shows files under the selected project directory<br>"
    "&#8226; Focuses on common result/data files (<code>.rst</code>, <code>.csv</code>, <code>.txt</code>)<br>"
    "&#8226; Double-click a file to open it with your default application<br><br>"
    "<b>Tip</b><br>"
    "Use <i>File &gt; Select Project Directory</i> to point the navigator to your working folder."
    "</html>"
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

TOOLTIP_DISPLAY_CONTOUR_TYPE = (
    "Select contour result family:\n"
    "- Stress\n"
    "- Forces\n"
    "- Deformation\n\n"
    "Only contour families available in current results are listed."
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
    "- UX, UY, UZ: Directional displacement components\n\n"
    "Works in both Envelope View and specific-combination view."
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
    "<html>"
    "<b>Select Elastic-Plastic Correction Method</b><br><br>"
    "<b>Methods</b><br>"
    "&#8226; <b>Neuber</b>: Local notch-yielding correction using equivalent stress-strain product. "
    "Best with proportional/monotonic loading.<br>"
    "&#8226; <b>Glinka</b>: Energy-density-based local correction; often predicts slightly higher plastic "
    "strain at the same elastic input.<br>"
    "&#8226; <b>IBG</b>: Incremental Buczynski-Glinka <i>(currently disabled)</i>.<br><br>"
    "<b>Applicability (Neuber/Glinka)</b><br>"
    "Representative linear-elastic FE stress, localized yielding (small plastic zone), "
    "monotonic/stabilized material curve, and limited path dependence.<br><br>"
    "<b>Prefer Full Nonlinear Analysis When</b><br>"
    "Loading is strongly non-proportional cyclic or plasticity is widespread."
    "</html>"
)

TOOLTIP_ITERATION_CONTROLS = (
    "Controls for nonlinear correction convergence.\n\n"
    "Max Iterations limits solver steps; Tolerance sets residual target.\n"
    "Tighter tolerance and more iterations improve accuracy but increase runtime."
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
