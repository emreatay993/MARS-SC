# MARS-SC: Solution Combination
## User Manual
**Version**: v1.1.x  
**Last updated**: February 14, 2026

> **Audience**: Structural/mechanical engineers using MARS-SC desktop GUI  
> **Format**: Designed for Word/PDF export with image placeholders

---

# PART I: OVERVIEW

---

## Chapter 1 - What Is MARS-SC

MARS-SC combines results from two static RST analyses using user-defined linear coefficients and provides envelope and per-combination visualization/export workflows.

### Key Capabilities

| Capability | Description |
| --- | --- |
| Two-analysis combination | Combine Analysis 1 (Base) and Analysis 2 (Combine) load-step results |
| Named-selection scoping | Scope solves to selected named selection nodes |
| Stress and optional outputs | Stress envelopes with optional forces/deformation workflows |
| Display workflows | Contour-family selection (`Stress`, `Forces`, `Deformation`) with envelope/single-combination visualization and CSV export |
| Guided UI | Detailed tooltips and global tooltip on/off setting |

[**Image Placeholder**: Main window with Solver and Display tabs]

---

# PART II: NAMED SELECTION WORKFLOW

---

## Chapter 2 - Named Selection Source Modes

When both RST files are loaded, the Solver tab provides:

- `Common (A1 & A2)`
- `Analysis 1 (Base)`
- `Analysis 2 (Combine)`

These modes control which names are listed in the named-selection dropdown.

### Mode Behavior

| Mode | What is listed |
| --- | --- |
| Common (A1 & A2) | Intersection of names in both analyses |
| Analysis 1 (Base) | All names from Analysis 1 |
| Analysis 2 (Combine) | All names from Analysis 2 |

[**Image Placeholder**: Named Selection Source dropdown with three modes]

---

## Chapter 3 - Same-Name Precedence Rule

If a named selection with the same name exists in both analyses but has different node content:

- MARS-SC always uses **Analysis 1 (Base)** node content for scoping.

This rule ensures deterministic solve behavior and avoids ambiguity.

### Additional Notes

- Analysis-2-only selections are supported when `Analysis 2 (Combine)` mode is selected.
- Validation and solve paths use the same scoping resolver.

[**Image Placeholder**: Diagram showing common-name NS resolved to Analysis 1 nodes]

---

## Chapter 4 - Dropdown Usability Improvements

Named-selection display was widened for long names:

- Wider field width
- Wider popup list
- Minimum content sizing tuned for longer strings

This reduces truncation and makes selection safer for similarly prefixed names.

[**Image Placeholder**: Long named-selection names fully visible in dropdown list]

---

# PART III: TOOLTIP SYSTEM

---

## Chapter 5 - Tooltip Coverage

Tooltips are provided for:

- Solver tab file inputs and named-selection controls
- Combination table controls and output options
- Display tab visualization/export controls

Tooltip text is centralized in:

- `src/utils/tooltips.py`

### CSV Guidance in Tooltips

Import/visualization controls include expected file format details, including required header columns and accepted naming patterns.

[**Image Placeholder**: HTML-style tooltip with CSV header guidance]

---

## Chapter 6 - Global Tooltip Toggle and Styling

Tooltips can be enabled/disabled globally from:

- `View -> Enable Tooltips`

Behavior:

- Checked: tooltips appear normally
- Unchecked: tooltip events are blocked application-wide
- State is persistent across restarts via `QSettings`

Styling:

- Global `QToolTip` style aligned with sibling `MARS_` project palette
- Implemented in `src/ui/styles/style_constants.py` as `TOOLTIP_STYLE`

[**Image Placeholder**: View menu with Enable Tooltips toggle and tooltip style sample]

---

# PART IV: DISPLAY CONTOUR WORKFLOW

---

## Chapter 7 - Visualization Controls Layout

The `Visualization Controls` group is intentionally split into two rows:

- **Row 1 (base rendering controls)**:
  - Node Point Size
  - Legend Range (Min/Max)
  - Deformation Scale Factor
- **Row 2 (result/contour controls)**:
  - Contour Type
  - Display
  - View Combination
  - Force Component
  - Displacement component
  - Export buttons

This layout keeps controls readable at common window widths and avoids overflow.

[**Image Placeholder**: Visualization Controls shown in two rows]

---

## Chapter 8 - Contour Type Behavior

### Contour Type values

- `Stress`
- `Forces`
- `Deformation`

### Visibility rules

- `Contour Type` appears only when **2 or more** contour families are available in current results.
- Only usable families are listed.

### Envelope vs specific-combination

- `View Combination = (Envelope View)`:
  - `Display` controls `Max/Min/Combo # of Max/Combo # of Min` for active family.
- `View Combination = N: <name>`:
  - The selected combination is shown.
  - `Display` selector is disabled for families that use specific-combination scalar arrays.

### Family-specific controls

- `Force Component` is shown only when `Contour Type = Forces`.
- `Displacement` component selector is shown only when `Contour Type = Deformation`.
- Deformation scale remains available whenever deformation results exist, independent of active contour type.

---

## Chapter 9 - Deformation Contours and Export

### Deformation contouring support

- Deformation contouring is supported in:
  - Envelope view
  - Specific-combination view
- Components:
  - `U_mag`
  - `UX`
  - `UY`
  - `UZ`

### Deformation envelope component logic

- `UX`, `UY`, `UZ` envelopes are signed extrema:
  - `Max` uses signed maximum
  - `Min` uses signed minimum
- Not absolute-value envelope behavior.

### Export Output CSV (Envelope View)

Deformation envelope export keeps existing magnitude columns and may include component envelope columns:

- Existing:
  - `Max Displacement Magnitude [unit]`
  - `Min Displacement Magnitude [unit]`
  - `Combination of Max (#)/(Name)`
  - `Combination of Min (#)/(Name)`
- Added (when available):
  - `Max/Min UX [unit]` + `Combination of Max/Min UX (#)/(Name)`
  - `Max/Min UY [unit]` + `Combination of Max/Min UY (#)/(Name)`
  - `Max/Min UZ [unit]` + `Combination of Max/Min UZ (#)/(Name)`

---

# PART V: QUICK TROUBLESHOOTING

---

## Chapter 10 - Common Questions

### Why is the named-selection list empty?

Check that:

1. Both RST files are loaded.
2. Selected source mode actually contains selections.
3. In common mode, names exist in both analyses.

### Why does my common-name NS not match Analysis 2 nodes?

This is by design. Common-name scoping uses Analysis 1 (Base) node content.

### Why are tooltips not appearing?

Check `View -> Enable Tooltips` is checked.

### Why is `Contour Type` not visible?

`Contour Type` is hidden when only one contour family is available. It appears only for mixed-family result sets.

### Why is `Display` disabled after choosing a specific combination?

In specific-combination mode, contour values come from selected-combination arrays; envelope `Display` toggles do not apply.

---

## Appendix A - Related Files

- `src/ui/builders/solver_ui.py`
- `src/ui/solver_tab.py`
- `src/ui/handlers/analysis_handler.py`
- `src/ui/builders/display_ui.py`
- `src/ui/display_tab.py`
- `src/ui/handlers/display_contour_types.py`
- `src/ui/handlers/display_contour_policy.py`
- `src/ui/handlers/display_contour_context.py`
- `src/ui/handlers/display_mesh_arrays.py`
- `src/ui/handlers/display_contour_sync_handler.py`
- `src/ui/handlers/display_visualization_handler.py`
- `src/ui/handlers/display_export_handler.py`
- `src/file_io/exporters.py`
- `src/ui/application_controller.py`
- `src/utils/tooltips.py`
- `src/ui/styles/style_constants.py`
- `tests/test_display_contour_policy.py`
- `tests/test_display_mesh_arrays.py`
- `tests/test_exporters.py`
- `MARS_SC_USER_ACCEPTANCE_TESTS.csv`
- `MARS_SC_USER_ACCEPTANCE_TESTS_GROUPED.csv`
- `MARS_SC_UAT_Consolidated_Tests.csv`
- `MARS_SC_USER_ACCEPTANCE_TESTS_TR.csv`
- `MARS_SC_USER_ACCEPTANCE_TESTS_GROUPED_TR.csv`
- `MARS_SC_UAT_Consolidated_Tests_TR.csv`
