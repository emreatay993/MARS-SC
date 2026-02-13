# Tooltip User Manual

This guide describes tooltip behavior in MARS-SC and how users can control it.

## Overview

Tooltips are available across Solver and Display tabs to explain controls, expected input formats, and workflow rules.

Tooltip text is centralized in:
- `src/utils/tooltips.py`

Tooltip style is centralized in:
- `src/ui/styles/style_constants.py` (`TOOLTIP_STYLE`)

## Enable/Disable Tooltips

Tooltips can be toggled globally from the menu bar:

1. Open `View`.
2. Toggle `Enable Tooltips`.

When disabled, tooltip popups are blocked application-wide.

The setting is persistent and restored on next launch using:
- `QSettings` key: `view/tooltips_enabled`

## Solver Tab Coverage

Tooltips are provided for:
- Analysis 1 / Analysis 2 input controls
- Named selection source filter and named selection dropdown
- Refresh named selection button
- Combination table group
- Import CSV / Export CSV / Add Row / Delete Row
- Output options and output-type details
- Plasticity controls

### Import CSV Tooltip

The Import CSV tooltip includes expected header format and accepted coefficient prefixes:
- Required leading columns: `Combination Name`, `Type`
- Recommended coefficient headers: `A1_Set_*`, `A2_Set_*`
- Also accepted: `Analysis1_` / `Base_`, `Analysis2_` / `Combine_`

## Display Tab Coverage

Tooltips are provided for:
- Load Visualization File button and file path
- Visualization Controls group
- Point size and legend range controls
- Scalar display and view-combination controls
- Force/deformation component selectors
- Export buttons
- Deformation scale factor
- Combination Point Controls group and buttons

### Load Visualization File Tooltip

The tooltip documents CSV expectations:
- Required columns: `X`, `Y`, `Z` (exact names)
- Optional: `NodeID`
- Scalar data: first non-coordinate column is used for contour in standard CSV files
- Envelope CSV exports from MARS-SC are supported

## Visual Style Consistency

Tooltip colors, border, spacing, and font are aligned with the `MARS_` project style.
