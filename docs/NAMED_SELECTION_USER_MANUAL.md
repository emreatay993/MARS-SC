# Named Selection Feature Manual

This manual covers the named-selection enhancements added to the Solver tab.

## What Changed

When both RST files are loaded, users can now choose which named-selection names appear in the dropdown:

- `Common (A1 & A2)`
- `Analysis 1 (Base)`
- `Analysis 2 (Combine)`

These controls are in the Solver tab under Input Files.

## Named Selection Source Modes

### Common (A1 & A2)

Shows only names present in both analyses.

Use this mode when you want to avoid selecting names that are missing in one file.

### Analysis 1 (Base)

Shows all named selections from Analysis 1.

Use this when Analysis 1 drives your scoping workflow.

### Analysis 2 (Combine)

Shows all named selections from Analysis 2.

Use this when only Analysis 2 contains the target named selection.

## Same-Name, Different-Node Behavior

If a named selection with the same name exists in both analyses but has different node content:

- Scoping always uses **Analysis 1 (Base)** node content.

This avoids ambiguity and ensures deterministic behavior during solve.

## Dropdown Width Improvement

The named-selection dropdown and popup list were widened so long named-selection names are visible without heavy truncation.

## Where Logic Is Implemented

- UI options and sizing:
  - `src/ui/builders/solver_ui.py`
- Source filtering and scoping resolution:
  - `src/ui/solver_tab.py`
- Engine scoping hookup:
  - `src/ui/handlers/analysis_handler.py`

