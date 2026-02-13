# Named Selection User Test Checklist

This checklist validates the original named-selection feature changes.

## Preconditions

1. Open Solver tab.
2. Load valid Analysis 1 and Analysis 2 RST files.
3. Ensure each file has at least one named selection.

## Source Filter Tests

1. Verify `Named Selection Source` dropdown is enabled after both RST files load.
2. Select `Common (A1 & A2)`.
3. Confirm only intersection names are shown.
4. Select `Analysis 1 (Base)`.
5. Confirm Analysis 1 names are shown.
6. Select `Analysis 2 (Combine)`.
7. Confirm Analysis 2 names are shown.

## Same-Name Precedence Tests

1. Prepare two RSTs where one named selection has the same name in both but different node sets.
2. Select that named selection and run solve.
3. Confirm scoping behavior follows Analysis 1 node content (base precedence).

## Analysis-2-Only Selection Test

1. Select `Analysis 2 (Combine)` mode.
2. Pick a named selection that exists only in Analysis 2.
3. Run solve and confirm scoping is accepted and analysis proceeds.

## Width/Readability Test

1. Use long named-selection names.
2. Open the named-selection dropdown list.
3. Confirm full/near-full names are readable in both field and popup.

## Refresh Behavior

1. Change source mode and click `Refresh`.
2. Confirm dropdown content updates consistently with active source mode.

