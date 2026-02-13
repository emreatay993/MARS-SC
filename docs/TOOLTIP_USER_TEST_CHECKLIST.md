# Tooltip User Test Checklist

Use this checklist to validate tooltip behavior from a user perspective.

## Preconditions

- Launch the application normally.
- Ensure both Solver and Display tabs are accessible.

## Global Toggle Tests

1. In menu bar, go to `View`.
2. Verify `Enable Tooltips` exists and is checkable.
3. With `Enable Tooltips` checked, hover controls and confirm tooltips appear.
4. Uncheck `Enable Tooltips`, hover same controls, confirm no tooltip appears.
5. Close and reopen app; confirm toggle state is persisted.

## Solver Tab Tooltip Tests

1. Hover `Select Base Analysis RST` and verify base-analysis guidance text.
2. Hover `Select Analysis to Combine RST` and verify Analysis 2 example text.
3. Hover named-selection source dropdown and verify mode descriptions.
4. Hover named-selection dropdown and verify base-precedence behavior description.
5. Hover `Import CSV` and confirm expected header/prefix guidance appears.
6. Hover `Add Row` and `Delete Row` and verify row action descriptions.
7. Hover `Output Options` group and verify exclusivity rule explanation.

## Display Tab Tooltip Tests

1. Hover `Load Visualization File`; verify CSV format guidance:
   - required `X,Y,Z`
   - optional `NodeID`
   - example header
2. Hover point size and legend range controls; verify descriptions.
3. Hover scalar display and view-combination controls; verify behavior descriptions.
4. Hover export buttons; verify export scope descriptions.

## Style Consistency Tests

1. Compare tooltip visual style with `MARS_` project:
   - background color
   - text color
   - border color/thickness
   - font size/family
   - corner radius/padding

Expected: styles should match.
