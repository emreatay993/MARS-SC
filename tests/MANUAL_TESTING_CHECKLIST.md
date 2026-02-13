# MARS-SC: Manual GUI Testing Checklist

**Version**: 1.0.x  
**Date**: ___________  
**Tester**: ___________

---

## Pre-Testing Setup

- [ ] Fresh Python environment activated
- [ ] Required dependencies installed
- [ ] Two RST test files available with named selections
- [ ] At least one case with same NS name but different node sets prepared

---

## 1. Application Launch

- [ ] Application opens successfully
- [ ] Main tabs visible: `Main Window`, `Display`
- [ ] Menu bar contains `File` and `View`
- [ ] View menu contains `Enable Tooltips` (checkable)

---

## 2. Named Selection Source Modes

- [ ] Load Analysis 1 RST
- [ ] Load Analysis 2 RST
- [ ] Confirm `Named Selection Source` is enabled
- [ ] Select `Common (A1 & A2)` and verify only intersection names appear
- [ ] Select `Analysis 1 (Base)` and verify Analysis 1 names appear
- [ ] Select `Analysis 2 (Combine)` and verify Analysis 2 names appear
- [ ] Click `Refresh` and verify list remains consistent with active mode

---

## 3. Same-Name Base Precedence

- [ ] Use two files where one named selection has same name in both analyses with different node sets
- [ ] Select that named selection
- [ ] Run analysis
- [ ] Confirm behavior matches Analysis 1 node scoping (base precedence)

---

## 4. Analysis-2-Only Named Selection

- [ ] Switch source mode to `Analysis 2 (Combine)`
- [ ] Select a named selection that exists only in Analysis 2
- [ ] Run solve
- [ ] Confirm solve proceeds with valid scoping

---

## 5. Dropdown Width and Readability

- [ ] Use long named-selection names
- [ ] Open dropdown
- [ ] Confirm names are readable in field and popup list

---

## 6. Tooltip Content Checks

### Solver Tab

- [ ] Base Analysis button tooltip explains reference/base role
- [ ] Combine Analysis button tooltip includes maneuver-analysis example
- [ ] Named Selection Source tooltip explains all three modes
- [ ] Named Selection tooltip explains base-precedence behavior
- [ ] Import CSV tooltip shows expected header/prefix format
- [ ] Output Options group tooltip explains exclusivity rules
- [ ] Add Row/Delete Row buttons have action tooltips

### Display Tab

- [ ] Load Visualization File tooltip explains required `X,Y,Z` columns
- [ ] Tooltip includes optional `NodeID` and example header
- [ ] Visualization controls show relevant tooltips

---

## 7. Global Tooltip Toggle

- [ ] With `Enable Tooltips` checked, tooltips appear
- [ ] Uncheck `Enable Tooltips`, hover controls, tooltips do not appear
- [ ] Close and reopen app
- [ ] Verify previous tooltip toggle state is preserved

---

## 8. Tooltip Visual Consistency

- [ ] Tooltip colors and border match sibling `MARS_` look
- [ ] Font and spacing are consistent and readable

---

## Final Sign-Off

- [ ] All checks pass
- [ ] Any failures logged with reproduction steps and screenshots
