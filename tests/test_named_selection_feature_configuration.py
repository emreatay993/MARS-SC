"""
Static configuration tests for named-selection feature updates.

These checks validate source wiring without requiring a live PyQt runtime.
"""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOLVER_UI_FILE = REPO_ROOT / "src" / "ui" / "builders" / "solver_ui.py"
SOLVER_TAB_FILE = REPO_ROOT / "src" / "ui" / "solver_tab.py"
ANALYSIS_HANDLER_FILE = REPO_ROOT / "src" / "ui" / "handlers" / "analysis_handler.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_named_selection_source_modes_are_defined_in_ui():
    """UI must expose common/A1/A2 source options."""
    src = _read(SOLVER_UI_FILE)

    assert 'named_selection_source_combo.addItem("Common (A1 & A2)", "common")' in src
    assert 'named_selection_source_combo.addItem("Analysis 1 (Base)", "analysis1")' in src
    assert 'named_selection_source_combo.addItem("Analysis 2 (Combine)", "analysis2")' in src


def test_named_selection_dropdown_width_is_increased():
    """Named selection field should use wider sizing for long names."""
    src = _read(SOLVER_UI_FILE)

    assert "named_selection_combo.setMinimumWidth(420)" in src
    assert "named_selection_combo.view().setMinimumWidth(520)" in src


def test_base_analysis_precedence_method_exists():
    """Solver tab must resolve same-name NS scoping to Analysis 1 first."""
    src = _read(SOLVER_TAB_FILE)

    assert "def get_scoping_reader_for_named_selection" in src
    assert "Analysis 1 (base precedence for common name)" in src
    assert "if in_analysis1 and base_reader is not None" in src


def test_analysis_engines_use_tab_scoping_resolver():
    """All engines should use unified tab resolver for named-selection scoping."""
    src = _read(ANALYSIS_HANDLER_FILE)

    assert src.count("get_nodal_scoping_for_selected_named_selection()") >= 3

