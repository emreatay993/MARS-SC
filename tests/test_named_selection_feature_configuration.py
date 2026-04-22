"""
Static configuration tests for named-selection feature updates.

These checks validate source wiring without requiring a live PyQt runtime.
"""

from pathlib import Path

from ui.handlers.solver_named_selection_handler import SolverNamedSelectionHandler


REPO_ROOT = Path(__file__).resolve().parents[1]
SOLVER_UI_FILE = REPO_ROOT / "src" / "ui" / "builders" / "solver_ui.py"
SOLVER_TAB_FILE = REPO_ROOT / "src" / "ui" / "solver_tab.py"
ANALYSIS_EXECUTOR_FILE = REPO_ROOT / "src" / "ui" / "handlers" / "solver_analysis_executor.py"
NAMED_SELECTION_HANDLER_FILE = REPO_ROOT / "src" / "ui" / "handlers" / "solver_named_selection_handler.py"
DPF_READER_FILE = REPO_ROOT / "src" / "file_io" / "dpf_reader.py"
DATA_MODELS_FILE = REPO_ROOT / "src" / "core" / "data_models.py"


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
    tab_src = _read(SOLVER_TAB_FILE)
    handler_src = _read(NAMED_SELECTION_HANDLER_FILE)

    assert "def get_scoping_reader_for_named_selection" in tab_src
    assert "def get_scoping_reader_for_named_selection" in handler_src
    assert "Analysis 1 (base precedence for common name)" in handler_src
    assert "if in_analysis1 and base_reader is not None" in handler_src


def test_analysis_engines_use_tab_scoping_resolver():
    """All engines should use unified tab resolver for named-selection scoping."""
    src = _read(ANALYSIS_EXECUTOR_FILE)

    assert "def _get_common_inputs" in src
    assert "get_nodal_scoping_for_selected_named_selection()" in src
    assert src.count("_get_common_inputs(") >= 3


def test_named_selection_locations_are_carried_from_reader_to_analysis_data():
    """Reader metadata should expose nodal/elemental named-selection locations."""
    reader_src = _read(DPF_READER_FILE)
    data_src = _read(DATA_MODELS_FILE)

    assert "def get_named_selection_locations" in reader_src
    assert "named_selection_locations=self.get_named_selection_locations()" in reader_src
    assert "named_selection_locations: Dict[str, str]" in data_src


def test_elemental_named_selection_entries_keep_raw_names():
    """UI labels may mark elemental selections, but item data keeps raw names."""
    src = _read(NAMED_SELECTION_HANDLER_FILE)

    assert "(Elemental)" in src
    assert "addItem(" in src
    assert "currentData()" in src


def test_named_selection_combo_is_searchable():
    """Named selection combo should use contains-filter completion."""
    src = _read(SOLVER_UI_FILE)

    assert "class SearchableComboBox" in src
    assert "named_selection_combo = SearchableComboBox()" in src
    assert "setEditable(True)" in src
    assert "setInsertPolicy(QComboBox.NoInsert)" in src
    assert "setCompletionMode(QCompleter.PopupCompletion)" in src
    assert "setFilterMode(Qt.MatchContains)" in src
    assert "setCaseSensitivity(Qt.CaseInsensitive)" in src


class _FakeCombo:
    def __init__(self, items, current_index=0, edit_text=None):
        self._items = items
        self._current_index = current_index
        self._edit_text = edit_text

    def currentText(self):
        if self._edit_text is not None:
            return self._edit_text
        return self.itemText(self._current_index)

    def currentData(self):
        return self.itemData(self._current_index)

    def currentIndex(self):
        return self._current_index

    def itemText(self, index):
        return self._items[index][0]

    def itemData(self, index):
        return self._items[index][1]

    def count(self):
        return len(self._items)

    def setCurrentIndex(self, index):
        self._current_index = index
        self._edit_text = None


class _FakeTab:
    def __init__(self, combo):
        self.named_selection_combo = combo


def test_selected_named_selection_accepts_display_or_raw_name():
    """Editable combo text should resolve to the raw named-selection value."""
    items = [("RIB_PANEL (Elemental)", "RIB_PANEL")]

    handler = SolverNamedSelectionHandler(_FakeTab(_FakeCombo(items)))
    assert handler.get_selected_named_selection() == "RIB_PANEL"

    handler = SolverNamedSelectionHandler(_FakeTab(_FakeCombo(items, edit_text="RIB_PANEL")))
    assert handler.get_selected_named_selection() == "RIB_PANEL"


def test_selected_named_selection_rejects_uncommitted_partial_search():
    """A partial typed filter should not reuse stale item data as a selection."""
    items = [("RIB_PANEL (Elemental)", "RIB_PANEL")]
    handler = SolverNamedSelectionHandler(_FakeTab(_FakeCombo(items, edit_text="RIB")))

    assert handler.get_selected_named_selection() is None
