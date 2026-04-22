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
FILE_HANDLER_FILE = REPO_ROOT / "src" / "ui" / "handlers" / "file_handler.py"
APP_CONTROLLER_FILE = REPO_ROOT / "src" / "ui" / "application_controller.py"


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
    assert "named_selection_sources=self.get_named_selection_sources()" in reader_src
    assert "named_selection_sources: Dict[str, str]" in data_src


def test_elemental_named_selection_entries_keep_raw_names():
    """UI labels may mark elemental selections, but item data keeps raw names."""
    src = _read(NAMED_SELECTION_HANDLER_FILE)

    assert 'suffix_parts.append("Elemental")' in src
    assert "CDB" in src
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


def test_cdb_named_selection_import_controls_are_wired():
    """Solver UI should expose one compact shared CDB named-selection import."""
    ui_src = _read(SOLVER_UI_FILE)
    tab_src = _read(SOLVER_TAB_FILE)
    handler_src = _read(FILE_HANDLER_FILE)
    app_src = _read(APP_CONTROLLER_FILE)

    assert 'import_cdb_button = QPushButton("Import CDB")' in ui_src
    assert "ns_row.addWidget(import_cdb_button)" in ui_src
    assert "base_cdb_path" not in ui_src
    assert "combine_cdb_path" not in ui_src
    assert "self.import_cdb_button.clicked.connect(self.file_handler.select_cdb_file)" in tab_src
    assert "select_cdb_file" in handler_src
    assert "_load_shared_cdb_named_selections" in handler_src
    assert "select_base_cdb_file" not in handler_src
    assert "select_combine_cdb_file" not in handler_src
    assert "CDBNamedSelectionReader.from_file" in handler_src
    assert "*.cdb" in app_src


def test_txt_named_selection_import_controls_are_wired():
    """Solver UI should expose one-file TXT nodal named-selection import."""
    ui_src = _read(SOLVER_UI_FILE)
    tab_src = _read(SOLVER_TAB_FILE)
    handler_src = _read(FILE_HANDLER_FILE)

    assert 'import_txt_ns_button = QPushButton("Import TXT NS")' in ui_src
    assert "ns_row.addWidget(import_txt_ns_button)" in ui_src
    assert "self.import_txt_ns_button.clicked.connect(self.file_handler.select_txt_named_selection_file)" in tab_src
    assert "on_txt_named_selection_loaded" in tab_src
    assert "select_txt_named_selection_file" in handler_src
    assert "QFileDialog.getOpenFileNames" in handler_src
    assert "Multiple file selection is not supported. Please select one TXT file for each import." in handler_src
    assert "TXTNamedSelectionReader.from_file" in handler_src


def test_named_selection_type_filter_controls_are_defined():
    """UI and handler should support source/location filtering of named selections."""
    ui_src = _read(SOLVER_UI_FILE)
    tab_src = _read(SOLVER_TAB_FILE)
    handler_src = _read(NAMED_SELECTION_HANDLER_FILE)

    assert 'named_selection_type_filter_combo.addItem("Nodal", "nodal")' in ui_src
    assert 'named_selection_type_filter_combo.addItem("Elemental", "elemental")' in ui_src
    assert 'named_selection_type_filter_combo.addItem("Body", "body")' in ui_src
    assert 'named_selection_type_filter_combo.addItem("Face", "face")' in ui_src
    assert 'named_selection_type_filter_combo.addItem("Vertex", "vertex")' in ui_src
    assert 'named_selection_type_filter_combo.addItem("Imported", "imported")' in ui_src
    assert "self.named_selection_type_filter_combo.currentIndexChanged.connect" in tab_src
    assert "def on_named_selection_type_filter_changed" in handler_src
    assert "def _matches_named_selection_type_filter" in handler_src
    assert '"txt" in source' in handler_src


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


class _MutableFakeCombo:
    def __init__(self, items=None, current_index=0):
        self._items = list(items or [])
        self._current_index = current_index
        self.enabled = True

    def currentData(self):
        if not self._items:
            return None
        return self._items[self._current_index][1]

    def currentText(self):
        if not self._items:
            return ""
        return self._items[self._current_index][0]

    def currentIndex(self):
        return self._current_index

    def clear(self):
        self._items = []
        self._current_index = -1

    def addItem(self, text, data=None):
        self._items.append((text, data))
        if self._current_index < 0:
            self._current_index = 0

    def itemText(self, index):
        return self._items[index][0]

    def itemData(self, index):
        return self._items[index][1]

    def count(self):
        return len(self._items)

    def setCurrentIndex(self, index):
        self._current_index = index

    def setEnabled(self, enabled):
        self.enabled = enabled


class _FakeAnalysisData:
    def __init__(self):
        self.named_selections = ["NODE_NS", "ELEM_NS", "TXT_NS"]
        self.named_selection_locations = {
            "NODE_NS": "nodal",
            "ELEM_NS": "elemental",
            "TXT_NS": "nodal",
        }
        self.named_selection_sources = {
            "NODE_NS": "rst",
            "ELEM_NS": "rst",
            "TXT_NS": "txt",
        }


class _FilterFakeTab:
    def __init__(self, type_filter):
        self.analysis1_data = _FakeAnalysisData()
        self.analysis2_data = _FakeAnalysisData()
        self.named_selection_source_combo = _MutableFakeCombo([("Analysis 1", "analysis1")])
        self.named_selection_type_filter_combo = _MutableFakeCombo([("Filter", type_filter)])
        self.named_selection_combo = _MutableFakeCombo()
        self.refresh_ns_button = _MutableFakeCombo()
        self.import_cdb_button = _MutableFakeCombo()
        self.import_txt_ns_button = _MutableFakeCombo()


def test_named_selection_type_filter_limits_dropdown_to_imported_sources():
    """Imported filter should show supplemental TXT/CDB selections only."""
    tab = _FilterFakeTab("imported")
    handler = SolverNamedSelectionHandler(tab)

    handler.update_named_selections()

    items = [tab.named_selection_combo.itemData(i) for i in range(tab.named_selection_combo.count())]
    assert items == ["TXT_NS"]


def test_named_selection_type_filter_limits_dropdown_to_elemental_locations():
    """Elemental filter should use named-selection location metadata."""
    tab = _FilterFakeTab("elemental")
    handler = SolverNamedSelectionHandler(tab)

    handler.update_named_selections()

    items = [tab.named_selection_combo.itemData(i) for i in range(tab.named_selection_combo.count())]
    assert items == ["ELEM_NS"]
