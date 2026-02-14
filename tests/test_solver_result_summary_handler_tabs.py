import numpy as np

from core.data_models import CombinationResult, SolverConfig
from ui.handlers.solver_result_summary_handler import SolverResultSummaryHandler


class _FakeConsole:
    def __init__(self):
        self.messages = []

    def append(self, message):
        self.messages.append(message)


class _FakeTabWidget:
    def __init__(self):
        self._pages = []
        self._titles = []
        self._visible = []
        self.current_index = 0

    def addTab(self, page, title):
        self._pages.append(page)
        self._titles.append(title)
        self._visible.append(True)
        return len(self._pages) - 1

    def removeTab(self, index):
        self._pages.pop(index)
        self._titles.pop(index)
        self._visible.pop(index)
        if self.current_index >= len(self._pages):
            self.current_index = max(0, len(self._pages) - 1)

    def indexOf(self, page):
        for idx, existing in enumerate(self._pages):
            if existing is page:
                return idx
        return -1

    def setTabText(self, index, title):
        self._titles[index] = title

    def setCurrentIndex(self, index):
        self.current_index = index

    def isTabVisible(self, index):
        return self._visible[index]

    def tabText(self, index):
        return self._titles[index]


class _FakeHistoryPlotTab:
    def __init__(self):
        self.calls = []

    def update_combination_history_plot(self, *args, **kwargs):
        self.calls.append((args, kwargs))


class _FakeEnvelopePlotTab:
    def __init__(self):
        self.max_calls = []
        self.env_calls = []

    def update_max_over_combinations_plot(self, **kwargs):
        self.max_calls.append(kwargs)

    def update_envelope_plot(self, **kwargs):
        self.env_calls.append(kwargs)


class _FakeCombinationTable:
    def __init__(self):
        self.combination_names = ["C1", "C2"]
        self.num_combinations = 2


class _FakeFileHandler:
    base_reader = None


class _FakeSolverTab:
    def __init__(self):
        self.console_textbox = _FakeConsole()
        self.show_output_tab_widget = _FakeTabWidget()
        self.plot_combo_history_tab = _FakeHistoryPlotTab()
        self.plot_max_combo_tab = _FakeEnvelopePlotTab()
        self.plot_min_combo_tab = _FakeEnvelopePlotTab()
        self.combination_table = _FakeCombinationTable()
        self.project_directory = None
        self.file_handler = _FakeFileHandler()
        self._history_popup_requested = False
        self.popup_calls = []

        self.show_output_tab_widget.addTab(self.console_textbox, "Console")

    def _show_history_popup(self, **kwargs):
        self.popup_calls.append(kwargs)


def test_handle_stress_history_result_restores_hidden_tab_as_separate_page():
    tab = _FakeSolverTab()
    handler = SolverResultSummaryHandler(tab)

    # Simulate prior hidden history page that can be selected but is not visible.
    hidden_idx = tab.show_output_tab_widget.addTab(tab.plot_combo_history_tab, "Plot (Combo History)")
    tab.show_output_tab_widget._visible[hidden_idx] = False

    result = CombinationResult(
        node_ids=np.array([101]),
        node_coords=np.array([[0.0, 0.0, 0.0]]),
        result_type="von_mises",
        all_combo_results=np.array([[12.0], [25.0]]),
    )
    result.metadata = {
        "node_id": 101,
        "combination_indices": np.array([0, 1]),
        "stress_values": np.array([12.0, 25.0]),
    }
    config = SolverConfig(combination_history_mode=True, selected_node_id=101)

    handler.handle_stress_history_result(result, config)

    combo_idx = tab.show_output_tab_widget.indexOf(tab.plot_combo_history_tab)
    assert combo_idx >= 0
    assert tab.show_output_tab_widget.isTabVisible(combo_idx) is True
    assert tab.show_output_tab_widget.current_index == combo_idx
    assert tab.show_output_tab_widget.tabText(combo_idx) == "Plot (Combo History)"
    assert tab.show_output_tab_widget.tabText(tab.show_output_tab_widget.current_index) != "Console"
    assert len(tab.plot_combo_history_tab.calls) == 1


def test_handle_stress_envelope_result_hides_min_tab_for_non_min_stress():
    tab = _FakeSolverTab()
    handler = SolverResultSummaryHandler(tab)

    # Pre-populate max/min tabs as if they were shown in a previous run.
    tab.show_output_tab_widget.addTab(tab.plot_max_combo_tab, "Maximum Over Combination")
    tab.show_output_tab_widget.addTab(tab.plot_min_combo_tab, "Minimum Over Combination")

    all_results = np.array(
        [
            [10.0, 20.0],
            [15.0, 18.0],
        ]
    )
    result = CombinationResult(
        node_ids=np.array([1, 2]),
        node_coords=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
        max_over_combo=np.max(all_results, axis=0),
        min_over_combo=np.min(all_results, axis=0),
        combo_of_max=np.argmax(all_results, axis=0),
        combo_of_min=np.argmin(all_results, axis=0),
        result_type="von_mises",
        all_combo_results=all_results,
    )
    config = SolverConfig(combination_history_mode=False)

    handler.handle_stress_envelope_result(result, config)

    assert tab.show_output_tab_widget.indexOf(tab.plot_max_combo_tab) >= 0
    assert tab.show_output_tab_widget.indexOf(tab.plot_min_combo_tab) == -1
    assert len(tab.plot_max_combo_tab.max_calls) == 1


def test_handle_stress_history_result_calls_popup_when_requested():
    tab = _FakeSolverTab()
    tab._history_popup_requested = True
    handler = SolverResultSummaryHandler(tab)

    result = CombinationResult(
        node_ids=np.array([42]),
        node_coords=np.array([[0.0, 0.0, 0.0]]),
        result_type="von_mises",
        all_combo_results=np.array([[1.0], [2.5]]),
    )
    result.metadata = {
        "node_id": 42,
        "combination_indices": np.array([0, 1]),
        "stress_values": np.array([1.0, 2.5]),
    }
    config = SolverConfig(combination_history_mode=True, selected_node_id=42)

    handler.handle_stress_history_result(result, config)

    assert len(tab.popup_calls) == 1
    call = tab.popup_calls[0]
    assert call["node_id"] == 42
    assert call["stress_type"] == "von_mises"
