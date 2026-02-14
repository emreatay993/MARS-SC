"""
Named-selection and scoping handling for the solver tab.

This module owns named-selection list population and reader/scoping resolution.
"""


class SolverNamedSelectionHandler:
    """Encapsulate named-selection source logic for ``SolverTab``."""

    def __init__(self, tab):
        self.tab = tab

    def on_named_selection_source_changed(self, _index: int) -> None:
        """Refresh list when the named selection source filter changes."""
        if self.tab.analysis1_data and self.tab.analysis2_data:
            self.update_named_selections()

    def get_named_selection_source_mode(self) -> str:
        """Return active source mode for named selection list."""
        mode = self.tab.named_selection_source_combo.currentData()
        if mode in ("common", "analysis1", "analysis2"):
            return mode
        return "common"

    def get_named_selection_sets(self):
        """Get named-selection name sets for both analyses."""
        ns1 = set(self.tab.analysis1_data.named_selections) if self.tab.analysis1_data else set()
        ns2 = set(self.tab.analysis2_data.named_selections) if self.tab.analysis2_data else set()
        return ns1, ns2

    def update_named_selections(self) -> None:
        """Update named-selection dropdown based on active source filter."""
        previous_selection = self.get_selected_named_selection()
        self.tab.named_selection_combo.clear()

        has_analysis1 = self.tab.analysis1_data is not None
        has_analysis2 = self.tab.analysis2_data is not None
        self.tab.named_selection_source_combo.setEnabled(has_analysis1 and has_analysis2)
        self.tab.refresh_ns_button.setEnabled(has_analysis1 or has_analysis2)

        if has_analysis1 and has_analysis2:
            ns1, ns2 = self.get_named_selection_sets()
            source_mode = self.get_named_selection_source_mode()

            if source_mode == "analysis1":
                selections = sorted(ns1)
                empty_text = "(No named selections in Analysis 1)"
            elif source_mode == "analysis2":
                selections = sorted(ns2)
                empty_text = "(No named selections in Analysis 2)"
            else:
                selections = sorted(ns1.intersection(ns2))
                empty_text = "(No common named selections)"

            if selections:
                self.tab.named_selection_combo.addItems(selections)
                self.tab.named_selection_combo.setEnabled(True)
                if previous_selection and previous_selection in selections:
                    self.tab.named_selection_combo.setCurrentText(previous_selection)
            else:
                self.tab.named_selection_combo.addItem(empty_text)
                self.tab.named_selection_combo.setEnabled(False)
        elif has_analysis1:
            if self.tab.analysis1_data.named_selections:
                self.tab.named_selection_combo.addItems(self.tab.analysis1_data.named_selections)
            else:
                self.tab.named_selection_combo.addItem("(No named selections)")
            self.tab.named_selection_combo.setEnabled(False)
        elif has_analysis2:
            if self.tab.analysis2_data.named_selections:
                self.tab.named_selection_combo.addItems(self.tab.analysis2_data.named_selections)
            else:
                self.tab.named_selection_combo.addItem("(No named selections)")
            self.tab.named_selection_combo.setEnabled(False)
        else:
            self.tab.named_selection_combo.addItem("(Load RST files first)")
            self.tab.named_selection_combo.setEnabled(False)

    def get_selected_named_selection(self):
        """Get currently selected named selection name."""
        text = self.tab.named_selection_combo.currentText()
        if not text or text.startswith("("):
            return None
        return text

    def get_scoping_reader_for_named_selection(self, ns_name: str):
        """
        Resolve which analysis reader should provide named-selection node scoping.

        If the same named selection exists in both analyses, Analysis 1 takes
        precedence to avoid mismatched node content.
        """
        if not ns_name:
            raise ValueError("Named selection name is required.")

        base_reader = self.tab.file_handler.base_reader
        combine_reader = self.tab.file_handler.combine_reader
        ns1, ns2 = self.get_named_selection_sets()

        in_analysis1 = ns_name in ns1
        in_analysis2 = ns_name in ns2

        if in_analysis1 and base_reader is not None:
            if in_analysis2:
                return base_reader, "Analysis 1 (base precedence for common name)"
            return base_reader, "Analysis 1"

        if in_analysis2 and combine_reader is not None:
            return combine_reader, "Analysis 2"

        source_mode = self.get_named_selection_source_mode()
        if source_mode == "analysis2" and combine_reader is not None:
            return combine_reader, "Analysis 2"
        if base_reader is not None:
            return base_reader, "Analysis 1"
        if combine_reader is not None:
            return combine_reader, "Analysis 2"

        raise ValueError("DPF readers are not available. Please reload RST files.")

    def get_nodal_scoping_for_selected_named_selection(self):
        """Get nodal scoping for current selection based on active source mode."""
        ns_name = self.get_selected_named_selection()
        if ns_name is None:
            raise ValueError("Please select a valid named selection.")

        reader, _ = self.get_scoping_reader_for_named_selection(ns_name)
        return reader.get_nodal_scoping_from_named_selection(ns_name)
