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

    def on_named_selection_type_filter_changed(self, _index: int) -> None:
        """Refresh list when the named selection type/source filter changes."""
        if self.tab.analysis1_data or self.tab.analysis2_data:
            self.update_named_selections()

    def get_named_selection_source_mode(self) -> str:
        """Return active source mode for named selection list."""
        mode = self.tab.named_selection_source_combo.currentData()
        if mode in ("common", "analysis1", "analysis2"):
            return mode
        return "common"

    def get_named_selection_type_filter(self) -> str:
        """Return active named-selection location/source filter."""
        combo = getattr(self.tab, "named_selection_type_filter_combo", None)
        if combo is None:
            return "all"

        filter_value = combo.currentData()
        if filter_value in ("all", "nodal", "elemental", "body", "face", "vertex", "imported"):
            return filter_value
        return "all"

    def get_named_selection_sets(self):
        """Get named-selection name sets for both analyses."""
        ns1 = set(self.tab.analysis1_data.named_selections) if self.tab.analysis1_data else set()
        ns2 = set(self.tab.analysis2_data.named_selections) if self.tab.analysis2_data else set()
        return ns1, ns2

    def _get_display_location(self, ns_name: str) -> str:
        """Return the location that matches the active scoping source precedence."""
        data = self._get_display_analysis_data()
        if data is None:
            return "unknown"

        locations = getattr(data, "named_selection_locations", {}) or {}
        return locations.get(ns_name, "unknown")

    def _get_display_source(self, ns_name: str) -> str:
        """Return the metadata source that matches active scoping precedence."""
        data = self._get_display_analysis_data()
        if data is None:
            return "unknown"

        sources = getattr(data, "named_selection_sources", {}) or {}
        return sources.get(ns_name, "rst")

    def _get_display_analysis_data(self):
        """Return analysis data used for labels under the current source mode."""
        source_mode = self.get_named_selection_source_mode()

        if source_mode == "analysis2":
            return self.tab.analysis2_data or self.tab.analysis1_data
        return self.tab.analysis1_data or self.tab.analysis2_data

    def _format_named_selection_label(self, ns_name: str) -> str:
        """Build the dropdown label while keeping raw name in item data."""
        location = self._get_display_location(ns_name)
        source = self._get_display_source(ns_name)
        suffix_parts = []
        if "cdb" in source:
            suffix_parts.append("CDB")
        if "txt" in source:
            suffix_parts.append("TXT")
        if location == "elemental":
            suffix_parts.append("Elemental")
        elif location in ("body", "face", "vertex"):
            suffix_parts.append(location.title())
        if suffix_parts:
            return f"{ns_name} ({' '.join(suffix_parts)})"
        return ns_name

    def _matches_named_selection_type_filter(self, ns_name: str) -> bool:
        """Return True if ``ns_name`` passes the active type/source filter."""
        filter_value = self.get_named_selection_type_filter()
        if filter_value == "all":
            return True

        location = self._get_display_location(ns_name)
        source = self._get_display_source(ns_name)

        if filter_value == "imported":
            return "cdb" in source or "txt" in source or source == "supplemental"

        return location == filter_value

    def _filter_named_selections(self, selections):
        """Apply the active type/source filter to a sequence of names."""
        return [
            ns_name
            for ns_name in selections
            if self._matches_named_selection_type_filter(ns_name)
        ]

    def _add_named_selection_items(self, selections) -> None:
        """Add named-selection entries with raw names stored as item data."""
        for ns_name in selections:
            self.tab.named_selection_combo.addItem(
                self._format_named_selection_label(ns_name),
                ns_name,
            )

    def _set_current_named_selection(self, ns_name: str) -> None:
        """Restore a previous raw named-selection value after repopulating."""
        combo = self.tab.named_selection_combo
        for index in range(combo.count()):
            if combo.itemData(index) == ns_name:
                combo.setCurrentIndex(index)
                return

    def update_named_selections(self) -> None:
        """Update named-selection dropdown based on active source filter."""
        previous_selection = self.get_selected_named_selection()
        self.tab.named_selection_combo.clear()

        has_analysis1 = self.tab.analysis1_data is not None
        has_analysis2 = self.tab.analysis2_data is not None
        self.tab.named_selection_source_combo.setEnabled(has_analysis1 and has_analysis2)
        if hasattr(self.tab, "named_selection_type_filter_combo"):
            self.tab.named_selection_type_filter_combo.setEnabled(has_analysis1 or has_analysis2)
        self.tab.refresh_ns_button.setEnabled(has_analysis1 or has_analysis2)
        if hasattr(self.tab, "import_cdb_button"):
            self.tab.import_cdb_button.setEnabled(has_analysis1 and has_analysis2)
        if hasattr(self.tab, "import_txt_ns_button"):
            self.tab.import_txt_ns_button.setEnabled(has_analysis1 and has_analysis2)

        if has_analysis1 and has_analysis2:
            ns1, ns2 = self.get_named_selection_sets()
            source_mode = self.get_named_selection_source_mode()

            if source_mode == "analysis1":
                selections = self._filter_named_selections(sorted(ns1))
                empty_text = "(No named selections in Analysis 1)"
            elif source_mode == "analysis2":
                selections = self._filter_named_selections(sorted(ns2))
                empty_text = "(No named selections in Analysis 2)"
            else:
                selections = self._filter_named_selections(sorted(ns1.intersection(ns2)))
                empty_text = "(No common named selections)"

            if selections:
                self._add_named_selection_items(selections)
                self.tab.named_selection_combo.setEnabled(True)
                if previous_selection and previous_selection in selections:
                    self._set_current_named_selection(previous_selection)
            else:
                self.tab.named_selection_combo.addItem(empty_text)
                self.tab.named_selection_combo.setEnabled(False)
        elif has_analysis1:
            selections = self._filter_named_selections(self.tab.analysis1_data.named_selections)
            if selections:
                self._add_named_selection_items(selections)
            else:
                self.tab.named_selection_combo.addItem("(No named selections)")
            self.tab.named_selection_combo.setEnabled(False)
        elif has_analysis2:
            selections = self._filter_named_selections(self.tab.analysis2_data.named_selections)
            if selections:
                self._add_named_selection_items(selections)
            else:
                self.tab.named_selection_combo.addItem("(No named selections)")
            self.tab.named_selection_combo.setEnabled(False)
        else:
            self.tab.named_selection_combo.addItem("(Load RST files first)")
            self.tab.named_selection_combo.setEnabled(False)

    def get_selected_named_selection(self):
        """Get currently selected named selection name."""
        combo = self.tab.named_selection_combo
        text = combo.currentText().strip()
        if not text or text.startswith("("):
            return None

        current_index = combo.currentIndex()
        if current_index >= 0 and combo.itemText(current_index) == text:
            raw_name = combo.itemData(current_index)
            return raw_name if raw_name else text

        for index in range(combo.count()):
            if combo.itemText(index) == text:
                raw_name = combo.itemData(index)
                combo.setCurrentIndex(index)
                return raw_name if raw_name else text

        for index in range(combo.count()):
            raw_name = combo.itemData(index)
            if raw_name == text:
                combo.setCurrentIndex(index)
                return raw_name

        return None

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
