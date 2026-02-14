"""
Combination-table handling for the solver tab.

This module owns delegate setup, coefficient highlighting, row operations,
and conversion between UI table content and ``CombinationTableData``.
"""

from typing import Optional

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QBrush, QColor, QDoubleValidator
from PyQt5.QtWidgets import QLineEdit, QStyledItemDelegate, QTableWidgetItem

from core.data_models import CombinationTableData
from ui.styles.style_constants import NONZERO_COEFFICIENT_BG_COLOR


class NumericDelegate(QStyledItemDelegate):
    """Delegate for coefficient columns: only numeric (float) input allowed."""

    def createEditor(self, parent, option, index):
        editor = QLineEdit(parent)
        validator = QDoubleValidator()
        validator.setNotation(QDoubleValidator.StandardNotation)
        editor.setValidator(validator)
        return editor

    def setEditorData(self, editor, index):
        value = index.model().data(index, Qt.EditRole)
        if value:
            editor.setText(str(value))
        else:
            editor.setText("0.0")

    def setModelData(self, editor, model, index):
        text = editor.text()
        try:
            value = float(text) if text else 0.0
            model.setData(index, str(value), Qt.EditRole)
        except ValueError:
            model.setData(index, "0.0", Qt.EditRole)


class ReadOnlyDelegate(QStyledItemDelegate):
    """Delegate for the Type column (read-only; only Linear is supported)."""

    def createEditor(self, parent, option, index):
        return None


class SolverCombinationTableHandler:
    """Encapsulate all combination-table behavior for ``SolverTab``."""

    def __init__(self, tab):
        self.tab = tab
        self._readonly_delegate: Optional[ReadOnlyDelegate] = None
        self._numeric_delegate: Optional[NumericDelegate] = None

    def setup_table_delegates(self) -> None:
        """Install delegates for table validation and edit restrictions."""
        self._readonly_delegate = ReadOnlyDelegate(self.tab.combo_table)
        self.tab.combo_table.setItemDelegateForColumn(1, self._readonly_delegate)

        self._numeric_delegate = NumericDelegate(self.tab.combo_table)
        self.apply_numeric_delegate_to_columns()

        self.tab.combo_table.model().columnsInserted.connect(self.apply_numeric_delegate_to_columns)

    def apply_numeric_delegate_to_columns(self) -> None:
        """Apply numeric delegate to all coefficient columns (column 2 onwards)."""
        if self._numeric_delegate is None:
            return
        for col in range(2, self.tab.combo_table.columnCount()):
            self.tab.combo_table.setItemDelegateForColumn(col, self._numeric_delegate)

    def update_combination_table_columns(self) -> None:
        """Update table columns based on currently loaded RST analyses."""
        columns = ["Combination Name", "Type"]

        if self.tab.analysis1_data:
            for step_id in self.tab.analysis1_data.load_step_ids:
                columns.append(self.tab.analysis1_data.format_time_label(step_id, prefix="A1"))

        if self.tab.analysis2_data:
            for step_id in self.tab.analysis2_data.load_step_ids:
                columns.append(self.tab.analysis2_data.format_time_label(step_id, prefix="A2"))

        current_rows = self.tab.combo_table.rowCount()
        self.tab.combo_table.setColumnCount(len(columns))
        self.tab.combo_table.setHorizontalHeaderLabels(columns)

        for row in range(current_rows):
            type_item = self.tab.combo_table.item(row, 1)
            if type_item:
                type_item.setFlags(type_item.flags() & ~Qt.ItemIsEditable)
                type_item.setToolTip("Only 'Linear' combination type is currently supported")

            for col in range(2, len(columns)):
                if self.tab.combo_table.item(row, col) is None:
                    self.tab.combo_table.setItem(row, col, QTableWidgetItem("0.0"))

        self.apply_numeric_delegate_to_columns()

    def update_coefficient_cell_highlight(self, item: Optional[QTableWidgetItem]) -> None:
        """Color non-zero coefficient cells for fast visual scanning."""
        if item is None:
            return

        try:
            value = float(item.text())
            is_nonzero = value != 0.0
        except (ValueError, TypeError):
            is_nonzero = False

        if is_nonzero:
            item.setBackground(QBrush(QColor(NONZERO_COEFFICIENT_BG_COLOR)))
        else:
            item.setBackground(QBrush())

    def on_coefficient_cell_changed(self, row: int, column: int) -> None:
        """React to user edits and keep coefficient highlighting up to date."""
        if column < 2:
            return
        item = self.tab.combo_table.item(row, column)
        self.update_coefficient_cell_highlight(item)

    def add_table_row(self) -> None:
        """Add a row with default combination values."""
        row_count = self.tab.combo_table.rowCount()
        self.tab.combo_table.insertRow(row_count)
        self.tab.combo_table.setItem(row_count, 0, QTableWidgetItem(f"Combination {row_count + 1}"))
        self.tab.combo_table.setItem(row_count, 1, self._create_type_item("Linear"))

        for col in range(2, self.tab.combo_table.columnCount()):
            self.tab.combo_table.setItem(row_count, col, QTableWidgetItem("0.0"))

    def delete_table_row(self) -> None:
        """Delete selected row or fallback to deleting the last row."""
        selected_rows = self.tab.combo_table.selectionModel().selectedRows()
        if selected_rows:
            self.tab.combo_table.removeRow(selected_rows[0].row())
        elif self.tab.combo_table.rowCount() > 1:
            self.tab.combo_table.removeRow(self.tab.combo_table.rowCount() - 1)

    def get_combination_table_data(self) -> Optional[CombinationTableData]:
        """Extract ``CombinationTableData`` from the table widget."""
        if not self.tab.analysis1_data or not self.tab.analysis2_data:
            return None

        row_count = self.tab.combo_table.rowCount()
        if row_count == 0:
            return None

        names = []
        types = []
        a1_coeffs = []
        a2_coeffs = []

        n_a1 = len(self.tab.analysis1_data.load_step_ids)
        n_a2 = len(self.tab.analysis2_data.load_step_ids)

        for row in range(row_count):
            name_item = self.tab.combo_table.item(row, 0)
            type_item = self.tab.combo_table.item(row, 1)

            names.append(name_item.text() if name_item else f"Combination {row + 1}")
            types.append(type_item.text() if type_item else "Linear")

            a1_row = []
            for col in range(2, 2 + n_a1):
                item = self.tab.combo_table.item(row, col)
                try:
                    a1_row.append(float(item.text()) if item else 0.0)
                except ValueError:
                    a1_row.append(0.0)
            a1_coeffs.append(a1_row)

            a2_row = []
            for col in range(2 + n_a1, 2 + n_a1 + n_a2):
                item = self.tab.combo_table.item(row, col)
                try:
                    a2_row.append(float(item.text()) if item else 0.0)
                except ValueError:
                    a2_row.append(0.0)
            a2_coeffs.append(a2_row)

        return CombinationTableData(
            combination_names=names,
            combination_types=types,
            analysis1_coeffs=np.array(a1_coeffs),
            analysis2_coeffs=np.array(a2_coeffs),
            analysis1_step_ids=list(self.tab.analysis1_data.load_step_ids),
            analysis2_step_ids=list(self.tab.analysis2_data.load_step_ids),
        )

    def set_combination_table_data(self, data: CombinationTableData) -> None:
        """Populate the table widget from ``CombinationTableData``."""
        self.tab.combination_table = data

        self.update_combination_table_columns()
        self.tab.combo_table.setRowCount(data.num_combinations)

        for row in range(data.num_combinations):
            self.tab.combo_table.setItem(row, 0, QTableWidgetItem(data.combination_names[row]))
            self.tab.combo_table.setItem(row, 1, self._create_type_item(data.combination_types[row]))

            for i, coeff in enumerate(data.analysis1_coeffs[row]):
                item = QTableWidgetItem(str(coeff))
                self.update_coefficient_cell_highlight(item)
                self.tab.combo_table.setItem(row, 2 + i, item)

            offset = 2 + data.num_analysis1_steps
            for i, coeff in enumerate(data.analysis2_coeffs[row]):
                item = QTableWidgetItem(str(coeff))
                self.update_coefficient_cell_highlight(item)
                self.tab.combo_table.setItem(row, offset + i, item)

    @staticmethod
    def _create_type_item(value: str) -> QTableWidgetItem:
        item = QTableWidgetItem(value)
        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
        item.setToolTip("Only 'Linear' combination type is currently supported")
        return item
