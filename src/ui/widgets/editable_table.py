"""
Generic editable table widget with copy/paste helpers and trailing blank row.
"""

from contextlib import contextmanager
from typing import Callable, Dict, Optional

import pandas as pd
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QTableWidget,
    QTableWidgetItem,
)


class EditableTableWidget(QTableWidget):
    """QTableWidget with copy/paste support and automatic blank-row management."""

    def __init__(
        self,
        headers,
        initial_rows: int = 30,
        display_formats: Optional[Dict[str, Callable[[float], str]]] = None,
        parent=None,
    ):
        super().__init__(0, len(headers), parent)
        self._headers = headers
        self._initial_rows = initial_rows
        self._display_formats = display_formats or {}
        self._updating = False

        self.setHorizontalHeaderLabels(headers)
        self.horizontalHeader().setStretchLastSection(True)
        self.verticalHeader().setVisible(False)
        self.setSelectionBehavior(QAbstractItemView.SelectItems)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setEditTriggers(
            QAbstractItemView.DoubleClicked
            | QAbstractItemView.EditKeyPressed
            | QAbstractItemView.AnyKeyPressed
        )

        self._ensure_min_rows()
        self.cellChanged.connect(self._on_cell_changed)

    @contextmanager
    def _block_updates(self):
        previous_state = self._updating
        self._updating = True
        self.blockSignals(True)
        try:
            yield
        finally:
            self.blockSignals(False)
            self._updating = previous_state

    def _format_value(self, column: int, value: float) -> str:
        header = self._headers[column]
        formatter = self._display_formats.get(header)
        if formatter is None:
            return f"{value:g}"
        if isinstance(formatter, str):
            if "{" in formatter:
                return formatter.format(value)
            return format(value, formatter)
        return formatter(value)

    def _set_item_value(self, row: int, column: int, value):
        item = self.item(row, column)
        if item is None:
            item = QTableWidgetItem()
            item.setTextAlignment(Qt.AlignCenter)
            super().setItem(row, column, item)

        if value is None or (isinstance(value, str) and value.strip() == ""):
            item.setText("")
            item.setData(Qt.UserRole, None)
            return

        try:
            float_value = float(value)
        except (TypeError, ValueError):
            item.setText(str(value))
            item.setData(Qt.UserRole, None)
            return

        item.setData(Qt.UserRole, float_value)
        item.setText(self._format_value(column, float_value))

    def _append_row(self, values=None):
        row_index = self.rowCount()
        super().insertRow(row_index)
        for col in range(self.columnCount()):
            if values is not None and col < len(values) and pd.notna(values[col]):
                self._set_item_value(row_index, col, values[col])
            else:
                self._set_item_value(row_index, col, None)

    def append_empty_row(self):
        with self._block_updates():
            self._append_row()

    def _ensure_min_rows(self):
        with self._block_updates():
            while self.rowCount() < self._initial_rows:
                self._append_row()

    def _row_is_blank(self, row: int) -> bool:
        for col in range(self.columnCount()):
            item = self.item(row, col)
            if item is not None and item.text().strip():
                return False
        return True

    def _row_has_data(self, row: int) -> bool:
        return not self._row_is_blank(row)

    def ensure_empty_row(self, force=False):
        if self._updating:
            return
        if self.rowCount() == 0:
            self.append_empty_row()
            return
        last_row = self.rowCount() - 1
        if force:
            if not self._row_is_blank(last_row):
                self.append_empty_row()
        else:
            if self._row_has_data(last_row):
                self.append_empty_row()
        self._ensure_min_rows()

    def _on_cell_changed(self, row, column):
        if self._updating:
            return
        item = self.item(row, column)
        if item is None:
            return
        text = item.text().strip()
        with self._block_updates():
            if text == "":
                item.setData(Qt.UserRole, None)
            else:
                try:
                    float_value = float(text)
                except ValueError:
                    item.setData(Qt.UserRole, None)
                else:
                    item.setData(Qt.UserRole, float_value)
                    item.setText(self._format_value(column, float_value))
        self.ensure_empty_row()

    def load_dataframe(self, dataframe: Optional[pd.DataFrame]):
        with self._block_updates():
            self.setRowCount(0)
            if dataframe is not None and not dataframe.empty:
                for _, row in dataframe.iterrows():
                    values = [row.get(header, None) for header in self._headers]
                    self._append_row(values)
            self._ensure_min_rows()
        self.ensure_empty_row(force=True)

    def to_dataframe(self) -> pd.DataFrame:
        data = []
        for row in range(self.rowCount()):
            row_values = []
            has_data = False
            for col in range(self.columnCount()):
                item = self.item(row, col)
                stored_value = item.data(Qt.UserRole) if item is not None else None
                if stored_value is not None:
                    row_values.append(stored_value)
                    has_data = True
                    continue

                text_value = item.text().strip() if item is not None else ""
                if text_value == "":
                    row_values.append(None)
                    continue

                try:
                    numeric_value = float(text_value)
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid numeric entry '{text_value}' in column '{self._headers[col]}'"
                    ) from exc

                row_values.append(numeric_value)
                has_data = True

            if not has_data:
                continue
            data.append(row_values)

        if not data:
            return pd.DataFrame(columns=self._headers)

        df = pd.DataFrame(data, columns=self._headers)
        for header in self._headers:
            df[header] = pd.to_numeric(df[header], errors="raise")
        return df

    def remove_selected_rows(self):
        selection = self.selectedIndexes()
        if not selection:
            return
        rows = sorted({index.row() for index in selection}, reverse=True)
        with self._block_updates():
            for row in rows:
                self.removeRow(row)
            self._ensure_min_rows()
        self.ensure_empty_row(force=True)

    def copy_selection_to_clipboard(self):
        selection = self.selectedIndexes()
        if not selection:
            return
        top_row = min(index.row() for index in selection)
        left_col = min(index.column() for index in selection)
        bottom_row = max(index.row() for index in selection)
        right_col = max(index.column() for index in selection)

        rows_text = []
        for row in range(top_row, bottom_row + 1):
            values = []
            for col in range(left_col, right_col + 1):
                item = self.item(row, col)
                values.append("" if item is None else item.text())
            rows_text.append("\t".join(values))

        QApplication.clipboard().setText("\n".join(rows_text))

    def paste_from_clipboard(self):
        clipboard_text = QApplication.clipboard().text()
        if not clipboard_text:
            return
        selection = self.selectedIndexes()
        if selection:
            start_row = min(index.row() for index in selection)
            start_col = min(index.column() for index in selection)
        else:
            start_row = self.currentRow() if self.currentRow() >= 0 else 0
            start_col = self.currentColumn() if self.currentColumn() >= 0 else 0

        rows = clipboard_text.splitlines()
        with self._block_updates():
            for r_offset, line in enumerate(rows):
                columns = line.split("\t")
                row = start_row + r_offset
                while row >= self.rowCount():
                    self._append_row()
                for c_offset, value in enumerate(columns):
                    col = start_col + c_offset
                    if col >= self.columnCount():
                        continue
                    self._set_item_value(row, col, value.strip())
        self.ensure_empty_row(force=True)

    def keyPressEvent(self, event):
        if event.matches(QKeySequence.Copy):
            self.copy_selection_to_clipboard()
            return
        if event.matches(QKeySequence.Paste):
            self.paste_from_clipboard()
            return
        super().keyPressEvent(event)
