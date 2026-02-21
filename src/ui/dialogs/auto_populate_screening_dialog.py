"""
Dialog for auto-generating screening combinations in the solver table.
"""

from typing import Tuple

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QComboBox,
    QVBoxLayout,
    QWidget,
)

from core.data_models import AnalysisData
from ui.styles.style_constants import DIALOG_STYLE


class AutoPopulateScreeningDialog(QDialog):
    """Collect user inputs for A1-fixed/A2-sweep table generation."""

    def __init__(
        self,
        analysis1_data: AnalysisData,
        analysis2_data: AnalysisData,
        parent: QWidget = None,
    ):
        super().__init__(parent)
        self.analysis1_data = analysis1_data
        self.analysis2_data = analysis2_data
        self.generate_button = None

        self.setWindowTitle("Auto Populate Screening Combinations")
        self.setMinimumWidth(660)
        self.setStyleSheet(DIALOG_STYLE)

        self._build_ui()
        self._populate_a1_steps()
        self._update_preview()
        self._validate_inputs()

    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)

        intro_label = QLabel(
            "<b>Screening generator:</b> create one row per Analysis 2 step while keeping "
            "one selected Analysis 1 step active in every row."
        )
        intro_label.setWordWrap(True)
        main_layout.addWidget(intro_label)

        note_label = QLabel(
            "Uses currently loaded steps. If Skip Substeps was enabled before loading, "
            "the generated rows follow those filtered steps."
        )
        note_label.setWordWrap(True)
        note_label.setStyleSheet("color: #4a4a4a;")
        main_layout.addWidget(note_label)

        input_group = QGroupBox("Generation Inputs")
        input_form = QFormLayout(input_group)

        self.a1_step_combo = QComboBox()
        self.a1_step_combo.setMinimumWidth(320)
        self.a1_step_combo.currentIndexChanged.connect(self._update_preview)
        self.a1_step_combo.currentIndexChanged.connect(self._validate_inputs)

        self.a1_coeff_input = QLineEdit("1.0")
        self.a1_coeff_input.setAlignment(Qt.AlignRight)
        self.a1_coeff_input.setValidator(self._create_coeff_validator(self.a1_coeff_input))
        self.a1_coeff_input.textChanged.connect(self._validate_inputs)

        self.a2_coeff_input = QLineEdit("1.0")
        self.a2_coeff_input.setAlignment(Qt.AlignRight)
        self.a2_coeff_input.setValidator(self._create_coeff_validator(self.a2_coeff_input))
        self.a2_coeff_input.textChanged.connect(self._validate_inputs)

        input_form.addRow("A1 Reference Time/Set:", self.a1_step_combo)
        input_form.addRow("A1 Coefficient (constant):", self.a1_coeff_input)
        input_form.addRow("A2 Coefficient (active step):", self.a2_coeff_input)
        main_layout.addWidget(input_group)

        preview_group = QGroupBox("Generation Preview")
        preview_layout = QVBoxLayout(preview_group)

        self.row_count_label = QLabel("-")
        self.first_row_label = QLabel("-")
        self.first_row_label.setWordWrap(True)
        self.last_row_label = QLabel("-")
        self.last_row_label.setWordWrap(True)
        self.formula_label = QLabel(
            "<code>sigma = (A1_selected * a1_const) + (A2_step_j * a2_const)</code>"
        )
        self.formula_label.setWordWrap(True)

        preview_layout.addWidget(self.row_count_label)
        preview_layout.addWidget(self.first_row_label)
        preview_layout.addWidget(self.last_row_label)
        preview_layout.addWidget(self.formula_label)
        main_layout.addWidget(preview_group)

        self.validation_label = QLabel("")
        self.validation_label.setStyleSheet("color: #a33;")
        self.validation_label.setWordWrap(True)
        main_layout.addWidget(self.validation_label)

        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.generate_button = button_box.button(QDialogButtonBox.Ok)
        if self.generate_button is not None:
            self.generate_button.setText("Generate")
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        button_layout.addWidget(button_box)
        main_layout.addLayout(button_layout)

    @staticmethod
    def _create_coeff_validator(parent: QWidget) -> QDoubleValidator:
        validator = QDoubleValidator(-1e12, 1e12, 12, parent)
        validator.setNotation(QDoubleValidator.StandardNotation)
        return validator

    def _populate_a1_steps(self) -> None:
        self.a1_step_combo.clear()
        for step_id in self.analysis1_data.load_step_ids:
            label = self.analysis1_data.format_time_label(step_id, prefix="A1")
            self.a1_step_combo.addItem(f"{label} (Set {step_id})", step_id)

    def _update_preview(self) -> None:
        a2_steps = list(self.analysis2_data.load_step_ids or [])
        row_count = len(a2_steps)
        self.row_count_label.setText(f"Rows to create: <b>{row_count}</b>")

        if row_count == 0 or self.a1_step_combo.currentIndex() < 0:
            self.first_row_label.setText("First generated name: -")
            self.last_row_label.setText("Last generated name: -")
            return

        a1_label = self.analysis1_data.format_time_label(
            self.get_selected_a1_step_id(), prefix="A1"
        )
        first_a2 = self.analysis2_data.format_time_label(a2_steps[0], prefix="A2")
        last_a2 = self.analysis2_data.format_time_label(a2_steps[-1], prefix="A2")
        self.first_row_label.setText(
            f"First generated name: <b>{self._build_row_name(a1_label, first_a2)}</b>"
        )
        self.last_row_label.setText(
            f"Last generated name: <b>{self._build_row_name(a1_label, last_a2)}</b>"
        )

    def _validate_inputs(self) -> None:
        errors = []
        if self.a1_step_combo.currentIndex() < 0:
            errors.append("Select an A1 reference time/set.")

        if self._parse_float(self.a1_coeff_input.text()) is None:
            errors.append("Enter a valid A1 coefficient.")

        if self._parse_float(self.a2_coeff_input.text()) is None:
            errors.append("Enter a valid A2 coefficient.")

        if not self.analysis2_data.load_step_ids:
            errors.append("No Analysis 2 steps are available to generate rows.")

        if errors:
            self.validation_label.setText(" ".join(errors))
            if self.generate_button is not None:
                self.generate_button.setEnabled(False)
            return

        self.validation_label.setText("")
        if self.generate_button is not None:
            self.generate_button.setEnabled(True)

    def get_selected_a1_step_id(self) -> int:
        return int(self.a1_step_combo.currentData())

    def get_inputs(self) -> Tuple[int, float, float]:
        step_id = self.get_selected_a1_step_id()
        a1_coeff = self._parse_float(self.a1_coeff_input.text())
        a2_coeff = self._parse_float(self.a2_coeff_input.text())
        if a1_coeff is None or a2_coeff is None:
            raise ValueError("Invalid coefficient input.")
        return step_id, a1_coeff, a2_coeff

    @staticmethod
    def _parse_float(text: str):
        try:
            return float(text.strip())
        except (ValueError, AttributeError):
            return None

    @staticmethod
    def _build_row_name(a1_label: str, a2_label: str) -> str:
        return f"Screen: {a1_label} + {a2_label}"

