"""Modal analysis main window."""

from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Optional

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QTextCursor
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QCheckBox,
    QComboBox,
    QProgressBar,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from modal_gui import dpf_modal_extractor
from modal_gui.worker import ExtractionJob, ModalExtractionWorker

try:
    from src.ui.styles.style_constants import (
        BUTTON_STYLE,
        CHECKBOX_STYLE,
        CONSOLE_STYLE,
        GROUP_BOX_STYLE,
        PROGRESS_BAR_STYLE,
        READONLY_INPUT_STYLE,
    )
except Exception:
    BUTTON_STYLE = ""
    CHECKBOX_STYLE = ""
    CONSOLE_STYLE = ""
    GROUP_BOX_STYLE = ""
    PROGRESS_BAR_STYLE = ""
    READONLY_INPUT_STYLE = ""

try:
    from src.ui.widgets.console import Logger
except Exception:
    Logger = None


class ModalMainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Modal DPF GUI")
        self.resize(1000, 720)

        self.worker: Optional[ModalExtractionWorker] = None
        self.stdout_logger = None

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        main_layout.addWidget(self._build_file_group())
        main_layout.addWidget(self._build_selection_group())
        main_layout.addWidget(self._build_mode_group())
        main_layout.addWidget(self._build_extract_group())
        main_layout.addWidget(self._build_output_group())
        main_layout.addWidget(self._build_progress_group())
        main_layout.addWidget(self._build_console_group())
        main_layout.addLayout(self._build_actions())

        self._apply_styles()
        self._configure_logger()

    def _build_file_group(self) -> QGroupBox:
        box = QGroupBox("RST File")
        layout = QHBoxLayout(box)

        self.rst_path_edit = QLineEdit()
        self.rst_path_edit.setReadOnly(True)
        self.rst_browse_btn = QPushButton("Browse...")
        self.rst_browse_btn.clicked.connect(self._browse_rst)

        layout.addWidget(self.rst_path_edit, 1)
        layout.addWidget(self.rst_browse_btn)
        return box

    def _build_selection_group(self) -> QGroupBox:
        box = QGroupBox("Named Selection")
        layout = QHBoxLayout(box)

        self.named_selection_combo = QComboBox()
        self.named_selection_combo.addItem("All Nodes")
        self.named_selection_combo.currentTextChanged.connect(self._on_named_selection_changed)
        
        self.selection_node_count_label = QLabel("")
        self.model_info_label = QLabel("No model loaded")

        layout.addWidget(self.named_selection_combo, 1)
        layout.addWidget(self.selection_node_count_label)
        layout.addWidget(QLabel(" | "))
        layout.addWidget(self.model_info_label)
        return box

    def _build_mode_group(self) -> QGroupBox:
        box = QGroupBox("Modes")
        layout = QHBoxLayout(box)

        self.mode_count_label = QLabel("Detected modes: 0")
        self.mode_count_spin = QSpinBox()
        self.mode_count_spin.setMinimum(1)
        self.mode_count_spin.setMaximum(1)
        self.mode_count_spin.setValue(1)
        
        self.specific_mode_check = QCheckBox("Extract specific mode:")
        self.specific_mode_spin = QSpinBox()
        self.specific_mode_spin.setMinimum(1)
        self.specific_mode_spin.setMaximum(1)
        self.specific_mode_spin.setValue(1)
        self.specific_mode_spin.setEnabled(False)
        self.specific_mode_check.toggled.connect(self._toggle_specific_mode)

        layout.addWidget(self.mode_count_label)
        layout.addWidget(QLabel("Export first N modes:"))
        layout.addWidget(self.mode_count_spin)
        layout.addSpacing(20)
        layout.addWidget(self.specific_mode_check)
        layout.addWidget(self.specific_mode_spin)
        layout.addStretch(1)
        return box
    
    def _toggle_specific_mode(self, checked: bool) -> None:
        self.specific_mode_spin.setEnabled(checked)
        self.mode_count_spin.setEnabled(not checked)

    def _build_extract_group(self) -> QGroupBox:
        box = QGroupBox("Results")
        layout = QHBoxLayout(box)

        self.stress_check = QCheckBox("Stress tensor")
        self.strain_check = QCheckBox("Strain tensor")
        self.disp_check = QCheckBox("Directional deformation")
        self.stress_check.setChecked(True)
        self.disp_check.setChecked(True)
        
        self.backend_combo = QComboBox()
        self.backend_combo.addItems(["Auto (PyMAPDL)", "DPF only", "PyMAPDL Reader"])
        self.backend_combo.setToolTip(
            "Auto: Uses PyMAPDL Reader (faster, stable for many modes)\n"
            "DPF only: Uses ANSYS DPF (slower, may crash on large mode counts)\n"
            "PyMAPDL Reader: Direct RST file reading, recommended"
        )

        layout.addWidget(self.stress_check)
        layout.addWidget(self.strain_check)
        layout.addWidget(self.disp_check)
        layout.addSpacing(20)
        layout.addWidget(QLabel("Backend:"))
        layout.addWidget(self.backend_combo)
        layout.addStretch(1)
        return box

    def _build_output_group(self) -> QGroupBox:
        box = QGroupBox("Output Folder")
        layout = QHBoxLayout(box)

        self.output_dir_edit = QLineEdit()
        self.output_browse_btn = QPushButton("Browse...")
        self.output_browse_btn.clicked.connect(self._browse_output_dir)

        layout.addWidget(self.output_dir_edit, 1)
        layout.addWidget(self.output_browse_btn)
        return box

    def _build_progress_group(self) -> QGroupBox:
        box = QGroupBox("Progress")
        layout = QHBoxLayout(box)

        self.progress_bar = QProgressBar()
        self.progress_status = QLabel("Idle")

        layout.addWidget(self.progress_bar, 1)
        layout.addWidget(self.progress_status)
        return box

    def _build_console_group(self) -> QGroupBox:
        box = QGroupBox("Console")
        layout = QVBoxLayout(box)

        self.console = QTextEdit()
        self.console.setReadOnly(True)
        layout.addWidget(self.console)
        return box

    def _build_actions(self) -> QHBoxLayout:
        layout = QHBoxLayout()

        self.estimate_btn = QPushButton("Estimate Time")
        self.estimate_btn.setToolTip("Run a quick test to estimate extraction time")
        self.extract_btn = QPushButton("Extract")
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)

        self.estimate_btn.clicked.connect(self._estimate_extraction_time)
        self.extract_btn.clicked.connect(self._start_extraction)
        self.cancel_btn.clicked.connect(self._cancel_extraction)

        layout.addStretch(1)
        layout.addWidget(self.estimate_btn)
        layout.addWidget(self.extract_btn)
        layout.addWidget(self.cancel_btn)
        return layout

    def _apply_styles(self) -> None:
        self.rst_path_edit.setStyleSheet(READONLY_INPUT_STYLE)
        self.rst_browse_btn.setStyleSheet(BUTTON_STYLE)
        self.output_browse_btn.setStyleSheet(BUTTON_STYLE)
        self.estimate_btn.setStyleSheet(BUTTON_STYLE)
        self.extract_btn.setStyleSheet(BUTTON_STYLE)
        self.cancel_btn.setStyleSheet(BUTTON_STYLE)
        self.console.setStyleSheet(CONSOLE_STYLE)
        self.progress_bar.setStyleSheet(PROGRESS_BAR_STYLE)

        for box in (
            self._find_group("RST File"),
            self._find_group("Named Selection"),
            self._find_group("Modes"),
            self._find_group("Results"),
            self._find_group("Output Folder"),
            self._find_group("Progress"),
            self._find_group("Console"),
        ):
            if box:
                box.setStyleSheet(GROUP_BOX_STYLE)

        for checkbox in (self.stress_check, self.strain_check, self.disp_check, self.specific_mode_check):
            checkbox.setStyleSheet(CHECKBOX_STYLE)

    def _find_group(self, title: str) -> Optional[QGroupBox]:
        for widget in self.findChildren(QGroupBox):
            if widget.title() == title:
                return widget
        return None

    def _configure_logger(self) -> None:
        if Logger is None:
            return
        self.stdout_logger = Logger(self.console)

    def _browse_rst(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select RST", "", "RST Files (*.rst)")
        if not path:
            return
        self.rst_path_edit.setText(path)
        self._load_model_info(path)

    def _browse_output_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if path:
            self.output_dir_edit.setText(path)

    def _load_model_info(self, rst_path: str) -> None:
        try:
            info = dpf_modal_extractor.get_model_info(rst_path)
        except Exception as exc:
            QMessageBox.critical(self, "Error", str(exc))
            return

        names = info.get("named_selections", [])
        n_sets = info.get("n_sets", 0)
        n_nodes = info.get("num_nodes", 0)
        n_elems = info.get("num_elements", 0)
        units = info.get("unit_system", "Unknown")

        self.named_selection_combo.clear()
        self.named_selection_combo.addItem("All Nodes")
        for name in names:
            self.named_selection_combo.addItem(name)

        self.mode_count_label.setText(f"Detected modes: {n_sets}")
        self.mode_count_spin.setMaximum(max(1, n_sets))
        self.mode_count_spin.setValue(min(10, max(1, n_sets)))
        self.specific_mode_spin.setMaximum(max(1, n_sets))
        self.specific_mode_spin.setValue(1)

        self.model_info_label.setText(
            f"Nodes: {n_nodes} | Elements: {n_elems} | Units: {units}"
        )

    def _start_extraction(self) -> None:
        rst_path = self.rst_path_edit.text().strip()
        output_dir = self.output_dir_edit.text().strip()

        if not rst_path or not os.path.exists(rst_path):
            QMessageBox.warning(self, "Missing file", "Select a valid RST file.")
            return
        if not output_dir or not os.path.isdir(output_dir):
            QMessageBox.warning(self, "Missing output", "Select a valid output folder.")
            return

        specific_mode = self.specific_mode_spin.value() if self.specific_mode_check.isChecked() else None
        backend_map = {0: "auto", 1: "dpf", 2: "pymapdl"}
        backend = backend_map.get(self.backend_combo.currentIndex(), "auto")
        
        job = ExtractionJob(
            rst_path=rst_path,
            output_dir=output_dir,
            named_selection=self.named_selection_combo.currentText(),
            mode_count=self.mode_count_spin.value(),
            do_stress=self.stress_check.isChecked(),
            do_strain=self.strain_check.isChecked(),
            do_displacement=self.disp_check.isChecked(),
            specific_mode=specific_mode,
            backend=backend,
        )

        self.worker = ModalExtractionWorker(job)
        self.worker.log.connect(self._append_log)
        self.worker.progress.connect(self._update_progress)
        self.worker.error.connect(self._handle_error)
        self.worker.finished.connect(self._handle_finished)
        self.worker.canceled.connect(self._handle_canceled)

        # Store start time and log
        self._extraction_start_time = time.perf_counter()
        self._extraction_start_datetime = datetime.now()
        self._append_log("=" * 55)
        self._append_log(f"EXTRACTION STARTED: {self._extraction_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        self._append_log("=" * 55)

        self._set_running(True)
        self.worker.start()

    def _cancel_extraction(self) -> None:
        if self.worker:
            self.worker.request_cancel()
            self.progress_status.setText("Canceling...")

    def _set_running(self, running: bool) -> None:
        self.extract_btn.setEnabled(not running)
        self.estimate_btn.setEnabled(not running)
        self.cancel_btn.setEnabled(running)
        self.rst_browse_btn.setEnabled(not running)
        self.output_browse_btn.setEnabled(not running)
        self.named_selection_combo.setEnabled(not running)
        self.specific_mode_check.setEnabled(not running)
        # Enable/disable spinboxes based on specific mode checkbox state
        if not running:
            self.mode_count_spin.setEnabled(not self.specific_mode_check.isChecked())
            self.specific_mode_spin.setEnabled(self.specific_mode_check.isChecked())
        else:
            self.mode_count_spin.setEnabled(False)
            self.specific_mode_spin.setEnabled(False)
        self.stress_check.setEnabled(not running)
        self.strain_check.setEnabled(not running)
        self.disp_check.setEnabled(not running)
        self.backend_combo.setEnabled(not running)
        if not running:
            self.progress_bar.setValue(0)
            self.progress_status.setText("Idle")

    def _append_log(self, message: str) -> None:
        if not message:
            return
        if self.stdout_logger is not None:
            self.stdout_logger.write(message + "\n")
            return
        self.console.moveCursor(QTextCursor.End)
        self.console.insertPlainText(message + "\n")
        self.console.moveCursor(QTextCursor.End)
        self.console.ensureCursorVisible()

    def _update_progress(self, current: int, total: int) -> None:
        if total <= 0:
            return
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.progress_status.setText(f"{current}/{total} chunks")

    def _handle_error(self, message: str) -> None:
        end_datetime = datetime.now()
        
        if hasattr(self, '_extraction_start_time'):
            duration = time.perf_counter() - self._extraction_start_time
            duration_str = f"{duration:.1f} seconds"
        else:
            duration_str = "unknown"
        
        self._append_log("=" * 55)
        self._append_log(f"EXTRACTION FAILED: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        self._append_log(f"TIME ELAPSED: {duration_str}")
        self._append_log(f"ERROR: {message}")
        self._append_log("=" * 55)
        
        self._set_running(False)
        QMessageBox.critical(self, "Extraction error", message)

    def _handle_finished(self) -> None:
        end_time = time.perf_counter()
        end_datetime = datetime.now()
        
        # Calculate duration
        if hasattr(self, '_extraction_start_time'):
            duration = end_time - self._extraction_start_time
            if duration < 60:
                duration_str = f"{duration:.1f} seconds"
            elif duration < 3600:
                duration_str = f"{duration / 60:.1f} minutes ({duration:.1f}s)"
            else:
                duration_str = f"{duration / 3600:.1f} hours ({duration:.1f}s)"
        else:
            duration_str = "unknown"
        
        self._append_log("=" * 55)
        self._append_log(f"EXTRACTION FINISHED: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        self._append_log(f"TOTAL DURATION: {duration_str}")
        self._append_log("=" * 55)
        
        self._set_running(False)
        QMessageBox.information(self, "Done", f"Extraction finished.\n\nDuration: {duration_str}")

    def _handle_canceled(self) -> None:
        end_datetime = datetime.now()
        
        if hasattr(self, '_extraction_start_time'):
            duration = time.perf_counter() - self._extraction_start_time
            duration_str = f"{duration:.1f} seconds"
        else:
            duration_str = "unknown"
        
        self._append_log("=" * 55)
        self._append_log(f"EXTRACTION CANCELED: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        self._append_log(f"TIME ELAPSED: {duration_str}")
        self._append_log("=" * 55)
        
        self._set_running(False)
        QMessageBox.information(self, "Canceled", "Extraction canceled.")

    def _on_named_selection_changed(self, selection_name: str) -> None:
        """Update node count label when named selection changes."""
        rst_path = self.rst_path_edit.text().strip()
        if not rst_path or not os.path.exists(rst_path):
            self.selection_node_count_label.setText("")
            return
        
        try:
            node_ids = dpf_modal_extractor.get_nodal_scoping_ids(rst_path, selection_name)
            count = len(node_ids)
            self.selection_node_count_label.setText(f"Nodes in selection: {count:,}")
            self._current_selection_node_count = count
        except Exception as e:
            self.selection_node_count_label.setText("(node count unavailable)")
            self._current_selection_node_count = 0

    def _estimate_extraction_time(self) -> None:
        """Run quick test extractions to estimate total time.
        
        Tests each selected result type separately with 1 and 3 modes to calculate:
        - Base overhead per result type
        - Per-mode cost per result type
        
        Shows breakdown per result type and total.
        """
        rst_path = self.rst_path_edit.text().strip()
        if not rst_path or not os.path.exists(rst_path):
            QMessageBox.warning(self, "Missing file", "Select a valid RST file first.")
            return
        
        named_selection = self.named_selection_combo.currentText()
        
        # Get node count for selection (for display only)
        try:
            node_ids = dpf_modal_extractor.get_nodal_scoping_ids(rst_path, named_selection)
            total_nodes = len(node_ids)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Cannot get node count: {e}")
            return
        
        if total_nodes == 0:
            QMessageBox.warning(self, "Error", "No nodes in selection.")
            return
        
        # Determine mode count
        if self.specific_mode_check.isChecked():
            total_modes = 1
        else:
            total_modes = self.mode_count_spin.value()
        
        # Collect selected result types
        result_types = []
        if self.stress_check.isChecked():
            result_types.append(("stress", "Stress (6 comp)", dpf_modal_extractor.extract_modal_stress_csv))
        if self.strain_check.isChecked():
            result_types.append(("strain", "Strain (6 comp)", dpf_modal_extractor.extract_modal_strain_csv))
        if self.disp_check.isChecked():
            result_types.append(("displacement", "Displacement (3 comp)", dpf_modal_extractor.extract_modal_displacement_csv))
        
        if not result_types:
            QMessageBox.warning(self, "Error", "Select at least one result type.")
            return
        
        # Get backend
        backend_map = {0: "auto", 1: "dpf", 2: "pymapdl"}
        backend = backend_map.get(self.backend_combo.currentIndex(), "auto")
        
        self._append_log("=" * 55)
        self._append_log("Running time estimation...")
        self._append_log(f"Selection: {named_selection} ({total_nodes:,} nodes)")
        self._append_log(f"Modes: {total_modes}, Backend: {backend}")
        self._append_log("-" * 55)
        
        # Disable UI during estimation
        self.estimate_btn.setEnabled(False)
        self.extract_btn.setEnabled(False)
        self.progress_status.setText("Estimating...")
        QApplication.processEvents()
        
        try:
            import tempfile
            temp_dir = tempfile.gettempdir()
            temp_csv = os.path.join(temp_dir, "_modal_estimate_test.csv")
            
            estimates = []
            total_estimated = 0.0
            
            for kind, label, extract_fn in result_types:
                self._append_log(f"  Testing {label}...")
                QApplication.processEvents()
                
                # Test 1: 1 mode
                start1 = time.perf_counter()
                extract_fn(
                    rst_path=rst_path,
                    output_csv_path=temp_csv,
                    mode_count=1,
                    named_selection=named_selection,
                    backend=backend,
                    log_cb=lambda msg: None,
                )
                t1 = time.perf_counter() - start1
                
                try:
                    os.remove(temp_csv)
                except Exception:
                    pass
                
                # Test 2: 3 modes
                start2 = time.perf_counter()
                extract_fn(
                    rst_path=rst_path,
                    output_csv_path=temp_csv,
                    mode_count=3,
                    named_selection=named_selection,
                    backend=backend,
                    log_cb=lambda msg: None,
                )
                t2 = time.perf_counter() - start2
                
                try:
                    os.remove(temp_csv)
                except Exception:
                    pass
                
                # Calculate timing model
                per_mode = max(0, (t2 - t1) / 2.0)
                base = max(0.1, t1 - per_mode)
                
                # Estimate for requested modes
                est_time = base + per_mode * total_modes
                total_estimated += est_time
                
                estimates.append((label, est_time, base, per_mode))
                self._append_log(f"    {label}: {est_time:.1f}s (base={base:.1f}s + {per_mode:.2f}s/mode)")
            
            # Format total time
            if total_estimated < 60:
                time_str = f"{total_estimated:.1f} seconds"
            elif total_estimated < 3600:
                time_str = f"{total_estimated / 60:.1f} minutes"
            else:
                time_str = f"{total_estimated / 3600:.1f} hours"
            
            self._append_log("-" * 55)
            self._append_log(f"ESTIMATED TOTAL TIME: {time_str}")
            self._append_log("=" * 55)
            
            # Build message box content
            msg_lines = [f"Estimated extraction time: {time_str}\n"]
            msg_lines.append(f"Nodes: {total_nodes:,}")
            msg_lines.append(f"Modes: {total_modes}")
            msg_lines.append(f"Backend: {backend}\n")
            msg_lines.append("Breakdown:")
            for label, est_time, base, per_mode in estimates:
                msg_lines.append(f"  {label}: {est_time:.1f}s")
            
            QMessageBox.information(self, "Time Estimate", "\n".join(msg_lines))
            
        except Exception as e:
            self._append_log(f"Estimation failed: {e}")
            QMessageBox.warning(self, "Estimation Error", f"Could not estimate time: {e}")
        finally:
            self.estimate_btn.setEnabled(True)
            self.extract_btn.setEnabled(True)
            self.progress_status.setText("Idle")


def run() -> None:
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication.instance() or QApplication([])
    window = ModalMainWindow()
    window.show()
    app.exec_()
