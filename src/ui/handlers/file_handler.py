"""
Solver tab file I/O: file dialogs, loading RST via DPF, import/export of combination
tables, and updating the tab state/UI.
"""

import os
from typing import Optional

import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QFileDialog, QInputDialog, QMessageBox
from PyQt5.QtCore import QThread, pyqtSignal

from file_io.dpf_reader import DPFAnalysisReader, DPFNotAvailableError
from file_io.cdb_reader import CDBNamedSelectionReader
from file_io.txt_named_selection_reader import TXTNamedSelectionReader
from file_io.combination_parser import CombinationTableParser, CombinationTableParseError
from file_io.loaders import load_temperature_field
from core.data_models import AnalysisData, CombinationResult


class RSTLoaderThread(QThread):
    """Background thread for loading RST files without freezing the GUI."""
    
    finished = pyqtSignal(object, str)  # Emits (AnalysisData, filename)
    error = pyqtSignal(str)  # Emits error message
    
    def __init__(self, rst_path: str, skip_substeps: bool = False):
        """
        Initialize the RST loader thread.
        
        Args:
            rst_path: Path to RST file to load.
            skip_substeps: If True, only load the last substep of each load step.
        """
        super().__init__()
        self.rst_path = rst_path
        self.skip_substeps = skip_substeps
    
    def run(self):
        """Run the loader in background thread."""
        try:
            reader = DPFAnalysisReader(self.rst_path)
            analysis_data = reader.get_analysis_data(skip_substeps=self.skip_substeps)
            self.finished.emit(analysis_data, self.rst_path)
        except DPFNotAvailableError as e:
            self.error.emit(f"DPF not available: {e}")
        except Exception as e:
            self.error.emit(str(e))


class SolverFileHandler:
    """
    Manages file selection, loading, and state updates for the SolverTab.
    
    Handles:
    - RST file loading for both analyses using DPF
    - Combination table CSV import/export
    - Temperature field loading for plasticity
    - Result export
    """

    def __init__(self, tab):
        """
        Initialize the file handler.

        Args:
            tab (SolverTab): The parent SolverTab instance.
        """
        self.tab = tab
        
        # Keep references to loader threads to prevent garbage collection
        self._base_loader_thread = None
        self._combine_loader_thread = None
        
        # DPF readers (kept for later use in analysis)
        self.base_reader = None
        self.combine_reader = None
        self.cdb_reader = None
        self.txt_named_selection_readers = []

    # ========== RST File Loading ==========

    def select_base_rst_file(self, checked=False):
        """Open file dialog for base analysis RST file."""
        file_name, _ = QFileDialog.getOpenFileName(
            self.tab, 'Select Base Analysis RST File', '',
            'ANSYS Result Files (*.rst);;All Files (*)'
        )
        if file_name:
            self._load_rst_file(file_name, is_base=True)

    def select_combine_rst_file(self, checked=False):
        """Open file dialog for analysis to combine RST file."""
        file_name, _ = QFileDialog.getOpenFileName(
            self.tab, 'Select Analysis to Combine RST File', '',
            'ANSYS Result Files (*.rst);;All Files (*)'
        )
        if file_name:
            self._load_rst_file(file_name, is_base=False)

    def _load_rst_file(self, filename: str, is_base: bool):
        """
        Load RST file using background thread.
        
        Args:
            filename: Path to RST file.
            is_base: True for base analysis (Analysis 1), False for combine (Analysis 2).
        """
        self._clear_shared_cdb_named_selections()
        self._clear_shared_txt_named_selections()

        # Disable UI during load
        self.tab.setEnabled(False)
        analysis_name = "Base Analysis" if is_base else "Analysis to Combine"
        
        # Check if skip substeps is enabled
        skip_substeps = self.tab.skip_substeps_checkbox.isChecked()
        skip_info = " (skipping intermediate substeps)" if skip_substeps else ""
        self.tab.console_textbox.append(f"Loading {analysis_name} RST file{skip_info}...\n")
        
        # Create and start loader thread
        loader_thread = RSTLoaderThread(filename, skip_substeps=skip_substeps)
        
        if is_base:
            loader_thread.finished.connect(
                lambda data, fname: self._on_base_rst_loaded(data, fname)
            )
            loader_thread.error.connect(
                lambda error: self._on_rst_load_error(error, "Base Analysis")
            )
            self._base_loader_thread = loader_thread
        else:
            loader_thread.finished.connect(
                lambda data, fname: self._on_combine_rst_loaded(data, fname)
            )
            loader_thread.error.connect(
                lambda error: self._on_rst_load_error(error, "Analysis to Combine")
            )
            self._combine_loader_thread = loader_thread
        
        loader_thread.start()
    
    def _on_base_rst_loaded(self, analysis_data: AnalysisData, filename: str):
        """Handle successful base RST file load."""
        self.tab.setEnabled(True)
        
        # Store the DPF reader for later use
        try:
            self.base_reader = DPFAnalysisReader(filename)
        except Exception as e:
            QMessageBox.warning(
                self.tab, "Reader Error",
                f"Failed to create DPF reader: {e}"
            )
            return
        
        # Notify the tab
        self.tab.on_base_rst_loaded(analysis_data, filename)
    
    def _on_combine_rst_loaded(self, analysis_data: AnalysisData, filename: str):
        """Handle successful combine RST file load."""
        self.tab.setEnabled(True)
        
        # Store the DPF reader for later use
        try:
            self.combine_reader = DPFAnalysisReader(filename)
        except Exception as e:
            QMessageBox.warning(
                self.tab, "Reader Error",
                f"Failed to create DPF reader: {e}"
            )
            return
        
        # Notify the tab
        self.tab.on_combine_rst_loaded(analysis_data, filename)
    
    def _on_rst_load_error(self, error: str, analysis_name: str):
        """Handle RST file load error."""
        self.tab.setEnabled(True)
        
        QMessageBox.warning(
            self.tab, "RST Load Error",
            f"Failed to load {analysis_name} RST file.\n\nError: {error}"
        )

    def refresh_named_selections(self, checked=False):
        """Refresh the named selections dropdown from loaded RST files."""
        if self.base_reader is None and self.combine_reader is None:
            QMessageBox.information(
                self.tab, "No Files Loaded",
                "Please load RST files first."
            )
            return
        
        # Update named selections through the tab
        self.tab._update_named_selections()
        self.tab.console_textbox.append("Named selections refreshed.\n")

    # ========== Optional Supplemental Named Selection Loading ==========

    def select_cdb_file(self, checked=False):
        """Open file dialog for shared CDB named-selection import."""
        file_name, _ = QFileDialog.getOpenFileName(
            self.tab,
            "Import CDB Named Selections",
            "",
            "ANSYS CDB Files (*.cdb *.CDB);;All Files (*)",
        )
        if file_name:
            self._load_shared_cdb_named_selections(file_name)

    def _load_shared_cdb_named_selections(self, filename: str):
        """
        Load one CDB as supplemental named selections for both analyses.

        CDB components are used only for scoping. The RST files remain the
        source for results, mesh, load steps, and units.
        """
        if (
            self.base_reader is None
            or self.combine_reader is None
            or self.tab.analysis1_data is None
            or self.tab.analysis2_data is None
        ):
            QMessageBox.information(
                self.tab,
                "Load RST Files First",
                "Please load both RST files before importing CDB named selections.",
            )
            return

        try:
            cdb_reader = CDBNamedSelectionReader.from_file(filename)
            if not cdb_reader.get_named_selections():
                QMessageBox.warning(
                    self.tab,
                    "No Named Selections Found",
                    "No supported node or element CMBLOCK components were found in the selected CDB file.",
                )
                return

            reserved_names = set(self.base_reader.get_rst_named_selections())
            reserved_names.update(self.combine_reader.get_rst_named_selections())
            for txt_reader in self.txt_named_selection_readers:
                reserved_names.update(txt_reader.get_named_selections())
            cdb_reader.rename_conflicting_selections(reserved_names)

            self.base_reader.attach_cdb_named_selection_reader(cdb_reader)
            self.combine_reader.attach_cdb_named_selection_reader(cdb_reader)
            self.cdb_reader = cdb_reader

            self._sync_analysis_named_selection_metadata(
                analysis_data=self.tab.analysis1_data,
                reader=self.base_reader,
                cdb_path=filename,
            )
            self._sync_analysis_named_selection_metadata(
                analysis_data=self.tab.analysis2_data,
                reader=self.combine_reader,
                cdb_path=filename,
            )
        except Exception as error:
            QMessageBox.warning(
                self.tab,
                "CDB Import Error",
                f"Failed to import CDB named selections:\n\n{error}",
            )
            return

        self.tab.on_cdb_loaded(cdb_reader, filename)

    def select_txt_named_selection_file(self, checked=False):
        """Open file dialog for one TXT nodal named-selection import."""
        file_names, _ = QFileDialog.getOpenFileNames(
            self.tab,
            "Import TXT Nodal Named Selection",
            "",
            "Text Files (*.txt *.TXT);;All Files (*)",
        )
        if not file_names:
            return

        if len(file_names) > 1:
            QMessageBox.warning(
                self.tab,
                "Select One TXT File",
                "Multiple file selection is not supported. Please select one TXT file for each import.",
            )
            return

        self._load_shared_txt_named_selection(file_names[0])

    def _load_shared_txt_named_selection(self, filename: str):
        """
        Load one nodal TXT table as a supplemental named selection.

        The imported TXT supplies node IDs only for scoping. Result values,
        mesh, units, and load steps still come from the loaded RST files.
        """
        if (
            self.base_reader is None
            or self.combine_reader is None
            or self.tab.analysis1_data is None
            or self.tab.analysis2_data is None
        ):
            QMessageBox.information(
                self.tab,
                "Load RST Files First",
                "Please load both RST files before importing TXT named selections.",
            )
            return

        reserved_names = set(self.base_reader.get_named_selections())
        reserved_names.update(self.combine_reader.get_named_selections())
        selection_name = self._prompt_txt_named_selection_name(reserved_names)
        if selection_name is None:
            return

        try:
            txt_reader = TXTNamedSelectionReader.from_file(filename, selection_name)

            self.base_reader.attach_txt_named_selection_reader(txt_reader)
            self.combine_reader.attach_txt_named_selection_reader(txt_reader)
            self.txt_named_selection_readers.append(txt_reader)

            self._sync_analysis_named_selection_metadata(
                analysis_data=self.tab.analysis1_data,
                reader=self.base_reader,
                cdb_path=self.tab.analysis1_data.cdb_file_path,
            )
            self._sync_analysis_named_selection_metadata(
                analysis_data=self.tab.analysis2_data,
                reader=self.combine_reader,
                cdb_path=self.tab.analysis2_data.cdb_file_path,
            )
        except Exception as error:
            QMessageBox.warning(
                self.tab,
                "TXT Import Error",
                f"Failed to import TXT named selection:\n\n{error}",
            )
            return

        self.tab.on_txt_named_selection_loaded(txt_reader, filename)

    def _prompt_txt_named_selection_name(self, reserved_names: set) -> Optional[str]:
        """Prompt until the user enters a valid unique imported NS name or cancels."""
        while True:
            name, accepted = QInputDialog.getText(
                self.tab,
                "Name Imported Named Selection",
                "Enter a named selection name using letters, numbers, and underscores:",
            )
            if not accepted:
                return None

            name = name.strip()
            if not TXTNamedSelectionReader.is_valid_selection_name(name):
                QMessageBox.warning(
                    self.tab,
                    "Invalid Name",
                    "Use a name that starts with a letter or underscore and contains only letters, numbers, and underscores.",
                )
                continue

            if name in reserved_names:
                QMessageBox.warning(
                    self.tab,
                    "Name Already Exists",
                    "That named selection already exists. Please choose a unique name for this import.",
                )
                continue

            return name

    def _clear_shared_cdb_named_selections(self) -> None:
        """Remove the currently attached shared CDB source from loaded readers."""
        if self.cdb_reader is None:
            return

        self.cdb_reader = None
        loaded_pairs = (
            (self.base_reader, self.tab.analysis1_data),
            (self.combine_reader, self.tab.analysis2_data),
        )
        for reader, analysis_data in loaded_pairs:
            if reader is None:
                continue
            reader.attach_cdb_named_selection_reader(None)
            if analysis_data is not None:
                self._sync_analysis_named_selection_metadata(
                    analysis_data=analysis_data,
                    reader=reader,
                    cdb_path=None,
                )
        self.tab._update_named_selections()

    def _clear_shared_txt_named_selections(self) -> None:
        """Remove currently attached TXT named-selection imports from loaded readers."""
        if not self.txt_named_selection_readers:
            return

        self.txt_named_selection_readers = []
        loaded_pairs = (
            (self.base_reader, self.tab.analysis1_data),
            (self.combine_reader, self.tab.analysis2_data),
        )
        for reader, analysis_data in loaded_pairs:
            if reader is None:
                continue
            reader.clear_txt_named_selection_readers()
            if analysis_data is not None:
                self._sync_analysis_named_selection_metadata(
                    analysis_data=analysis_data,
                    reader=reader,
                    cdb_path=analysis_data.cdb_file_path,
                )
        self.tab._update_named_selections()

    @staticmethod
    def _sync_analysis_named_selection_metadata(
        analysis_data: AnalysisData,
        reader: DPFAnalysisReader,
        cdb_path: Optional[str],
    ) -> None:
        """Refresh mutable AnalysisData named-selection metadata from the reader."""
        analysis_data.named_selections = reader.get_named_selections()
        analysis_data.named_selection_locations = reader.get_named_selection_locations()
        analysis_data.named_selection_sources = reader.get_named_selection_sources()
        analysis_data.cdb_file_path = cdb_path

    # ========== Combination Table Import/Export ==========

    def import_combination_table(self, checked=False):
        """Open file dialog and import combination table from CSV."""
        file_name, _ = QFileDialog.getOpenFileName(
            self.tab, 'Import Combination Table', '',
            'CSV Files (*.csv);;All Files (*)'
        )
        if file_name:
            self._load_combination_csv(file_name)

    def _load_combination_csv(self, filename: str):
        """
        Load combination table from CSV file.
        
        Args:
            filename: Path to CSV file.
        """
        try:
            combo_data = CombinationTableParser.parse_csv(filename)
            
            # Validate against loaded analyses if available
            if self.tab.analysis1_data and self.tab.analysis2_data:
                is_valid, error_msg = CombinationTableParser.validate_against_analyses(
                    combo_data,
                    self.tab.analysis1_data.num_load_steps,
                    self.tab.analysis2_data.num_load_steps
                )
                if not is_valid:
                    QMessageBox.warning(
                        self.tab, "Validation Warning",
                        f"Combination table may not match loaded analyses:\n\n{error_msg}\n\n"
                        "The table will be loaded anyway, but coefficients may need adjustment."
                    )
            
            # Update the tab's table widget
            self.tab.set_combination_table_data(combo_data)
            
            self.tab.console_textbox.append(
                f"Imported combination table from {os.path.basename(filename)}\n"
                f"  Combinations: {combo_data.num_combinations}\n"
                f"  Analysis 1 steps: {combo_data.num_analysis1_steps}\n"
                f"  Analysis 2 steps: {combo_data.num_analysis2_steps}\n"
            )
            
        except CombinationTableParseError as e:
            QMessageBox.warning(
                self.tab, "Import Error",
                f"Failed to parse combination table:\n\n{e}"
            )
        except Exception as e:
            QMessageBox.warning(
                self.tab, "Import Error",
                f"Failed to import combination table:\n\n{e}"
            )

    def export_combination_table(self, checked=False):
        """Export current combination table to CSV file."""
        # Get data from table widget
        combo_data = self.tab.get_combination_table_data()
        
        if combo_data is None or combo_data.num_combinations == 0:
            QMessageBox.information(
                self.tab, "Nothing to Export",
                "The combination table is empty."
            )
            return
        
        # Get save path
        file_name, _ = QFileDialog.getSaveFileName(
            self.tab, 'Export Combination Table', 'combination_table.csv',
            'CSV Files (*.csv)'
        )
        
        if file_name:
            try:
                CombinationTableParser.export_csv(combo_data, file_name)
                self.tab.console_textbox.append(
                    f"Exported combination table to {os.path.basename(file_name)}\n"
                )
            except Exception as e:
                QMessageBox.warning(
                    self.tab, "Export Error",
                    f"Failed to export combination table:\n\n{e}"
                )

    # ========== Temperature Field Loading ==========

    def select_temperature_field_file(self, checked=False):
        """Open file dialog for temperature field file."""
        file_name, _ = QFileDialog.getOpenFileName(
            self.tab, 'Open Temperature Field File', '',
            'Text Files (*.txt);;CSV Files (*.csv);;All Files (*)'
        )
        if file_name:
            self._load_temperature_field_file(file_name)

    def _load_temperature_field_file(self, filename: str):
        """Load temperature field file into a DataFrame."""
        try:
            temperature_data = load_temperature_field(filename)
            self.tab.temperature_field_file_path.setText(filename)
            self.tab.on_temperature_field_loaded(temperature_data, filename)
        except ValueError as e:
            self.tab.temperature_field_data = None
            self.tab.temperature_field_file_path.clear()
            QMessageBox.warning(
                self.tab, "Invalid File",
                f"The selected Temperature Field File is not valid.\n\nError: {e}"
            )

    # ========== Result Export ==========

    def export_single_combination_result(
        self, 
        result: CombinationResult, 
        combo_index: int, 
        filename: str
    ):
        """
        Export a single combination result to CSV.
        
        Args:
            result: CombinationResult containing all results.
            combo_index: Index of the combination to export.
            filename: Output file path.
        """
        if result.all_combo_results is None:
            raise ValueError("No combination results available.")
        
        # Get stress values for this combination
        stress_values = result.all_combo_results[combo_index, :]
        
        # Build DataFrame
        data = {
            'NodeID': result.node_ids,
            'X': result.node_coords[:, 0],
            'Y': result.node_coords[:, 1],
            'Z': result.node_coords[:, 2],
            f'{result.result_type}': stress_values,
        }
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)

    def export_envelope_results(
        self,
        result: CombinationResult,
        filename: str,
        include_combo_indices: bool = True
    ):
        """
        Export envelope (max/min over combinations) results to CSV.
        
        Args:
            result: CombinationResult containing envelope data.
            filename: Output file path.
            include_combo_indices: Whether to include which combo caused max/min.
        """
        data = {
            'NodeID': result.node_ids,
            'X': result.node_coords[:, 0],
            'Y': result.node_coords[:, 1],
            'Z': result.node_coords[:, 2],
        }
        
        if result.max_over_combo is not None:
            data[f'Max_{result.result_type}'] = result.max_over_combo
            if include_combo_indices and result.combo_of_max is not None:
                data['Combo_of_Max'] = result.combo_of_max
        
        if result.min_over_combo is not None:
            data[f'Min_{result.result_type}'] = result.min_over_combo
            if include_combo_indices and result.combo_of_min is not None:
                data['Combo_of_Min'] = result.combo_of_min
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
