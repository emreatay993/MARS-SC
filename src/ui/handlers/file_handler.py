"""
Solver tab file I/O: file dialogs, loading RST via DPF, import/export of combination
tables, and updating the tab state/UI.
"""

import os
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtCore import QThread, pyqtSignal

from file_io.dpf_reader import DPFAnalysisReader, DPFNotAvailableError
from file_io.cdb_reader import CDBNamedSelectionReader
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
        self.base_cdb_reader = None
        self.combine_cdb_reader = None

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
        # Disable UI during load
        self.tab.setEnabled(False)
        analysis_name = "Base Analysis" if is_base else "Analysis to Combine"
        if is_base:
            self.base_cdb_reader = None
        else:
            self.combine_cdb_reader = None
        
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

    # ========== Optional CDB Named Selection Loading ==========

    def select_base_cdb_file(self, checked=False):
        """Open file dialog for base-analysis CDB named selections."""
        self._select_cdb_file(is_base=True)

    def select_combine_cdb_file(self, checked=False):
        """Open file dialog for combine-analysis CDB named selections."""
        self._select_cdb_file(is_base=False)

    def _select_cdb_file(self, is_base: bool):
        analysis_name = "Base Analysis" if is_base else "Analysis to Combine"
        file_name, _ = QFileDialog.getOpenFileName(
            self.tab,
            f"Select {analysis_name} CDB File",
            "",
            "ANSYS CDB Files (*.cdb *.CDB);;All Files (*)",
        )
        if file_name:
            self._load_cdb_named_selections(file_name, is_base=is_base)

    def _load_cdb_named_selections(self, filename: str, is_base: bool):
        """
        Load CDB component blocks as supplemental named selections.

        CDB components are used only for scoping. The RST file remains the
        source for results, mesh, load steps, and units.
        """
        reader = self.base_reader if is_base else self.combine_reader
        analysis_data = self.tab.analysis1_data if is_base else self.tab.analysis2_data
        analysis_name = "Base Analysis" if is_base else "Analysis to Combine"

        if reader is None or analysis_data is None:
            QMessageBox.information(
                self.tab,
                "Load RST First",
                f"Please load the {analysis_name} RST file before importing its CDB named selections.",
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

            reader.attach_cdb_named_selection_reader(cdb_reader)
            if is_base:
                self.base_cdb_reader = cdb_reader
            else:
                self.combine_cdb_reader = cdb_reader

            self._sync_analysis_named_selection_metadata(
                analysis_data=analysis_data,
                reader=reader,
                cdb_path=filename,
            )
        except Exception as error:
            QMessageBox.warning(
                self.tab,
                "CDB Import Error",
                f"Failed to import CDB named selections:\n\n{error}",
            )
            return

        if is_base:
            self.tab.on_base_cdb_loaded(cdb_reader, filename)
        else:
            self.tab.on_combine_cdb_loaded(cdb_reader, filename)

    @staticmethod
    def _sync_analysis_named_selection_metadata(
        analysis_data: AnalysisData,
        reader: DPFAnalysisReader,
        cdb_path: str,
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
