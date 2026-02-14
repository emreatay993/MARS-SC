"""
File loading logic for the Display tab.
"""

import os
import re
from typing import Optional, Sequence, Tuple, List
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QFileDialog, QMessageBox

from ui.handlers.display_base_handler import DisplayBaseHandler
from core.visualization import VisualizationManager
from ui.display_payload import SolverOutputFlags


class DisplayFileHandler(DisplayBaseHandler):
    """Handles loading external data sources for visualization."""

    def __init__(self, tab, state, viz_manager: VisualizationManager):
        super().__init__(tab, state)
        self.viz_manager = viz_manager
        self.plotting_handler = None

    def set_plotting_handler(self, plotting_handler: Optional[object]) -> None:
        """Assign plotting handler reference (mirrors DisplayTab API)."""
        self.plotting_handler = plotting_handler

    def open_file_dialog(self) -> None:
        """Open a file dialog for selecting a visualization CSV and load it."""
        file_name, _ = QFileDialog.getOpenFileName(
            self.tab, "Open Visualization File", "", "CSV Files (*.csv)"
        )
        if file_name:
            self.load_from_file(file_name)

    def load_from_file(self, filename: str) -> None:
        """
        Load visualization data from a CSV file and update the view.

        Args:
            filename: Absolute path to the CSV file selected by the user.
        """
        try:
            df = pd.read_csv(filename)
        except Exception as exc:
            QMessageBox.critical(
                self.tab, "Error Loading File",
                f"Failed to load file:\n{exc}"
            )
            return

        if not self._has_required_columns(df.columns):
            QMessageBox.warning(
                self.tab, "Invalid File",
                "File must contain X, Y, Z coordinate columns."
            )
            return

        coords = df[["X", "Y", "Z"]].to_numpy(dtype=float)
        node_ids = df["NodeID"].to_numpy(dtype=int) if "NodeID" in df.columns else None

        mesh = self.viz_manager.create_mesh_from_coords(coords, node_ids)

        # Loading external CSV replaces solver-backed result references.
        self.tab.stress_result = None
        self.tab.all_combo_results = None
        self.tab.nodal_forces_result = None
        self.tab.deformation_result = None
        self.tab.output_flags = SolverOutputFlags()

        # Check if this is an envelope file and handle accordingly
        is_envelope, result_type = self._detect_envelope_file(filename, df.columns)
        
        if is_envelope:
            scalar_name = self._apply_envelope_data(df, mesh, result_type)
        else:
            scalar_name = self._apply_scalar_data(df, mesh)
            # Hide scalar display controls for non-envelope files
            self.tab.scalar_display_label.setVisible(False)
            self.tab.scalar_display_combo.setVisible(False)
            self.tab.current_result_type = None
            self.tab.combination_names = []
        
        if scalar_name:
            self.state.data_column = scalar_name
            self.tab.data_column = scalar_name

        self.state.current_mesh = mesh
        self.tab.current_mesh = mesh
        self.tab.file_path.setText(filename)

        # Refresh the 3D view via contour-sync pipeline
        if hasattr(self.tab, "contour_sync_handler") and self.tab.contour_sync_handler is not None:
            self.tab.contour_sync_handler.sync_from_current_state()
        else:
            self.tab.update_visualization()
        self.tab.plotter.reset_camera()

    @staticmethod
    def _has_required_columns(columns: Sequence[str]) -> bool:
        """Return True if the minimum coordinate columns are present."""
        required = {"X", "Y", "Z"}
        return required.issubset(set(columns))

    def _detect_envelope_file(self, filename: str, columns: Sequence[str]) -> Tuple[bool, Optional[str]]:
        """
        Detect if the file is an envelope results file.
        
        Envelope files have characteristic patterns:
        - Filename: envelope_{result_type}.csv
        - Columns: "Max {Type} [MPa]", "Combination of Max (#)", etc.
        
        Args:
            filename: Path to the CSV file.
            columns: List of column names from the CSV.
            
        Returns:
            Tuple of (is_envelope, result_type).
        """
        basename = os.path.basename(filename).lower()
        columns_lower = [col.lower() for col in columns]
        
        # Check for envelope filename pattern
        result_type = None
        if basename.startswith("envelope_"):
            # Extract result type from filename: envelope_von_mises.csv -> von_mises
            match = re.match(r"envelope_(.+)\.csv", basename)
            if match:
                result_type = match.group(1)
        
        # Check for envelope column patterns
        has_max_col = any("max" in col and "[mpa]" in col for col in columns_lower)
        has_combo_max_col = any("combination of max" in col for col in columns_lower)
        
        is_envelope = has_max_col and has_combo_max_col
        
        # If result_type not determined from filename, try to infer from column names
        if is_envelope and not result_type:
            for col in columns:
                col_lower = col.lower()
                if "von mises" in col_lower:
                    result_type = "von_mises"
                    break
                elif "max principal" in col_lower:
                    result_type = "max_principal"
                    break
                elif "min principal" in col_lower:
                    result_type = "min_principal"
                    break
        
        return is_envelope, result_type

    def _apply_envelope_data(self, df: pd.DataFrame, mesh, result_type: Optional[str]) -> Optional[str]:
        """
        Apply envelope file data to the mesh with all columns.
        
        Loads Max/Min stress values and Combination indices into the mesh
        and populates the scalar display dropdown.
        
        Args:
            df: DataFrame loaded from envelope CSV.
            mesh: PyVista mesh to update.
            result_type: Detected result type (e.g., "von_mises", "min_principal").
            
        Returns:
            Name of the primary scalar column applied.
        """
        columns = list(df.columns)
        
        # Find envelope columns by pattern matching
        max_stress_col = None
        min_stress_col = None
        combo_of_max_col = None
        combo_of_min_col = None
        combo_name_max_col = None
        combo_name_min_col = None
        
        for col in columns:
            col_lower = col.lower()
            if "max" in col_lower and "[mpa]" in col_lower:
                max_stress_col = col
            elif "min" in col_lower and "[mpa]" in col_lower:
                min_stress_col = col
            elif "combination of max (#)" in col_lower:
                combo_of_max_col = col
            elif "combination of min (#)" in col_lower:
                combo_of_min_col = col
            elif "combination of max (name)" in col_lower:
                combo_name_max_col = col
            elif "combination of min (name)" in col_lower:
                combo_name_min_col = col
        
        # Apply max stress data
        if max_stress_col:
            max_data = df[max_stress_col].to_numpy(dtype=float)
            mesh['Max_Stress'] = max_data
            mesh.set_active_scalars('Max_Stress')
        
        # Apply min stress data
        has_min_data = False
        if min_stress_col:
            min_data = df[min_stress_col].to_numpy(dtype=float)
            mesh['Min_Stress'] = min_data
            has_min_data = True
        
        # Apply combination indices (convert from 1-based to 0-based)
        if combo_of_max_col:
            combo_max_data = df[combo_of_max_col].to_numpy(dtype=int) - 1
            mesh['Combo_of_Max'] = combo_max_data
        
        if combo_of_min_col:
            combo_min_data = df[combo_of_min_col].to_numpy(dtype=int) - 1
            mesh['Combo_of_Min'] = combo_min_data
        
        # Extract unique combination names if available
        combination_names = []
        if combo_name_max_col:
            # Get unique combination names in order of their indices
            combo_names_series = df[[combo_of_max_col, combo_name_max_col]].drop_duplicates()
            combo_names_series = combo_names_series.sort_values(by=combo_of_max_col)
            combination_names = combo_names_series[combo_name_max_col].tolist()
        
        # Store combination names and result type in the tab
        self.tab.combination_names = combination_names
        self.tab.current_result_type = result_type or "von_mises"
        
        # Store result type as mesh field data
        mesh.field_data['result_type'] = [self.tab.current_result_type]
        
        # Populate scalar display dropdown
        self.tab.populate_scalar_display_options(self.tab.current_result_type, has_min_data)
        
        # Set scalar range from the max stress data
        if max_stress_col:
            data_min = float(np.min(max_data))
            data_max = float(np.max(max_data))
            
            self.tab.scalar_min_spin.blockSignals(True)
            self.tab.scalar_max_spin.blockSignals(True)
            self.tab.scalar_min_spin.setRange(data_min, data_max)
            self.tab.scalar_max_spin.setRange(data_min, 1e30)
            self.tab.scalar_min_spin.setValue(data_min)
            self.tab.scalar_max_spin.setValue(data_max)
            self.tab.scalar_min_spin.blockSignals(False)
            self.tab.scalar_max_spin.blockSignals(False)
        
        return 'Max_Stress' if max_stress_col else None

    def _apply_scalar_data(self, df: pd.DataFrame, mesh) -> Optional[str]:
        """
        Attach scalar data to the mesh if present.

        Args:
            df: DataFrame loaded from CSV.
            mesh: PyVista mesh to update.

        Returns:
            Name of the scalar column applied, or None if none present.
        """
        scalar_cols = [
            col for col in df.columns
            if col not in {"X", "Y", "Z", "NodeID", "Index"}
        ]

        if not scalar_cols:
            return None

        scalar_name = scalar_cols[0]
        scalar_data = df[scalar_name].to_numpy()
        self.viz_manager.update_mesh_scalars(mesh, scalar_data, scalar_name)
        return scalar_name
