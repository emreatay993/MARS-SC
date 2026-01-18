"""
Log Handler for the SolverTab.

This class encapsulates all logic related to formatting log messages
and appending them to the UI's console.
"""

import os
from typing import Sequence
import pandas as pd


class SolverLogHandler:
    """Manages formatting and writing log messages to the console."""

    def __init__(self, tab):
        """
        Initialize the log handler.

        Args:
            tab (SolverTab): The parent SolverTab instance (to access console).
        """
        self.tab = tab

    def _get_node_column_name(self, df: pd.DataFrame) -> str | None:
        """Find the node column name from a list of candidates."""
        candidates = ["Node Number", "Node", "NodeID", "Node Id"]
        for name in candidates:
            if name in df.columns:
                return name
        return None

    def _log_coordinate_load(self, filename, modal_data):
        """Log successful coordinate file load."""
        self.tab.console_textbox.append(
            f"Successfully validated and loaded modal coordinate file: "
            f"{os.path.basename(filename)}\n"
        )
        self.tab.console_textbox.append(
            f"Modes: {modal_data.num_modes}, Time Points: {modal_data.num_time_points:,}\n"
        )

    def _log_stress_load(self, filename, stress_data):
        """Log successful stress file load."""
        self.tab.console_textbox.append(
            f"Successfully validated and loaded modal stress file: "
            f"{os.path.basename(filename)}\n"
        )
        self.tab.console_textbox.append(
            f"📊 Stress Components: 6 components (SX, SY, SZ, SXY, SYZ, SXZ)\n"
            f"   Nodes: {stress_data.num_nodes:,}\n"
            f"   Modes: {stress_data.num_modes}\n"
        )
        self.tab.console_textbox.verticalScrollBar().setValue(
            self.tab.console_textbox.verticalScrollBar().maximum()
        )

    def _log_deformation_load(self, filename, deform_data):
        """Log successful deformation file load."""
        self.tab.console_textbox.append(
            f"Successfully validated and loaded modal deformations file: "
            f"{os.path.basename(filename)}\n"
        )
        self.tab.console_textbox.append(
            f"📐 Deformation Components: 3 components (UX, UY, UZ)\n"
            f"   Nodes: {deform_data.num_nodes:,}\n"
            f"   Modes: {deform_data.num_modes}\n"
        )

    def _log_steady_state_load(self, filename, steady_data):
        """Log successful steady-state file load."""
        self.tab.console_textbox.append(
            f"Successfully validated and loaded steady-state stress file: "
            f"{os.path.basename(filename)}\n"
        )
        self.tab.console_textbox.append(
            f"Steady-state stress data shape: {steady_data.node_ids.shape}"
        )

    def _log_temperature_field_load(self, filename, temperature_data):
        """Log successful temperature field file load."""
        self.tab.console_textbox.append(
            f"Successfully loaded temperature field file: "
            f"{os.path.basename(filename)}\n"
        )
        
        num_rows = temperature_data.num_nodes
        node_col = self._get_node_column_name(temperature_data.dataframe)
        if node_col:
            num_unique = temperature_data.dataframe[node_col].nunique()
            self.tab.console_textbox.append(
                f"Temperature entries: {num_unique} unique nodes from {num_rows} rows"
            )
        else:
            self.tab.console_textbox.append(
                f"Temperature entries: {num_rows} rows (could not determine node column)"
            )

        self.tab.console_textbox.verticalScrollBar().setValue(
            self.tab.console_textbox.verticalScrollBar().maximum()
        )

    def _log_material_profile_update(self, material_data):
        """Log updates to the material profile."""
        youngs_rows = len(material_data.youngs_modulus.index)
        poisson_rows = len(material_data.poisson_ratio.index)
        plastic_sets = len(material_data.plastic_curves)
        self.tab.console_textbox.append("Material profile updated:")
        self.tab.console_textbox.append(
            f" - Young's modulus entries: {youngs_rows}\n"
            f" - Poisson's ratio entries: {poisson_rows}\n"
            f" - Plasticity curves: {plastic_sets} temperature sets\n"
        )
        self.tab.console_textbox.verticalScrollBar().setValue(
            self.tab.console_textbox.verticalScrollBar().maximum()
        )
