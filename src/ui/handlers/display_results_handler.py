"""
Helpers for applying solver output datasets to the Display tab.
"""

import os
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from ui.handlers.display_base_handler import DisplayBaseHandler


class DisplayResultsHandler(DisplayBaseHandler):
    """Loads solver results and applies them to the visualization."""

    def __init__(self, tab, state, visual_handler):
        super().__init__(tab, state)
        self.visual_handler = visual_handler

    def apply_solver_results(
        self,
        solver,
        dataset_options: Iterable[Tuple[str, str]],
        current_field: Optional[str]
    ) -> None:
        """
        Load a solver-generated per-node dataset and apply it to the display.

        Args:
            solver: Solver instance that produced the results.
            dataset_options: Iterable of (field_name, filename) tuples.
            current_field: Name of the field currently visible on the display.
        """
        if solver is None or not hasattr(solver, "output_directory"):
            return

        options = list(dataset_options)
        if not options:
            return

        field_name, file_name = self._select_dataset(options, current_field)
        values = self._load_solver_array(solver, file_name)
        if values is None:
            return

        self._update_scalar_controls(values)
        try:
            self.visual_handler.apply_scalar_field(field_name, values)
        except Exception as exc:
            print(f"Could not apply scalar field '{field_name}': {exc}")

    def _select_dataset(
        self,
        options: Iterable[Tuple[str, str]],
        current_field: Optional[str]
    ) -> Tuple[str, str]:
        """Choose the dataset that best matches the current visualization."""
        if current_field:
            for field_name, file_name in options:
                if field_name == current_field:
                    return field_name, file_name
        return options[0]

    def _load_solver_array(self, solver, filename: str) -> Optional[np.ndarray]:
        """Load a NumPy array from the solver output directory."""
        try:
            base_dir = getattr(solver, "output_directory", None)
            if not base_dir:
                return None

            file_path = os.path.join(base_dir, filename)
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                candidate_cols = [col for col in df.columns if col.lower() != 'nodeid']
                if not candidate_cols:
                    return None
                values = df[candidate_cols[0]].to_numpy(dtype=float, copy=True)
                return values

            return None
        except Exception as exc:
            print(f"Failed to load solver result array from {filename}: {exc}")
            return None

    def _update_scalar_controls(self, values: np.ndarray) -> None:
        """Set display tab scalar spinboxes based on provided values."""
        if values.size == 0:
            return

        scalar_min = float(np.min(values))
        scalar_max = float(np.max(values))

        if scalar_min == scalar_max:
            epsilon = abs(scalar_min) * 0.01 or 0.01
            scalar_min -= epsilon
            scalar_max += epsilon

        spin_min = self.tab.scalar_min_spin
        spin_max = self.tab.scalar_max_spin

        spin_min.blockSignals(True)
        spin_max.blockSignals(True)
        try:
            spin_min.setRange(scalar_min, scalar_max)
            spin_max.setRange(scalar_min, 1e30)
            spin_min.setValue(scalar_min)
            spin_max.setValue(scalar_max)
        finally:
            spin_min.blockSignals(False)
            spin_max.blockSignals(False)
