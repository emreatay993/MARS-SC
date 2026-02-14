"""
Lifecycle/UI concern for solver analysis runs.

This module owns progress updates, user-visible error handling, run logging,
and post-solve result routing.
"""

from datetime import datetime
from typing import Optional

from PyQt5.QtWidgets import QApplication, QMessageBox

from core.data_models import (
    CombinationResult,
    DeformationResult,
    NodalForcesResult,
    SolverConfig,
)
from file_io.dpf_reader import (
    DisplacementNotAvailableError,
    NodalForcesNotAvailableError,
)
from solver.deformation_engine import CylindricalCSNotFoundError
from ui.handlers.solver_result_summary_handler import SolverResultSummaryHandler
from utils.constants import MSG_NODAL_FORCES_ANSYS


class SolverRunLifecycleHandler:
    """Manage solve lifecycle behavior for the Solver tab."""

    def __init__(self, tab):
        self.tab = tab
        self.results_handler = SolverResultSummaryHandler(tab)
        self._solve_start_time: Optional[datetime] = None

    def begin_solve(self, config: SolverConfig) -> None:
        """Log solve start and initialize the progress UI."""
        self._log_solve_start(config)
        self.tab.progress_bar.setVisible(True)
        self.tab.progress_bar.setValue(0)
        self.tab.console_textbox.append("Running combination analysis...\n")
        QApplication.processEvents()

    def announce_stage(self, message: str) -> None:
        """Emit a stage message while keeping the UI responsive."""
        self.tab.console_textbox.append(message)
        QApplication.processEvents()

    def update_progress(self, current: int, total: int, message: str) -> None:
        """Update progress bar value/label from analysis callbacks."""
        if total > 0:
            percent = int((current / total) * 100)
            self.tab.progress_bar.setValue(percent)
            self.tab.progress_bar.setFormat(f"{message} ({percent}%)")
        else:
            self.tab.progress_bar.setFormat(message)
        QApplication.processEvents()

    def complete_solve(
        self,
        stress_result: Optional[CombinationResult],
        config: SolverConfig,
        forces_result: Optional[NodalForcesResult] = None,
        deformation_result: Optional[DeformationResult] = None,
    ) -> None:
        """Finalize solve state and route all produced results to UI handlers."""
        self.tab.setEnabled(True)
        self.tab.progress_bar.setVisible(False)

        self.tab.combination_result = stress_result
        self.tab.nodal_forces_result = forces_result
        self.tab.deformation_result = deformation_result

        if stress_result is not None:
            if config.combination_history_mode:
                self.results_handler.handle_stress_history_result(stress_result, config)
            else:
                self.results_handler.handle_stress_envelope_result(stress_result, config)

        if forces_result is not None:
            if config.combination_history_mode:
                self.results_handler.handle_forces_history_result(forces_result, config)
            else:
                self.results_handler.handle_forces_envelope_result(forces_result, config)

        if deformation_result is not None:
            if config.combination_history_mode:
                self.results_handler.handle_deformation_history_result(deformation_result, config)
            else:
                self.results_handler.handle_deformation_envelope_result(deformation_result, config)

        if not config.combination_history_mode:
            if stress_result is not None:
                self.tab.on_analysis_complete(stress_result)
            elif forces_result is not None:
                self.tab.on_forces_analysis_complete(forces_result)

            if deformation_result is not None:
                is_standalone = stress_result is None and forces_result is None
                self.tab.on_deformation_analysis_complete(
                    deformation_result,
                    is_standalone=is_standalone,
                )

        self._log_solve_complete()

    def finish_without_results(self) -> None:
        """Reset run-state UI when solve exits without producing any results."""
        self.tab.setEnabled(True)
        self.tab.progress_bar.setVisible(False)

    def fail_solve(self, error_msg: str) -> None:
        """Handle unexpected solve failure."""
        self.tab.setEnabled(True)
        self.tab.progress_bar.setVisible(False)
        self.tab.console_textbox.append(f"\nSolver Error:\n{error_msg}\n")
        QMessageBox.critical(
            self.tab,
            "Solver Error",
            f"An error occurred during analysis:\n\n{error_msg}",
        )

    def handle_engine_creation_error(self, engine_label: str, error: Exception) -> None:
        """Report engine creation failures."""
        QMessageBox.critical(
            self.tab,
            "Engine Creation Error",
            f"Failed to create {engine_label} engine:\n\n{error}",
        )

    def handle_nodal_forces_unavailable(self, error: NodalForcesNotAvailableError) -> None:
        """Report nodal-forces output unavailability."""
        self.tab.progress_bar.setVisible(False)
        error_msg = (
            f"Nodal Forces Not Available\n\n"
            f"{str(error)}\n\n"
            f"{MSG_NODAL_FORCES_ANSYS}"
        )
        QMessageBox.critical(self.tab, "Nodal Forces Error", error_msg)

    def handle_displacement_unavailable(self, error: DisplacementNotAvailableError) -> None:
        """Report displacement output unavailability."""
        self.tab.progress_bar.setVisible(False)
        error_msg = (
            f"Displacement Results Not Available\n\n"
            f"{str(error)}\n\n"
            f"Ensure displacement output is enabled in ANSYS Output Controls."
        )
        QMessageBox.critical(self.tab, "Displacement Error", error_msg)

    def handle_cylindrical_cs_error(self, error: CylindricalCSNotFoundError) -> None:
        """Report invalid/missing cylindrical coordinate system."""
        self.tab.progress_bar.setVisible(False)
        error_msg = f"Cylindrical Coordinate System Error\n\n{str(error)}"
        QMessageBox.critical(self.tab, "Coordinate System Error", error_msg)

    def handle_memory_error(self, error: MemoryError) -> None:
        """Report out-of-memory failure with practical guidance."""
        self.tab.progress_bar.setVisible(False)
        error_msg = (
            f"Out of Memory Error\n\n"
            f"The analysis requires more RAM than available.\n\n"
            f"Suggestions:\n"
            f"- Use a smaller Named Selection to reduce nodes\n"
            f"- Close other applications to free memory\n\n"
            f"Details: {str(error)}"
        )
        QMessageBox.critical(self.tab, "Memory Error", error_msg)

    def _log_solve_start(self, config: SolverConfig) -> None:
        current_time = datetime.now()
        self._solve_start_time = current_time

        output_types = []
        if config.calculate_von_mises:
            output_types.append("Von Mises")
        if config.calculate_max_principal_stress:
            output_types.append("Max Principal")
        if config.calculate_min_principal_stress:
            output_types.append("Min Principal")
        if config.calculate_nodal_forces:
            csys = "Global" if config.nodal_forces_rotate_to_global else "Local (Element)"
            output_types.append(f"Nodal Forces ({csys})")
        if config.calculate_deformation:
            output_types.append("Deformation")

        mode = "Combination History" if config.combination_history_mode else "Envelope"
        self.tab.console_textbox.append(
            f"\n{'=' * 60}\n"
            f"BEGIN COMBINATION ANALYSIS\n"
            f"{'=' * 60}\n"
            f"Datetime: {current_time}\n"
            f"Mode: {mode}\n"
            f"Outputs: {', '.join(output_types)}\n"
        )

        if config.combination_history_mode and config.selected_node_id:
            self.tab.console_textbox.append(f"Node ID: {config.selected_node_id}\n")

    def _log_solve_complete(self) -> None:
        end_time = datetime.now()
        elapsed = None
        if self._solve_start_time:
            elapsed = (end_time - self._solve_start_time).total_seconds()

        msg = f"\n{'=' * 60}\nANALYSIS COMPLETE\n{'=' * 60}\n"
        if elapsed is not None:
            msg += f"Elapsed time: {elapsed:.3f} seconds\n"

        self.tab.console_textbox.append(msg)
        self._solve_start_time = None
