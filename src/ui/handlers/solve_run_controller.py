"""
Orchestration for solver-tab analysis runs.

This controller keeps the solve path focused on flow coordination:
validation -> execution -> lifecycle completion/error handling.
"""

import traceback
from typing import Optional

from core.data_models import CombinationResult, SolverConfig
from file_io.dpf_reader import (
    DisplacementNotAvailableError,
    NodalForcesNotAvailableError,
)
from solver.deformation_engine import CylindricalCSNotFoundError
from solver.stress_engine import StressCombinationEngine
from ui.handlers.progress_coordinator import ProgressCoordinator, StageSpec
from ui.handlers.solver_analysis_executor import (
    SolverRunEngineCreationError,
    SolverAnalysisExecutor,
)
from ui.handlers.solver_run_ui_handler import SolverRunUiHandler
from ui.handlers.solver_engine_factory import SolverEngineFactory
from ui.handlers.solver_input_validator import SolverInputValidator


class SolveRunController:
    """Coordinate the complete analysis run workflow for SolverTab."""

    MEMORY_THRESHOLD_GB = 2.0

    def __init__(self, tab):
        self.tab = tab
        self.input_validator = SolverInputValidator(tab)
        self.engine_factory = SolverEngineFactory()
        self.execution_handler = SolverAnalysisExecutor(
            tab=tab,
            engine_factory=self.engine_factory,
            memory_threshold_gb=self.MEMORY_THRESHOLD_GB,
        )
        self.lifecycle_handler = SolverRunUiHandler(tab)
        self._solve_in_progress = False

    def solve(self, config: Optional[SolverConfig] = None) -> None:
        """Run selected analyses in the main thread while preserving UI responsiveness."""
        if self._solve_in_progress:
            self.lifecycle_handler.announce_stage(
                "\nSolve request ignored: analysis is already running.\n"
            )
            return

        if config is None:
            config = self.tab._build_solver_config()

        if not self.input_validator.validate_inputs(config):
            return

        self._solve_in_progress = True
        stress_type = self.input_validator.get_selected_stress_type(config)
        self.lifecycle_handler.begin_solve(config)
        coordinator = self._build_progress_coordinator(config, stress_type)

        stress_result = None
        forces_result = None
        deformation_result = None

        try:
            if stress_type is not None:
                try:
                    stress_progress = (
                        coordinator.stage_callback("stress")
                        if coordinator is not None
                        else self.lifecycle_handler.update_progress
                    )
                    stress_result = self.execution_handler.run_stress_analysis(
                        config=config,
                        stress_type=stress_type,
                        progress_callback=stress_progress,
                    )
                except SolverRunEngineCreationError as error:
                    self.lifecycle_handler.handle_engine_creation_error(
                        engine_label=error.engine_label,
                        error=error.cause,
                    )
                    self.lifecycle_handler.finish_without_results()
                    return

            if config.calculate_nodal_forces:
                self.lifecycle_handler.announce_stage("\nRunning nodal forces combination...\n")
                try:
                    forces_progress = (
                        coordinator.stage_callback("forces")
                        if coordinator is not None
                        else self.lifecycle_handler.update_progress
                    )
                    forces_result = self.execution_handler.run_nodal_forces_analysis(
                        config=config,
                        progress_callback=forces_progress,
                    )
                except SolverRunEngineCreationError as error:
                    self.lifecycle_handler.handle_engine_creation_error(
                        engine_label=error.engine_label,
                        error=error.cause,
                    )

            if config.calculate_deformation:
                self.lifecycle_handler.announce_stage("\nRunning deformation combination...\n")
                try:
                    deformation_progress = (
                        coordinator.stage_callback("deformation")
                        if coordinator is not None
                        else self.lifecycle_handler.update_progress
                    )
                    deformation_result = self.execution_handler.run_deformation_analysis(
                        config=config,
                        progress_callback=deformation_progress,
                    )
                except SolverRunEngineCreationError as error:
                    self.lifecycle_handler.handle_engine_creation_error(
                        engine_label=error.engine_label,
                        error=error.cause,
                    )

            if stress_result is not None or forces_result is not None or deformation_result is not None:
                self.lifecycle_handler.complete_solve(
                    stress_result=stress_result,
                    config=config,
                    forces_result=forces_result,
                    deformation_result=deformation_result,
                )
            else:
                self.lifecycle_handler.finish_without_results()

        except NodalForcesNotAvailableError as error:
            self.lifecycle_handler.handle_nodal_forces_unavailable(error)
        except DisplacementNotAvailableError as error:
            self.lifecycle_handler.handle_displacement_unavailable(error)
        except CylindricalCSNotFoundError as error:
            self.lifecycle_handler.handle_cylindrical_cs_error(error)
        except MemoryError as error:
            self.lifecycle_handler.handle_memory_error(error)
        except Exception as error:
            error_msg = f"{str(error)}\n\n{traceback.format_exc()}"
            self.lifecycle_handler.fail_solve(error_msg)
        finally:
            self._solve_in_progress = False

    def recompute_stress_combination_for_display(
        self,
        config: SolverConfig,
        stress_type: str,
        combination_index: int,
        combo_table_override=None,
    ) -> Optional[CombinationResult]:
        """Run a lightweight single-combination stress recompute for Display on-demand usage."""
        self.tab.progress_bar.setVisible(True)
        self.tab.progress_bar.setValue(0)
        self.tab.progress_bar.setFormat("Recomputing selected combination...")

        try:
            return self.execution_handler.run_stress_single_combination(
                config=config,
                stress_type=stress_type,
                combination_index=combination_index,
                progress_callback=self.lifecycle_handler.update_progress,
                combo_table_override=combo_table_override,
            )
        except NodalForcesNotAvailableError as error:
            self.lifecycle_handler.handle_nodal_forces_unavailable(error)
            return None
        except DisplacementNotAvailableError as error:
            self.lifecycle_handler.handle_displacement_unavailable(error)
            return None
        except CylindricalCSNotFoundError as error:
            self.lifecycle_handler.handle_cylindrical_cs_error(error)
            return None
        except MemoryError as error:
            self.lifecycle_handler.handle_memory_error(error)
            return None
        except SolverRunEngineCreationError as error:
            self.lifecycle_handler.handle_engine_creation_error(
                engine_label=error.engine_label,
                error=error.cause,
            )
            return None
        except Exception as error:
            error_msg = f"{str(error)}\n\n{traceback.format_exc()}"
            self.lifecycle_handler.fail_solve(error_msg)
            return None
        finally:
            self.tab.setEnabled(True)
            self.tab.progress_bar.setVisible(False)

    def get_combination_engine(self) -> Optional[StressCombinationEngine]:
        """Return the current stress engine instance if one exists."""
        return self.execution_handler.get_stress_engine()

    def _build_progress_coordinator(
        self,
        config: SolverConfig,
        stress_type: Optional[str],
    ) -> Optional[ProgressCoordinator]:
        """Create staged global progress mapping for active outputs."""
        stages = []
        if stress_type is not None:
            stages.append(StageSpec(key="stress", label="Stress", weight=1.0))
        if config.calculate_nodal_forces:
            stages.append(StageSpec(key="forces", label="Nodal Forces", weight=1.0))
        if config.calculate_deformation:
            stages.append(StageSpec(key="deformation", label="Deformation", weight=1.0))
        if not stages:
            return None
        return ProgressCoordinator(
            sink_callback=self.lifecycle_handler.update_progress,
            stages=stages,
        )
