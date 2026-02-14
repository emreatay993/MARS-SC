"""
Orchestration for solver-tab analysis runs.

This controller keeps the solve path focused on flow coordination:
validation -> execution -> lifecycle completion/error handling.
"""

import traceback
from typing import Optional

from core.data_models import SolverConfig
from file_io.dpf_reader import (
    DisplacementNotAvailableError,
    NodalForcesNotAvailableError,
)
from solver.deformation_engine import CylindricalCSNotFoundError
from solver.stress_engine import StressCombinationEngine
from ui.handlers.solver_run_execution_handler import (
    SolverRunEngineCreationError,
    SolverRunExecutionHandler,
)
from ui.handlers.solver_run_lifecycle_handler import SolverRunLifecycleHandler
from ui.handlers.solver_engine_factory import SolverEngineFactory
from ui.handlers.solver_input_validator import SolverInputValidator


class SolveRunController:
    """Coordinate the complete analysis run workflow for SolverTab."""

    MEMORY_THRESHOLD_GB = 2.0

    def __init__(self, tab):
        self.tab = tab
        self.input_validator = SolverInputValidator(tab)
        self.engine_factory = SolverEngineFactory()
        self.execution_handler = SolverRunExecutionHandler(
            tab=tab,
            engine_factory=self.engine_factory,
            memory_threshold_gb=self.MEMORY_THRESHOLD_GB,
        )
        self.lifecycle_handler = SolverRunLifecycleHandler(tab)

    def solve(self, config: Optional[SolverConfig] = None) -> None:
        """Run selected analyses in the main thread while preserving UI responsiveness."""
        if config is None:
            config = self.tab._build_solver_config()

        if not self.input_validator.validate_inputs(config):
            return

        stress_type = self.input_validator.get_selected_stress_type(config)
        self.lifecycle_handler.begin_solve(config)

        stress_result = None
        forces_result = None
        deformation_result = None

        try:
            if stress_type is not None:
                try:
                    stress_result = self.execution_handler.run_stress_analysis(
                        config=config,
                        stress_type=stress_type,
                        progress_callback=self.lifecycle_handler.update_progress,
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
                    forces_result = self.execution_handler.run_nodal_forces_analysis(
                        config=config,
                        progress_callback=self.lifecycle_handler.update_progress,
                    )
                except SolverRunEngineCreationError as error:
                    self.lifecycle_handler.handle_engine_creation_error(
                        engine_label=error.engine_label,
                        error=error.cause,
                    )

            if config.calculate_deformation:
                self.lifecycle_handler.announce_stage("\nRunning deformation combination...\n")
                try:
                    deformation_result = self.execution_handler.run_deformation_analysis(
                        config=config,
                        progress_callback=self.lifecycle_handler.update_progress,
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

    def get_combination_engine(self) -> Optional[StressCombinationEngine]:
        """Return the current stress engine instance if one exists."""
        return self.execution_handler.get_stress_engine()
