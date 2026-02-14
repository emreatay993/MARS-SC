"""
Execution concern for solver analysis runs.

This module owns engine creation and computation for stress, nodal forces,
and deformation outputs.
"""

from typing import Callable, Optional

import numpy as np

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
from solver.deformation_engine import (
    CylindricalCSNotFoundError,
    DeformationCombinationEngine,
)
from solver.nodal_forces_engine import NodalForcesCombinationEngine
from solver.stress_engine import StressCombinationEngine
from ui.handlers.solver_engine_factory import SolverEngineFactory


class SolverRunEngineCreationError(RuntimeError):
    """Raised when a solver engine cannot be constructed from current UI inputs."""

    def __init__(self, engine_label: str, cause: Exception):
        super().__init__(str(cause))
        self.engine_label = engine_label
        self.cause = cause


class SolverRunExecutionHandler:
    """Run stress, force, and deformation analyses for the current solver configuration."""

    def __init__(
        self,
        tab,
        engine_factory: SolverEngineFactory,
        memory_threshold_gb: float = 2.0,
    ):
        self.tab = tab
        self.engine_factory = engine_factory
        self.memory_threshold_gb = memory_threshold_gb

        self._stress_engine: Optional[StressCombinationEngine] = None
        self._nodal_forces_engine: Optional[NodalForcesCombinationEngine] = None
        self._deformation_engine: Optional[DeformationCombinationEngine] = None

    def run_stress_analysis(
        self,
        config: SolverConfig,
        stress_type: str,
        progress_callback: Callable[[int, int, str], None],
    ) -> CombinationResult:
        """Run stress-combination analysis with envelope or single-node history mode."""
        engine = self._create_stress_engine()
        self._stress_engine = engine

        use_chunked = self._should_use_chunked_processing(engine)

        progress_callback(0, 100, "Estimating memory requirements...")
        try:
            is_sufficient, estimates = engine.check_memory_available(raise_on_insufficient=False)
            if not is_sufficient:
                self._append_console(
                    f"\n[Warning] Memory Warning: Limited RAM detected.\n"
                    f"  Available: {estimates['available_bytes'] / 1e9:.2f} GB\n"
                    f"  Using memory-efficient processing.\n"
                )
        except Exception as memory_error:
            progress_callback(0, 100, f"Memory check skipped: {memory_error}")

        if config.combination_history_mode and config.selected_node_id:
            progress_callback(0, 100, f"Computing history for node {config.selected_node_id}...")
            combo_indices, stress_values = engine.compute_single_node_history_fast(
                config.selected_node_id,
                stress_type,
                progress_callback=progress_callback,
            )

            result = CombinationResult(
                node_ids=np.array([config.selected_node_id]),
                node_coords=np.array([[0, 0, 0]]),
                result_type=stress_type,
                all_combo_results=stress_values.reshape(1, -1).T,
            )
            result.metadata = {
                "mode": "history",
                "node_id": config.selected_node_id,
                "combination_indices": combo_indices,
                "stress_values": stress_values,
            }
        else:
            progress_callback(5, 100, "Starting envelope analysis...")
            if use_chunked:
                result = engine.compute_full_analysis_chunked(
                    stress_type=stress_type,
                    progress_callback=progress_callback,
                )
            else:
                result = engine.compute_full_analysis_auto(
                    stress_type=stress_type,
                    progress_callback=progress_callback,
                    memory_threshold_gb=self.memory_threshold_gb,
                )

        progress_callback(100, 100, "Complete")
        return result

    def run_nodal_forces_analysis(
        self,
        config: SolverConfig,
        progress_callback: Callable[[int, int, str], None],
    ) -> NodalForcesResult:
        """Run nodal-forces combination analysis with envelope or single-node history mode."""
        engine = self._create_nodal_forces_engine(config)
        self._nodal_forces_engine = engine

        progress_callback(0, 100, "Validating nodal forces availability...")
        is_valid, error_msg = engine.validate_nodal_forces_availability()
        if not is_valid:
            raise NodalForcesNotAvailableError(error_msg)

        progress_callback(5, 100, "Loading nodal forces from RST files...")
        engine.preload_force_data(progress_callback=progress_callback)

        if config.combination_history_mode and config.selected_node_id:
            progress_callback(50, 100, f"Computing force history for node {config.selected_node_id}...")
            combo_indices, fx, fy, fz, magnitude = engine.compute_single_node_history(
                config.selected_node_id
            )

            result = NodalForcesResult(
                node_ids=np.array([config.selected_node_id]),
                node_coords=np.array([[0, 0, 0]]),
                all_combo_fx=fx.reshape(-1, 1),
                all_combo_fy=fy.reshape(-1, 1),
                all_combo_fz=fz.reshape(-1, 1),
                force_unit=engine.force_unit,
            )
            result.metadata = {
                "mode": "history",
                "node_id": config.selected_node_id,
                "combination_indices": combo_indices,
                "fx": fx,
                "fy": fy,
                "fz": fz,
                "magnitude": magnitude,
            }
        else:
            progress_callback(10, 100, "Computing nodal forces for all combinations...")
            result = engine.compute_full_analysis(progress_callback=progress_callback)

        progress_callback(100, 100, "Nodal forces complete")
        return result

    def run_deformation_analysis(
        self,
        config: SolverConfig,
        progress_callback: Callable[[int, int, str], None],
    ) -> DeformationResult:
        """Run displacement-combination analysis with envelope or single-node history mode."""
        engine = self._create_deformation_engine(config)
        self._deformation_engine = engine

        progress_callback(0, 100, "Validating displacement availability...")
        is_valid, error_msg = engine.validate_displacement_availability()
        if not is_valid:
            raise DisplacementNotAvailableError(error_msg)

        if engine.uses_cylindrical_cs:
            progress_callback(2, 100, f"Validating coordinate system {config.deformation_cylindrical_cs_id}...")
            is_valid, error_msg = engine.validate_cylindrical_cs()
            if not is_valid:
                raise CylindricalCSNotFoundError(error_msg)
            self._append_console(
                f"  Cylindrical CS {config.deformation_cylindrical_cs_id} validated - "
                f"results will be in cylindrical coordinates (R, Theta, Z)\n"
            )

        progress_callback(5, 100, "Loading displacement from RST files...")
        engine.preload_displacement_data(progress_callback=progress_callback)

        if config.combination_history_mode and config.selected_node_id:
            progress_callback(
                50, 100, f"Computing displacement history for node {config.selected_node_id}..."
            )
            combo_indices, ux, uy, uz, magnitude = engine.compute_single_node_history(
                config.selected_node_id
            )

            result = DeformationResult(
                node_ids=np.array([config.selected_node_id]),
                node_coords=np.array([[0, 0, 0]]),
                all_combo_ux=ux.reshape(-1, 1),
                all_combo_uy=uy.reshape(-1, 1),
                all_combo_uz=uz.reshape(-1, 1),
                displacement_unit=engine.displacement_unit,
            )
            result.metadata = {
                "mode": "history",
                "node_id": config.selected_node_id,
                "combination_indices": combo_indices,
                "ux": ux,
                "uy": uy,
                "uz": uz,
                "magnitude": magnitude,
            }
        else:
            progress_callback(10, 100, "Computing deformation for all combinations...")
            result = engine.compute_full_analysis(progress_callback=progress_callback)

        progress_callback(100, 100, "Deformation complete")
        return result

    def get_stress_engine(self) -> Optional[StressCombinationEngine]:
        """Return the most recent stress engine instance if one was created."""
        return self._stress_engine

    def _create_stress_engine(self) -> StressCombinationEngine:
        try:
            reader1, reader2, nodal_scoping, combo_table = self._get_common_inputs()
            return self.engine_factory.create_stress_engine(
                reader1=reader1,
                reader2=reader2,
                nodal_scoping=nodal_scoping,
                combo_table=combo_table,
            )
        except Exception as error:
            raise SolverRunEngineCreationError("stress combination", error) from error

    def _create_nodal_forces_engine(self, config: SolverConfig) -> NodalForcesCombinationEngine:
        try:
            reader1, reader2, nodal_scoping, combo_table = self._get_common_inputs()
            return self.engine_factory.create_nodal_forces_engine(
                reader1=reader1,
                reader2=reader2,
                nodal_scoping=nodal_scoping,
                combo_table=combo_table,
                rotate_to_global=config.nodal_forces_rotate_to_global,
            )
        except Exception as error:
            raise SolverRunEngineCreationError("nodal forces combination", error) from error

    def _create_deformation_engine(self, config: SolverConfig) -> DeformationCombinationEngine:
        try:
            reader1, reader2, nodal_scoping, combo_table = self._get_common_inputs()
            return self.engine_factory.create_deformation_engine(
                reader1=reader1,
                reader2=reader2,
                nodal_scoping=nodal_scoping,
                combo_table=combo_table,
                cylindrical_cs_id=config.deformation_cylindrical_cs_id,
            )
        except Exception as error:
            raise SolverRunEngineCreationError("deformation combination", error) from error

    def _get_common_inputs(self):
        reader1 = self.tab.file_handler.base_reader
        reader2 = self.tab.file_handler.combine_reader
        nodal_scoping = self.tab.get_nodal_scoping_for_selected_named_selection()
        combo_table = self.tab.get_combination_table_data()
        return reader1, reader2, nodal_scoping, combo_table

    def _should_use_chunked_processing(self, engine: StressCombinationEngine) -> bool:
        """Determine if chunked processing is recommended for current model size."""
        try:
            num_nodes = len(engine.scoping.ids)
            estimates = engine.estimate_memory_requirements(num_nodes)
            available = engine._get_available_memory()
            threshold = available * 0.8

            if estimates["total_bytes"] > threshold:
                self._append_console(
                    f"Large model detected ({num_nodes:,} nodes, "
                    f"~{estimates['total_bytes'] / 1e9:.2f} GB estimated).\n"
                    f"Using memory-efficient chunked processing.\n"
                )
                return True

            return False
        except Exception:
            return False

    def _append_console(self, message: str) -> None:
        self.tab.console_textbox.append(message)
