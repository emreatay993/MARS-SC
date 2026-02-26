"""
Execution concern for solver analysis runs.

This module owns engine creation and computation for stress, nodal forces,
and deformation outputs.
"""

from typing import Callable, Optional

import numpy as np

from core.plasticity import (
    PlasticityDataError,
    build_material_db_from_profile,
    map_temperature_field_to_nodes,
)
from core.data_models import (
    CombinationTableData,
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
from solver.plasticity_engine import (
    apply_glinka_correction,
    apply_neuber_correction,
)
from solver.stress_engine import StressCombinationEngine
from ui.handlers.solver_engine_factory import SolverEngineFactory


class SolverRunEngineCreationError(RuntimeError):
    """Raised when a solver engine cannot be constructed from current UI inputs."""

    def __init__(self, engine_label: str, cause: Exception):
        super().__init__(str(cause))
        self.engine_label = engine_label
        self.cause = cause


class SolverAnalysisExecutor:
    """Run stress, force, and deformation analyses for the current solver configuration."""

    _PROGRESS_UNITS = 1000

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

        scalar_plasticity_requested = self._is_scalar_plasticity_requested(config, stress_type)
        use_chunked = self._should_use_chunked_processing(
            engine,
            calculate_scalar_plasticity=scalar_plasticity_requested,
        )

        self._emit_progress(progress_callback, 0.00, "Estimating memory requirements...")
        try:
            is_sufficient, estimates = engine.check_memory_available(
                raise_on_insufficient=False,
                calculate_scalar_plasticity=scalar_plasticity_requested,
            )
            if not is_sufficient:
                self._append_console(
                    f"\n[Warning] Memory Warning: Limited RAM detected.\n"
                    f"  Available: {estimates['available_bytes'] / 1e9:.2f} GB\n"
                    f"  Using memory-efficient processing.\n"
                )
        except Exception as memory_error:
            self._emit_progress(progress_callback, 0.00, f"Memory check skipped: {memory_error}")

        plasticity_ctx = None
        if self._should_apply_scalar_plasticity(config, stress_type):
            self._emit_progress(progress_callback, 0.03, "Preparing plasticity inputs...")
            try:
                plasticity_ctx = self._prepare_scalar_plasticity_context(config)
            except PlasticityDataError as error:
                raise ValueError(f"Plasticity input error: {error}") from error

        if config.combination_history_mode and config.selected_node_id:
            self._emit_progress(
                progress_callback,
                0.08,
                f"Computing history for node {config.selected_node_id}...",
            )
            engine_progress = self._map_progress_callback(progress_callback, 0.08, 0.95)
            combo_indices, stress_values = engine.compute_single_node_history_fast(
                config.selected_node_id,
                stress_type,
                progress_callback=engine_progress,
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
            if plasticity_ctx is not None:
                self._emit_progress(
                    progress_callback,
                    0.95,
                    "Applying plasticity correction to history...",
                )
                self._apply_plasticity_to_history_result(result, plasticity_ctx)
        else:
            self._emit_progress(progress_callback, 0.08, "Starting envelope analysis...")
            chunk_transform = None
            chunk_plasticity_state = None
            if plasticity_ctx is not None:
                chunk_transform, chunk_plasticity_state = self._build_chunked_scalar_plasticity_transform(
                    engine,
                    plasticity_ctx,
                )

            engine_progress = self._map_progress_callback(progress_callback, 0.08, 0.95)
            if use_chunked:
                result = engine.compute_full_analysis_chunked(
                    stress_type=stress_type,
                    progress_callback=engine_progress,
                    calculate_scalar_plasticity=scalar_plasticity_requested,
                    chunk_results_transform=chunk_transform,
                )
            else:
                result = engine.compute_full_analysis_auto(
                    stress_type=stress_type,
                    progress_callback=engine_progress,
                    memory_threshold_gb=self.memory_threshold_gb,
                    calculate_scalar_plasticity=scalar_plasticity_requested,
                    chunk_results_transform=chunk_transform,
                )
            if plasticity_ctx is not None:
                chunked_plasticity_applied = (
                    chunk_plasticity_state is not None
                    and chunk_plasticity_state["chunks_processed"] > 0
                    and result.all_combo_results is None
                )
                if chunked_plasticity_applied:
                    self._apply_chunked_plasticity_metadata(
                        result,
                        plasticity_ctx,
                        chunk_plasticity_state,
                    )
                else:
                    self._emit_progress(
                        progress_callback,
                        0.95,
                        "Applying plasticity correction to envelope...",
                    )
                    self._apply_plasticity_to_envelope_result(result, plasticity_ctx)

        self._emit_progress(progress_callback, 1.0, "Complete")
        return result

    def run_stress_single_combination(
        self,
        config: SolverConfig,
        stress_type: str,
        combination_index: int,
        progress_callback: Callable[[int, int, str], None],
        combo_table_override: Optional[CombinationTableData] = None,
    ) -> CombinationResult:
        """Recompute one stress combination over all scoped nodes."""
        engine = self._create_stress_engine(combo_table_override=combo_table_override)
        self._stress_engine = engine

        scalar_plasticity_requested = self._is_scalar_plasticity_requested(config, stress_type)
        plasticity_ctx = None
        if self._should_apply_scalar_plasticity(config, stress_type):
            self._emit_progress(progress_callback, 0.02, "Preparing plasticity inputs...")
            try:
                plasticity_ctx = self._prepare_scalar_plasticity_context(config)
            except PlasticityDataError as error:
                raise ValueError(f"Plasticity input error: {error}") from error

        chunk_transform = None
        if plasticity_ctx is not None:
            full_node_ids = np.asarray(engine.scoping.ids, dtype=int)
            temperatures = np.asarray(
                self._resolve_temperatures(full_node_ids, plasticity_ctx),
                dtype=np.float64,
            )

            def _transform_chunk_values(
                chunk_values: np.ndarray,
                _chunk_node_ids: np.ndarray,
                chunk_start: int,
                chunk_end: int,
            ) -> np.ndarray:
                temp_chunk = temperatures[chunk_start:chunk_end]
                corrected, _strain = self._apply_scalar_plasticity(
                    np.asarray(chunk_values, dtype=np.float64),
                    temp_chunk,
                    plasticity_ctx,
                )
                return corrected

            chunk_transform = _transform_chunk_values

        chunk_progress = self._map_progress_callback(progress_callback, 0.05, 0.99)
        result = engine.compute_single_combination_chunked(
            combination_index=combination_index,
            stress_type=stress_type,
            progress_callback=chunk_progress,
            calculate_scalar_plasticity=scalar_plasticity_requested,
            chunk_values_transform=chunk_transform,
        )

        metadata = getattr(result, "metadata", None)
        if metadata is None:
            metadata = {}
        metadata["mode"] = "single_combination_recompute"
        metadata["combination_index"] = int(combination_index)
        if plasticity_ctx is not None:
            metadata["plasticity"] = {
                "method": plasticity_ctx["method"],
                "note": "On-demand corrected single combination.",
            }
        setattr(result, "metadata", metadata)
        self._emit_progress(progress_callback, 1.0, "Recompute complete")
        return result

    def run_nodal_forces_analysis(
        self,
        config: SolverConfig,
        progress_callback: Callable[[int, int, str], None],
    ) -> NodalForcesResult:
        """Run nodal-forces combination analysis with envelope or single-node history mode."""
        engine = self._create_nodal_forces_engine(config)
        self._nodal_forces_engine = engine

        if config.combination_history_mode and config.selected_node_id:
            single_node_scoping = engine.reader1.create_single_node_scoping(
                config.selected_node_id,
                engine.scoping,
            )
            self._emit_progress(progress_callback, 0.00, "Validating nodal forces availability...")
            is_valid, error_msg = engine.validate_nodal_forces_availability(
                nodal_scoping=single_node_scoping
            )
            if not is_valid:
                raise NodalForcesNotAvailableError(error_msg)

            history_progress = self._map_progress_callback(progress_callback, 0.08, 0.98)
            combo_indices, fx, fy, fz, magnitude = engine.compute_single_node_history_fast(
                config.selected_node_id,
                progress_callback=history_progress,
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
            self._emit_progress(progress_callback, 0.00, "Validating nodal forces availability...")
            is_valid, error_msg = engine.validate_nodal_forces_availability()
            if not is_valid:
                raise NodalForcesNotAvailableError(error_msg)

            self._emit_progress(progress_callback, 0.08, "Loading nodal forces from RST files...")
            preload_progress = self._map_progress_callback(progress_callback, 0.08, 0.45)
            engine.preload_force_data(progress_callback=preload_progress)

            self._emit_progress(progress_callback, 0.45, "Computing nodal forces for all combinations...")
            compute_progress = self._map_progress_callback(progress_callback, 0.45, 0.98)
            result = engine.compute_full_analysis(progress_callback=compute_progress)

        self._emit_progress(progress_callback, 1.0, "Nodal forces complete")
        return result

    def run_deformation_analysis(
        self,
        config: SolverConfig,
        progress_callback: Callable[[int, int, str], None],
    ) -> DeformationResult:
        """Run displacement-combination analysis with envelope or single-node history mode."""
        engine = self._create_deformation_engine(config)
        self._deformation_engine = engine

        is_history_mode = bool(config.combination_history_mode and config.selected_node_id)

        if is_history_mode and not engine.uses_cylindrical_cs:
            single_node_scoping = engine.reader1.create_single_node_scoping(
                config.selected_node_id,
                engine.scoping,
            )
            self._emit_progress(progress_callback, 0.00, "Validating displacement availability...")
            is_valid, error_msg = engine.validate_displacement_availability(
                nodal_scoping=single_node_scoping
            )
            if not is_valid:
                raise DisplacementNotAvailableError(error_msg)

            history_progress = self._map_progress_callback(progress_callback, 0.12, 0.98)
            combo_indices, ux, uy, uz, magnitude = engine.compute_single_node_history_fast(
                config.selected_node_id,
                progress_callback=history_progress,
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
            self._emit_progress(progress_callback, 0.00, "Validating displacement availability...")
            is_valid, error_msg = engine.validate_displacement_availability()
            if not is_valid:
                raise DisplacementNotAvailableError(error_msg)

            if engine.uses_cylindrical_cs:
                self._emit_progress(
                    progress_callback,
                    0.06,
                    f"Validating coordinate system {config.deformation_cylindrical_cs_id}...",
                )
                is_valid, error_msg = engine.validate_cylindrical_cs()
                if not is_valid:
                    raise CylindricalCSNotFoundError(error_msg)
                self._append_console(
                    f"  Cylindrical CS {config.deformation_cylindrical_cs_id} validated - "
                    f"results will be in cylindrical coordinates (R, Theta, Z)\n"
                )

            preload_start = 0.12
            preload_end = 0.50 if is_history_mode else 0.45
            self._emit_progress(progress_callback, preload_start, "Loading displacement from RST files...")
            preload_progress = self._map_progress_callback(
                progress_callback,
                preload_start,
                preload_end,
            )
            engine.preload_displacement_data(progress_callback=preload_progress)

            if is_history_mode:
                self._emit_progress(
                    progress_callback,
                    preload_end,
                    f"Computing displacement history for node {config.selected_node_id}...",
                )
                combo_indices, ux, uy, uz, magnitude = engine.compute_single_node_history(
                    config.selected_node_id
                )
                self._emit_progress(
                    progress_callback,
                    0.98,
                    f"Computed displacement history for node {config.selected_node_id}.",
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
                self._emit_progress(progress_callback, 0.45, "Computing deformation for all combinations...")
                compute_progress = self._map_progress_callback(progress_callback, 0.45, 0.98)
                result = engine.compute_full_analysis(progress_callback=compute_progress)

        self._emit_progress(progress_callback, 1.0, "Deformation complete")
        return result

    def get_stress_engine(self) -> Optional[StressCombinationEngine]:
        """Return the most recent stress engine instance if one was created."""
        return self._stress_engine

    def _create_stress_engine(
        self,
        combo_table_override: Optional[CombinationTableData] = None,
    ) -> StressCombinationEngine:
        try:
            reader1, reader2, nodal_scoping, combo_table = self._get_common_inputs(
                combo_table_override=combo_table_override
            )
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

    def _get_common_inputs(
        self,
        combo_table_override: Optional[CombinationTableData] = None,
    ):
        reader1 = self.tab.file_handler.base_reader
        reader2 = self.tab.file_handler.combine_reader
        nodal_scoping = self.tab.get_nodal_scoping_for_selected_named_selection()
        combo_table = (
            combo_table_override
            if combo_table_override is not None
            else self.tab.get_combination_table_data()
        )
        return reader1, reader2, nodal_scoping, combo_table

    def _should_use_chunked_processing(
        self,
        engine: StressCombinationEngine,
        calculate_scalar_plasticity: bool = False,
    ) -> bool:
        """Determine if chunked processing is recommended for current model size."""
        try:
            num_nodes = len(engine.scoping.ids)
            estimates = engine.estimate_memory_requirements(
                num_nodes,
                calculate_scalar_plasticity=calculate_scalar_plasticity,
            )
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

    def _map_progress_callback(
        self,
        progress_callback: Callable[[int, int, str], None],
        start_fraction: float,
        end_fraction: float,
    ) -> Callable[[int, int, str], None]:
        """Map a local progress callback into one fractional range of global progress."""
        start = min(max(start_fraction, 0.0), 1.0)
        end = min(max(end_fraction, start), 1.0)

        def _mapped(current: int, total: int, message: str) -> None:
            if total <= 0:
                progress_callback(0, 0, message)
                return
            ratio = float(current) / float(total)
            ratio = min(max(ratio, 0.0), 1.0)
            mapped_ratio = start + ((end - start) * ratio)
            mapped_current = int(round(mapped_ratio * self._PROGRESS_UNITS))
            progress_callback(mapped_current, self._PROGRESS_UNITS, message)

        return _mapped

    def _emit_progress(
        self,
        progress_callback: Callable[[int, int, str], None],
        fraction: float,
        message: str,
    ) -> None:
        """Emit direct progress update using shared precision."""
        clamped = min(max(fraction, 0.0), 1.0)
        current = int(round(clamped * self._PROGRESS_UNITS))
        progress_callback(current, self._PROGRESS_UNITS, message)

    @staticmethod
    def _is_scalar_plasticity_requested(config: SolverConfig, stress_type: str) -> bool:
        """Cheap pre-check used for memory estimation before building material DB."""
        if config.plasticity is None or not config.plasticity.enabled:
            return False
        if stress_type != "von_mises":
            return False
        method_raw = (config.plasticity.method or "").lower()
        return ("neuber" in method_raw) or ("glinka" in method_raw)

    def _should_apply_scalar_plasticity(self, config: SolverConfig, stress_type: str) -> bool:
        """Return True when Neuber/Glinka correction should run for stress output."""
        if config.plasticity is None or not config.plasticity.is_active:
            return False

        if stress_type != "von_mises":
            self._append_console(
                "Plasticity correction currently applies only to Von Mises output; "
                "skipping for selected stress type.\n"
            )
            return False

        method_raw = (config.plasticity.method or "").lower()
        if "neuber" not in method_raw and "glinka" not in method_raw:
            self._append_console(
                f"Unsupported plasticity method '{config.plasticity.method}'. "
                "Only Neuber/Glinka are enabled in this workflow.\n"
            )
            return False

        if "ibg" in method_raw or "buczynski" in method_raw:
            self._append_console(
                "IBG is disabled in this workflow. Select Neuber or Glinka.\n"
            )
            return False

        return True

    def _prepare_scalar_plasticity_context(self, config: SolverConfig) -> dict:
        """Build shared runtime context used by Neuber/Glinka stress correction."""
        assert config.plasticity is not None
        pcfg = config.plasticity

        method_raw = (pcfg.method or "neuber").lower()
        method = "neuber" if "neuber" in method_raw else "glinka"

        material_db = build_material_db_from_profile(pcfg.material_profile)
        default_temperature = (
            float(pcfg.default_temperature)
            if pcfg.default_temperature is not None
            else float(material_db.TEMP[0])
        )
        use_plateau = (pcfg.extrapolation_mode or "linear").strip().lower() == "plateau"

        self._append_console(
            f"Applying {method.title()} plasticity correction "
            f"(extrapolation: {'Plateau' if use_plateau else 'Linear'}).\n"
        )

        return {
            "method": method,
            "material_db": material_db,
            "tolerance": float(pcfg.tolerance),
            "max_iterations": int(pcfg.max_iterations),
            "use_plateau": use_plateau,
            "temperature_field": pcfg.temperature_field,
            "temperature_column": pcfg.temperature_column,
            "default_temperature": default_temperature,
        }

    def _resolve_temperatures(self, node_ids: np.ndarray, plasticity_ctx: dict) -> np.ndarray:
        """Resolve nodal temperatures aligned with ``node_ids``."""
        temperature_field = plasticity_ctx["temperature_field"]
        if temperature_field is None:
            return np.full(node_ids.shape[0], plasticity_ctx["default_temperature"], dtype=np.float64)

        return map_temperature_field_to_nodes(
            temperature_data=temperature_field,
            node_ids=np.asarray(node_ids, dtype=int),
            column_name=plasticity_ctx["temperature_column"],
            default_temperature=plasticity_ctx["default_temperature"],
        )

    def _apply_scalar_plasticity(
        self,
        stress_values: np.ndarray,
        temperature_values: np.ndarray,
        plasticity_ctx: dict,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply Neuber or Glinka to scalar stress values."""
        kwargs = {
            "tol": plasticity_ctx["tolerance"],
            "max_iterations": plasticity_ctx["max_iterations"],
            "use_plateau": plasticity_ctx["use_plateau"],
        }
        material_db = plasticity_ctx["material_db"]
        if plasticity_ctx["method"] == "neuber":
            return apply_neuber_correction(stress_values, temperature_values, material_db, **kwargs)
        return apply_glinka_correction(stress_values, temperature_values, material_db, **kwargs)

    def _build_chunked_scalar_plasticity_transform(
        self,
        engine: StressCombinationEngine,
        plasticity_ctx: dict,
    ) -> tuple[Callable[[np.ndarray, np.ndarray, int, int], np.ndarray], dict]:
        """
        Build per-chunk scalar-plasticity transform for chunked envelope processing.

        The returned callback receives the full combo-vs-node matrix for each chunk
        and must return the corrected matrix before envelope reduction.
        """
        full_node_ids = np.asarray(engine.scoping.ids, dtype=int)
        temperatures = np.asarray(
            self._resolve_temperatures(full_node_ids, plasticity_ctx),
            dtype=np.float64,
        )
        num_nodes = full_node_ids.size

        state = {
            "chunks_processed": 0,
            "temperatures": temperatures,
            "elastic_max_over_combo": np.full(num_nodes, -np.inf, dtype=np.float64),
            "elastic_min_over_combo": np.full(num_nodes, np.inf, dtype=np.float64),
            "elastic_combo_of_max": np.zeros(num_nodes, dtype=np.int32),
            "elastic_combo_of_min": np.zeros(num_nodes, dtype=np.int32),
            "plastic_strain_at_max": np.zeros(num_nodes, dtype=np.float64),
        }

        def _transform_chunk(
            chunk_results: np.ndarray,
            _chunk_node_ids: np.ndarray,
            chunk_start: int,
            chunk_end: int,
        ) -> np.ndarray:
            chunk_slice = slice(chunk_start, chunk_end)
            elastic_chunk = np.asarray(chunk_results, dtype=np.float64)
            chunk_size = elastic_chunk.shape[1]
            num_combos = elastic_chunk.shape[0]

            state["elastic_max_over_combo"][chunk_slice] = np.max(elastic_chunk, axis=0)
            state["elastic_min_over_combo"][chunk_slice] = np.min(elastic_chunk, axis=0)
            state["elastic_combo_of_max"][chunk_slice] = np.argmax(elastic_chunk, axis=0)
            state["elastic_combo_of_min"][chunk_slice] = np.argmin(elastic_chunk, axis=0)

            temp_chunk = temperatures[chunk_slice]
            flat_stress = elastic_chunk.reshape(-1)
            flat_temp = np.tile(temp_chunk, num_combos)
            corrected_flat, strain_flat = self._apply_scalar_plasticity(
                flat_stress,
                flat_temp,
                plasticity_ctx,
            )
            corrected_chunk = corrected_flat.reshape(elastic_chunk.shape)
            strain_chunk = strain_flat.reshape(elastic_chunk.shape)

            corrected_argmax = np.argmax(corrected_chunk, axis=0)
            node_idx = np.arange(chunk_size)
            state["plastic_strain_at_max"][chunk_slice] = strain_chunk[corrected_argmax, node_idx]
            state["chunks_processed"] += 1
            return corrected_chunk

        return _transform_chunk, state

    @staticmethod
    def _apply_chunked_plasticity_metadata(
        result: CombinationResult,
        plasticity_ctx: dict,
        state: dict,
    ) -> None:
        """Attach elastic and corrected scalar-plasticity metadata for chunked runs."""
        metadata = getattr(result, "metadata", None)
        if metadata is None:
            metadata = {}

        metadata["plasticity"] = {
            "method": plasticity_ctx["method"],
            "plastic_strain_at_max": np.asarray(state["plastic_strain_at_max"], dtype=np.float64),
            "temperatures": np.asarray(state["temperatures"], dtype=np.float64),
            "note": "Corrected from full combination matrix in chunked mode.",
        }
        metadata["elastic_max_over_combo"] = np.asarray(state["elastic_max_over_combo"], dtype=np.float64)
        metadata["elastic_min_over_combo"] = np.asarray(state["elastic_min_over_combo"], dtype=np.float64)
        metadata["elastic_combo_of_max"] = np.asarray(state["elastic_combo_of_max"], dtype=np.int32)
        metadata["elastic_combo_of_min"] = np.asarray(state["elastic_combo_of_min"], dtype=np.int32)
        setattr(result, "metadata", metadata)

    def _apply_plasticity_to_history_result(self, result: CombinationResult, plasticity_ctx: dict) -> None:
        """Apply scalar plasticity to a single-node combination history result."""
        metadata = getattr(result, "metadata", None)
        if metadata is None:
            metadata = {}
        node_id = int(metadata["node_id"])
        stress_values = np.asarray(metadata["stress_values"], dtype=np.float64)

        node_temp = float(self._resolve_temperatures(np.array([node_id]), plasticity_ctx)[0])
        temp_array = np.full(stress_values.shape, node_temp, dtype=np.float64)
        corrected, strain = self._apply_scalar_plasticity(stress_values, temp_array, plasticity_ctx)

        result.all_combo_results = corrected.reshape(-1, 1)
        metadata["stress_values"] = corrected
        metadata["elastic_vm"] = stress_values
        metadata["plasticity_overlay"] = {
            "corrected_vm": corrected,
            "plastic_strain": strain,
            "elastic_vm": stress_values,
        }
        metadata["plasticity"] = {
            "method": plasticity_ctx["method"],
            "temperature": node_temp,
        }
        setattr(result, "metadata", metadata)

    def _apply_plasticity_to_envelope_result(self, result: CombinationResult, plasticity_ctx: dict) -> None:
        """Apply scalar plasticity to envelope outputs for all nodes."""
        temperatures = self._resolve_temperatures(result.node_ids, plasticity_ctx)
        metadata = getattr(result, "metadata", None)
        if metadata is None:
            metadata = {}

        elastic_max = None
        elastic_min = None
        elastic_combo_of_max = None
        elastic_combo_of_min = None

        if result.all_combo_results is not None:
            elastic_all = np.asarray(result.all_combo_results, dtype=np.float64)
            num_combos, num_nodes = elastic_all.shape
            elastic_max = np.max(elastic_all, axis=0)
            elastic_min = np.min(elastic_all, axis=0)
            elastic_combo_of_max = np.argmax(elastic_all, axis=0)
            elastic_combo_of_min = np.argmin(elastic_all, axis=0)

            flat_stress = elastic_all.reshape(-1)
            flat_temp = np.tile(np.asarray(temperatures, dtype=np.float64), num_combos)
            corrected_flat, strain_flat = self._apply_scalar_plasticity(flat_stress, flat_temp, plasticity_ctx)

            corrected_all = corrected_flat.reshape(num_combos, num_nodes)
            strain_all = strain_flat.reshape(num_combos, num_nodes)

            result.all_combo_results = corrected_all
            result.max_over_combo = np.max(corrected_all, axis=0)
            result.min_over_combo = np.min(corrected_all, axis=0)
            result.combo_of_max = np.argmax(corrected_all, axis=0)
            result.combo_of_min = np.argmin(corrected_all, axis=0)

            node_idx = np.arange(num_nodes)
            peak_strain = strain_all[result.combo_of_max, node_idx]

            metadata["plasticity"] = {
                "method": plasticity_ctx["method"],
                "plastic_strain_at_max": peak_strain,
                "temperatures": temperatures,
            }
        elif result.max_over_combo is not None:
            elastic_max = np.asarray(result.max_over_combo, dtype=np.float64).copy()
            if result.min_over_combo is not None:
                elastic_min = np.asarray(result.min_over_combo, dtype=np.float64).copy()
            if result.combo_of_max is not None:
                elastic_combo_of_max = np.asarray(result.combo_of_max, dtype=np.int32).copy()
            if result.combo_of_min is not None:
                elastic_combo_of_min = np.asarray(result.combo_of_min, dtype=np.int32).copy()

            corrected_max, strain = self._apply_scalar_plasticity(elastic_max, temperatures, plasticity_ctx)
            result.max_over_combo = corrected_max
            metadata["plasticity"] = {
                "method": plasticity_ctx["method"],
                "plastic_strain_at_max": strain,
                "temperatures": temperatures,
                "note": "Corrected from envelope values (fallback path).",
            }

        if elastic_max is not None:
            metadata["elastic_max_over_combo"] = np.asarray(elastic_max, dtype=np.float64)
        if elastic_min is not None:
            metadata["elastic_min_over_combo"] = np.asarray(elastic_min, dtype=np.float64)
        if elastic_combo_of_max is not None:
            metadata["elastic_combo_of_max"] = np.asarray(elastic_combo_of_max, dtype=np.int32)
        if elastic_combo_of_min is not None:
            metadata["elastic_combo_of_min"] = np.asarray(elastic_combo_of_min, dtype=np.int32)

        setattr(result, "metadata", metadata)
