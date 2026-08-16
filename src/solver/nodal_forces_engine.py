"""
Nodal Forces Combination Engine for MARS-SC (Solution Combination).

Performs linear combination of nodal forces from two analyses and computes
force envelopes over combinations.

The combination formula is:
    F_combined = Σ(α_i × F_A1_i) + Σ(β_j × F_A2_j)

Where:
    - α_i are coefficients for Analysis 1 load steps
    - β_j are coefficients for Analysis 2 load steps
    - F_A1_i and F_A2_j are force vectors for each load step
"""

from typing import Dict, Tuple, Optional, Callable, List
import numpy as np

MATRIX_TEMP_TARGET_BYTES = 64 * 1024 * 1024

from file_io.dpf_reader import (
    DPFAnalysisReader,
    NodalForcesNotAvailableError,
    scale_force_field,
    add_force_fields,
    compute_force_magnitude,
    DPF_AVAILABLE,
)
from core.data_models import CombinationTableData, NodalForcesResult
from utils.constants import MSG_NODAL_FORCES_ANSYS

if DPF_AVAILABLE:
    from ansys.dpf import core as dpf


class NodalForcesCombinationEngine:
    """
    Performs linear combination of nodal forces from two analyses.
    
    This engine preloads force data from both analyses and then computes
    combined forces for each combination defined in the combination table.
    
    Attributes:
        reader1: DPFAnalysisReader for Analysis 1 (base analysis).
        reader2: DPFAnalysisReader for Analysis 2 (analysis to combine).
        scoping: DPF Scoping defining which nodes to process.
        table: CombinationTableData with combination coefficients.
    """
    
    def __init__(
        self,
        reader1: DPFAnalysisReader,
        reader2: DPFAnalysisReader,
        nodal_scoping,  # dpf.Scoping
        combination_table: CombinationTableData,
        rotate_to_global: bool = True
    ):
        """
        Initialize the nodal forces combination engine.
        
        Args:
            reader1: DPFAnalysisReader for Analysis 1 (base).
            reader2: DPFAnalysisReader for Analysis 2 (to combine).
            nodal_scoping: DPF Scoping with node IDs to process.
            combination_table: CombinationTableData with coefficients.
            rotate_to_global: If True (default), rotate forces to global coordinate
                             system. If False, keep forces in element (local) 
                             coordinate system.
        """
        self.reader1 = reader1
        self.reader2 = reader2
        self.scoping = nodal_scoping
        self.table = combination_table
        self.rotate_to_global = rotate_to_global
        
        self._active_step_keys, self._active_coefficients = self.table.get_active_step_matrix()
        self._force_data: Optional[np.ndarray] = None  # (active steps, 3, nodes)
        
        # DPF field cache for native DPF operations (not populated by default to save memory)
        self._field_cache: Dict[Tuple[int, int], 'dpf.Field'] = {}
        
        # Node information (populated during preload)
        self._node_ids: Optional[np.ndarray] = None
        self._node_coords: Optional[np.ndarray] = None
        
        # Force unit
        self._force_unit: str = "N"
        
        # Element type information (populated during preload)
        self._node_element_types: Optional[np.ndarray] = None
        self._has_beam_nodes: bool = False
    
    @property
    def node_ids(self) -> np.ndarray:
        """Node IDs (available after preload)."""
        if self._node_ids is None:
            raise RuntimeError("Force data not preloaded. Call preload_force_data() first.")
        return self._node_ids
    
    @property
    def node_coords(self) -> np.ndarray:
        """Node coordinates (available after preload)."""
        if self._node_coords is None:
            raise RuntimeError("Force data not preloaded. Call preload_force_data() first.")
        return self._node_coords
    
    @property
    def num_nodes(self) -> int:
        """Number of nodes being processed."""
        return len(self.node_ids)
    
    @property
    def force_unit(self) -> str:
        """Force unit string."""
        return self._force_unit
    
    @property
    def node_element_types(self) -> Optional[np.ndarray]:
        """Element types per node ('beam' or 'solid_shell')."""
        return self._node_element_types
    
    @property
    def has_beam_nodes(self) -> bool:
        """True if any node has beam elements attached."""
        return self._has_beam_nodes
    
    @property
    def coordinate_system(self) -> str:
        """Coordinate system for forces ('Global' or 'Local')."""
        return "Global" if self.rotate_to_global else "Local"
    
    def validate_nodal_forces_availability(
        self,
        nodal_scoping=None,  # dpf.Scoping
    ) -> Tuple[bool, str]:
        """
        Validate that nodal forces are available in both RST files
        for all active load steps (those with non-zero coefficients).

        Args:
            nodal_scoping: Optional scoping to validate against. If None,
                uses the engine's full scoping.
        
        Returns:
            Tuple of (is_valid, error_message). If valid, error_message is empty.
        """
        errors = []
        target_scoping = nodal_scoping if nodal_scoping is not None else self.scoping
        active_a1_steps, active_a2_steps = self.table.get_active_step_ids()

        if active_a1_steps:
            if not self.reader1.check_nodal_forces_available():
                errors.append(
                    "Analysis 1 RST file does not contain nodal forces.\n"
                    + MSG_NODAL_FORCES_ANSYS
                )
            else:
                for step_id in active_a1_steps:
                    try:
                        self.reader1.read_nodal_forces_for_loadstep(
                            step_id,
                            target_scoping,
                            rotate_to_global=self.rotate_to_global,
                        )
                    except NodalForcesNotAvailableError as error:
                        errors.append(f"Analysis 1, Load Step {step_id}: {error}")

        if active_a2_steps:
            if not self.reader2.check_nodal_forces_available():
                errors.append(
                    "Analysis 2 RST file does not contain nodal forces.\n"
                    + MSG_NODAL_FORCES_ANSYS
                )
            else:
                for step_id in active_a2_steps:
                    try:
                        self.reader2.read_nodal_forces_for_loadstep(
                            step_id,
                            target_scoping,
                            rotate_to_global=self.rotate_to_global,
                        )
                    except NodalForcesNotAvailableError as error:
                        errors.append(f"Analysis 2, Load Step {step_id}: {error}")
        
        if errors:
            return False, "\n\n".join(errors)
        return True, ""
    
    def preload_force_data(self, progress_callback: Optional[Callable[[int, int, str], None]] = None):
        """Load and cache nodal forces for active load steps (avoids repeated I/O). Caches numpy only."""
        # Get only active steps (those with non-zero coefficients)
        a1_steps, a2_steps = self.table.get_active_step_ids()
        total_steps = len(a1_steps) + len(a2_steps)
        current = 0
        
        if total_steps == 0:
            raise ValueError("No active load steps found. All coefficients are zero.")
        
        # Get force unit from first file
        self._force_unit = self.reader1.get_force_unit()
        
        force_cache = {}
        for analysis_idx, reader, step_ids in (
            (1, self.reader1, a1_steps),
            (2, self.reader2, a2_steps),
        ):
            for step_id in step_ids:
                result = reader.read_nodal_forces_for_loadstep(
                    step_id,
                    self.scoping,
                    rotate_to_global=self.rotate_to_global,
                )
                force_cache[(analysis_idx, step_id)] = result
                current += 1
                if progress_callback:
                    progress_callback(
                        current,
                        total_steps,
                        f"Loading A{analysis_idx} Forces Step {step_id}...",
                    )
        
        if progress_callback:
            progress_callback(total_steps, total_steps, "Force data loading complete.")
        
        self._force_data = np.ascontiguousarray(
            np.stack(
                [np.stack(force_cache[key][1:4], axis=0) for key in self._active_step_keys],
                axis=0,
            )
        )
        
        # Get node coordinates
        self._node_ids, self._node_coords = self.reader1.get_node_coordinates(self.scoping)
        
        # Get element type information per node (beam vs solid/shell)
        if progress_callback:
            progress_callback(total_steps, total_steps, "Detecting element types...")
        self._node_element_types, self._has_beam_nodes = self.reader1.get_node_element_types(self.scoping)
    
    def compute_combination_numpy(self, combo_index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute combined force vector for a single combination using numpy.
        
        Args:
            combo_index: Index of the combination (0-based).
            
        Returns:
            Tuple of (fx, fy, fz) combined arrays, each shape (num_nodes,).
        """
        force_data, coefficients = self._get_packed_force_data()
        combined = np.matmul(
            coefficients[combo_index],
            force_data.reshape(force_data.shape[0], -1),
        ).reshape(3, self.num_nodes)
        return tuple(combined)

    def _get_packed_force_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return production packed data, packing legacy test caches only when present."""
        force_data = getattr(self, "_force_data", None)
        keys, coefficients = self.table.get_active_step_matrix()
        if not isinstance(force_data, np.ndarray):
            cache = getattr(self, "_force_cache", None)
            if not cache:
                raise RuntimeError("Force data not preloaded. Call preload_force_data() first.")
            force_data = np.ascontiguousarray(
                np.stack([np.stack(cache[key][1:4], axis=0) for key in keys], axis=0)
            )
            self._force_data = force_data
        self._active_step_keys = keys
        self._active_coefficients = coefficients
        return force_data, coefficients

    def _compute_component_matrices(self, progress_callback=None):
        force_data, coefficients = self._get_packed_force_data()
        components = []
        for component in range(3):
            combined = np.matmul(coefficients, force_data[:, component, :])
            self._restore_cancellation_order(combined, coefficients, force_data[:, component, :])
            components.append(combined)
        if progress_callback:
            progress_callback(
                self.table.num_combinations,
                self.table.num_combinations,
                "Force computation complete.",
            )
        return tuple(components)

    @staticmethod
    def _restore_cancellation_order(combined, coefficients, step_values):
        """Recompute only ill-conditioned entries in the legacy step order."""
        num_combos, num_nodes = combined.shape
        coefficient_norm = np.sum(np.abs(coefficients), axis=1)
        value_scale = np.max(np.abs(step_values), axis=0)
        tolerance = 8 * np.finfo(np.float64).eps * coefficients.shape[1]
        block_size = max(
            1,
            min(num_nodes, MATRIX_TEMP_TARGET_BYTES // max(1, num_combos * 16)),
        )
        for start in range(0, num_nodes, block_size):
            end = min(start + block_size, num_nodes)
            unstable = np.abs(combined[:, start:end]) <= (
                tolerance * coefficient_norm[:, None] * value_scale[None, start:end]
            )
            for local_node in np.flatnonzero(np.any(unstable, axis=0)):
                corrected = np.zeros(num_combos)
                node_index = start + local_node
                for step_index in range(coefficients.shape[1]):
                    corrected += coefficients[:, step_index] * step_values[step_index, node_index]
                mask = unstable[:, local_node]
                combined[mask, node_index] = corrected[mask]

    @staticmethod
    def _magnitude_envelopes(x, y, z):
        num_combos, num_nodes = x.shape
        block_size = max(
            1,
            min(num_nodes, MATRIX_TEMP_TARGET_BYTES // max(1, num_combos * 8)),
        )
        maximum = np.empty(num_nodes)
        minimum = np.empty(num_nodes)
        argmax = np.empty(num_nodes, dtype=np.intp)
        argmin = np.empty(num_nodes, dtype=np.intp)
        for start in range(0, num_nodes, block_size):
            end = min(start + block_size, num_nodes)
            magnitude = np.square(x[:, start:end])
            magnitude += np.square(y[:, start:end])
            magnitude += np.square(z[:, start:end])
            np.sqrt(magnitude, out=magnitude)
            maximum[start:end] = np.max(magnitude, axis=0)
            minimum[start:end] = np.min(magnitude, axis=0)
            argmax[start:end] = np.argmax(magnitude, axis=0)
            argmin[start:end] = np.argmin(magnitude, axis=0)
        return maximum, minimum, argmax, argmin
    
    def compute_combination_dpf(self, combo_index: int) -> 'dpf.Field':
        """
        Compute combined force vector using DPF operators.
        
        Args:
            combo_index: Index of the combination (0-based).
            
        Returns:
            DPF Field containing the combined force vector.
        """
        a1_coeffs, a2_coeffs = self.table.get_coeffs_for_combination(combo_index)
        
        combined_field = None
        
        # Add contributions from Analysis 1 (check cache membership for active-only loading)
        for i, step_id in enumerate(self.table.analysis1_step_ids):
            coeff = a1_coeffs[i]
            if coeff != 0.0 and (1, step_id) in self._field_cache:
                field = self._field_cache[(1, step_id)]
                scaled = scale_force_field(field, coeff)
                
                if combined_field is None:
                    combined_field = scaled
                else:
                    combined_field = add_force_fields(combined_field, scaled)
        
        # Add contributions from Analysis 2 (check cache membership for active-only loading)
        for i, step_id in enumerate(self.table.analysis2_step_ids):
            coeff = a2_coeffs[i]
            if coeff != 0.0 and (2, step_id) in self._field_cache:
                field = self._field_cache[(2, step_id)]
                scaled = scale_force_field(field, coeff)
                
                if combined_field is None:
                    combined_field = scaled
                else:
                    combined_field = add_force_fields(combined_field, scaled)
        
        if combined_field is None:
            combo_name = self.table.combination_names[combo_index]
            raise ValueError(
                f"Combination '{combo_name}' (row {combo_index + 1}) has all-zero coefficients."
            )
        
        return combined_field
    
    @staticmethod
    def compute_magnitude(fx: np.ndarray, fy: np.ndarray, fz: np.ndarray) -> np.ndarray:
        """
        Compute force magnitude from 3-component vector.
        
        Args:
            fx, fy, fz: Force components.
            
        Returns:
            Force magnitude array.
        """
        return np.sqrt(fx**2 + fy**2 + fz**2)
    
    def compute_all_combinations(
        self,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute combined forces for ALL combinations.
        
        Args:
            progress_callback: Optional callback(current, total, message) for progress.
            
        Returns:
            Tuple of (fx_all, fy_all, fz_all, magnitude_all) arrays,
            each of shape (num_combinations, num_nodes).
        """
        num_combos = self.table.num_combinations
        if num_combos <= 0:
            raise ValueError("No combinations defined.")
        fx_all, fy_all, fz_all = self._compute_component_matrices(progress_callback)
        magnitude_all = self.compute_magnitude(fx_all, fy_all, fz_all)
        return (fx_all, fy_all, fz_all, magnitude_all)
    
    def compute_envelope(
        self,
        magnitude_results: np.ndarray,
        envelope_type: str = "max"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute envelope across combinations based on force magnitude.
        
        Args:
            magnitude_results: Array of shape (num_combinations, num_nodes).
            envelope_type: "max" or "min".
            
        Returns:
            Tuple of (envelope_values, combo_indices) both shape (num_nodes,).
        """
        if envelope_type == "max":
            envelope_values = np.max(magnitude_results, axis=0)
            combo_indices = np.argmax(magnitude_results, axis=0)
        elif envelope_type == "min":
            envelope_values = np.min(magnitude_results, axis=0)
            combo_indices = np.argmin(magnitude_results, axis=0)
        else:
            raise ValueError(f"Unknown envelope type: {envelope_type}. Use 'max' or 'min'.")
        
        return (envelope_values, combo_indices)
    
    def compute_full_analysis(
        self,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        auto_cleanup: bool = True
    ) -> NodalForcesResult:
        """
        Compute complete force envelope analysis and return NodalForcesResult.
        
        Args:
            progress_callback: Optional callback for progress updates.
            auto_cleanup: If True, clear cached force data after computation to
                          free memory. Default is True for memory efficiency.
            
        Returns:
            NodalForcesResult with all envelope data.
        """
        fx_all, fy_all, fz_all = self._compute_component_matrices(progress_callback)
        max_values, min_values, combo_of_max, combo_of_min = self._magnitude_envelopes(
            fx_all,
            fy_all,
            fz_all,
        )
        
        result = NodalForcesResult(
            node_ids=self.node_ids.copy(),
            node_coords=self.node_coords.copy(),
            max_magnitude_over_combo=max_values,
            min_magnitude_over_combo=min_values,
            combo_of_max=combo_of_max,
            combo_of_min=combo_of_min,
            all_combo_fx=fx_all,
            all_combo_fy=fy_all,
            all_combo_fz=fz_all,
            force_unit=self.force_unit,
            node_element_types=self._node_element_types.copy() if self._node_element_types is not None else None,
            has_beam_nodes=self._has_beam_nodes,
            coordinate_system=self.coordinate_system,
        )
        
        # Auto-cleanup cached data to free memory
        if auto_cleanup:
            self.clear_cache()
        
        return result
    
    def compute_single_node_history_fast(
        self,
        node_id: int,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute force history for a single node without loading full-node arrays.

        This is optimized for combination-history mode and scales with active
        load steps and combinations rather than total scoped node count.
        """
        single_node_scoping = self.reader1.create_single_node_scoping(node_id, self.scoping)

        if progress_callback:
            progress_callback(0, 100, f"Loading force data for node {node_id}...")

        self._force_unit = self.reader1.get_force_unit()
        self._node_ids, self._node_coords = self.reader1.get_node_coordinates(single_node_scoping)

        single_force_cache: Dict[Tuple[int, int], Tuple] = {}
        active_a1_steps, active_a2_steps = self.table.get_active_step_ids()
        active_a1_set = set(active_a1_steps)
        active_a2_set = set(active_a2_steps)
        total_steps = len(active_a1_steps) + len(active_a2_steps)
        if total_steps <= 0:
            raise ValueError("No active load steps found. All coefficients are zero.")
        current_step = 0

        for step_id in active_a1_steps:
            result = self.reader1.read_nodal_forces_for_loadstep(
                step_id,
                single_node_scoping,
                rotate_to_global=self.rotate_to_global,
            )
            single_force_cache[(1, step_id)] = result
            current_step += 1
            if progress_callback and total_steps > 0:
                progress = int((current_step / total_steps) * 40)
                progress_callback(progress, 100, f"Loading A1 step {step_id}...")

        for step_id in active_a2_steps:
            result = self.reader2.read_nodal_forces_for_loadstep(
                step_id,
                single_node_scoping,
                rotate_to_global=self.rotate_to_global,
            )
            single_force_cache[(2, step_id)] = result
            current_step += 1
            if progress_callback and total_steps > 0:
                progress = int((current_step / total_steps) * 40)
                progress_callback(progress, 100, f"Loading A2 step {step_id}...")

        if progress_callback:
            progress_callback(50, 100, f"Computing combinations for node {node_id}...")

        num_combos = self.table.num_combinations
        if num_combos <= 0:
            raise ValueError("No combinations defined.")
        fx = np.zeros(num_combos, dtype=np.float64)
        fy = np.zeros(num_combos, dtype=np.float64)
        fz = np.zeros(num_combos, dtype=np.float64)

        all_a1_steps = self.table.analysis1_step_ids
        all_a2_steps = self.table.analysis2_step_ids

        for combo_idx in range(num_combos):
            a1_coeffs, a2_coeffs = self.table.get_coeffs_for_combination(combo_idx)
            combo_fx = 0.0
            combo_fy = 0.0
            combo_fz = 0.0

            for i, step_id in enumerate(all_a1_steps):
                coeff = a1_coeffs[i]
                if coeff != 0.0 and step_id in active_a1_set:
                    _, s_fx, s_fy, s_fz = single_force_cache[(1, step_id)]
                    combo_fx += coeff * s_fx[0]
                    combo_fy += coeff * s_fy[0]
                    combo_fz += coeff * s_fz[0]

            for i, step_id in enumerate(all_a2_steps):
                coeff = a2_coeffs[i]
                if coeff != 0.0 and step_id in active_a2_set:
                    _, s_fx, s_fy, s_fz = single_force_cache[(2, step_id)]
                    combo_fx += coeff * s_fx[0]
                    combo_fy += coeff * s_fy[0]
                    combo_fz += coeff * s_fz[0]

            fx[combo_idx] = combo_fx
            fy[combo_idx] = combo_fy
            fz[combo_idx] = combo_fz

            if progress_callback:
                progress = 50 + int((combo_idx + 1) / num_combos * 50)
                progress_callback(progress, 100, f"Computing combination {combo_idx + 1}/{num_combos}...")

        magnitude = self.compute_magnitude(fx, fy, fz)
        combination_indices = np.arange(num_combos)
        return (combination_indices, fx, fy, fz, magnitude)
    
    def get_combination_names(self) -> List[str]:
        """Get list of combination names."""
        return self.table.combination_names
    
    def clear_cache(self):
        """Clear cached force data to free memory."""
        self._force_data = None
        if hasattr(self, "_force_cache"):
            self._force_cache.clear()
        self._field_cache.clear()
