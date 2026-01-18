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
import gc

from file_io.dpf_reader import (
    DPFAnalysisReader,
    NodalForcesNotAvailableError,
    scale_force_field,
    add_force_fields,
    compute_force_magnitude,
    DPF_AVAILABLE,
)
from core.data_models import CombinationTableData, NodalForcesResult

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
        
        # Force cache: maps (analysis_idx, step_id) -> force components tuple
        # Each tuple is (node_ids, fx, fy, fz)
        self._force_cache: Dict[Tuple[int, int], Tuple] = {}
        
        # DPF field cache for native DPF operations (not populated by default to save memory)
        self._field_cache: Dict[Tuple[int, int], 'dpf.Field'] = {}
        
        # Node information (populated during preload)
        self._node_ids: Optional[np.ndarray] = None
        self._node_coords: Optional[np.ndarray] = None
        
        # Force unit
        self._force_unit: str = "N"
    
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
    
    def validate_nodal_forces_availability(self) -> Tuple[bool, str]:
        """
        Validate that nodal forces are available in both RST files
        for all active load steps (those with non-zero coefficients).
        
        Returns:
            Tuple of (is_valid, error_message). If valid, error_message is empty.
        """
        errors = []
        
        # Get only active steps (those with non-zero coefficients)
        active_a1_steps, active_a2_steps = self.table.get_active_step_ids()
        
        # Check Analysis 1 (only if it has active steps)
        if active_a1_steps:
            if not self.reader1.check_nodal_forces_available():
                errors.append(
                    "Analysis 1 RST file does not contain nodal forces.\n"
                    "Ensure 'Write element nodal forces' is enabled in ANSYS Output Controls."
                )
            else:
                # Check only active load steps
                for step_id in active_a1_steps:
                    try:
                        self.reader1.read_nodal_forces_for_loadstep(step_id, self.scoping)
                    except NodalForcesNotAvailableError as e:
                        errors.append(f"Analysis 1, Load Step {step_id}: {str(e)}")
        
        # Check Analysis 2 (only if it has active steps)
        if active_a2_steps:
            if not self.reader2.check_nodal_forces_available():
                errors.append(
                    "Analysis 2 RST file does not contain nodal forces.\n"
                    "Ensure 'Write element nodal forces' is enabled in ANSYS Output Controls."
                )
            else:
                # Check only active load steps
                for step_id in active_a2_steps:
                    try:
                        self.reader2.read_nodal_forces_for_loadstep(step_id, self.scoping)
                    except NodalForcesNotAvailableError as e:
                        errors.append(f"Analysis 2, Load Step {step_id}: {str(e)}")
        
        if errors:
            return False, "\n\n".join(errors)
        return True, ""
    
    def preload_force_data(self, progress_callback: Optional[Callable[[int, int, str], None]] = None):
        """
        Cache nodal forces for load steps with non-zero coefficients.
        
        This method reads force data upfront to avoid repeated file I/O
        during combination calculations. Only steps that have at least one
        non-zero coefficient across all combinations are loaded, which can
        dramatically reduce I/O time when only a subset of steps are used.
        
        Note: Only numpy arrays are cached (not DPF fields) to minimize memory usage.
        
        Args:
            progress_callback: Optional callback(current, total, message) for progress updates.
            
        Raises:
            NodalForcesNotAvailableError: If nodal forces are not available.
        """
        # Get only active steps (those with non-zero coefficients)
        a1_steps, a2_steps = self.table.get_active_step_ids()
        total_steps = len(a1_steps) + len(a2_steps)
        current = 0
        
        if total_steps == 0:
            raise ValueError("No active load steps found. All coefficients are zero.")
        
        # Get force unit from first file
        self._force_unit = self.reader1.get_force_unit()
        
        # Load Analysis 1 force data (only active steps, numpy arrays only)
        for step_id in a1_steps:
            if progress_callback:
                progress_callback(current, total_steps, f"Loading A1 Forces Step {step_id}...")
            
            result = self.reader1.read_nodal_forces_for_loadstep(
                step_id, self.scoping, rotate_to_global=self.rotate_to_global
            )
            self._force_cache[(1, step_id)] = result
            current += 1
        
        # Load Analysis 2 force data (only active steps, numpy arrays only)
        for step_id in a2_steps:
            if progress_callback:
                progress_callback(current, total_steps, f"Loading A2 Forces Step {step_id}...")
            
            result = self.reader2.read_nodal_forces_for_loadstep(
                step_id, self.scoping, rotate_to_global=self.rotate_to_global
            )
            self._force_cache[(2, step_id)] = result
            current += 1
        
        if progress_callback:
            progress_callback(total_steps, total_steps, "Force data loading complete.")
        
        # Store node information from first loaded step
        if a1_steps:
            first_result = self._force_cache[(1, a1_steps[0])]
        elif a2_steps:
            first_result = self._force_cache[(2, a2_steps[0])]
        else:
            raise ValueError("No load steps defined in combination table.")
        
        self._node_ids = first_result[0]
        
        # Get node coordinates
        self._node_ids, self._node_coords = self.reader1.get_node_coordinates(self.scoping)
    
    def compute_combination_numpy(self, combo_index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute combined force vector for a single combination using numpy.
        
        Args:
            combo_index: Index of the combination (0-based).
            
        Returns:
            Tuple of (fx, fy, fz) combined arrays, each shape (num_nodes,).
        """
        a1_coeffs, a2_coeffs = self.table.get_coeffs_for_combination(combo_index)
        
        # Initialize combined force components
        num_nodes = self.num_nodes
        fx = np.zeros(num_nodes)
        fy = np.zeros(num_nodes)
        fz = np.zeros(num_nodes)
        
        # Add contributions from Analysis 1 (check cache membership for active-only loading)
        for i, step_id in enumerate(self.table.analysis1_step_ids):
            coeff = a1_coeffs[i]
            if coeff != 0.0 and (1, step_id) in self._force_cache:
                _, s_fx, s_fy, s_fz = self._force_cache[(1, step_id)]
                fx += coeff * s_fx
                fy += coeff * s_fy
                fz += coeff * s_fz
        
        # Add contributions from Analysis 2 (check cache membership for active-only loading)
        for i, step_id in enumerate(self.table.analysis2_step_ids):
            coeff = a2_coeffs[i]
            if coeff != 0.0 and (2, step_id) in self._force_cache:
                _, s_fx, s_fy, s_fz = self._force_cache[(2, step_id)]
                fx += coeff * s_fx
                fy += coeff * s_fy
                fz += coeff * s_fz
        
        return (fx, fy, fz)
    
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
        num_nodes = self.num_nodes
        
        fx_all = np.zeros((num_combos, num_nodes))
        fy_all = np.zeros((num_combos, num_nodes))
        fz_all = np.zeros((num_combos, num_nodes))
        magnitude_all = np.zeros((num_combos, num_nodes))
        
        for combo_idx in range(num_combos):
            if progress_callback:
                combo_name = self.table.combination_names[combo_idx]
                progress_callback(combo_idx, num_combos, f"Computing forces: {combo_name}...")
            
            fx, fy, fz = self.compute_combination_numpy(combo_idx)
            fx_all[combo_idx, :] = fx
            fy_all[combo_idx, :] = fy
            fz_all[combo_idx, :] = fz
            magnitude_all[combo_idx, :] = self.compute_magnitude(fx, fy, fz)
        
        if progress_callback:
            progress_callback(num_combos, num_combos, "Force computation complete.")
        
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
        # Compute all combinations
        fx_all, fy_all, fz_all, magnitude_all = self.compute_all_combinations(
            progress_callback=progress_callback
        )
        
        # Compute envelopes based on magnitude
        max_values, combo_of_max = self.compute_envelope(magnitude_all, "max")
        min_values, combo_of_min = self.compute_envelope(magnitude_all, "min")
        
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
        )
        
        # Auto-cleanup cached data to free memory
        if auto_cleanup:
            self.clear_cache()
            gc.collect()
        
        return result
    
    def compute_single_node_history(
        self,
        node_id: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute force history for a single node across all combinations.
        
        Args:
            node_id: Node ID to analyze.
            
        Returns:
            Tuple of (combination_indices, fx, fy, fz, magnitude) arrays.
        """
        # Find node index
        node_idx = np.where(self.node_ids == node_id)[0]
        if len(node_idx) == 0:
            raise ValueError(f"Node ID {node_id} not found in scoping.")
        node_idx = node_idx[0]
        
        # Compute all combinations and extract single node
        fx_all, fy_all, fz_all, magnitude_all = self.compute_all_combinations()
        
        fx = fx_all[:, node_idx]
        fy = fy_all[:, node_idx]
        fz = fz_all[:, node_idx]
        magnitude = magnitude_all[:, node_idx]
        
        combination_indices = np.arange(self.table.num_combinations)
        
        return (combination_indices, fx, fy, fz, magnitude)
    
    def get_combination_names(self) -> List[str]:
        """Get list of combination names."""
        return self.table.combination_names
    
    def clear_cache(self):
        """Clear cached force data to free memory."""
        self._force_cache.clear()
        self._field_cache.clear()
