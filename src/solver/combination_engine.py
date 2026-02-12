"""
Combination Engine for MARS-SC (Solution Combination).

Performs linear combination of stress tensors from two analyses and computes
stress envelopes over combinations.

The combination formula is:
    σ_combined = Σ(α_i × σ_A1_i) + Σ(β_j × σ_A2_j)

Where:
    - α_i are coefficients for Analysis 1 load steps
    - β_j are coefficients for Analysis 2 load steps
    - σ_A1_i and σ_A2_j are stress tensors for each load step

Reference: https://dpf.docs.pyansys.com/version/stable/examples/06-plotting/02-solution_combination.html
"""

from typing import Dict, Tuple, Optional, Callable, List
import numpy as np
import gc

# Try to import psutil for memory estimation
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# RAM usage fraction (use 90% of available RAM as limit)
RAM_USAGE_FRACTION = 0.9
# Default chunk size if memory estimation fails
DEFAULT_CHUNK_SIZE = 50000
# Minimum chunk size to avoid excessive overhead
MIN_CHUNK_SIZE = 1000

from file_io.dpf_reader import (
    DPFAnalysisReader,
    scale_stress_field,
    add_stress_fields,
    compute_principal_stresses,
    compute_von_mises_from_field,
    DPF_AVAILABLE,
)
from core.data_models import CombinationTableData, CombinationResult

if DPF_AVAILABLE:
    from ansys.dpf import core as dpf


class CombinationEngine:
    """
    Performs linear combination of stress tensors from two analyses.
    
    This engine preloads stress data from both analyses and then computes
    combined stresses for each combination defined in the combination table.
    
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
        combination_table: CombinationTableData
    ):
        """
        Initialize the combination engine.
        
        Args:
            reader1: DPFAnalysisReader for Analysis 1 (base).
            reader2: DPFAnalysisReader for Analysis 2 (to combine).
            nodal_scoping: DPF Scoping with node IDs to process.
            combination_table: CombinationTableData with coefficients.
        """
        self.reader1 = reader1
        self.reader2 = reader2
        self.scoping = nodal_scoping
        self.table = combination_table
        
        # Stress tensor cache: maps (analysis_idx, step_id) -> stress components tuple
        # Each tuple is (node_ids, sx, sy, sz, sxy, syz, sxz)
        self._stress_cache: Dict[Tuple[int, int], Tuple] = {}
        
        # DPF field cache for native DPF operations (not populated by default to save memory)
        # Use preload_stress_fields() explicitly if DPF operations are needed
        self._field_cache: Dict[Tuple[int, int], 'dpf.Field'] = {}
        
        # Node information (populated during preload)
        self._node_ids: Optional[np.ndarray] = None
        self._node_coords: Optional[np.ndarray] = None
    
    @property
    def node_ids(self) -> np.ndarray:
        """Node IDs (available after preload)."""
        if self._node_ids is None:
            raise RuntimeError("Stress data not preloaded. Call preload_stress_data() first.")
        return self._node_ids
    
    @property
    def node_coords(self) -> np.ndarray:
        """Node coordinates (available after preload)."""
        if self._node_coords is None:
            raise RuntimeError("Stress data not preloaded. Call preload_stress_data() first.")
        return self._node_coords
    
    @property
    def num_nodes(self) -> int:
        """Number of nodes being processed."""
        return len(self.node_ids)
    
    def preload_stress_data(self, progress_callback: Optional[Callable[[int, int, str], None]] = None):
        """
        Cache stress tensors for load steps with non-zero coefficients.
        
        This method reads stress data upfront to avoid repeated file I/O
        during combination calculations. Only steps that have at least one
        non-zero coefficient across all combinations are loaded, which can
        dramatically reduce I/O time when only a subset of steps are used.
        
        Note: Only numpy arrays are cached (not DPF fields) to minimize memory usage.
        
        Args:
            progress_callback: Optional callback(current, total, message) for progress updates.
        """
        # Get only active steps (those with non-zero coefficients)
        a1_steps, a2_steps = self.table.get_active_step_ids()
        total_steps = len(a1_steps) + len(a2_steps)
        current = 0
        
        if total_steps == 0:
            raise ValueError("No active load steps found. All coefficients are zero.")
        
        # Load Analysis 1 stress data (only active steps, numpy arrays only)
        for step_id in a1_steps:
            if progress_callback:
                progress_callback(current, total_steps, f"Loading A1 Step {step_id}...")
            
            result = self.reader1.read_stress_tensor_for_loadstep(step_id, self.scoping)
            self._stress_cache[(1, step_id)] = result
            current += 1
        
        # Load Analysis 2 stress data (only active steps, numpy arrays only)
        for step_id in a2_steps:
            if progress_callback:
                progress_callback(current, total_steps, f"Loading A2 Step {step_id}...")
            
            result = self.reader2.read_stress_tensor_for_loadstep(step_id, self.scoping)
            self._stress_cache[(2, step_id)] = result
            current += 1
        
        if progress_callback:
            progress_callback(total_steps, total_steps, "Loading complete.")
        
        # Store node information from first loaded step
        if a1_steps:
            first_result = self._stress_cache[(1, a1_steps[0])]
        elif a2_steps:
            first_result = self._stress_cache[(2, a2_steps[0])]
        else:
            raise ValueError("No load steps defined in combination table.")
        
        self._node_ids = first_result[0]
        
        # Get node coordinates
        self._node_ids, self._node_coords = self.reader1.get_node_coordinates(self.scoping)
    
    def compute_combination_numpy(self, combo_index: int) -> Tuple[np.ndarray, ...]:
        """
        Compute combined stress tensor for a single combination using numpy.
        
        This method uses cached numpy arrays for the calculation, providing
        maximum flexibility and control over the computation.
        
        Args:
            combo_index: Index of the combination (0-based).
            
        Returns:
            Tuple of (sx, sy, sz, sxy, syz, sxz) combined arrays, each shape (num_nodes,).
        """
        a1_coeffs, a2_coeffs = self.table.get_coeffs_for_combination(combo_index)
        
        # Initialize combined stress components
        num_nodes = self.num_nodes
        sx = np.zeros(num_nodes)
        sy = np.zeros(num_nodes)
        sz = np.zeros(num_nodes)
        sxy = np.zeros(num_nodes)
        syz = np.zeros(num_nodes)
        sxz = np.zeros(num_nodes)
        
        # Add contributions from Analysis 1 (check cache membership for active-only loading)
        for i, step_id in enumerate(self.table.analysis1_step_ids):
            coeff = a1_coeffs[i]
            if coeff != 0.0 and (1, step_id) in self._stress_cache:
                _, s_sx, s_sy, s_sz, s_sxy, s_syz, s_sxz = self._stress_cache[(1, step_id)]
                sx += coeff * s_sx
                sy += coeff * s_sy
                sz += coeff * s_sz
                sxy += coeff * s_sxy
                syz += coeff * s_syz
                sxz += coeff * s_sxz
        
        # Add contributions from Analysis 2 (check cache membership for active-only loading)
        for i, step_id in enumerate(self.table.analysis2_step_ids):
            coeff = a2_coeffs[i]
            if coeff != 0.0 and (2, step_id) in self._stress_cache:
                _, s_sx, s_sy, s_sz, s_sxy, s_syz, s_sxz = self._stress_cache[(2, step_id)]
                sx += coeff * s_sx
                sy += coeff * s_sy
                sz += coeff * s_sz
                sxy += coeff * s_sxy
                syz += coeff * s_syz
                sxz += coeff * s_sxz
        
        return (sx, sy, sz, sxy, syz, sxz)
    
    def compute_combination_dpf(self, combo_index: int) -> 'dpf.Field':
        """
        Compute combined stress tensor using DPF operators.
        
        This method uses native DPF operations (scale and add) for potentially
        better performance with large datasets.
        
        Args:
            combo_index: Index of the combination (0-based).
            
        Returns:
            DPF Field containing the combined stress tensor.
            
        Raises:
            ValueError: If all coefficients for the combination are zero.
            
        Reference:
            https://dpf.docs.pyansys.com/version/stable/examples/06-plotting/02-solution_combination.html
        """
        a1_coeffs, a2_coeffs = self.table.get_coeffs_for_combination(combo_index)
        
        combined_field = None
        
        # Add contributions from Analysis 1 (check cache membership for active-only loading)
        for i, step_id in enumerate(self.table.analysis1_step_ids):
            coeff = a1_coeffs[i]
            if coeff != 0.0 and (1, step_id) in self._field_cache:
                field = self._field_cache[(1, step_id)]
                scaled = scale_stress_field(field, coeff)
                
                if combined_field is None:
                    combined_field = scaled
                else:
                    combined_field = add_stress_fields(combined_field, scaled)
        
        # Add contributions from Analysis 2 (check cache membership for active-only loading)
        for i, step_id in enumerate(self.table.analysis2_step_ids):
            coeff = a2_coeffs[i]
            if coeff != 0.0 and (2, step_id) in self._field_cache:
                field = self._field_cache[(2, step_id)]
                scaled = scale_stress_field(field, coeff)
                
                if combined_field is None:
                    combined_field = scaled
                else:
                    combined_field = add_stress_fields(combined_field, scaled)
        
        # Check if all coefficients were zero
        if combined_field is None:
            combo_name = self.table.combination_names[combo_index]
            raise ValueError(
                f"Combination '{combo_name}' (row {combo_index + 1}) has all-zero coefficients. "
                f"Please enter at least one non-zero coefficient or delete this row."
            )
        
        return combined_field
    
    @staticmethod
    def compute_von_mises(sx, sy, sz, sxy, syz, sxz) -> np.ndarray:
        """
        Compute von Mises stress from 6-component tensor.
        
        Formula: σ_vm = √(0.5 * [(σx-σy)² + (σy-σz)² + (σz-σx)² + 6*(τxy² + τyz² + τxz²)])
        
        Args:
            sx, sy, sz: Normal stress components.
            sxy, syz, sxz: Shear stress components.
            
        Returns:
            Von Mises equivalent stress array.
        """
        return np.sqrt(
            0.5 * (
                (sx - sy)**2 + (sy - sz)**2 + (sz - sx)**2 +
                6 * (sxy**2 + syz**2 + sxz**2)
            )
        )
    
    @staticmethod
    def compute_principal_stresses_numpy(sx, sy, sz, sxy, syz, sxz) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute principal stresses (S1, S2, S3) from tensor components.
        
        Principal stresses are eigenvalues of the stress tensor, sorted
        such that S1 >= S2 >= S3.
        
        Args:
            sx, sy, sz: Normal stress components.
            sxy, syz, sxz: Shear stress components.
            
        Returns:
            Tuple of (S1, S2, S3) arrays where S1 is maximum principal.
        """
        num_nodes = len(sx)
        s1 = np.zeros(num_nodes)
        s2 = np.zeros(num_nodes)
        s3 = np.zeros(num_nodes)
        
        for i in range(num_nodes):
            # Build 3x3 stress tensor
            tensor = np.array([
                [sx[i], sxy[i], sxz[i]],
                [sxy[i], sy[i], syz[i]],
                [sxz[i], syz[i], sz[i]]
            ])
            
            # Compute eigenvalues (principal stresses)
            eigenvalues = np.linalg.eigvalsh(tensor)
            
            # Sort in descending order (S1 >= S2 >= S3)
            sorted_eig = np.sort(eigenvalues)[::-1]
            s1[i] = sorted_eig[0]
            s2[i] = sorted_eig[1]
            s3[i] = sorted_eig[2]
        
        return (s1, s2, s3)
    
    def compute_all_combinations(
        self,
        stress_type: str = "von_mises",
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        use_dpf: bool = False
    ) -> np.ndarray:
        """
        Compute requested stress for ALL combinations.
        
        Args:
            stress_type: One of "von_mises", "max_principal", "min_principal".
            progress_callback: Optional callback(current, total, message) for progress.
            use_dpf: If True, use DPF operators (requires _field_cache to be populated).
                     Default is False to use memory-efficient numpy-based computation.
            
        Returns:
            Array of shape (num_combinations, num_nodes).
        """
        num_combos = self.table.num_combinations
        num_nodes = self.num_nodes
        results = np.zeros((num_combos, num_nodes))
        
        for combo_idx in range(num_combos):
            if progress_callback:
                combo_name = self.table.combination_names[combo_idx]
                progress_callback(combo_idx, num_combos, f"Computing {combo_name}...")
            
            if use_dpf and DPF_AVAILABLE:
                # Use DPF for combination and invariant computation
                combined_field = self.compute_combination_dpf(combo_idx)
                
                if stress_type == "von_mises":
                    result_field = compute_von_mises_from_field(combined_field)
                    results[combo_idx, :] = result_field.data.flatten()
                elif stress_type == "max_principal":
                    s1, _, _ = compute_principal_stresses(combined_field)
                    results[combo_idx, :] = s1.data.flatten()
                elif stress_type == "min_principal":
                    _, _, s3 = compute_principal_stresses(combined_field)
                    results[combo_idx, :] = s3.data.flatten()
                else:
                    raise ValueError(f"Unknown stress type: {stress_type}")
            else:
                # Use numpy for all calculations
                sx, sy, sz, sxy, syz, sxz = self.compute_combination_numpy(combo_idx)
                
                if stress_type == "von_mises":
                    results[combo_idx, :] = self.compute_von_mises(sx, sy, sz, sxy, syz, sxz)
                elif stress_type == "max_principal":
                    s1, _, _ = self.compute_principal_stresses_numpy(sx, sy, sz, sxy, syz, sxz)
                    results[combo_idx, :] = s1
                elif stress_type == "min_principal":
                    _, _, s3 = self.compute_principal_stresses_numpy(sx, sy, sz, sxy, syz, sxz)
                    results[combo_idx, :] = s3
                else:
                    raise ValueError(f"Unknown stress type: {stress_type}")
        
        if progress_callback:
            progress_callback(num_combos, num_combos, "Computation complete.")
        
        return results
    
    def compute_envelope(
        self,
        all_combo_results: np.ndarray,
        envelope_type: str = "max"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute envelope across combinations.
        
        Finds the maximum or minimum stress value at each node across all
        combinations, and records which combination produced that value.
        
        Args:
            all_combo_results: Array of shape (num_combinations, num_nodes).
            envelope_type: "max" or "min".
            
        Returns:
            Tuple of (envelope_values, combo_indices) both shape (num_nodes,).
            combo_indices contains the 0-based combination index that caused
            the extreme value at each node.
        """
        if envelope_type == "max":
            envelope_values = np.max(all_combo_results, axis=0)
            combo_indices = np.argmax(all_combo_results, axis=0)
        elif envelope_type == "min":
            envelope_values = np.min(all_combo_results, axis=0)
            combo_indices = np.argmin(all_combo_results, axis=0)
        else:
            raise ValueError(f"Unknown envelope type: {envelope_type}. Use 'max' or 'min'.")
        
        return (envelope_values, combo_indices)
    
    def compute_full_analysis(
        self,
        stress_type: str = "von_mises",
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        auto_cleanup: bool = True
    ) -> CombinationResult:
        """
        Compute complete envelope analysis and return CombinationResult.
        
        This is a convenience method that computes all combinations, both
        max and min envelopes, and packages everything into a CombinationResult.
        
        Args:
            stress_type: One of "von_mises", "max_principal", "min_principal".
            progress_callback: Optional callback for progress updates.
            auto_cleanup: If True, clear cached stress data after computation to
                          free memory. Default is True for memory efficiency.
            
        Returns:
            CombinationResult with all envelope data.
        """
        # Compute all combinations
        all_results = self.compute_all_combinations(
            stress_type=stress_type,
            progress_callback=progress_callback
        )
        
        # Compute envelopes
        max_values, combo_of_max = self.compute_envelope(all_results, "max")
        min_values, combo_of_min = self.compute_envelope(all_results, "min")
        
        result = CombinationResult(
            node_ids=self.node_ids.copy(),
            node_coords=self.node_coords.copy(),
            max_over_combo=max_values,
            min_over_combo=min_values,
            combo_of_max=combo_of_max,
            combo_of_min=combo_of_min,
            result_type=stress_type,
            all_combo_results=all_results,
        )
        
        # Auto-cleanup cached data to free memory
        if auto_cleanup:
            self.clear_cache()
            gc.collect()
        
        return result
    
    def compute_single_node_history(
        self,
        node_id: int,
        stress_type: str = "von_mises"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Stress vs combo index for one node. Slow (computes all nodes then slices); use compute_single_node_history_fast for big models."""
        # Find node index
        node_idx = np.where(self.node_ids == node_id)[0]
        if len(node_idx) == 0:
            raise ValueError(f"Node ID {node_id} not found in scoping.")
        node_idx = node_idx[0]
        
        # Compute all combinations and extract single node
        all_results = self.compute_all_combinations(stress_type=stress_type)
        stress_values = all_results[:, node_idx]
        
        combination_indices = np.arange(self.table.num_combinations)
        
        return (combination_indices, stress_values)
    
    def compute_single_node_history_fast(
        self,
        node_id: int,
        stress_type: str = "von_mises",
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute stress history for a single node (optimized version).
        
        This method only loads and computes stress data for the specified node,
        avoiding the overhead of processing all nodes. Much faster than
        compute_single_node_history() for large models.
        
        Only load steps with non-zero coefficients are read from the RST files,
        which can dramatically reduce I/O time when only a subset of steps are used.
        
        Args:
            node_id: Node ID to analyze.
            stress_type: One of "von_mises", "max_principal", "min_principal".
            progress_callback: Optional callback(current, total, message) for progress.
            
        Returns:
            Tuple of (combination_indices, stress_values) arrays.
            
        Raises:
            ValueError: If node_id is not found in the scoping.
        """
        # Create single-node scoping
        single_node_scoping = self.reader1.create_single_node_scoping(node_id, self.scoping)
        
        if progress_callback:
            progress_callback(0, 100, f"Loading stress data for node {node_id}...")
        
        # Load stress data for this single node only from both analyses
        # Only load active steps (those with non-zero coefficients)
        single_stress_cache: Dict[Tuple[int, int], Tuple] = {}
        
        active_a1_steps, active_a2_steps = self.table.get_active_step_ids()
        active_a1_set = set(active_a1_steps)
        active_a2_set = set(active_a2_steps)
        total_steps = len(active_a1_steps) + len(active_a2_steps)
        current_step = 0
        
        # Load Analysis 1 stress data for single node (only active steps)
        for step_id in active_a1_steps:
            result = self.reader1.read_stress_tensor_for_loadstep(step_id, single_node_scoping)
            single_stress_cache[(1, step_id)] = result
            current_step += 1
            if progress_callback:
                progress = int((current_step / total_steps) * 40)  # 0-40% for loading
                progress_callback(progress, 100, f"Loading A1 step {step_id}...")
        
        # Load Analysis 2 stress data for single node (only active steps)
        for step_id in active_a2_steps:
            result = self.reader2.read_stress_tensor_for_loadstep(step_id, single_node_scoping)
            single_stress_cache[(2, step_id)] = result
            current_step += 1
            if progress_callback:
                progress = int((current_step / total_steps) * 40)  # 0-40% for loading
                progress_callback(progress, 100, f"Loading A2 step {step_id}...")
        
        if progress_callback:
            progress_callback(50, 100, f"Computing combinations for node {node_id}...")
        
        # Compute combinations for single node
        num_combos = self.table.num_combinations
        stress_values = np.zeros(num_combos)
        
        # Keep full step ID lists for correct coefficient indexing
        all_a1_steps = self.table.analysis1_step_ids
        all_a2_steps = self.table.analysis2_step_ids
        
        for combo_idx in range(num_combos):
            a1_coeffs, a2_coeffs = self.table.get_coeffs_for_combination(combo_idx)
            
            # Initialize combined stress components (single node = scalar)
            sx, sy, sz = 0.0, 0.0, 0.0
            sxy, syz, sxz = 0.0, 0.0, 0.0
            
            # Add contributions from Analysis 1 (iterate over all for correct indexing)
            for i, step_id in enumerate(all_a1_steps):
                coeff = a1_coeffs[i]
                if coeff != 0.0 and step_id in active_a1_set:
                    _, s_sx, s_sy, s_sz, s_sxy, s_syz, s_sxz = single_stress_cache[(1, step_id)]
                    sx += coeff * s_sx[0]
                    sy += coeff * s_sy[0]
                    sz += coeff * s_sz[0]
                    sxy += coeff * s_sxy[0]
                    syz += coeff * s_syz[0]
                    sxz += coeff * s_sxz[0]
            
            # Add contributions from Analysis 2 (iterate over all for correct indexing)
            for i, step_id in enumerate(all_a2_steps):
                coeff = a2_coeffs[i]
                if coeff != 0.0 and step_id in active_a2_set:
                    _, s_sx, s_sy, s_sz, s_sxy, s_syz, s_sxz = single_stress_cache[(2, step_id)]
                    sx += coeff * s_sx[0]
                    sy += coeff * s_sy[0]
                    sz += coeff * s_sz[0]
                    sxy += coeff * s_sxy[0]
                    syz += coeff * s_syz[0]
                    sxz += coeff * s_sxz[0]
            
            # Compute stress invariant for this combination
            if stress_type == "von_mises":
                stress_values[combo_idx] = np.sqrt(
                    0.5 * (
                        (sx - sy)**2 + (sy - sz)**2 + (sz - sx)**2 +
                        6 * (sxy**2 + syz**2 + sxz**2)
                    )
                )
            elif stress_type == "max_principal":
                # Build tensor and compute eigenvalues for single point
                tensor = np.array([
                    [sx, sxy, sxz],
                    [sxy, sy, syz],
                    [sxz, syz, sz]
                ])
                eigenvalues = np.linalg.eigvalsh(tensor)
                stress_values[combo_idx] = np.max(eigenvalues)  # S1
            elif stress_type == "min_principal":
                tensor = np.array([
                    [sx, sxy, sxz],
                    [sxy, sy, syz],
                    [sxz, syz, sz]
                ])
                eigenvalues = np.linalg.eigvalsh(tensor)
                stress_values[combo_idx] = np.min(eigenvalues)  # S3
            else:
                raise ValueError(f"Unknown stress type: {stress_type}")
            
            if progress_callback:
                progress = 50 + int((combo_idx + 1) / num_combos * 50)  # 50-100% for combinations
                progress_callback(progress, 100, f"Computing combination {combo_idx + 1}/{num_combos}...")
        
        if progress_callback:
            progress_callback(100, 100, "Complete")
        
        combination_indices = np.arange(num_combos)
        return (combination_indices, stress_values)
    
    def get_combination_names(self) -> List[str]:
        """Get list of combination names."""
        return self.table.combination_names
    
    def clear_cache(self):
        """Clear cached stress data to free memory."""
        self._stress_cache.clear()
        self._field_cache.clear()
    
    # =========================================================================
    # RAM-Saving Methods for Large Models
    # =========================================================================
    
    def estimate_memory_requirements(self, num_nodes: Optional[int] = None) -> Dict[str, int]:
        """
        Estimate memory requirements for the full analysis.
        
        Calculates memory needed for stress caching and result arrays based on
        the number of nodes, active load steps (those with non-zero coefficients),
        and combinations.
        
        Args:
            num_nodes: Number of nodes. If None, uses scoping size.
            
        Returns:
            Dictionary with memory estimates in bytes:
              - stress_cache_bytes: Memory for caching all stress tensors
              - results_array_bytes: Memory for (combos × nodes) results
              - envelope_bytes: Memory for envelope arrays (max/min/indices)
              - total_bytes: Total estimated memory
              - minimum_required_bytes: Minimum memory for chunked processing
        """
        if num_nodes is None:
            num_nodes = len(self.scoping.ids)
        
        # Use active step count (only steps with non-zero coefficients)
        active_a1_steps, active_a2_steps = self.table.get_active_step_ids()
        num_a1_steps = len(active_a1_steps)
        num_a2_steps = len(active_a2_steps)
        total_steps = num_a1_steps + num_a2_steps
        num_combos = self.table.num_combinations
        
        bytes_per_float = 8  # float64
        
        # Stress cache: 6 components × nodes × active_steps (numpy arrays only, no DPF fields)
        # Plus node IDs array per step
        stress_cache_bytes = (6 * num_nodes * total_steps * bytes_per_float +
                             total_steps * num_nodes * bytes_per_float)  # node IDs
        
        # Full results array: (num_combos, num_nodes)
        results_array_bytes = num_combos * num_nodes * bytes_per_float
        
        # Envelope arrays: max, min, combo_of_max, combo_of_min (4 arrays × num_nodes)
        envelope_bytes = 4 * num_nodes * bytes_per_float
        
        # Total for full (non-chunked) processing
        total_bytes = stress_cache_bytes + results_array_bytes + envelope_bytes
        
        # Minimum for chunked processing (envelope only, single chunk of stress data)
        # Memory per node per chunk: 6 × active_steps × 8 (stress) + num_combos × 8 (results)
        memory_per_node = 6 * total_steps * bytes_per_float + num_combos * bytes_per_float
        minimum_chunk_memory = MIN_CHUNK_SIZE * memory_per_node + envelope_bytes
        
        return {
            'stress_cache_bytes': stress_cache_bytes,
            'results_array_bytes': results_array_bytes,
            'envelope_bytes': envelope_bytes,
            'total_bytes': total_bytes,
            'minimum_required_bytes': minimum_chunk_memory,
            'memory_per_node': memory_per_node,
            'num_nodes': num_nodes,
            'num_combinations': num_combos,
            'num_load_steps': total_steps,
            'num_active_a1_steps': num_a1_steps,
            'num_active_a2_steps': num_a2_steps,
        }
    
    def _get_available_memory(self) -> int:
        """
        Get available system RAM in bytes.
        
        Returns:
            Available memory in bytes, scaled by RAM_USAGE_FRACTION.
            Returns a default large value if psutil is not available.
        """
        if PSUTIL_AVAILABLE:
            return int(psutil.virtual_memory().available * RAM_USAGE_FRACTION)
        else:
            # Default to 4GB if psutil not available
            return int(4 * 1024 * 1024 * 1024 * RAM_USAGE_FRACTION)
    
    def _calculate_chunk_size(self, available_memory: int, num_nodes: int) -> int:
        """
        Calculate optimal node chunk size based on available memory.
        
        Memory per node per chunk:
          - Stress tensor: 6 components × num_load_steps × 8 bytes
          - Combined stress: 6 × 8 bytes (temporary)
          - Scalar result: num_combos × 8 bytes
        
        Args:
            available_memory: Available memory in bytes.
            num_nodes: Total number of nodes to process.
            
        Returns:
            Optimal chunk size (number of nodes per chunk).
        """
        estimates = self.estimate_memory_requirements(num_nodes)
        memory_per_node = estimates['memory_per_node']
        
        # Reserve memory for envelope arrays
        envelope_memory = estimates['envelope_bytes']
        usable_memory = available_memory - envelope_memory
        
        if usable_memory <= 0 or memory_per_node <= 0:
            return min(DEFAULT_CHUNK_SIZE, num_nodes)
        
        # Calculate chunk size that fits in memory
        chunk_size = int(usable_memory / memory_per_node)
        
        # Clamp to reasonable bounds
        chunk_size = max(MIN_CHUNK_SIZE, min(chunk_size, num_nodes))
        
        return chunk_size
    
    def check_memory_available(self, raise_on_insufficient: bool = True) -> Tuple[bool, Dict]:
        """
        Check if sufficient memory is available for analysis.
        
        Args:
            raise_on_insufficient: If True, raises MemoryError when insufficient.
            
        Returns:
            Tuple of (is_sufficient, estimates_dict).
            
        Raises:
            MemoryError: If memory is insufficient and raise_on_insufficient is True.
        """
        num_nodes = len(self.scoping.ids)
        estimates = self.estimate_memory_requirements(num_nodes)
        available = self._get_available_memory()
        
        # Check if minimum chunked processing is possible
        is_sufficient = estimates['minimum_required_bytes'] < available
        
        estimates['available_bytes'] = available
        estimates['is_sufficient'] = is_sufficient
        estimates['recommended_chunk_size'] = self._calculate_chunk_size(available, num_nodes)
        
        if not is_sufficient and raise_on_insufficient:
            raise MemoryError(
                f"Insufficient RAM for this analysis.\n"
                f"Minimum required: {estimates['minimum_required_bytes'] / 1e9:.2f} GB\n"
                f"Available: {available / 1e9:.2f} GB\n"
                f"Nodes: {num_nodes:,}\n"
                f"Tip: Reduce nodes via Named Selection or close other applications."
            )
        
        return (is_sufficient, estimates)
    
    def load_stress_for_node_range(
        self,
        start_idx: int,
        end_idx: int,
        full_node_ids: np.ndarray
    ) -> Dict[Tuple[int, int], Tuple]:
        """
        Load stress data for a specific node range only.
        
        Creates a temporary scoping for just the requested nodes,
        then reads stress tensors from both RST files. Only load steps
        that have non-zero coefficients to minimize I/O.
        
        Args:
            start_idx: Start index (0-based, inclusive).
            end_idx: End index (0-based, exclusive).
            full_node_ids: Full array of node IDs for reference.
            
        Returns:
            Dictionary mapping (analysis_idx, step_id) to stress tuple.
            Each tuple is (node_ids, sx, sy, sz, sxy, syz, sxz).
        """
        # Create sub-scoping for this chunk
        chunk_scoping = self.reader1.create_sub_scoping(self.scoping, start_idx, end_idx)
        chunk_cache = {}
        
        # Get only active steps (those with non-zero coefficients)
        active_a1_steps, active_a2_steps = self.table.get_active_step_ids()
        
        # Load Analysis 1 stress data for chunk (only active steps)
        for step_id in active_a1_steps:
            result = self.reader1.read_stress_tensor_for_loadstep(step_id, chunk_scoping)
            chunk_cache[(1, step_id)] = result
        
        # Load Analysis 2 stress data for chunk (only active steps)
        for step_id in active_a2_steps:
            result = self.reader2.read_stress_tensor_for_loadstep(step_id, chunk_scoping)
            chunk_cache[(2, step_id)] = result
        
        return chunk_cache
    
    def _compute_chunk_combinations(
        self,
        chunk_cache: Dict[Tuple[int, int], Tuple],
        chunk_size: int,
        stress_type: str = "von_mises"
    ) -> np.ndarray:
        """
        Compute all combinations for a chunk of nodes.
        
        Args:
            chunk_cache: Stress cache for this chunk only (contains only active steps).
            chunk_size: Number of nodes in this chunk.
            stress_type: One of "von_mises", "max_principal", "min_principal".
            
        Returns:
            Array of shape (num_combinations, chunk_size).
        """
        num_combos = self.table.num_combinations
        results = np.zeros((num_combos, chunk_size))
        
        for combo_idx in range(num_combos):
            a1_coeffs, a2_coeffs = self.table.get_coeffs_for_combination(combo_idx)
            
            # Initialize combined stress components for this chunk
            sx = np.zeros(chunk_size)
            sy = np.zeros(chunk_size)
            sz = np.zeros(chunk_size)
            sxy = np.zeros(chunk_size)
            syz = np.zeros(chunk_size)
            sxz = np.zeros(chunk_size)
            
            # Add contributions from Analysis 1 (check cache membership for active-only loading)
            for i, step_id in enumerate(self.table.analysis1_step_ids):
                coeff = a1_coeffs[i]
                if coeff != 0.0 and (1, step_id) in chunk_cache:
                    _, s_sx, s_sy, s_sz, s_sxy, s_syz, s_sxz = chunk_cache[(1, step_id)]
                    sx += coeff * s_sx
                    sy += coeff * s_sy
                    sz += coeff * s_sz
                    sxy += coeff * s_sxy
                    syz += coeff * s_syz
                    sxz += coeff * s_sxz
            
            # Add contributions from Analysis 2 (check cache membership for active-only loading)
            for i, step_id in enumerate(self.table.analysis2_step_ids):
                coeff = a2_coeffs[i]
                if coeff != 0.0 and (2, step_id) in chunk_cache:
                    _, s_sx, s_sy, s_sz, s_sxy, s_syz, s_sxz = chunk_cache[(2, step_id)]
                    sx += coeff * s_sx
                    sy += coeff * s_sy
                    sz += coeff * s_sz
                    sxy += coeff * s_sxy
                    syz += coeff * s_syz
                    sxz += coeff * s_sxz
            
            # Compute stress invariant
            if stress_type == "von_mises":
                results[combo_idx, :] = self.compute_von_mises(sx, sy, sz, sxy, syz, sxz)
            elif stress_type == "max_principal":
                s1, _, _ = self.compute_principal_stresses_numpy(sx, sy, sz, sxy, syz, sxz)
                results[combo_idx, :] = s1
            elif stress_type == "min_principal":
                _, _, s3 = self.compute_principal_stresses_numpy(sx, sy, sz, sxy, syz, sxz)
                results[combo_idx, :] = s3
            else:
                raise ValueError(f"Unknown stress type: {stress_type}")
        
        return results
    
    def _update_envelope_for_chunk(
        self,
        chunk_results: np.ndarray,
        chunk_start_idx: int,
        max_envelope: np.ndarray,
        min_envelope: np.ndarray,
        combo_of_max: np.ndarray,
        combo_of_min: np.ndarray
    ):
        """
        Update envelope arrays with chunk results (in-place).
        
        Uses np.maximum/np.minimum with in-place updates for memory efficiency.
        
        Args:
            chunk_results: Array of shape (num_combos, chunk_size).
            chunk_start_idx: Starting index in the full arrays.
            max_envelope: Full max envelope array (modified in-place).
            min_envelope: Full min envelope array (modified in-place).
            combo_of_max: Full combo indices for max (modified in-place).
            combo_of_min: Full combo indices for min (modified in-place).
        """
        chunk_size = chunk_results.shape[1]
        chunk_end_idx = chunk_start_idx + chunk_size
        
        # Get max/min and their indices for this chunk
        chunk_max = np.max(chunk_results, axis=0)
        chunk_argmax = np.argmax(chunk_results, axis=0)
        chunk_min = np.min(chunk_results, axis=0)
        chunk_argmin = np.argmin(chunk_results, axis=0)
        
        # Update envelope arrays
        max_envelope[chunk_start_idx:chunk_end_idx] = chunk_max
        min_envelope[chunk_start_idx:chunk_end_idx] = chunk_min
        combo_of_max[chunk_start_idx:chunk_end_idx] = chunk_argmax
        combo_of_min[chunk_start_idx:chunk_end_idx] = chunk_argmin
    
    def compute_full_analysis_chunked(
        self,
        stress_type: str = "von_mises",
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        chunk_size: Optional[int] = None
    ) -> CombinationResult:
        """
        Compute envelope analysis with chunked processing for memory efficiency.
        
        This method processes nodes in chunks to avoid loading all stress data
        into memory at once. Ideal for large models (>100K nodes).
        
        Algorithm:
        1. Pre-check memory requirements
        2. Initialize envelope arrays (max/min/combo_of_max/combo_of_min)
        3. For each node chunk:
           a. Load stress data for chunk from DPF
           b. Compute all combinations for chunk nodes
           c. Update envelope arrays incrementally
           d. Clear chunk data, gc.collect()
        4. Return CombinationResult with envelope data
        
        Args:
            stress_type: One of "von_mises", "max_principal", "min_principal".
            progress_callback: Optional callback(current, total, message) for progress.
            chunk_size: Optional explicit chunk size. If None, calculated automatically.
            
        Returns:
            CombinationResult with envelope data (without all_combo_results to save memory).
            
        Raises:
            MemoryError: If insufficient memory even for chunked processing.
        """
        # Get node information
        full_node_ids = np.array(self.scoping.ids)
        num_nodes = len(full_node_ids)
        
        # Check memory and calculate chunk size
        _, estimates = self.check_memory_available(raise_on_insufficient=True)
        
        if chunk_size is None:
            chunk_size = estimates['recommended_chunk_size']
        
        if progress_callback:
            progress_callback(0, num_nodes, f"Starting chunked analysis (chunk size: {chunk_size:,})...")
        
        # Get node coordinates upfront
        self._node_ids, self._node_coords = self.reader1.get_node_coordinates(self.scoping)
        
        # Initialize envelope arrays
        max_envelope = np.full(num_nodes, -np.inf)
        min_envelope = np.full(num_nodes, np.inf)
        combo_of_max = np.zeros(num_nodes, dtype=np.int32)
        combo_of_min = np.zeros(num_nodes, dtype=np.int32)
        
        # Process nodes in chunks
        num_chunks = (num_nodes + chunk_size - 1) // chunk_size
        
        for chunk_idx, chunk_start in enumerate(range(0, num_nodes, chunk_size)):
            chunk_end = min(chunk_start + chunk_size, num_nodes)
            actual_chunk_size = chunk_end - chunk_start
            
            if progress_callback:
                progress_callback(
                    chunk_start, 
                    num_nodes, 
                    f"Processing nodes {chunk_start+1:,}-{chunk_end:,} (chunk {chunk_idx+1}/{num_chunks})..."
                )
            
            # Load stress data for this chunk
            chunk_cache = self.load_stress_for_node_range(
                chunk_start, chunk_end, full_node_ids
            )
            
            # Compute all combinations for this chunk
            chunk_results = self._compute_chunk_combinations(
                chunk_cache, actual_chunk_size, stress_type
            )
            
            # Update envelope arrays
            self._update_envelope_for_chunk(
                chunk_results, chunk_start,
                max_envelope, min_envelope,
                combo_of_max, combo_of_min
            )
            
            # Clear chunk data to free memory
            del chunk_cache
            del chunk_results
            gc.collect()
        
        if progress_callback:
            progress_callback(num_nodes, num_nodes, "Chunked analysis complete.")
        
        return CombinationResult(
            node_ids=self._node_ids.copy(),
            node_coords=self._node_coords.copy(),
            max_over_combo=max_envelope,
            min_over_combo=min_envelope,
            combo_of_max=combo_of_max,
            combo_of_min=combo_of_min,
            result_type=stress_type,
            all_combo_results=None,  # Not stored to save memory
        )
    
    def compute_full_analysis_auto(
        self,
        stress_type: str = "von_mises",
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        memory_threshold_gb: float = 2.0,
        auto_cleanup: bool = True
    ) -> CombinationResult:
        """
        Automatically choose between standard and chunked processing.
        
        Uses chunked processing if estimated memory exceeds threshold.
        
        Args:
            stress_type: One of "von_mises", "max_principal", "min_principal".
            progress_callback: Optional callback for progress updates.
            memory_threshold_gb: Memory threshold in GB to switch to chunked.
            auto_cleanup: If True, clear cached data after computation to free memory.
            
        Returns:
            CombinationResult with envelope data.
        """
        num_nodes = len(self.scoping.ids)
        estimates = self.estimate_memory_requirements(num_nodes)
        
        threshold_bytes = memory_threshold_gb * 1024 * 1024 * 1024
        
        if estimates['total_bytes'] > threshold_bytes:
            # Use chunked processing for large models (inherently memory-efficient)
            if progress_callback:
                progress_callback(
                    0, 1, 
                    f"Large model detected ({num_nodes:,} nodes). Using chunked processing..."
                )
            return self.compute_full_analysis_chunked(
                stress_type=stress_type,
                progress_callback=progress_callback
            )
        else:
            # Use standard processing for smaller models
            self.preload_stress_data(progress_callback=progress_callback)
            return self.compute_full_analysis(
                stress_type=stress_type,
                progress_callback=progress_callback,
                auto_cleanup=auto_cleanup
            )
