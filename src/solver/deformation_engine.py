"""
Deformation (Displacement) Combination Engine for MARS-SC (Solution Combination).

Performs linear combination of displacement results from two analyses and computes
displacement envelopes over combinations.

The combination formula is:
    U_combined = Σ(α_i × U_A1_i) + Σ(β_j × U_A2_j)

Where:
    - α_i are coefficients for Analysis 1 load steps
    - β_j are coefficients for Analysis 2 load steps
    - U_A1_i and U_A2_j are displacement vectors for each load step
"""

from typing import Dict, Tuple, Optional, Callable, List
import numpy as np
import gc

from file_io.dpf_reader import (
    DPFAnalysisReader,
    DisplacementNotAvailableError,
    DPF_AVAILABLE,
)
from core.data_models import CombinationTableData, DeformationResult

if DPF_AVAILABLE:
    from ansys.dpf import core as dpf


class DeformationCombinationEngine:
    """
    Performs linear combination of displacement from two analyses.
    
    This engine preloads displacement data from both analyses and then computes
    combined displacements for each combination defined in the combination table.
    
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
    ):
        """
        Initialize the deformation combination engine.
        
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
        
        # Displacement cache: maps (analysis_idx, step_id) -> displacement components tuple
        # Each tuple is (node_ids, ux, uy, uz)
        self._displacement_cache: Dict[Tuple[int, int], Tuple] = {}
        
        # Node information (populated during preload)
        self._node_ids: Optional[np.ndarray] = None
        self._node_coords: Optional[np.ndarray] = None
        
        # Displacement unit
        self._displacement_unit: str = "mm"
    
    @property
    def node_ids(self) -> np.ndarray:
        """Node IDs (available after preload)."""
        if self._node_ids is None:
            raise RuntimeError("Displacement data not preloaded. Call preload_displacement_data() first.")
        return self._node_ids
    
    @property
    def node_coords(self) -> np.ndarray:
        """Node coordinates (available after preload)."""
        if self._node_coords is None:
            raise RuntimeError("Displacement data not preloaded. Call preload_displacement_data() first.")
        return self._node_coords
    
    @property
    def num_nodes(self) -> int:
        """Number of nodes being processed."""
        return len(self.node_ids)
    
    @property
    def displacement_unit(self) -> str:
        """Displacement unit string."""
        return self._displacement_unit
    
    def validate_displacement_availability(self) -> Tuple[bool, str]:
        """
        Validate that displacement results are available in both RST files
        for all active load steps (those with non-zero coefficients).
        
        Returns:
            Tuple of (is_valid, error_message). If valid, error_message is empty.
        """
        errors = []
        
        # Get only active steps (those with non-zero coefficients)
        active_a1_steps, active_a2_steps = self.table.get_active_step_ids()
        
        # Check Analysis 1 (only if it has active steps)
        if active_a1_steps:
            if not self.reader1.check_displacement_available():
                errors.append(
                    "Analysis 1 RST file does not contain displacement results.\n"
                    "Ensure displacement output is enabled in ANSYS Output Controls."
                )
            else:
                # Check only active load steps
                for step_id in active_a1_steps:
                    try:
                        self.reader1.read_displacement_for_loadstep(step_id, self.scoping)
                    except DisplacementNotAvailableError as e:
                        errors.append(f"Analysis 1, Load Step {step_id}: {str(e)}")
        
        # Check Analysis 2 (only if it has active steps)
        if active_a2_steps:
            if not self.reader2.check_displacement_available():
                errors.append(
                    "Analysis 2 RST file does not contain displacement results.\n"
                    "Ensure displacement output is enabled in ANSYS Output Controls."
                )
            else:
                # Check only active load steps
                for step_id in active_a2_steps:
                    try:
                        self.reader2.read_displacement_for_loadstep(step_id, self.scoping)
                    except DisplacementNotAvailableError as e:
                        errors.append(f"Analysis 2, Load Step {step_id}: {str(e)}")
        
        if errors:
            return False, "\n\n".join(errors)
        return True, ""
    
    def preload_displacement_data(self, progress_callback: Optional[Callable[[int, int, str], None]] = None):
        """
        Cache displacement data for load steps with non-zero coefficients.
        
        This method reads displacement data upfront to avoid repeated file I/O
        during combination calculations. Only steps that have at least one
        non-zero coefficient across all combinations are loaded.
        
        Args:
            progress_callback: Optional callback(current, total, message) for progress updates.
            
        Raises:
            DisplacementNotAvailableError: If displacement results are not available.
        """
        # Get only active steps (those with non-zero coefficients)
        a1_steps, a2_steps = self.table.get_active_step_ids()
        total_steps = len(a1_steps) + len(a2_steps)
        current = 0
        
        if total_steps == 0:
            raise ValueError("No active load steps found. All coefficients are zero.")
        
        # Get displacement unit from first file
        self._displacement_unit = self.reader1.get_displacement_unit()
        
        # Load Analysis 1 displacement data (only active steps)
        for step_id in a1_steps:
            if progress_callback:
                progress_callback(current, total_steps, f"Loading A1 Displacement Step {step_id}...")
            
            result = self.reader1.read_displacement_for_loadstep(step_id, self.scoping)
            self._displacement_cache[(1, step_id)] = result
            current += 1
        
        # Load Analysis 2 displacement data (only active steps)
        for step_id in a2_steps:
            if progress_callback:
                progress_callback(current, total_steps, f"Loading A2 Displacement Step {step_id}...")
            
            result = self.reader2.read_displacement_for_loadstep(step_id, self.scoping)
            self._displacement_cache[(2, step_id)] = result
            current += 1
        
        if progress_callback:
            progress_callback(total_steps, total_steps, "Displacement data loading complete.")
        
        # Store node information from first loaded step
        if a1_steps:
            first_result = self._displacement_cache[(1, a1_steps[0])]
        elif a2_steps:
            first_result = self._displacement_cache[(2, a2_steps[0])]
        else:
            raise ValueError("No load steps defined in combination table.")
        
        self._node_ids = first_result[0]
        
        # Get node coordinates
        self._node_ids, self._node_coords = self.reader1.get_node_coordinates(self.scoping)
    
    def compute_combination_numpy(self, combo_index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute combined displacement vector for a single combination using numpy.
        
        Args:
            combo_index: Index of the combination (0-based).
            
        Returns:
            Tuple of (ux, uy, uz) combined arrays, each shape (num_nodes,).
        """
        a1_coeffs, a2_coeffs = self.table.get_coeffs_for_combination(combo_index)
        
        # Initialize combined displacement components
        num_nodes = self.num_nodes
        ux = np.zeros(num_nodes)
        uy = np.zeros(num_nodes)
        uz = np.zeros(num_nodes)
        
        # Add contributions from Analysis 1 (check cache membership for active-only loading)
        for i, step_id in enumerate(self.table.analysis1_step_ids):
            coeff = a1_coeffs[i]
            if coeff != 0.0 and (1, step_id) in self._displacement_cache:
                _, s_ux, s_uy, s_uz = self._displacement_cache[(1, step_id)]
                ux += coeff * s_ux
                uy += coeff * s_uy
                uz += coeff * s_uz
        
        # Add contributions from Analysis 2 (check cache membership for active-only loading)
        for i, step_id in enumerate(self.table.analysis2_step_ids):
            coeff = a2_coeffs[i]
            if coeff != 0.0 and (2, step_id) in self._displacement_cache:
                _, s_ux, s_uy, s_uz = self._displacement_cache[(2, step_id)]
                ux += coeff * s_ux
                uy += coeff * s_uy
                uz += coeff * s_uz
        
        return (ux, uy, uz)
    
    @staticmethod
    def compute_magnitude(ux: np.ndarray, uy: np.ndarray, uz: np.ndarray) -> np.ndarray:
        """
        Compute displacement magnitude from 3-component vector.
        
        Args:
            ux, uy, uz: Displacement components.
            
        Returns:
            Displacement magnitude array.
        """
        return np.sqrt(ux**2 + uy**2 + uz**2)
    
    def compute_all_combinations(
        self,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute combined displacements for ALL combinations.
        
        Args:
            progress_callback: Optional callback(current, total, message) for progress.
            
        Returns:
            Tuple of (ux_all, uy_all, uz_all, magnitude_all) arrays,
            each of shape (num_combinations, num_nodes).
        """
        num_combos = self.table.num_combinations
        num_nodes = self.num_nodes
        
        ux_all = np.zeros((num_combos, num_nodes))
        uy_all = np.zeros((num_combos, num_nodes))
        uz_all = np.zeros((num_combos, num_nodes))
        magnitude_all = np.zeros((num_combos, num_nodes))
        
        for combo_idx in range(num_combos):
            if progress_callback:
                combo_name = self.table.combination_names[combo_idx]
                progress_callback(combo_idx, num_combos, f"Computing displacement: {combo_name}...")
            
            ux, uy, uz = self.compute_combination_numpy(combo_idx)
            ux_all[combo_idx, :] = ux
            uy_all[combo_idx, :] = uy
            uz_all[combo_idx, :] = uz
            magnitude_all[combo_idx, :] = self.compute_magnitude(ux, uy, uz)
        
        if progress_callback:
            progress_callback(num_combos, num_combos, "Displacement computation complete.")
        
        return (ux_all, uy_all, uz_all, magnitude_all)
    
    def compute_envelope(
        self,
        magnitude_results: np.ndarray,
        envelope_type: str = "max"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute envelope across combinations based on displacement magnitude.
        
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
    ) -> DeformationResult:
        """
        Compute complete displacement envelope analysis and return DeformationResult.
        
        Args:
            progress_callback: Optional callback for progress updates.
            auto_cleanup: If True, clear cached displacement data after computation to
                          free memory. Default is True for memory efficiency.
            
        Returns:
            DeformationResult with all envelope data.
        """
        # Compute all combinations
        ux_all, uy_all, uz_all, magnitude_all = self.compute_all_combinations(
            progress_callback=progress_callback
        )
        
        # Compute envelopes based on magnitude
        max_values, combo_of_max = self.compute_envelope(magnitude_all, "max")
        min_values, combo_of_min = self.compute_envelope(magnitude_all, "min")
        
        result = DeformationResult(
            node_ids=self.node_ids.copy(),
            node_coords=self.node_coords.copy(),
            max_magnitude_over_combo=max_values,
            min_magnitude_over_combo=min_values,
            combo_of_max=combo_of_max,
            combo_of_min=combo_of_min,
            all_combo_ux=ux_all,
            all_combo_uy=uy_all,
            all_combo_uz=uz_all,
            displacement_unit=self.displacement_unit,
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
        Compute displacement history for a single node across all combinations.
        
        Args:
            node_id: Node ID to analyze.
            
        Returns:
            Tuple of (combination_indices, ux, uy, uz, magnitude) arrays.
        """
        # Find node index
        node_idx = np.where(self.node_ids == node_id)[0]
        if len(node_idx) == 0:
            raise ValueError(f"Node ID {node_id} not found in scoping.")
        node_idx = node_idx[0]
        
        # Compute all combinations and extract single node
        ux_all, uy_all, uz_all, magnitude_all = self.compute_all_combinations()
        
        ux = ux_all[:, node_idx]
        uy = uy_all[:, node_idx]
        uz = uz_all[:, node_idx]
        magnitude = magnitude_all[:, node_idx]
        
        combination_indices = np.arange(self.table.num_combinations)
        
        return (combination_indices, ux, uy, uz, magnitude)
    
    def get_combination_names(self) -> List[str]:
        """Get list of combination names."""
        return self.table.combination_names
    
    def clear_cache(self):
        """Clear cached displacement data to free memory."""
        self._displacement_cache.clear()
