"""
Data models for MARS-SC (Solution Combination).

Defines dataclasses and structures to hold analysis data in a structured and type-safe manner.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List
import pandas as pd
import numpy as np


# =============================================================================
# Combination Analysis Data Models (MARS-SC specific)
# =============================================================================

@dataclass
class AnalysisData:
    """
    Container for a single RST file analysis.
    
    Attributes:
        file_path: Path to the RST file.
        num_load_steps: Number of load steps/time sets in the analysis.
        load_step_ids: List of load step IDs available in the RST.
        time_values: List of time values (in seconds) for each set.
        named_selections: List of named selection names in the RST.
        unit_system: Unit system string from the RST file (e.g., "MKS: m, kg, N, s...").
        stress_unit: Original stress unit from the RST file (e.g., "Pa").
        stress_conversion_factor: Factor to convert stress to MPa.
        nodal_forces_available: Whether nodal forces are available in the RST.
        force_unit: Force unit from the RST file (e.g., "N").
        displacement_available: Whether displacement results are available in the RST.
        displacement_unit: Displacement unit from the RST file (e.g., "mm", "m").
    """
    file_path: str
    num_load_steps: int
    load_step_ids: List[int]
    named_selections: List[str]
    time_values: Optional[List[float]] = None
    unit_system: str = "Unknown"
    stress_unit: str = "Pa"
    stress_conversion_factor: float = 1e-6  # Default Pa to MPa
    nodal_forces_available: bool = False
    force_unit: str = "N"
    displacement_available: bool = False
    displacement_unit: str = "mm"
    
    def get_time_for_step(self, step_id: int) -> Optional[float]:
        """
        Get the time value for a specific load step ID.
        
        Args:
            step_id: Load step ID (1-based).
            
        Returns:
            Time value in seconds, or None if not available.
        """
        if self.time_values is None:
            return None
        try:
            idx = self.load_step_ids.index(step_id)
            return self.time_values[idx]
        except (ValueError, IndexError):
            return None
    
    def format_time_label(self, step_id: int, prefix: str = "A") -> str:
        """
        Format a column label with time value for a load step.
        
        Args:
            step_id: Load step ID (1-based).
            prefix: Prefix for the label (e.g., "A1", "A2").
            
        Returns:
            Formatted label like "A1_Time_0.5s" or "A1_Set_1" if time not available.
        """
        time_val = self.get_time_for_step(step_id)
        if time_val is not None:
            # Format time value nicely
            if time_val == int(time_val):
                # Integer time value
                return f"{prefix}_Time_{int(time_val)}s"
            elif abs(time_val) < 0.001:
                # Very small time, use scientific notation
                return f"{prefix}_Time_{time_val:.2e}s"
            elif abs(time_val) < 1:
                # Sub-second, show more decimal places
                return f"{prefix}_Time_{time_val:.3f}s"
            else:
                # Normal time value
                return f"{prefix}_Time_{time_val:.2f}s"
        else:
            # Fallback to set ID
            return f"{prefix}_Set_{step_id}"


@dataclass
class CombinationTableData:
    """
    Container for combination coefficients table.
    
    The combination table defines how to linearly combine stress results from
    two analyses. Each row represents a combination with coefficients for each
    load step from both analyses.
    
    Attributes:
        combination_names: List of combination names (one per row).
        combination_types: List of combination types (e.g., "Linear").
        analysis1_coeffs: Coefficients for analysis 1, shape (num_combos, num_a1_steps).
        analysis2_coeffs: Coefficients for analysis 2, shape (num_combos, num_a2_steps).
        analysis1_step_ids: Load step IDs from analysis 1 (column headers).
        analysis2_step_ids: Load step IDs from analysis 2 (column headers).
    """
    combination_names: List[str]
    combination_types: List[str]
    analysis1_coeffs: np.ndarray
    analysis2_coeffs: np.ndarray
    analysis1_step_ids: List[int]
    analysis2_step_ids: List[int]
    
    @property
    def num_combinations(self) -> int:
        """Number of combinations defined in the table."""
        return len(self.combination_names)
    
    @property
    def num_analysis1_steps(self) -> int:
        """Number of load steps from analysis 1."""
        return len(self.analysis1_step_ids)
    
    @property
    def num_analysis2_steps(self) -> int:
        """Number of load steps from analysis 2."""
        return len(self.analysis2_step_ids)
    
    def get_coeffs_for_combination(self, combo_index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get coefficients for a specific combination.
        
        Args:
            combo_index: Index of the combination (0-based).
            
        Returns:
            Tuple of (analysis1_coeffs, analysis2_coeffs) arrays for this combination.
        """
        return (self.analysis1_coeffs[combo_index, :], 
                self.analysis2_coeffs[combo_index, :])
    
    def is_combination_all_zeros(self, combo_index: int) -> bool:
        """
        Check if all coefficients for a combination are zero.
        
        Args:
            combo_index: Index of the combination (0-based).
            
        Returns:
            True if all coefficients (from both analyses) are zero.
        """
        a1_coeffs, a2_coeffs = self.get_coeffs_for_combination(combo_index)
        return np.allclose(a1_coeffs, 0.0) and np.allclose(a2_coeffs, 0.0)
    
    def get_zero_coefficient_combinations(self) -> List[Tuple[int, str]]:
        """
        Find all combinations that have all-zero coefficients.
        
        Returns:
            List of (index, name) tuples for combinations with all-zero coefficients.
        """
        zero_combos = []
        for i in range(self.num_combinations):
            if self.is_combination_all_zeros(i):
                zero_combos.append((i, self.combination_names[i]))
        return zero_combos
    
    def validate(self) -> Tuple[bool, str]:
        """
        Validate the combination table data.
        
        Returns:
            Tuple of (is_valid, error_message). If valid, error_message is empty.
        """
        if self.num_combinations == 0:
            return False, "No combinations defined in the table."
        
        # Check for all-zero coefficient combinations
        zero_combos = self.get_zero_coefficient_combinations()
        if zero_combos:
            combo_list = ", ".join([f"'{name}' (row {idx + 1})" for idx, name in zero_combos])
            return False, (
                f"The following combinations have all-zero coefficients:\n{combo_list}\n\n"
                f"Please enter at least one non-zero coefficient for each combination, "
                f"or delete the rows with all-zero coefficients."
            )
        
        return True, ""
    
    def get_active_step_ids(self) -> Tuple[List[int], List[int]]:
        """
        Get step IDs that have at least one non-zero coefficient across all combinations.
        
        This method identifies which load steps are actually used in the combination
        calculations, allowing callers to skip loading data for steps where all
        coefficients are zero. This can dramatically reduce I/O and memory usage
        when only a subset of available steps are used.
        
        Returns:
            Tuple of (active_a1_step_ids, active_a2_step_ids) where each list
            contains only the step IDs that have at least one non-zero coefficient.
        
        Example:
            If analysis1_step_ids = [1, 2, 3, 4, 5] and the coefficient matrix has
            non-zero values only in columns 0 and 2, this returns ([1, 3], ...).
        """
        # For Analysis 1: find columns (steps) where any row (combination) has non-zero
        a1_active_mask = np.any(self.analysis1_coeffs != 0.0, axis=0)
        active_a1 = [step_id for i, step_id in enumerate(self.analysis1_step_ids) 
                     if a1_active_mask[i]]
        
        # For Analysis 2: same logic
        a2_active_mask = np.any(self.analysis2_coeffs != 0.0, axis=0)
        active_a2 = [step_id for i, step_id in enumerate(self.analysis2_step_ids) 
                     if a2_active_mask[i]]
        
        return (active_a1, active_a2)


@dataclass
class CombinationResult:
    """
    Container for combination analysis results.
    
    Holds the results of computing stress envelopes (max/min) across all
    combinations, including which combination caused the extreme values.
    
    Attributes:
        node_ids: Array of node IDs.
        node_coords: Node coordinates, shape (num_nodes, 3).
        max_over_combo: Maximum stress values over all combinations, shape (num_nodes,).
        min_over_combo: Minimum stress values over all combinations, shape (num_nodes,).
        combo_of_max: Index of combination that caused max at each node, shape (num_nodes,).
        combo_of_min: Index of combination that caused min at each node, shape (num_nodes,).
        result_type: Type of stress result (e.g., "von_mises", "max_principal", "min_principal").
        all_combo_results: Optional full results array, shape (num_combinations, num_nodes).
    """
    node_ids: np.ndarray
    node_coords: np.ndarray
    max_over_combo: Optional[np.ndarray] = None
    min_over_combo: Optional[np.ndarray] = None
    combo_of_max: Optional[np.ndarray] = None
    combo_of_min: Optional[np.ndarray] = None
    result_type: str = "von_mises"
    all_combo_results: Optional[np.ndarray] = None
    
    @property
    def num_nodes(self) -> int:
        """Number of nodes in the result."""
        return len(self.node_ids)
    
    @property
    def num_combinations(self) -> int:
        """Number of combinations (if full results available)."""
        if self.all_combo_results is not None:
            return self.all_combo_results.shape[0]
        return 0


@dataclass
class NodalForcesResult:
    """
    Container for nodal forces combination analysis results.
    
    Holds the results of computing nodal force envelopes (max/min) across all
    combinations for each force component.
    
    Attributes:
        node_ids: Array of node IDs.
        node_coords: Node coordinates, shape (num_nodes, 3).
        combined_forces: Dict mapping combo_index to force arrays.
            Each entry has (fx, fy, fz) arrays of shape (num_nodes,).
        max_magnitude_over_combo: Maximum force magnitude over all combinations, shape (num_nodes,).
        min_magnitude_over_combo: Minimum force magnitude over all combinations, shape (num_nodes,).
        combo_of_max: Index of combination that caused max at each node, shape (num_nodes,).
        combo_of_min: Index of combination that caused min at each node, shape (num_nodes,).
        all_combo_fx: Full FX results array, shape (num_combinations, num_nodes).
        all_combo_fy: Full FY results array, shape (num_combinations, num_nodes).
        all_combo_fz: Full FZ results array, shape (num_combinations, num_nodes).
        force_unit: Unit of force (e.g., "N").
        node_element_types: Array of element types per node ('beam' or 'solid_shell').
        has_beam_nodes: True if any node has beam elements attached.
        coordinate_system: Coordinate system used for forces ('Global' or 'Local').
    """
    node_ids: np.ndarray
    node_coords: np.ndarray
    max_magnitude_over_combo: Optional[np.ndarray] = None
    min_magnitude_over_combo: Optional[np.ndarray] = None
    combo_of_max: Optional[np.ndarray] = None
    combo_of_min: Optional[np.ndarray] = None
    all_combo_fx: Optional[np.ndarray] = None
    all_combo_fy: Optional[np.ndarray] = None
    all_combo_fz: Optional[np.ndarray] = None
    force_unit: str = "N"
    node_element_types: Optional[np.ndarray] = None
    has_beam_nodes: bool = False
    coordinate_system: str = "Global"

    @property
    def num_nodes(self) -> int:
        """Number of nodes in the result."""
        return len(self.node_ids)

    @property
    def num_combinations(self) -> int:
        """Number of combinations (if full results available)."""
        if self.all_combo_fx is not None:
            return self.all_combo_fx.shape[0]
        return 0

    def get_force_magnitude(self, combo_idx: int) -> np.ndarray:
        """
        Compute force magnitude for a specific combination.
        
        Args:
            combo_idx: Index of the combination.
            
        Returns:
            Array of force magnitudes, shape (num_nodes,).
        """
        if self.all_combo_fx is None or self.all_combo_fy is None or self.all_combo_fz is None:
            raise ValueError("Force components not available.")
        fx = self.all_combo_fx[combo_idx, :]
        fy = self.all_combo_fy[combo_idx, :]
        fz = self.all_combo_fz[combo_idx, :]
        return np.sqrt(fx**2 + fy**2 + fz**2)


@dataclass
class DeformationResult:
    """
    Container for deformation (displacement) combination analysis results.
    
    Holds the results of computing displacement envelopes (max/min) across all
    combinations for each displacement component.
    
    Attributes:
        node_ids: Array of node IDs.
        node_coords: Original (undeformed) node coordinates, shape (num_nodes, 3).
        max_magnitude_over_combo: Maximum displacement magnitude over all combinations, shape (num_nodes,).
        min_magnitude_over_combo: Minimum displacement magnitude over all combinations, shape (num_nodes,).
        combo_of_max: Index of combination that caused max at each node, shape (num_nodes,).
        combo_of_min: Index of combination that caused min at each node, shape (num_nodes,).
        all_combo_ux: Full UX (X displacement) results array, shape (num_combinations, num_nodes).
        all_combo_uy: Full UY (Y displacement) results array, shape (num_combinations, num_nodes).
        all_combo_uz: Full UZ (Z displacement) results array, shape (num_combinations, num_nodes).
        displacement_unit: Unit of displacement (e.g., "mm", "m").
    """
    node_ids: np.ndarray
    node_coords: np.ndarray
    max_magnitude_over_combo: Optional[np.ndarray] = None
    min_magnitude_over_combo: Optional[np.ndarray] = None
    combo_of_max: Optional[np.ndarray] = None
    combo_of_min: Optional[np.ndarray] = None
    all_combo_ux: Optional[np.ndarray] = None
    all_combo_uy: Optional[np.ndarray] = None
    all_combo_uz: Optional[np.ndarray] = None
    displacement_unit: str = "mm"
    
    @property
    def num_nodes(self) -> int:
        """Number of nodes in the result."""
        return len(self.node_ids)
    
    @property
    def num_combinations(self) -> int:
        """Number of combinations (if full results available)."""
        if self.all_combo_ux is not None:
            return self.all_combo_ux.shape[0]
        return 0
    
    def get_displacement_magnitude(self, combo_idx: int) -> np.ndarray:
        """
        Compute displacement magnitude for a specific combination.
        
        Args:
            combo_idx: Index of the combination.
            
        Returns:
            Array of displacement magnitudes, shape (num_nodes,).
        """
        if self.all_combo_ux is None or self.all_combo_uy is None or self.all_combo_uz is None:
            raise ValueError("Displacement components not available.")
        ux = self.all_combo_ux[combo_idx, :]
        uy = self.all_combo_uy[combo_idx, :]
        uz = self.all_combo_uz[combo_idx, :]
        return np.sqrt(ux**2 + uy**2 + uz**2)
    
    def get_displacement_component(self, combo_idx: int, component: str) -> np.ndarray:
        """
        Get a specific displacement component for a combination.
        
        Args:
            combo_idx: Index of the combination.
            component: Displacement component name - 'UX', 'UY', 'UZ', or 'U_mag'.
            
        Returns:
            Array of displacement values for the specified component, shape (num_nodes,).
        """
        component_upper = component.upper()
        
        if component_upper == 'UX':
            if self.all_combo_ux is None:
                raise ValueError("UX component not available.")
            return self.all_combo_ux[combo_idx, :]
        elif component_upper == 'UY':
            if self.all_combo_uy is None:
                raise ValueError("UY component not available.")
            return self.all_combo_uy[combo_idx, :]
        elif component_upper == 'UZ':
            if self.all_combo_uz is None:
                raise ValueError("UZ component not available.")
            return self.all_combo_uz[combo_idx, :]
        elif component_upper in ('U_MAG', 'UMAG', 'MAGNITUDE'):
            return self.get_displacement_magnitude(combo_idx)
        else:
            raise ValueError(f"Unknown displacement component: {component}. "
                           f"Valid options: 'UX', 'UY', 'UZ', 'U_mag'")


# =============================================================================
# Configuration Models
# =============================================================================

@dataclass
class PlasticityConfig:
    """Configuration for plasticity corrections."""

    enabled: bool = False
    method: str = "neuber"  # allowable: "neuber", "glinka", "ibg"
    max_iterations: int = 60
    tolerance: float = 1e-10
    material_profile: Optional['MaterialProfileData'] = None
    temperature_field: Optional['TemperatureFieldData'] = None
    default_temperature: Optional[float] = None
    temperature_column: Optional[str] = None
    poisson_ratio: Optional[float] = None
    extrapolation_mode: str = "linear"  # "linear" or "plateau"

    @property
    def is_active(self) -> bool:
        return self.enabled and (self.material_profile is not None)


@dataclass
class SolverConfig:
    """
    Configuration settings for the MARS-SC combination solver.
    
    Attributes:
        calculate_von_mises: Whether to calculate von Mises stress.
        calculate_max_principal_stress: Whether to calculate max principal stress.
        calculate_min_principal_stress: Whether to calculate min principal stress.
        calculate_nodal_forces: Whether to calculate combined nodal forces.
        calculate_deformation: Whether to calculate combined displacement/deformation.
        nodal_forces_rotate_to_global: If True, rotate nodal forces to global coordinate 
            system. If False, keep forces in element (local) coordinate system.
        deformation_cylindrical_cs_id: Optional coordinate system ID for transforming
            deformation results to cylindrical coordinates. If None, results remain
            in global Cartesian coordinates.
        combination_history_mode: Whether in combination history mode (single node).
        selected_node_id: Node ID for combination history mode.
        selected_combination_index: Combination index for single combination export.
        output_directory: Directory for output files.
        plasticity: Optional plasticity correction configuration.
    """
    calculate_von_mises: bool = True
    calculate_max_principal_stress: bool = False
    calculate_min_principal_stress: bool = False
    calculate_nodal_forces: bool = False
    calculate_deformation: bool = False
    nodal_forces_rotate_to_global: bool = True
    deformation_cylindrical_cs_id: Optional[int] = None
    combination_history_mode: bool = False
    selected_node_id: Optional[int] = None
    selected_combination_index: Optional[int] = None
    output_directory: Optional[str] = None
    plasticity: Optional[PlasticityConfig] = None


# =============================================================================
# Material and Temperature Data Models
# =============================================================================

@dataclass
class TemperatureFieldData:
    """Container for nodal temperature field information."""

    dataframe: pd.DataFrame

    @property
    def num_nodes(self) -> int:
        """Number of temperature entries."""
        return len(self.dataframe.index)

    def get_column(self, column_name: str) -> pd.Series:
        """Return a specific column from the underlying dataframe."""
        return self.dataframe[column_name]


@dataclass
class MaterialProfileData:
    """Container for temperature-dependent material properties."""

    youngs_modulus: pd.DataFrame
    poisson_ratio: pd.DataFrame
    plastic_curves: Dict[float, pd.DataFrame] = field(default_factory=dict)

    @classmethod
    def empty(cls):
        return cls(
            youngs_modulus=pd.DataFrame(columns=["Temperature (°C)", "Young's Modulus [MPa]"]),
            poisson_ratio=pd.DataFrame(columns=["Temperature (°C)", "Poisson's Ratio"]),
            plastic_curves={}
        )

    @property
    def has_data(self) -> bool:
        return not (self.youngs_modulus.empty and self.poisson_ratio.empty and not self.plastic_curves)

    def curve_for_temperature(self, temperature: float) -> Optional[pd.DataFrame]:
        return self.plastic_curves.get(temperature)


# =============================================================================
# Result Models
# =============================================================================

@dataclass
class AnalysisResult:
    """
    Container for analysis results.
    
    Used for both combination history (single node over combinations) and
    batch results (all nodes for envelope).
    
    Attributes:
        combination_indices: Array of combination indices (for history mode).
        stress_values: Computed stress or other values.
        result_type: Type of result (e.g., 'von_mises', 'max_principal').
        node_id: Node ID (for single node results).
        metadata: Additional metadata dictionary.
    """
    combination_indices: Optional[np.ndarray] = None
    stress_values: Optional[np.ndarray] = None
    result_type: str = "unknown"
    node_id: Optional[int] = None
    metadata: dict = field(default_factory=dict)


# =============================================================================
# Legacy Models (kept for compatibility during transition)
# =============================================================================

@dataclass
class ModalData:
    """
    Container for modal coordinate data (legacy - kept for compatibility).
    
    Attributes:
        modal_coord: Modal coordinates array, shape (num_modes, num_time_points).
        time_values: Array of time values.
        num_modes: Number of modes in the modal coordinate data.
        num_time_points: Number of time points in the analysis.
    """
    modal_coord: np.ndarray
    time_values: np.ndarray
    
    @property
    def num_modes(self) -> int:
        """Number of modes."""
        return self.modal_coord.shape[0]
    
    @property
    def num_time_points(self) -> int:
        """Number of time points."""
        return self.modal_coord.shape[1]


@dataclass
class ModalStressData:
    """
    Container for modal stress components (legacy - kept for compatibility).
    
    Attributes:
        node_ids: Array of node IDs.
        modal_sx: Modal stress in X direction, shape (num_nodes, num_modes).
        modal_sy: Modal stress in Y direction, shape (num_nodes, num_modes).
        modal_sz: Modal stress in Z direction, shape (num_nodes, num_modes).
        modal_sxy: Modal shear stress XY, shape (num_nodes, num_modes).
        modal_syz: Modal shear stress YZ, shape (num_nodes, num_modes).
        modal_sxz: Modal shear stress XZ, shape (num_nodes, num_modes).
        node_coords: Optional node coordinates, shape (num_nodes, 3).
    """
    node_ids: np.ndarray
    modal_sx: np.ndarray
    modal_sy: np.ndarray
    modal_sz: np.ndarray
    modal_sxy: np.ndarray
    modal_syz: np.ndarray
    modal_sxz: np.ndarray
    node_coords: Optional[np.ndarray] = None
    
    @property
    def num_nodes(self) -> int:
        """Number of nodes."""
        return len(self.node_ids)
    
    @property
    def num_modes(self) -> int:
        """Number of modes."""
        return self.modal_sx.shape[1]


@dataclass
class DeformationData:
    """
    Container for modal deformation components (legacy - kept for compatibility).
    
    Attributes:
        node_ids: Array of node IDs.
        modal_ux: Modal displacement in X direction, shape (num_nodes, num_modes).
        modal_uy: Modal displacement in Y direction, shape (num_nodes, num_modes).
        modal_uz: Modal displacement in Z direction, shape (num_nodes, num_modes).
    """
    node_ids: np.ndarray
    modal_ux: np.ndarray
    modal_uy: np.ndarray
    modal_uz: np.ndarray
    
    @property
    def num_nodes(self) -> int:
        """Number of nodes."""
        return len(self.node_ids)
    
    @property
    def num_modes(self) -> int:
        """Number of modes."""
        return self.modal_ux.shape[1]
    
    def as_tuple(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return deformations as a tuple (ux, uy, uz)."""
        return (self.modal_ux, self.modal_uy, self.modal_uz)


@dataclass
class SteadyStateData:
    """
    Container for steady-state stress data (legacy - kept for compatibility).
    
    Attributes:
        node_ids: Array of node IDs.
        steady_sx: Steady-state stress in X direction.
        steady_sy: Steady-state stress in Y direction.
        steady_sz: Steady-state stress in Z direction.
        steady_sxy: Steady-state shear stress XY.
        steady_syz: Steady-state shear stress YZ.
        steady_sxz: Steady-state shear stress XZ.
    """
    node_ids: np.ndarray
    steady_sx: np.ndarray
    steady_sy: np.ndarray
    steady_sz: np.ndarray
    steady_sxy: np.ndarray
    steady_syz: np.ndarray
    steady_sxz: np.ndarray
    
    @property
    def num_nodes(self) -> int:
        """Number of nodes."""
        return len(self.node_ids)
