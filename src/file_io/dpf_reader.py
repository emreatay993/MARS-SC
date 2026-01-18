"""
DPF Analysis Reader for MARS-SC (Solution Combination).

Provides functionality to read RST files using ansys-dpf-core and extract
stress tensor data for linear combination analysis.

Reference: https://dpf.docs.pyansys.com/version/stable/examples/06-plotting/02-solution_combination.html
"""

from typing import List, Tuple, Optional, Dict
import numpy as np

try:
    from ansys.dpf import core as dpf
    DPF_AVAILABLE = True
except ImportError:
    DPF_AVAILABLE = False
    dpf = None

from core.data_models import AnalysisData


# Target stress unit for MARS-SC output (MPa = N/mm² = Pa * 1e-6)
TARGET_STRESS_UNIT = "Pa"  # DPF uses "Pa" for Pascals


def get_stress_unit_conversion_factor(source_unit: str) -> float:
    """
    Get the conversion factor to convert from source unit to MPa.
    
    MARS-SC reports stresses in MPa (millimeter-Newton standard).
    
    Args:
        source_unit: The source unit string from DPF (e.g., "Pa", "psi", etc.)
        
    Returns:
        Multiplication factor to convert to MPa.
    """
    # Common stress unit conversion factors to MPa
    # 1 MPa = 1 N/mm² = 1e6 Pa
    conversion_to_mpa = {
        'Pa': 1e-6,           # Pascal to MPa
        'pa': 1e-6,           # Pascal (lowercase)
        'N/m^2': 1e-6,        # Newton per square meter
        'N/m**2': 1e-6,       # Alternative notation
        'kPa': 1e-3,          # kiloPascal to MPa
        'MPa': 1.0,           # Already MPa
        'mpa': 1.0,           # Already MPa (lowercase)
        'GPa': 1e3,           # GigaPascal to MPa
        'psi': 0.00689476,    # Pounds per square inch to MPa
        'ksi': 6.89476,       # kilo-psi to MPa
        'bar': 0.1,           # bar to MPa
        'N/mm^2': 1.0,        # Already MPa (N/mm² = MPa)
        'N/mm**2': 1.0,       # Alternative notation
    }
    
    return conversion_to_mpa.get(source_unit, 1e-6)  # Default to Pa->MPa if unknown


class DPFNotAvailableError(Exception):
    """Raised when DPF is not available but required."""
    pass


class NodalForcesNotAvailableError(Exception):
    """Raised when nodal forces are not available in the RST file."""
    pass


class BeamElementNotSupportedError(Exception):
    """Raised when beam elements are detected (not supported for this operation)."""
    pass


class DPFAnalysisReader:
    """
    Reads RST files using ansys-dpf-core.
    
    Provides methods to extract metadata, named selections, and stress tensor
    data from ANSYS result files for use in solution combination analysis.
    
    Attributes:
        model: DPF Model object for the RST file.
        rst_path: Path to the RST file.
    """
    
    def __init__(self, rst_path: str):
        """
        Initialize the DPF reader with an RST file.
        
        Args:
            rst_path: Path to the ANSYS RST result file.
            
        Raises:
            DPFNotAvailableError: If ansys-dpf-core is not installed.
            FileNotFoundError: If the RST file does not exist.
        """
        if not DPF_AVAILABLE:
            raise DPFNotAvailableError(
                "ansys-dpf-core is not installed. Please install it with: "
                "pip install ansys-dpf-core"
            )
        
        self.rst_path = rst_path
        self.model = dpf.Model(rst_path)
        self._mesh = None
        self._time_freq_support = None
        self._unit_system = None
        self._stress_unit = None
        self._stress_conversion_factor = None
        self._nodal_forces_available = None
        self._force_unit = None
    
    @property
    def mesh(self):
        """Get the meshed region from the model (cached)."""
        if self._mesh is None:
            self._mesh = self.model.metadata.meshed_region
        return self._mesh
    
    @property
    def time_freq_support(self):
        """Get the time/frequency support from the model (cached)."""
        if self._time_freq_support is None:
            self._time_freq_support = self.model.metadata.time_freq_support
        return self._time_freq_support
    
    @property
    def unit_system(self) -> str:
        """
        Get the unit system from the RST file.
        
        Returns:
            String describing the unit system (e.g., "MKS: m, kg, N, s, V, A, degC").
        """
        if self._unit_system is None:
            try:
                result_info = self.model.metadata.result_info
                self._unit_system = str(result_info.unit_system)
            except Exception:
                self._unit_system = "Unknown"
        return self._unit_system
    
    @property
    def stress_unit(self) -> str:
        """
        Get the stress unit from the RST file by reading a sample stress field.
        
        Returns:
            String describing the stress unit (e.g., "Pa", "psi").
        """
        if self._stress_unit is None:
            try:
                # Read a sample stress field to get the unit
                stress_op = self.model.results.stress()
                time_scoping = dpf.Scoping()
                time_scoping.ids = [1]  # First load step
                stress_op.inputs.time_scoping.connect(time_scoping)
                stress_op.inputs.requested_location.connect(dpf.locations.nodal)
                
                fields_container = stress_op.outputs.fields_container()
                if fields_container and len(fields_container) > 0:
                    self._stress_unit = fields_container[0].unit or "Pa"
                else:
                    self._stress_unit = "Pa"
            except Exception:
                self._stress_unit = "Pa"  # Default to Pa
        return self._stress_unit
    
    @property
    def stress_conversion_factor(self) -> float:
        """
        Get the conversion factor to convert stress from RST units to MPa.
        
        Returns:
            Multiplication factor to convert stress values to MPa.
        """
        if self._stress_conversion_factor is None:
            self._stress_conversion_factor = get_stress_unit_conversion_factor(self.stress_unit)
        return self._stress_conversion_factor
    
    def get_named_selections(self) -> List[str]:
        """
        Return list of named selection names in RST.
        
        Returns:
            List of named selection names available in the model.
        """
        try:
            available_ns = self.model.metadata.available_named_selections
            return list(available_ns) if available_ns else []
        except Exception:
            return []
    
    def get_load_step_count(self) -> int:
        """
        Return number of load steps/time sets.
        
        Returns:
            Number of result sets (load steps/substeps) in the model.
        """
        try:
            return self.time_freq_support.n_sets
        except Exception:
            return 0
    
    def get_load_step_ids(self) -> List[int]:
        """
        Return list of load step IDs (1-based set IDs).
        
        Returns:
            List of set IDs available in the result file.
        """
        try:
            n_sets = self.time_freq_support.n_sets
            return list(range(1, n_sets + 1))
        except Exception:
            return []
    
    def get_last_substep_ids(self) -> List[int]:
        """
        Return list of set IDs corresponding to the last substep of each load step.
        
        This method identifies load steps and returns only the final substep
        of each, which is useful for reducing the number of result sets when
        the RST file contains many intermediate substeps.
        
        Uses multiple strategies:
        1. DPF's get_cumulative_index(step, substep) API
        2. Time value analysis to detect load step boundaries
        
        Returns:
            List of set IDs representing the last substep of each load step.
        """
        try:
            tf_support = self.time_freq_support
            n_sets = tf_support.n_sets
            
            if n_sets == 0:
                return []
            
            # Strategy 1: Try DPF's step/substep API
            result = self._get_last_substep_ids_via_dpf_api(tf_support, n_sets)
            if result is not None:
                return result
            
            # Strategy 2: Analyze time values to detect load step boundaries
            result = self._get_last_substep_ids_via_time_analysis(n_sets)
            if result is not None:
                return result
            
            # Fallback: return all sets
            return list(range(1, n_sets + 1))
            
        except Exception:
            return self.get_load_step_ids()
    
    def _get_last_substep_ids_via_dpf_api(self, tf_support, n_sets: int) -> Optional[List[int]]:
        """
        Try to find last substep IDs using DPF's get_cumulative_index API.
        
        Returns:
            List of last substep IDs if successful, None if not enough coverage.
        """
        step_to_substeps = {}
        found_indices = set()
        
        for step in range(1, n_sets + 10):
            try:
                cumulative_idx = tf_support.get_cumulative_index(step=step, substep=1)
                
                if cumulative_idx is not None and 0 < cumulative_idx <= n_sets:
                    step_to_substeps[step] = []
                    substep = 1
                    
                    while True:
                        try:
                            idx = tf_support.get_cumulative_index(step=step, substep=substep)
                            if idx is not None and 0 < idx <= n_sets and idx not in found_indices:
                                found_indices.add(idx)
                                step_to_substeps[step].append((substep, idx))
                                substep += 1
                            else:
                                break
                        except Exception:
                            break
                        if substep > 1000:
                            break
            except Exception:
                continue
            
            if len(found_indices) >= n_sets:
                break
        
        # Need at least 80% coverage to use this method
        if len(found_indices) < n_sets * 0.8:
            return None
        
        if step_to_substeps:
            last_substep_ids = []
            for load_step in sorted(step_to_substeps.keys()):
                substeps = step_to_substeps[load_step]
                if substeps:
                    last_cumulative_idx = max(substeps, key=lambda x: x[0])[1]
                    last_substep_ids.append(last_cumulative_idx)
            
            if last_substep_ids:
                return sorted(last_substep_ids)
        
        return None
    
    def _get_last_substep_ids_via_time_analysis(self, n_sets: int) -> Optional[List[int]]:
        """
        Detect load step boundaries by analyzing time values.
        
        Looks for patterns like:
        - Time values that are integer seconds (likely end of load step)
        - Time resets (next time < current time)
        
        Returns:
            List of last substep IDs if pattern detected, None otherwise.
        """
        time_values = self.get_time_values()
        
        if not time_values or len(time_values) < 2:
            return None
        
        # Strategy 1: Find integer time values (1.0, 2.0, 3.0, etc.)
        # These typically indicate end of load step in ANSYS
        integer_time_indices = []
        non_integer_count = 0
        
        for i, t in enumerate(time_values):
            set_id = i + 1  # 1-based set ID
            
            # Check if time is an integer (or very close to integer)
            if abs(t - round(t)) < 1e-6 and t > 0:
                integer_time_indices.append(set_id)
            else:
                non_integer_count += 1
        
        # If ALL times are integers, there are no substeps to skip
        # (each set is its own load step)
        if non_integer_count == 0:
            return None  # No filtering needed
        
        # If we found integer time boundaries and they make sense, use them
        # Need at least some non-integer times (substeps) for this to make sense
        if len(integer_time_indices) >= 1 and non_integer_count > 0:
            # Verify the pattern makes sense
            # Also ensure the last set is included if it's an integer time
            last_set_id = n_sets
            if last_set_id not in integer_time_indices:
                last_time = time_values[-1]
                if abs(last_time - round(last_time)) < 1e-6:
                    integer_time_indices.append(last_set_id)
            
            # Sort and return if we have a reasonable number
            # (fewer than total sets, meaning we're actually filtering)
            integer_time_indices = sorted(set(integer_time_indices))
            if len(integer_time_indices) >= 1 and len(integer_time_indices) < n_sets:
                return integer_time_indices
        
        # Strategy 2: Detect time resets (next time <= current time)
        reset_indices = []
        for i, t in enumerate(time_values):
            set_id = i + 1
            
            if i == len(time_values) - 1:
                # Always include the last set
                reset_indices.append(set_id)
                continue
            
            next_t = time_values[i + 1]
            if next_t <= t:
                reset_indices.append(set_id)
        
        if len(reset_indices) >= 1 and len(reset_indices) < n_sets:
            return reset_indices
        
        return None
    
    def get_time_values(self, set_ids: Optional[List[int]] = None) -> List[float]:
        """
        Return list of time values (in seconds) for each set.
        
        For static/transient analyses, these are the time points at which
        results were stored.
        
        Args:
            set_ids: Optional list of specific set IDs to get time values for.
                    If None, returns time values for all sets.
        
        Returns:
            List of time values in seconds, one per set.
        """
        try:
            tf_support = self.time_freq_support
            all_time_values = None
            
            # Try different methods to get time/frequency values
            # (API varies slightly between DPF versions)
            try:
                # Primary method: time_frequencies field
                time_field = tf_support.time_frequencies
                if time_field is not None:
                    all_time_values = list(time_field.data)
            except AttributeError:
                pass
            
            if all_time_values is None:
                try:
                    # Alternative: frequencies field (used for both time and freq)
                    freq_field = tf_support.frequencies
                    if freq_field is not None:
                        all_time_values = list(freq_field.data)
                except AttributeError:
                    pass
            
            if all_time_values is None:
                # Fallback: return set indices as float
                n_sets = tf_support.n_sets
                all_time_values = [float(i) for i in range(1, n_sets + 1)]
            
            # Filter by specific set IDs if provided
            if set_ids is not None:
                # set_ids are 1-based, array indices are 0-based
                return [all_time_values[sid - 1] for sid in set_ids if 0 < sid <= len(all_time_values)]
            
            return all_time_values
            
        except Exception:
            # Ultimate fallback: return sequential integers
            n_sets = self.get_load_step_count()
            if set_ids is not None:
                return [float(sid) for sid in set_ids]
            return [float(i) for i in range(1, n_sets + 1)]
    
    def get_analysis_data(self, skip_substeps: bool = False) -> AnalysisData:
        """
        Create an AnalysisData object with metadata from this RST file.
        
        Args:
            skip_substeps: If True, only include the last substep of each load step.
                          This reduces the number of result sets when the RST file
                          contains many intermediate substeps.
        
        Returns:
            AnalysisData object containing file metadata including unit information.
        """
        # Check nodal forces availability once (result is cached)
        nodal_forces_available = self.check_nodal_forces_available()
        
        # Get appropriate set IDs based on skip_substeps option
        if skip_substeps:
            load_step_ids = self.get_last_substep_ids()
        else:
            load_step_ids = self.get_load_step_ids()
        
        # Get time values for the selected sets
        time_values = self.get_time_values(set_ids=load_step_ids)
        
        return AnalysisData(
            file_path=self.rst_path,
            num_load_steps=len(load_step_ids),
            load_step_ids=load_step_ids,
            time_values=time_values,
            named_selections=self.get_named_selections(),
            unit_system=self.unit_system,
            stress_unit=self.stress_unit,
            stress_conversion_factor=self.stress_conversion_factor,
            nodal_forces_available=nodal_forces_available,
            force_unit=self.get_force_unit() if nodal_forces_available else "N",
        )
    
    def get_nodal_scoping_from_named_selection(self, ns_name: str) -> 'dpf.Scoping':
        """
        Convert named selection to nodal scoping.
        
        Args:
            ns_name: Name of the named selection.
            
        Returns:
            DPF Scoping object with node IDs from the named selection.
            
        Raises:
            ValueError: If the named selection is not found.
        """
        try:
            # Get the named selection scoping
            ns_scoping = self.model.metadata.named_selection(ns_name)
            
            # If it's already a nodal scoping, return it
            if ns_scoping.location == dpf.locations.nodal:
                return ns_scoping
            
            # If it's elemental, we need to convert to nodal
            # Use the mesh to get nodes from elements
            mesh = self.mesh
            
            # Try using transpose operator (works for elemental -> nodal)
            try:
                transpose_op = dpf.operators.scoping.transpose()
                transpose_op.inputs.mesh_scoping.connect(ns_scoping)
                transpose_op.inputs.meshed_region.connect(mesh)
                transpose_op.inputs.inclusive.connect(1)  # Include all nodes of elements
                
                # Try different output attribute names for API compatibility
                if hasattr(transpose_op.outputs, 'mesh_scoping'):
                    return transpose_op.outputs.mesh_scoping()
                elif hasattr(transpose_op.outputs, 'mesh_scoping_as_scoping'):
                    return transpose_op.outputs.mesh_scoping_as_scoping()
                else:
                    # Evaluate and get first output
                    result = transpose_op.eval()
                    if isinstance(result, dpf.Scoping):
                        return result
                    raise AttributeError("Could not get scoping from transpose operator")
            except Exception:
                # Fallback: manually get nodes from elements
                if ns_scoping.location == dpf.locations.elemental:
                    element_ids = ns_scoping.ids
                    node_ids_set = set()
                    for elem_id in element_ids:
                        try:
                            elem_idx = mesh.elements.scoping.index(elem_id)
                            connectivity = mesh.elements.element_by_index(elem_idx).connectivity
                            node_ids_set.update(connectivity)
                        except Exception:
                            continue
                    
                    nodal_scoping = dpf.Scoping(location=dpf.locations.nodal)
                    nodal_scoping.ids = list(node_ids_set)
                    return nodal_scoping
                else:
                    raise
            
        except Exception as e:
            raise ValueError(f"Failed to get nodal scoping from named selection '{ns_name}': {e}")
    
    def get_all_nodes_scoping(self) -> 'dpf.Scoping':
        """
        Get a scoping containing all nodes in the mesh.
        
        Returns:
            DPF Scoping object with all node IDs.
        """
        node_ids = self.mesh.nodes.scoping.ids
        scoping = dpf.Scoping(location=dpf.locations.nodal)
        scoping.ids = node_ids
        return scoping
    
    def read_stress_tensor_for_loadstep(
        self, 
        load_step: int, 
        nodal_scoping: Optional['dpf.Scoping'] = None,
        convert_to_mpa: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Read 6-component stress tensor for a load step.
        
        Extracts nodal stress tensor components from the RST file for the
        specified load step. Optionally filters to specific nodes via scoping.
        
        Stress values are converted to MPa (N/mm²) by default for consistent
        reporting in MARS-SC millimeter-Newton standard units.
        
        Args:
            load_step: Load step/set ID (1-based).
            nodal_scoping: Optional DPF Scoping to filter nodes. If None, 
                          returns results for all nodes.
            convert_to_mpa: If True, convert stress values to MPa. Default True.
            
        Returns:
            Tuple of (node_ids, sx, sy, sz, sxy, syz, sxz) arrays.
            Each stress component array has shape (num_nodes,).
            Stress values are in MPa if convert_to_mpa=True.
            
        Note:
            Uses elemental nodal stress averaged to nodes.
        """
        # Create stress operator
        stress_op = self.model.results.stress()
        
        # Set time scoping for the specific load step
        time_scoping = dpf.Scoping()
        time_scoping.ids = [load_step]
        stress_op.inputs.time_scoping.connect(time_scoping)
        
        # Request nodal location (averages elemental nodal to nodes)
        stress_op.inputs.requested_location.connect(dpf.locations.nodal)
        
        # Apply mesh scoping if provided
        if nodal_scoping is not None:
            stress_op.inputs.mesh_scoping.connect(nodal_scoping)
        
        # Get the stress field (first field in container for single time step)
        fields_container = stress_op.outputs.fields_container()
        stress_field = fields_container[0]
        
        # Get node IDs from the field scoping
        node_ids = np.array(stress_field.scoping.ids)
        
        # Get stress data - DPF returns tensor components
        # Stress tensor has 6 components: SX, SY, SZ, SXY, SYZ, SXZ
        stress_data = stress_field.data
        
        # Extract individual components
        # DPF stress tensor component order: XX, YY, ZZ, XY, YZ, XZ
        sx = stress_data[:, 0].copy()
        sy = stress_data[:, 1].copy()
        sz = stress_data[:, 2].copy()
        sxy = stress_data[:, 3].copy()
        syz = stress_data[:, 4].copy()
        sxz = stress_data[:, 5].copy()
        
        # Convert to MPa if requested
        if convert_to_mpa:
            factor = self.stress_conversion_factor
            sx *= factor
            sy *= factor
            sz *= factor
            sxy *= factor
            syz *= factor
            sxz *= factor
        
        return (node_ids, sx, sy, sz, sxy, syz, sxz)
    
    def read_stress_field_for_loadstep(
        self, 
        load_step: int, 
        nodal_scoping: Optional['dpf.Scoping'] = None,
        convert_to_mpa: bool = True
    ) -> 'dpf.Field':
        """
        Read stress field for a load step (returns DPF Field object).
        
        This method returns the DPF Field, useful for direct DPF operations
        like scaling and adding fields.
        
        Stress values are converted to MPa (N/mm²) by default for consistent
        reporting in MARS-SC millimeter-Newton standard units.
        
        Args:
            load_step: Load step/set ID (1-based).
            nodal_scoping: Optional DPF Scoping to filter nodes.
            convert_to_mpa: If True, convert stress values to MPa. Default True.
            
        Returns:
            DPF Field object containing the stress tensor (in MPa if convert_to_mpa=True).
        """
        # Create stress operator
        stress_op = self.model.results.stress()
        
        # Set time scoping for the specific load step
        time_scoping = dpf.Scoping()
        time_scoping.ids = [load_step]
        stress_op.inputs.time_scoping.connect(time_scoping)
        
        # Request nodal location
        stress_op.inputs.requested_location.connect(dpf.locations.nodal)
        
        # Apply mesh scoping if provided
        if nodal_scoping is not None:
            stress_op.inputs.mesh_scoping.connect(nodal_scoping)
        
        # Get the stress field
        fields_container = stress_op.outputs.fields_container()
        stress_field = fields_container[0]
        
        # Convert to MPa if requested
        if convert_to_mpa:
            factor = self.stress_conversion_factor
            if factor != 1.0:
                # Use DPF scale operator to convert units
                stress_field = scale_stress_field(stress_field, factor)
        
        return stress_field
    
    def get_node_coordinates(self, nodal_scoping: Optional['dpf.Scoping'] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get XYZ coordinates for scoped nodes.
        
        Args:
            nodal_scoping: Optional DPF Scoping to filter nodes. If None,
                          returns coordinates for all nodes.
            
        Returns:
            Tuple of (node_ids, coordinates) where coordinates has shape (num_nodes, 3).
        """
        mesh = self.mesh
        
        if nodal_scoping is not None:
            node_ids = np.array(nodal_scoping.ids)
        else:
            node_ids = np.array(mesh.nodes.scoping.ids)
        
        # Get coordinates for each node
        coords = []
        for node_id in node_ids:
            try:
                # DPF node indexing is 0-based internally, but IDs are 1-based
                node_idx = mesh.nodes.scoping.index(node_id)
                coord = mesh.nodes.coordinates_field.data[node_idx]
                coords.append(coord)
            except Exception:
                # If node not found, use zeros
                coords.append([0.0, 0.0, 0.0])
        
        return (node_ids, np.array(coords))
    
    def get_model_info(self) -> Dict:
        """
        Get summary information about the model.
        
        Returns:
            Dictionary with model metadata including unit information.
        """
        info = {
            'file_path': self.rst_path,
            'num_nodes': self.mesh.nodes.n_nodes,
            'num_elements': self.mesh.elements.n_elements,
            'num_load_steps': self.get_load_step_count(),
            'named_selections': self.get_named_selections(),
        }
        
        # Try to get analysis type
        try:
            info['analysis_type'] = str(self.model.metadata.result_info.analysis_type)
        except Exception:
            info['analysis_type'] = 'Unknown'
        
        # Get unit information
        info['unit_system'] = self.unit_system
        info['stress_unit'] = self.stress_unit
        info['stress_conversion_to_mpa'] = self.stress_conversion_factor
        
        # Check nodal forces availability
        info['nodal_forces_available'] = self.check_nodal_forces_available()
        
        return info
    
    def check_nodal_forces_available(self) -> bool:
        """
        Check if nodal forces (element nodal forces) are available in the RST file.
        
        Returns:
            True if nodal forces are available, False otherwise.
        """
        if self._nodal_forces_available is not None:
            return self._nodal_forces_available
        
        try:
            # Try to access element nodal forces operator
            nforce_op = self.model.results.element_nodal_forces()
            
            # Set time scoping to first load step to test
            time_scoping = dpf.Scoping()
            time_scoping.ids = [1]
            nforce_op.inputs.time_scoping.connect(time_scoping)
            nforce_op.inputs.requested_location.connect(dpf.locations.nodal)
            
            # Try to get the fields container
            fields_container = nforce_op.outputs.fields_container()
            
            # Check if we got valid data
            if fields_container and len(fields_container) > 0:
                field = fields_container[0]
                if field.data is not None and len(field.data) > 0:
                    self._nodal_forces_available = True
                    return True
            self._nodal_forces_available = False
            return False
            
        except Exception:
            self._nodal_forces_available = False
            return False
    
    def get_force_unit(self) -> str:
        """
        Get the force unit from the RST file by reading a sample force field.
        
        Returns:
            String describing the force unit (e.g., "N").
        """
        if self._force_unit is not None:
            return self._force_unit
        
        try:
            nforce_op = self.model.results.element_nodal_forces()
            time_scoping = dpf.Scoping()
            time_scoping.ids = [1]
            nforce_op.inputs.time_scoping.connect(time_scoping)
            nforce_op.inputs.requested_location.connect(dpf.locations.nodal)
            
            fields_container = nforce_op.outputs.fields_container()
            if fields_container and len(fields_container) > 0:
                self._force_unit = fields_container[0].unit or "N"
                return self._force_unit
            self._force_unit = "N"
            return "N"
        except Exception:
            self._force_unit = "N"
            return "N"
    
    def read_nodal_forces_for_loadstep(
        self, 
        load_step: int, 
        nodal_scoping: Optional['dpf.Scoping'] = None,
        rotate_to_global: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Read nodal forces (element nodal forces summed at nodes) for a load step.
        
        Extracts nodal force components from the RST file for the specified load step.
        Optionally filters to specific nodes via scoping.
        
        Args:
            load_step: Load step/set ID (1-based).
            nodal_scoping: Optional DPF Scoping to filter nodes. If None, 
                          returns results for all nodes.
            rotate_to_global: If True (default), rotate forces to global coordinate
                             system. If False, keep forces in element (local) 
                             coordinate system. Note: Beam/pipe element forces are 
                             always in local coordinates regardless of this setting.
            
        Returns:
            Tuple of (node_ids, fx, fy, fz) arrays.
            Each force component array has shape (num_nodes,).
            
        Raises:
            NodalForcesNotAvailableError: If nodal forces are not available.
        """
        try:
            # Create element nodal forces operator
            nforce_op = self.model.results.element_nodal_forces()
            
            # Set time scoping for the specific load step
            time_scoping = dpf.Scoping()
            time_scoping.ids = [load_step]
            nforce_op.inputs.time_scoping.connect(time_scoping)
            
            # Set coordinate system rotation option
            nforce_op.inputs.bool_rotate_to_global.connect(rotate_to_global)
            
            # Request nodal location (sums element nodal forces at shared nodes)
            nforce_op.inputs.requested_location.connect(dpf.locations.nodal)
            
            # Apply mesh scoping if provided
            if nodal_scoping is not None:
                nforce_op.inputs.mesh_scoping.connect(nodal_scoping)
            
            # Get the force field
            fields_container = nforce_op.outputs.fields_container()
            
            if not fields_container or len(fields_container) == 0:
                raise NodalForcesNotAvailableError(
                    f"No nodal forces data available for load step {load_step}. "
                    "Ensure 'Write element nodal forces' is enabled in ANSYS Output Controls."
                )
            
            force_field = fields_container[0]
            
            # Get node IDs from the field scoping
            node_ids = np.array(force_field.scoping.ids)
            
            # Get force data - DPF returns 3-component vector (FX, FY, FZ)
            force_data = force_field.data
            
            # Extract individual components
            fx = force_data[:, 0].copy()
            fy = force_data[:, 1].copy()
            fz = force_data[:, 2].copy()
            
            return (node_ids, fx, fy, fz)
            
        except NodalForcesNotAvailableError:
            raise
        except Exception as e:
            raise NodalForcesNotAvailableError(
                f"Failed to read nodal forces for load step {load_step}: {e}"
            )
    
    def read_nodal_forces_field_for_loadstep(
        self, 
        load_step: int, 
        nodal_scoping: Optional['dpf.Scoping'] = None,
        rotate_to_global: bool = True
    ) -> 'dpf.Field':
        """
        Read nodal forces field for a load step (returns DPF Field object).
        
        This method returns the DPF Field, useful for direct DPF operations
        like scaling and adding fields.
        
        Args:
            load_step: Load step/set ID (1-based).
            nodal_scoping: Optional DPF Scoping to filter nodes.
            rotate_to_global: If True (default), rotate forces to global coordinate
                             system. If False, keep forces in element (local) 
                             coordinate system.
            
        Returns:
            DPF Field object containing the nodal forces vector.
            
        Raises:
            NodalForcesNotAvailableError: If nodal forces are not available.
        """
        try:
            # Create element nodal forces operator
            nforce_op = self.model.results.element_nodal_forces()
            
            # Set time scoping for the specific load step
            time_scoping = dpf.Scoping()
            time_scoping.ids = [load_step]
            nforce_op.inputs.time_scoping.connect(time_scoping)
            
            # Set coordinate system rotation option
            nforce_op.inputs.bool_rotate_to_global.connect(rotate_to_global)
            
            # Request nodal location
            nforce_op.inputs.requested_location.connect(dpf.locations.nodal)
            
            # Apply mesh scoping if provided
            if nodal_scoping is not None:
                nforce_op.inputs.mesh_scoping.connect(nodal_scoping)
            
            # Get the force field
            fields_container = nforce_op.outputs.fields_container()
            
            if not fields_container or len(fields_container) == 0:
                raise NodalForcesNotAvailableError(
                    f"No nodal forces data available for load step {load_step}."
                )
            
            return fields_container[0]
            
        except NodalForcesNotAvailableError:
            raise
        except Exception as e:
            raise NodalForcesNotAvailableError(
                f"Failed to read nodal forces field for load step {load_step}: {e}"
            )
    
    def create_sub_scoping(
        self, 
        full_scoping: 'dpf.Scoping', 
        start_idx: int, 
        end_idx: int
    ) -> 'dpf.Scoping':
        """
        Create a sub-scoping for a range of nodes from a full scoping.
        
        This method is used for chunked processing to create scopings for
        specific node ranges without loading the full dataset.
        
        Args:
            full_scoping: The full DPF Scoping containing all node IDs.
            start_idx: Start index (0-based, inclusive) in the scoping.
            end_idx: End index (0-based, exclusive) in the scoping.
            
        Returns:
            DPF Scoping containing only the nodes in the specified range.
        """
        all_ids = np.array(full_scoping.ids)
        sub_ids = all_ids[start_idx:end_idx]
        
        sub_scoping = dpf.Scoping(location=dpf.locations.nodal)
        sub_scoping.ids = sub_ids.tolist()
        
        return sub_scoping
    
    def create_single_node_scoping(
        self, 
        node_id: int, 
        full_scoping: Optional['dpf.Scoping'] = None
    ) -> 'dpf.Scoping':
        """
        Create a scoping containing a single node ID.
        
        This method is used for optimized single-node queries to avoid
        loading stress data for the entire model when only one node is needed.
        
        Args:
            node_id: The node ID to include in the scoping.
            full_scoping: Optional full scoping to validate node existence.
                          If provided and node_id is not in it, raises ValueError.
            
        Returns:
            DPF Scoping containing only the specified node.
            
        Raises:
            ValueError: If node_id is not found in full_scoping (when provided).
        """
        # Validate node exists in full scoping if provided
        if full_scoping is not None:
            scoping_ids = list(full_scoping.ids)
            if node_id not in scoping_ids:
                raise ValueError(
                    f"Node ID {node_id} not found in the provided scoping. "
                    f"Available nodes: {len(scoping_ids):,}"
                )
        
        # Create single-node scoping
        single_scoping = dpf.Scoping(location=dpf.locations.nodal)
        single_scoping.ids = [node_id]
        
        return single_scoping
    
    def get_scoping_node_count(self, scoping: 'dpf.Scoping') -> int:
        """
        Get the number of nodes in a scoping.
        
        Args:
            scoping: DPF Scoping object.
            
        Returns:
            Number of node IDs in the scoping.
        """
        return len(scoping.ids)


def scale_stress_field(field: 'dpf.Field', scale_factor: float) -> 'dpf.Field':
    """
    Scale a stress field by a coefficient.
    
    This is a utility function that wraps DPF's scale operator for use
    in linear combination calculations.
    
    Args:
        field: DPF Field containing stress tensor.
        scale_factor: Coefficient to multiply the field by.
        
    Returns:
        Scaled DPF Field.
        
    Reference:
        https://dpf.docs.pyansys.com/version/stable/examples/06-plotting/02-solution_combination.html
    """
    if not DPF_AVAILABLE:
        raise DPFNotAvailableError("ansys-dpf-core is not installed.")
    
    # Convert to Python native float to avoid numpy float64 type issues
    scale_op = dpf.operators.math.scale(field=field, ponderation=float(scale_factor))
    return scale_op.outputs.field()


def add_stress_fields(field_a: 'dpf.Field', field_b: 'dpf.Field') -> 'dpf.Field':
    """
    Add two stress fields together.
    
    This is a utility function that wraps DPF's add operator for use
    in linear combination calculations.
    
    Args:
        field_a: First DPF Field.
        field_b: Second DPF Field (must have compatible scoping).
        
    Returns:
        Sum of the two fields as a DPF Field.
        
    Reference:
        https://dpf.docs.pyansys.com/version/stable/examples/06-plotting/02-solution_combination.html
    """
    if not DPF_AVAILABLE:
        raise DPFNotAvailableError("ansys-dpf-core is not installed.")
    
    add_op = dpf.operators.math.add(fieldA=field_a, fieldB=field_b)
    return add_op.outputs.field()


def compute_principal_stresses(stress_field: 'dpf.Field') -> Tuple['dpf.Field', 'dpf.Field', 'dpf.Field']:
    """
    Compute principal stresses (S1, S2, S3) from a stress tensor field.
    
    Principal stresses are the eigenvalues of the stress tensor.
    S1 is maximum principal, S3 is minimum principal.
    
    Args:
        stress_field: DPF Field containing 6-component stress tensor.
        
    Returns:
        Tuple of (S1, S2, S3) DPF Fields (max, mid, min principal).
        
    Reference:
        https://dpf.docs.pyansys.com/version/stable/examples/06-plotting/02-solution_combination.html
    """
    if not DPF_AVAILABLE:
        raise DPFNotAvailableError("ansys-dpf-core is not installed.")
    
    # Use principal invariants operator
    p_inv = dpf.operators.invariant.principal_invariants()
    p_inv.inputs.field.connect(stress_field)
    
    # Get principal stresses
    s1 = p_inv.outputs.field_eig_1()  # Maximum principal
    s2 = p_inv.outputs.field_eig_2()  # Middle principal
    s3 = p_inv.outputs.field_eig_3()  # Minimum principal
    
    return (s1, s2, s3)


def compute_von_mises_from_field(stress_field: 'dpf.Field') -> 'dpf.Field':
    """
    Compute von Mises equivalent stress from a stress tensor field.
    
    Args:
        stress_field: DPF Field containing 6-component stress tensor.
        
    Returns:
        DPF Field containing von Mises stress values.
    """
    if not DPF_AVAILABLE:
        raise DPFNotAvailableError("ansys-dpf-core is not installed.")
    
    # Use equivalent stress operator (von Mises)
    eqv_op = dpf.operators.invariant.von_mises_eqv()
    eqv_op.inputs.field.connect(stress_field)
    
    return eqv_op.outputs.field()


def scale_force_field(field: 'dpf.Field', scale_factor: float) -> 'dpf.Field':
    """
    Scale a force field by a coefficient.
    
    This is a utility function that wraps DPF's scale operator for use
    in linear combination calculations of nodal forces.
    
    Args:
        field: DPF Field containing force vector.
        scale_factor: Coefficient to multiply the field by.
        
    Returns:
        Scaled DPF Field.
    """
    if not DPF_AVAILABLE:
        raise DPFNotAvailableError("ansys-dpf-core is not installed.")
    
    # Convert to Python native float to avoid numpy float64 type issues
    scale_op = dpf.operators.math.scale(field=field, ponderation=float(scale_factor))
    return scale_op.outputs.field()


def add_force_fields(field_a: 'dpf.Field', field_b: 'dpf.Field') -> 'dpf.Field':
    """
    Add two force fields together.
    
    This is a utility function that wraps DPF's add operator for use
    in linear combination calculations of nodal forces.
    
    Args:
        field_a: First DPF Field.
        field_b: Second DPF Field (must have compatible scoping).
        
    Returns:
        Sum of the two fields as a DPF Field.
    """
    if not DPF_AVAILABLE:
        raise DPFNotAvailableError("ansys-dpf-core is not installed.")
    
    add_op = dpf.operators.math.add(fieldA=field_a, fieldB=field_b)
    return add_op.outputs.field()


def compute_force_magnitude(force_field: 'dpf.Field') -> 'dpf.Field':
    """
    Compute magnitude of force vector field.
    
    Args:
        force_field: DPF Field containing 3-component force vector.
        
    Returns:
        DPF Field containing force magnitude values.
    """
    if not DPF_AVAILABLE:
        raise DPFNotAvailableError("ansys-dpf-core is not installed.")
    
    # Use norm operator to compute magnitude
    norm_op = dpf.operators.math.norm()
    norm_op.inputs.field.connect(force_field)
    
    return norm_op.outputs.field()
