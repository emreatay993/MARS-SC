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


def get_displacement_unit_conversion_factor(source_unit: str) -> Tuple[float, str]:
    """
    Get the conversion factor to convert from source unit to millimeters.
    
    MARS-SC reports displacements in millimeters (mm) for consistency with 
    the millimeter-Newton unit system used throughout.
    
    Args:
        source_unit: The source unit string from DPF (e.g., "m", "mm", "in", etc.)
        
    Returns:
        Tuple of (multiplication_factor, target_unit).
        - multiplication_factor: Factor to multiply values by to convert to mm.
        - target_unit: The target unit string ("mm").
    """
    # Common displacement unit conversion factors to mm
    # 1 m = 1000 mm
    conversion_to_mm = {
        'm': 1000.0,          # meters to mm
        'M': 1000.0,          # meters (uppercase)
        'meter': 1000.0,      # meters (spelled out)
        'meters': 1000.0,     # meters (plural)
        'mm': 1.0,            # Already mm
        'MM': 1.0,            # Already mm (uppercase)
        'millimeter': 1.0,    # Already mm (spelled out)
        'millimeters': 1.0,   # Already mm (plural)
        'cm': 10.0,           # centimeters to mm
        'in': 25.4,           # inches to mm
        'inch': 25.4,         # inches (spelled out)
        'inches': 25.4,       # inches (plural)
        'ft': 304.8,          # feet to mm
        'um': 0.001,          # micrometers to mm
        'µm': 0.001,          # micrometers (symbol)
    }
    
    factor = conversion_to_mm.get(source_unit, 1000.0)  # Default to m->mm if unknown
    return factor, "mm"


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
        
        # Check displacement availability
        displacement_available = self.check_displacement_available()
        
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
            displacement_available=displacement_available,
            displacement_unit=self.get_displacement_unit() if displacement_available else "mm",
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
    
    def check_named_selection_has_beam_elements(self, ns_name: str) -> bool:
        """
        Check if a named selection contains any beam elements.
        
        Beam elements do not support stress tensor output, so this check is used
        to prevent users from selecting stress outputs for beam-only selections.
        
        Args:
            ns_name: Name of the named selection.
            
        Returns:
            True if the named selection contains any beam elements, False otherwise.
        """
        try:
            # Get the named selection scoping
            ns_scoping = self.model.metadata.named_selection(ns_name)
            
            # If it's a nodal scoping, we need to find associated elements
            if ns_scoping.location == dpf.locations.nodal:
                # For nodal named selections, we need to find elements that use these nodes
                # This is more complex - try to get elemental scoping using transpose
                try:
                    transpose_op = dpf.operators.scoping.transpose()
                    transpose_op.inputs.mesh_scoping.connect(ns_scoping)
                    transpose_op.inputs.meshed_region.connect(self.mesh)
                    transpose_op.inputs.inclusive.connect(0)  # Elements fully contained
                    
                    if hasattr(transpose_op.outputs, 'mesh_scoping'):
                        elem_scoping = transpose_op.outputs.mesh_scoping()
                    elif hasattr(transpose_op.outputs, 'mesh_scoping_as_scoping'):
                        elem_scoping = transpose_op.outputs.mesh_scoping_as_scoping()
                    else:
                        result = transpose_op.eval()
                        if isinstance(result, dpf.Scoping):
                            elem_scoping = result
                        else:
                            # Cannot determine elements, assume no beams
                            return False
                    
                    element_ids = list(elem_scoping.ids)
                except Exception:
                    # If we can't get element scoping from nodal, assume no beams
                    return False
            else:
                # Elemental scoping - directly get element IDs
                element_ids = list(ns_scoping.ids)
            
            if not element_ids:
                return False
            
            # Check element types for beam elements
            mesh = self.mesh
            for elem_id in element_ids:
                try:
                    elem_idx = mesh.elements.scoping.index(elem_id)
                    element = mesh.elements.element_by_index(elem_idx)
                    
                    # Check if element is a beam using the shape property
                    if hasattr(element, 'shape'):
                        if element.shape == "beam":
                            return True
                    
                    # Alternative: check element type descriptor
                    if hasattr(element, 'type'):
                        elem_type = element.type
                        # Check if element type descriptor indicates beam
                        if hasattr(elem_type, 'shape') and elem_type.shape == "beam":
                            return True
                        # Some DPF versions use different attribute
                        if hasattr(elem_type, 'name'):
                            type_name = str(elem_type.name).lower()
                            if 'beam' in type_name or 'line' in type_name:
                                return True
                except Exception:
                    continue
            
            return False
            
        except Exception:
            # If we can't determine, assume no beams (fail open for usability)
            return False
    
    def get_node_element_types(
        self, 
        nodal_scoping: 'dpf.Scoping'
    ) -> Tuple[np.ndarray, bool]:
        """
        Determine element type for each node in the scoping.
        
        For each node, checks all attached elements and marks the node as 'beam'
        if ANY attached element is a beam type, otherwise marks as 'solid_shell'.
        
        Args:
            nodal_scoping: DPF Scoping containing node IDs to analyze.
            
        Returns:
            Tuple of (element_types_array, has_beam_nodes) where:
            - element_types_array: numpy array of strings ('beam' or 'solid_shell')
                                   aligned with nodal_scoping.ids order
            - has_beam_nodes: True if any node has beam elements attached
        """
        node_ids = np.array(nodal_scoping.ids)
        num_nodes = len(node_ids)
        
        # Default all nodes to 'solid_shell'
        element_types = np.array(['solid_shell'] * num_nodes, dtype=object)
        has_beam_nodes = False
        
        try:
            mesh = self.mesh
            
            # Build a mapping from node_id to index in our array
            node_id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}
            
            # Build reverse connectivity: node_id -> list of element_ids
            # This is more efficient than checking each node individually
            node_to_elements: Dict[int, List[int]] = {nid: [] for nid in node_ids}
            
            # Iterate through all elements and build connectivity
            for elem_idx in range(mesh.elements.n_elements):
                try:
                    element = mesh.elements.element_by_index(elem_idx)
                    elem_id = mesh.elements.scoping.ids[elem_idx]
                    connectivity = element.connectivity
                    
                    # Add this element to all its nodes that are in our scoping
                    for node_id in connectivity:
                        if node_id in node_to_elements:
                            node_to_elements[node_id].append((elem_id, element))
                except Exception:
                    continue
            
            # Now check each node's elements for beam types
            for node_id, elements in node_to_elements.items():
                if node_id not in node_id_to_idx:
                    continue
                    
                node_idx = node_id_to_idx[node_id]
                
                for elem_id, element in elements:
                    is_beam = False
                    
                    # Check if element is a beam using the shape property
                    if hasattr(element, 'shape'):
                        if element.shape == "beam":
                            is_beam = True
                    
                    # Alternative: check element type descriptor
                    if not is_beam and hasattr(element, 'type'):
                        elem_type = element.type
                        # Check if element type descriptor indicates beam
                        if hasattr(elem_type, 'shape') and elem_type.shape == "beam":
                            is_beam = True
                        # Some DPF versions use different attribute
                        elif hasattr(elem_type, 'name'):
                            type_name = str(elem_type.name).lower()
                            if 'beam' in type_name or 'line' in type_name or 'pipe' in type_name:
                                is_beam = True
                    
                    if is_beam:
                        element_types[node_idx] = 'beam'
                        has_beam_nodes = True
                        break  # No need to check more elements for this node
            
            return (element_types, has_beam_nodes)
            
        except Exception as e:
            # If we can't determine element types, return all as solid_shell
            # This is a safe fallback that won't break functionality
            return (element_types, False)
    
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
        
        IMPORTANT: When nodal_scoping is provided, the returned arrays are 
        guaranteed to be ordered according to nodal_scoping.ids, NOT the order
        that DPF returns them. This ensures consistent alignment with coordinates
        obtained via get_node_coordinates().
        
        Args:
            load_step: Load step/set ID (1-based).
            nodal_scoping: Optional DPF Scoping to filter nodes. If None, 
                          returns results for all nodes.
            convert_to_mpa: If True, convert stress values to MPa. Default True.
            
        Returns:
            Tuple of (node_ids, sx, sy, sz, sxy, syz, sxz) arrays.
            Each stress component array has shape (num_nodes,).
            Stress values are in MPa if convert_to_mpa=True.
            Node IDs and stress arrays are ordered to match input scoping.
            
        Note:
            Uses elemental nodal stress averaged to nodes.
        """
        # Store input scoping order for later reordering
        # DPF may return results in a different order than the input scoping
        input_node_ids_order = None
        if nodal_scoping is not None:
            input_node_ids_order = np.array(nodal_scoping.ids)
        
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
        
        # Get node IDs from the field scoping (DPF's returned order)
        dpf_node_ids = np.array(stress_field.scoping.ids)
        
        # Get stress data - DPF returns tensor components
        # Stress tensor has 6 components: SX, SY, SZ, SXY, SYZ, SXZ
        stress_data = stress_field.data
        
        # Handle case where stress data is 1D (e.g., beam elements)
        # Beam elements don't have a full 6-component stress tensor
        if stress_data.ndim == 1:
            # Single component data (beam axial stress or similar)
            # Cannot extract 6-component tensor - raise informative error
            raise BeamElementNotSupportedError(
                f"Stress data is 1-dimensional (shape: {stress_data.shape}). "
                f"This typically occurs when the scoping includes beam or line elements "
                f"which don't have a 6-component stress tensor. "
                f"Please ensure your named selection contains only solid/shell elements."
            )
        
        # Verify we have 6 components
        if stress_data.shape[1] < 6:
            raise ValueError(
                f"Expected 6 stress components but got {stress_data.shape[1]}. "
                f"Stress data shape: {stress_data.shape}. "
                f"This may occur with element types that don't support full stress tensors."
            )
        
        # Extract individual components (in DPF returned order)
        # DPF stress tensor component order: XX, YY, ZZ, XY, YZ, XZ
        sx = stress_data[:, 0].copy()
        sy = stress_data[:, 1].copy()
        sz = stress_data[:, 2].copy()
        sxy = stress_data[:, 3].copy()
        syz = stress_data[:, 4].copy()
        sxz = stress_data[:, 5].copy()
        
        # CRITICAL FIX: Reorder/align arrays to match input scoping order
        # DPF may return results in a different order than the input scoping,
        # and may also return fewer nodes (if some nodes have no stress data).
        # This ensures scalar values are correctly mapped to node positions.
        if input_node_ids_order is not None:
            # Build a lookup: node_id -> index in DPF results
            dpf_id_to_idx = {nid: idx for idx, nid in enumerate(dpf_node_ids)}
            
            # Check if DPF returned exactly the nodes we requested in the same order
            if len(input_node_ids_order) == len(dpf_node_ids) and np.array_equal(dpf_node_ids, input_node_ids_order):
                # Perfect match - no reordering needed
                pass
            elif len(input_node_ids_order) == len(dpf_node_ids):
                # Same count but different order - reorder to match input
                reorder_indices = np.array([dpf_id_to_idx[nid] for nid in input_node_ids_order])
                sx = sx[reorder_indices]
                sy = sy[reorder_indices]
                sz = sz[reorder_indices]
                sxy = sxy[reorder_indices]
                syz = syz[reorder_indices]
                sxz = sxz[reorder_indices]
                dpf_node_ids = input_node_ids_order
            else:
                # DPF returned different node count than requested (some nodes may lack data)
                # Create full-size arrays aligned to input scoping, with zeros for missing nodes
                # (zeros represent no stress contribution, which is safe for combination)
                num_requested = len(input_node_ids_order)
                sx_full = np.zeros(num_requested)
                sy_full = np.zeros(num_requested)
                sz_full = np.zeros(num_requested)
                sxy_full = np.zeros(num_requested)
                syz_full = np.zeros(num_requested)
                sxz_full = np.zeros(num_requested)
                
                # Fill in values for nodes that have data
                for req_idx, nid in enumerate(input_node_ids_order):
                    if nid in dpf_id_to_idx:
                        dpf_idx = dpf_id_to_idx[nid]
                        sx_full[req_idx] = sx[dpf_idx]
                        sy_full[req_idx] = sy[dpf_idx]
                        sz_full[req_idx] = sz[dpf_idx]
                        sxy_full[req_idx] = sxy[dpf_idx]
                        syz_full[req_idx] = syz[dpf_idx]
                        sxz_full[req_idx] = sxz[dpf_idx]
                
                sx, sy, sz = sx_full, sy_full, sz_full
                sxy, syz, sxz = sxy_full, syz_full, sxz_full
                dpf_node_ids = input_node_ids_order
        
        # Convert to MPa if requested
        if convert_to_mpa:
            factor = self.stress_conversion_factor
            sx *= factor
            sy *= factor
            sz *= factor
            sxy *= factor
            syz *= factor
            sxz *= factor
        
        return (dpf_node_ids, sx, sy, sz, sxy, syz, sxz)
    
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
        Get XYZ coordinates for scoped nodes, converted to millimeters.
        
        MARS-SC uses the mm-N-MPa unit system throughout. This method converts
        coordinates from DPF's native unit (typically meters) to millimeters.
        
        Args:
            nodal_scoping: Optional DPF Scoping to filter nodes. If None,
                          returns coordinates for all nodes.
            
        Returns:
            Tuple of (node_ids, coordinates) where coordinates has shape (num_nodes, 3).
            Coordinates are in millimeters.
        """
        mesh = self.mesh
        
        if nodal_scoping is not None:
            node_ids = np.array(nodal_scoping.ids)
        else:
            node_ids = np.array(mesh.nodes.scoping.ids)
        
        # Get the coordinate unit from the mesh and compute conversion factor to mm
        try:
            coord_unit = mesh.nodes.coordinates_field.unit or "m"
        except Exception:
            coord_unit = "m"  # Default assumption
        
        conversion_factor, _ = get_displacement_unit_conversion_factor(coord_unit)
        
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
        
        # Convert coordinates to mm
        coords_array = np.array(coords) * conversion_factor
        
        return (node_ids, coords_array)
    
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
        
        # Check displacement availability
        info['displacement_available'] = self.check_displacement_available()
        
        return info
    
    def check_nodal_forces_available(self) -> bool:
        """
        Check if nodal forces (element nodal forces) are available in the RST file.
        
        Uses a two-step approach:
        1. Check if element_nodal_forces is listed in available_results
        2. Try to actually read data to confirm availability (tries both with and 
           without global rotation as some element types fail with rotation)
        
        Returns:
            True if nodal forces are available, False otherwise.
        """
        if self._nodal_forces_available is not None:
            return self._nodal_forces_available
        
        try:
            # Step 1: Check available_results first (faster, more reliable)
            result_info = self.model.metadata.result_info
            available = result_info.available_results
            
            # Check if element_nodal_forces is in available results
            has_enf = any(
                "element_nodal_forces" in str(res).lower() or 
                "enf" in str(getattr(res, 'name', '')).lower()
                for res in available
            )
            
            if not has_enf:
                self._nodal_forces_available = False
                return False
            
            # Step 2: Try to actually read data to confirm
            # Use first available load step (don't hardcode to 1)
            load_step_ids = self.get_load_step_ids()
            if not load_step_ids:
                self._nodal_forces_available = False
                return False
            
            # Try with default rotation first, then without rotation
            # Some element types (e.g., elements with more integration points than nodes)
            # fail when rotating to global coordinate system
            for rotate_to_global in [True, False]:
                try:
                    nforce_op = self.model.results.element_nodal_forces()
                    
                    time_scoping = dpf.Scoping()
                    time_scoping.ids = [load_step_ids[0]]
                    nforce_op.inputs.time_scoping.connect(time_scoping)
                    nforce_op.inputs.bool_rotate_to_global.connect(rotate_to_global)
                    # NOTE: Do NOT connect requested_location - let DPF use defaults
                    
                    # Try to get the fields container
                    fields_container = nforce_op.outputs.fields_container()
                    
                    # Check if we got valid data
                    if fields_container and len(fields_container) > 0:
                        field = fields_container[0]
                        if field.data is not None and len(field.data) > 0:
                            self._nodal_forces_available = True
                            return True
                except Exception:
                    # Try the next rotation setting
                    continue
            
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
            
            # Use first available load step
            load_step_ids = self.get_load_step_ids()
            if not load_step_ids:
                self._force_unit = "N"
                return "N"
            
            time_scoping = dpf.Scoping()
            time_scoping.ids = [load_step_ids[0]]
            nforce_op.inputs.time_scoping.connect(time_scoping)
            # NOTE: Do NOT connect requested_location - let DPF use defaults
            
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
        
        IMPORTANT: When nodal_scoping is provided, the returned arrays are 
        guaranteed to be ordered according to nodal_scoping.ids, NOT the order
        that DPF returns them. This ensures consistent alignment with coordinates
        obtained via get_node_coordinates().
        
        Args:
            load_step: Load step/set ID (1-based).
            nodal_scoping: Optional DPF Scoping to filter nodes. If None, 
                          returns results for all nodes.
            rotate_to_global: If True (default), rotate forces to global coordinate
                             system. If False, keep forces in element (local) 
                             coordinate system. Note: Per DPF documentation, beam/pipe 
                             element forces CAN be rotated to global when this is True.
            
        Returns:
            Tuple of (node_ids, fx, fy, fz) arrays.
            Each force component array has shape (num_nodes,).
            Node IDs and force arrays are ordered to match input scoping.
            
        Raises:
            NodalForcesNotAvailableError: If nodal forces are not available.
        """
        try:
            # Store input scoping order for later reordering
            # DPF may return results in a different order than the input scoping
            input_node_ids_order = None
            if nodal_scoping is not None:
                input_node_ids_order = np.array(nodal_scoping.ids)
            
            # Create element nodal forces operator
            nforce_op = self.model.results.element_nodal_forces()
            
            # Set time scoping for the specific load step
            time_scoping = dpf.Scoping()
            time_scoping.ids = [load_step]
            nforce_op.inputs.time_scoping.connect(time_scoping)
            
            # Set coordinate system rotation option
            nforce_op.inputs.bool_rotate_to_global.connect(rotate_to_global)
            
            # NOTE: Do NOT connect requested_location - let DPF use defaults
            # The previous code used dpf.locations.nodal which caused issues with
            # the element_nodal_forces operator
            
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
            
            # Get node IDs from the field scoping (DPF's returned order)
            dpf_node_ids = np.array(force_field.scoping.ids)
            
            # Get force data - DPF returns 3-component vector (FX, FY, FZ)
            force_data = force_field.data
            
            # Handle case where force data is 1D (unexpected format)
            if force_data.ndim == 1:
                raise ValueError(
                    f"Force data is 1-dimensional (shape: {force_data.shape}). "
                    f"Expected 2D array with 3 components (FX, FY, FZ). "
                    f"This may indicate incompatible element types in the scoping."
                )
            
            # Verify we have at least 3 components
            if force_data.shape[1] < 3:
                raise ValueError(
                    f"Expected 3 force components but got {force_data.shape[1]}. "
                    f"Force data shape: {force_data.shape}."
                )
            
            # Extract individual components (in DPF returned order)
            fx = force_data[:, 0].copy()
            fy = force_data[:, 1].copy()
            fz = force_data[:, 2].copy()
            
            # CRITICAL FIX: Reorder/align arrays to match input scoping order
            # DPF may return results in a different order than the input scoping,
            # and may also return fewer nodes (if some nodes have no force data).
            # This ensures scalar values are correctly mapped to node positions.
            if input_node_ids_order is not None:
                # Build a lookup: node_id -> index in DPF results
                dpf_id_to_idx = {nid: idx for idx, nid in enumerate(dpf_node_ids)}
                
                # Check if DPF returned exactly the nodes we requested in the same order
                if len(input_node_ids_order) == len(dpf_node_ids) and np.array_equal(dpf_node_ids, input_node_ids_order):
                    # Perfect match - no reordering needed
                    pass
                elif len(input_node_ids_order) == len(dpf_node_ids):
                    # Same count but different order - reorder to match input
                    reorder_indices = np.array([dpf_id_to_idx[nid] for nid in input_node_ids_order])
                    fx = fx[reorder_indices]
                    fy = fy[reorder_indices]
                    fz = fz[reorder_indices]
                    dpf_node_ids = input_node_ids_order
                else:
                    # DPF returned different node count than requested (some nodes may lack data)
                    # Create full-size arrays aligned to input scoping, with zeros for missing nodes
                    # (zeros represent no force contribution, which is safe for combination)
                    num_requested = len(input_node_ids_order)
                    fx_full = np.zeros(num_requested)
                    fy_full = np.zeros(num_requested)
                    fz_full = np.zeros(num_requested)
                    
                    # Fill in values for nodes that have data
                    for req_idx, nid in enumerate(input_node_ids_order):
                        if nid in dpf_id_to_idx:
                            dpf_idx = dpf_id_to_idx[nid]
                            fx_full[req_idx] = fx[dpf_idx]
                            fy_full[req_idx] = fy[dpf_idx]
                            fz_full[req_idx] = fz[dpf_idx]
                    
                    fx, fy, fz = fx_full, fy_full, fz_full
                    dpf_node_ids = input_node_ids_order
            
            return (dpf_node_ids, fx, fy, fz)
            
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
            
            # NOTE: Do NOT connect requested_location - let DPF use defaults
            # The previous code used dpf.locations.nodal which caused issues
            
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
    
    # =========================================================================
    # Displacement (Deformation) Methods
    # =========================================================================
    
    def check_displacement_available(self) -> bool:
        """
        Check if displacement results are available in the RST file.
        
        Returns:
            True if displacement results are available, False otherwise.
        """
        try:
            # Check if displacement is in available_results
            result_info = self.model.metadata.result_info
            available = result_info.available_results
            
            # Check for displacement result
            has_displacement = any(
                "displacement" in str(res).lower() or
                "u" == str(getattr(res, 'name', '')).lower()
                for res in available
            )
            
            if not has_displacement:
                return False
            
            # Try to actually read data to confirm
            load_step_ids = self.get_load_step_ids()
            if not load_step_ids:
                return False
            
            disp_op = self.model.results.displacement()
            time_scoping = dpf.Scoping()
            time_scoping.ids = [load_step_ids[0]]
            disp_op.inputs.time_scoping.connect(time_scoping)
            
            fields_container = disp_op.outputs.fields_container()
            
            if fields_container and len(fields_container) > 0:
                field = fields_container[0]
                if field.data is not None and len(field.data) > 0:
                    return True
            
            return False
            
        except Exception:
            return False
    
    def get_displacement_unit(self) -> str:
        """
        Get the displacement unit for output display.
        
        MARS-SC always converts displacement values to millimeters internally,
        so this method always returns "mm" regardless of the source file's unit.
        
        Returns:
            "mm" - displacements are always reported in millimeters.
        """
        # MARS-SC converts all displacement values to mm internally
        # (similar to how stresses are converted to MPa)
        return "mm"
    
    def read_displacement_for_loadstep(
        self, 
        load_step: int, 
        nodal_scoping: Optional['dpf.Scoping'] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Read displacement (UX, UY, UZ) for a load step.
        
        Extracts nodal displacement components from the RST file for the
        specified load step. Optionally filters to specific nodes via scoping.
        
        IMPORTANT: When nodal_scoping is provided, the returned arrays are 
        guaranteed to be ordered according to nodal_scoping.ids, NOT the order
        that DPF returns them. This ensures consistent alignment with coordinates
        obtained via get_node_coordinates().
        
        Args:
            load_step: Load step/set ID (1-based).
            nodal_scoping: Optional DPF Scoping to filter nodes. If None, 
                          returns results for all nodes.
            
        Returns:
            Tuple of (node_ids, ux, uy, uz) arrays.
            Each displacement component array has shape (num_nodes,).
            Node IDs and displacement arrays are ordered to match input scoping.
            
        Raises:
            DisplacementNotAvailableError: If displacement results are not available.
        """
        try:
            # Store input scoping order for later reordering
            input_node_ids_order = None
            if nodal_scoping is not None:
                input_node_ids_order = np.array(nodal_scoping.ids)
            
            # Create displacement operator
            disp_op = self.model.results.displacement()
            
            # Set time scoping for the specific load step
            time_scoping = dpf.Scoping()
            time_scoping.ids = [load_step]
            disp_op.inputs.time_scoping.connect(time_scoping)
            
            # Apply mesh scoping if provided
            if nodal_scoping is not None:
                disp_op.inputs.mesh_scoping.connect(nodal_scoping)
            
            # Get the displacement field
            fields_container = disp_op.outputs.fields_container()
            
            if not fields_container or len(fields_container) == 0:
                raise DisplacementNotAvailableError(
                    f"No displacement data available for load step {load_step}."
                )
            
            disp_field = fields_container[0]
            
            # Get the displacement unit and compute conversion factor to mm
            source_unit = disp_field.unit or "m"
            conversion_factor, _ = get_displacement_unit_conversion_factor(source_unit)
            
            # Get node IDs from the field scoping (DPF's returned order)
            dpf_node_ids = np.array(disp_field.scoping.ids)
            
            # Get displacement data - DPF returns 3-component vector (UX, UY, UZ)
            disp_data = disp_field.data
            
            # Handle case where displacement data is 1D (unexpected format)
            if disp_data.ndim == 1:
                raise ValueError(
                    f"Displacement data is 1-dimensional (shape: {disp_data.shape}). "
                    f"Expected 2D array with 3 components (UX, UY, UZ)."
                )
            
            # Verify we have at least 3 components
            if disp_data.shape[1] < 3:
                raise ValueError(
                    f"Expected 3 displacement components but got {disp_data.shape[1]}. "
                    f"Displacement data shape: {disp_data.shape}."
                )
            
            # Extract individual components (in DPF returned order) and convert to mm
            ux = disp_data[:, 0].copy() * conversion_factor
            uy = disp_data[:, 1].copy() * conversion_factor
            uz = disp_data[:, 2].copy() * conversion_factor
            
            # CRITICAL FIX: Reorder/align arrays to match input scoping order
            if input_node_ids_order is not None:
                # Build a lookup: node_id -> index in DPF results
                dpf_id_to_idx = {nid: idx for idx, nid in enumerate(dpf_node_ids)}
                
                # Check if DPF returned exactly the nodes we requested in the same order
                if len(input_node_ids_order) == len(dpf_node_ids) and np.array_equal(dpf_node_ids, input_node_ids_order):
                    # Perfect match - no reordering needed
                    pass
                elif len(input_node_ids_order) == len(dpf_node_ids):
                    # Same count but different order - reorder to match input
                    reorder_indices = np.array([dpf_id_to_idx[nid] for nid in input_node_ids_order])
                    ux = ux[reorder_indices]
                    uy = uy[reorder_indices]
                    uz = uz[reorder_indices]
                    dpf_node_ids = input_node_ids_order
                else:
                    # DPF returned different node count than requested
                    # Create full-size arrays aligned to input scoping, with zeros for missing nodes
                    num_requested = len(input_node_ids_order)
                    ux_full = np.zeros(num_requested)
                    uy_full = np.zeros(num_requested)
                    uz_full = np.zeros(num_requested)
                    
                    # Fill in values for nodes that have data
                    for req_idx, nid in enumerate(input_node_ids_order):
                        if nid in dpf_id_to_idx:
                            dpf_idx = dpf_id_to_idx[nid]
                            ux_full[req_idx] = ux[dpf_idx]
                            uy_full[req_idx] = uy[dpf_idx]
                            uz_full[req_idx] = uz[dpf_idx]
                    
                    ux, uy, uz = ux_full, uy_full, uz_full
                    dpf_node_ids = input_node_ids_order
            
            return (dpf_node_ids, ux, uy, uz)
            
        except DisplacementNotAvailableError:
            raise
        except Exception as e:
            raise DisplacementNotAvailableError(
                f"Failed to read displacement for load step {load_step}: {e}"
            )


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


class DisplacementNotAvailableError(Exception):
    """Raised when displacement results are not available in the RST file."""
    pass
