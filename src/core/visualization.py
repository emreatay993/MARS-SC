"""
Visualization management classes for MARS-SC (Solution Combination).

Contains manager classes that handle complex visualisation tasks, separating
business logic from UI code.
"""

import numpy as np
import pandas as pd
import pyvista as pv
from typing import Optional, Tuple, Dict


class VisualizationManager:
    """PyVista mesh creation, scalar updates, and range/percentile helpers."""
    
    @staticmethod
    def create_mesh_from_coords(node_coords: np.ndarray, 
                               node_ids: Optional[np.ndarray] = None) -> pv.PolyData:
        """
        Create a PyVista mesh from node coordinates.
        
        Args:
            node_coords: Array of node coordinates (n_nodes, 3).
            node_ids: Optional array of node IDs.
        
        Returns:
            pv.PolyData: PyVista mesh object.
        """
        mesh = pv.PolyData(node_coords)
        
        if node_ids is not None:
            mesh["NodeID"] = node_ids.astype(int)
        
        mesh["Index"] = np.arange(mesh.n_points)
        
        return mesh
    
    @staticmethod
    def update_mesh_scalars(mesh: pv.PolyData, 
                           scalar_data: np.ndarray, 
                           scalar_name: str) -> pv.PolyData:
        """
        Update mesh with new scalar data.
        
        Args:
            mesh: PyVista mesh to update.
            scalar_data: Array of scalar values.
            scalar_name: Name for the scalar field.
        
        Returns:
            pv.PolyData: Updated mesh.
        """
        mesh[scalar_name] = scalar_data
        mesh.set_active_scalars(scalar_name)
        return mesh
    
    @staticmethod
    def compute_scalar_range(data: np.ndarray, 
                            percentile: float = 100.0) -> Tuple[float, float]:
        """
        Compute scalar range for visualization.
        
        Args:
            data: Array of scalar values.
            percentile: Percentile for range calculation (default: 100).
        
        Returns:
            Tuple of (min_value, max_value).
        """
        if percentile >= 100.0:
            return np.min(data), np.max(data)
        else:
            lower = (100.0 - percentile) / 2.0
            upper = 100.0 - lower
            return np.percentile(data, lower), np.percentile(data, upper)
    
    @staticmethod
    def apply_deformation(mesh: pv.PolyData, 
                         original_coords: np.ndarray,
                         deformation_vectors: np.ndarray,
                         scale_factor: float = 1.0) -> pv.PolyData:
        """
        Apply deformation to mesh coordinates.
        
        Args:
            mesh: PyVista mesh to deform.
            original_coords: Original node coordinates.
            deformation_vectors: Deformation vectors (n_nodes, 3).
            scale_factor: Scale factor for deformation.
        
        Returns:
            pv.PolyData: Mesh with updated coordinates.
        """
        new_coords = original_coords + scale_factor * deformation_vectors
        mesh.points = new_coords
        return mesh


class HotspotDetector:
    """Finds nodes with extreme scalar values and filters them (top N, max/min)."""
    
    @staticmethod
    def detect_hotspots(scalar_data: np.ndarray,
                       node_ids: np.ndarray,
                       node_coords: Optional[np.ndarray] = None,
                       top_n: int = 10,
                       mode: str = 'max') -> pd.DataFrame:
        """
        Detect hotspot nodes with extreme scalar values.
        
        Args:
            scalar_data: Array of scalar values at each node.
            node_ids: Array of node IDs.
            node_coords: Optional array of node coordinates.
            top_n: Number of top hotspots to return.
            mode: 'max' for maximum values, 'min' for minimum values, 'abs' for absolute.
        
        Returns:
            pd.DataFrame: DataFrame with hotspot information.
        """
        if mode == 'max':
            sorted_indices = np.argsort(scalar_data)[::-1][:top_n]
        elif mode == 'min':
            sorted_indices = np.argsort(scalar_data)[:top_n]
        elif mode == 'abs':
            sorted_indices = np.argsort(np.abs(scalar_data))[::-1][:top_n]
        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'max', 'min', or 'abs'.")
        
        hotspot_data = {
            'Rank': np.arange(1, len(sorted_indices) + 1),
            'NodeID': node_ids[sorted_indices],
            'Value': scalar_data[sorted_indices]
        }
        
        if node_coords is not None:
            hotspot_data['X'] = node_coords[sorted_indices, 0]
            hotspot_data['Y'] = node_coords[sorted_indices, 1]
            hotspot_data['Z'] = node_coords[sorted_indices, 2]
        
        return pd.DataFrame(hotspot_data)
    
    @staticmethod
    def filter_by_region(scalar_data: np.ndarray,
                        node_ids: np.ndarray,
                        node_coords: np.ndarray,
                        region_bounds: Dict[str, Tuple[float, float]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Filter nodes by spatial region.
        
        Args:
            scalar_data: Array of scalar values.
            node_ids: Array of node IDs.
            node_coords: Array of node coordinates (n_nodes, 3).
            region_bounds: Dict with 'x', 'y', 'z' keys containing (min, max) tuples.
        
        Returns:
            Tuple of (filtered_scalar_data, filtered_node_ids, filtered_coords).
        """
        mask = np.ones(len(node_ids), dtype=bool)
        
        if 'x' in region_bounds:
            x_min, x_max = region_bounds['x']
            mask &= (node_coords[:, 0] >= x_min) & (node_coords[:, 0] <= x_max)
        
        if 'y' in region_bounds:
            y_min, y_max = region_bounds['y']
            mask &= (node_coords[:, 1] >= y_min) & (node_coords[:, 1] <= y_max)
        
        if 'z' in region_bounds:
            z_min, z_max = region_bounds['z']
            mask &= (node_coords[:, 2] >= z_min) & (node_coords[:, 2] <= z_max)
        
        return scalar_data[mask], node_ids[mask], node_coords[mask]
    
    @staticmethod
    def filter_by_threshold(scalar_data: np.ndarray,
                           node_ids: np.ndarray,
                           threshold: float,
                           mode: str = 'above') -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter nodes by scalar threshold.
        
        Args:
            scalar_data: Array of scalar values.
            node_ids: Array of node IDs.
            threshold: Threshold value.
            mode: 'above' for values above threshold, 'below' for values below.
        
        Returns:
            Tuple of (filtered_scalar_data, filtered_node_ids).
        """
        if mode == 'above':
            mask = scalar_data >= threshold
        elif mode == 'below':
            mask = scalar_data <= threshold
        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'above' or 'below'.")
        
        return scalar_data[mask], node_ids[mask]
