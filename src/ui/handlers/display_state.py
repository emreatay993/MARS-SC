"""
Shared state container for the Display tab.

This dataclass holds mutable state that needs to be shared between the
DisplayTab widget and its supporting handler classes.
"""

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pyvista as pv


@dataclass
class DisplayState:
    """Aggregates runtime state for the Display tab."""

    current_mesh: Optional[pv.PolyData] = None
    current_actor: Optional[Any] = None
    camera_state: Optional[dict] = None
    camera_widget: Optional[Any] = None
    hover_annotation: Optional[Any] = None
    hover_observer: Optional[int] = None
    last_hover_time: float = 0.0
    data_column: str = "Result"
    temp_solver: Optional[Any] = None
    time_values: Optional[np.ndarray] = None
    original_node_coords: Optional[np.ndarray] = None
    last_valid_deformation_scale: float = 1.0
    highlight_actor: Optional[Any] = None
    box_widget: Optional[Any] = None
    hotspot_dialog: Optional[Any] = None
    is_point_picking_active: bool = False
    target_node_index: Optional[int] = None
    target_node_id: Optional[int] = None
    target_node_label_actor: Optional[Any] = None
    label_point_data: Optional[Any] = None
    marker_poly: Optional[Any] = None
    target_node_marker_actor: Optional[Any] = None
    last_goto_node_id: Optional[int] = None
    pick_indicator_actor: Optional[Any] = None
    current_contour_type: Optional[str] = None
    scalar_bar_digits: int = 4  # Number of decimal places for legend labels
