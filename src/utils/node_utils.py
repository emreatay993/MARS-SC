"""
Node-related utility helpers for MARS-SC (Solution Combination).

Provides helper functions for node ID mapping and manipulation.
"""

import numpy as np


def get_node_index_from_id(node_id, node_ids):
    """
    Map the given node_id to its corresponding index in the node array.
    
    Args:
        node_id: The node ID to map.
        node_ids: Array of node IDs (numpy array or similar).
    
    Returns:
        int: The index of the node ID in the array, or None if not found.
    """
    try:
        # Find the index of the node ID in the list of node IDs
        return np.where(node_ids == node_id)[0][0]
    except IndexError:
        print(f"Node ID {node_id} not found in the list of nodes.")
        return None
