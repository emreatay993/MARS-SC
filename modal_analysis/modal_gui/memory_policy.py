"""Memory policy helpers for modal extraction."""

from __future__ import annotations

from dataclasses import dataclass

import psutil


@dataclass(frozen=True)
class ChunkPolicy:
    min_nodes: int = 200
    max_nodes: int = 50_000
    max_ram_fraction: float = 0.15
    overhead_multiplier: float = 3.0
    # Maximum modes to process before restarting DPF server to clear internal cache
    # This prevents DPF server-side memory accumulation that causes access violations
    max_modes_per_server_session: int = 50


def estimate_bytes_per_node(n_modes: int, n_components: int, include_coords: bool = True) -> int:
    """Estimate bytes per node for a chunk.

    Uses float64 sizing and a conservative overhead multiplier.
    """
    base = 0
    if include_coords:
        base += 3 * 8
    base += n_modes * n_components * 8
    return base


def compute_chunk_size(
    n_nodes: int,
    n_modes: int,
    n_components: int,
    policy: ChunkPolicy | None = None,
    available_bytes: int | None = None,
) -> int:
    """Compute a chunk size based on available RAM and result shape."""
    if policy is None:
        policy = ChunkPolicy()
    if available_bytes is None:
        available_bytes = psutil.virtual_memory().available

    bytes_per_node = estimate_bytes_per_node(n_modes, n_components)
    safe_bytes = int(available_bytes * policy.max_ram_fraction)

    if bytes_per_node <= 0:
        chunk = policy.min_nodes
    else:
        chunk = max(policy.min_nodes, int(safe_bytes / (bytes_per_node * policy.overhead_multiplier)))

    chunk = min(chunk, policy.max_nodes)
    if n_nodes > 0:
        chunk = min(chunk, n_nodes)
    return max(chunk, policy.min_nodes)


def compute_chunk_count(n_nodes: int, chunk_size: int) -> int:
    if n_nodes <= 0:
        return 0
    return (n_nodes + chunk_size - 1) // chunk_size


def should_restart_server(mode_index: int, policy: ChunkPolicy | None = None) -> bool:
    """Check if DPF server should be restarted to clear internal cache.
    
    DPF servers accumulate internal state/cache that can cause access violations
    after processing many modes. This function determines when to restart.
    """
    if policy is None:
        policy = ChunkPolicy()
    max_modes = policy.max_modes_per_server_session
    # Restart every max_modes_per_server_session modes (but not at mode 0)
    return mode_index > 0 and mode_index % max_modes == 0
