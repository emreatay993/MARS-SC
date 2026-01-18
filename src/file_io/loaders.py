"""
File loading helpers for MARS-SC (Solution Combination).

Provides functions for loading input files and converting them into structured
data models.
"""

import json
import os
import sys
import time
import pandas as pd
import numpy as np
from typing import Optional
from pathlib import Path

from core.data_models import (
    ModalData,
    ModalStressData,
    DeformationData,
    SteadyStateData,
    TemperatureFieldData,
    MaterialProfileData,
)
from file_io.validators import (
    validate_mcf_file,
    validate_pch_file,
    validate_modal_stress_file,
    validate_deformation_file,
    validate_steady_state_file,
    validate_material_profile_payload,
)
from utils.file_utils import unwrap_mcf_file, parse_nastran_pch_modal_coordinates
from utils.constants import NP_DTYPE

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


# File size threshold for showing progress (in MB)
PROGRESS_THRESHOLD_MB = 100

# Performance tracking for adaptive ETA estimation
_MAX_HISTORY_SIZE = 5  # Keep last 5 measurements for rolling average
_PERFORMANCE_CACHE_FILE = Path.home() / '.mars_loader_performance.json'

# Initialize performance history (will be loaded from cache)
_performance_history = {
    'stress': [],  # List of (file_size_mb, throughput_mbps) tuples
    'deformation': []
}
_history_loaded = False  # Flag to ensure we only load once


def _load_performance_history():
    """Load performance history from cache file."""
    global _performance_history, _history_loaded
    
    if _history_loaded:
        return  # Already loaded
    
    try:
        if _PERFORMANCE_CACHE_FILE.exists():
            with open(_PERFORMANCE_CACHE_FILE, 'r') as f:
                cached_data = json.load(f)
                
            # Convert lists back to tuples and validate
            for file_type in ['stress', 'deformation']:
                if file_type in cached_data:
                    history = cached_data[file_type]
                    # Keep only last _MAX_HISTORY_SIZE entries
                    _performance_history[file_type] = [
                        tuple(entry) for entry in history[-_MAX_HISTORY_SIZE:]
                    ]
    except Exception:
        # If loading fails, just start fresh (no big deal)
        pass
    
    _history_loaded = True


def _save_performance_history():
    """Save performance history to cache file."""
    try:
        # Convert to JSON-serializable format
        cache_data = {
            file_type: list(history[-_MAX_HISTORY_SIZE:])
            for file_type, history in _performance_history.items()
        }
        
        # Write atomically (write to temp file, then rename)
        temp_file = _PERFORMANCE_CACHE_FILE.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
        
        # Atomic rename
        temp_file.replace(_PERFORMANCE_CACHE_FILE)
        
    except Exception:
        # If saving fails, no big deal - just won't persist
        pass


def _get_estimated_throughput(file_type: str) -> float:
    """
    Get estimated throughput based on historical performance.
    Loads cached history on first call.
    
    Args:
        file_type: 'stress' or 'deformation'
    
    Returns:
        Estimated throughput in MB/s. Falls back to conservative estimate if no history.
    """
    # Load cached history if not already loaded
    _load_performance_history()
    
    history = _performance_history.get(file_type, [])
    
    if not history:
        # Conservative fallback estimate for first load
        return 100.0  # MB/s
    
    # Calculate weighted average (recent measurements weighted more)
    total_weight = 0
    weighted_sum = 0
    
    for i, (size_mb, throughput) in enumerate(history):
        # More recent = higher weight
        weight = i + 1
        weighted_sum += throughput * weight
        total_weight += weight
    
    return weighted_sum / total_weight if total_weight > 0 else 100.0


def _record_performance(file_type: str, file_size_mb: float, throughput_mbps: float):
    """
    Record loading performance for future ETA estimation.
    Saves to persistent cache file.
    
    Args:
        file_type: 'stress' or 'deformation'
        file_size_mb: File size in MB
        throughput_mbps: Achieved throughput in MB/s
    """
    # Ensure history is loaded
    _load_performance_history()
    
    if file_type not in _performance_history:
        _performance_history[file_type] = []
    
    history = _performance_history[file_type]
    history.append((file_size_mb, throughput_mbps))
    
    # Keep only recent measurements
    if len(history) > _MAX_HISTORY_SIZE:
        history.pop(0)
    
    # Save to cache file for persistence across sessions
    _save_performance_history()


def _should_show_progress(filename: str) -> tuple[bool, float]:
    """
    Determine if progress indicator should be shown based on file size.
    
    Args:
        filename: Path to the file.
    
    Returns:
        Tuple of (should_show, file_size_mb).
    """
    try:
        file_size_bytes = os.path.getsize(filename)
        file_size_mb = file_size_bytes / (1024 * 1024)
        return file_size_mb >= PROGRESS_THRESHOLD_MB, file_size_mb
    except Exception:
        return False, 0.0


def _log_loading_start(filename: str, file_type: str, file_size_mb: float):
    """Log the start of file loading with size information."""
    print(f"\n{'='*70}")
    print(f"📂 Loading {file_type} file...")
    print(f"   File: {os.path.basename(filename)}")
    print(f"   Size: {file_size_mb:.2f} MB")
    if file_size_mb >= PROGRESS_THRESHOLD_MB:
        print(f"   ⏱️  Large file detected - this may take a moment...")
    print(f"{'='*70}")
    sys.stdout.flush()


def _log_loading_complete(file_type: str, elapsed_time: float, num_nodes: int = None, num_modes: int = None):
    """Log the completion of file loading with timing information."""
    print(f"\n✅ {file_type} file loaded successfully!")
    print(f"   Time: {elapsed_time:.2f}s")
    if num_nodes is not None:
        print(f"   Nodes: {num_nodes:,}")
    if num_modes is not None:
        print(f"   Modes: {num_modes}")
    print(f"{'='*70}\n")
    sys.stdout.flush()


def _read_csv_with_progress(filename: str, file_size_mb: float, file_type: str) -> pd.DataFrame:
    """
    Read CSV file with progress indication for large files.
    Uses adaptive ETA based on historical performance.
    Shows live progress updates during read.
    
    Args:
        filename: Path to CSV file.
        file_size_mb: File size in megabytes.
        file_type: Type of file being loaded ('stress' or 'deformation').
    
    Returns:
        Loaded DataFrame.
    """
    show_progress = file_size_mb >= PROGRESS_THRESHOLD_MB
    
    if show_progress:
        # Get estimated throughput based on historical performance
        estimated_throughput = _get_estimated_throughput(file_type)
        estimated_seconds = file_size_mb / estimated_throughput
        
        print(f"📊 Reading CSV data... (estimated time: ~{estimated_seconds:.1f}s)")
        sys.stdout.flush()
        
        if TQDM_AVAILABLE:
            # Show progress bar with ETA
            with tqdm(total=100, desc="Progress", unit="%", 
                     bar_format='{desc}: {percentage:3.0f}%|{bar:40}| [{elapsed}<{remaining}]',
                     file=sys.stdout) as pbar:
                
                start_read = time.time()
                
                # Start a background thread to update progress based on time
                import threading
                stop_progress = threading.Event()
                
                def update_progress():
                    """Update progress bar based on elapsed time vs estimated time."""
                    while not stop_progress.is_set():
                        elapsed = time.time() - start_read
                        progress_pct = min(95, (elapsed / estimated_seconds) * 100)
                        pbar.n = progress_pct
                        pbar.refresh()
                        time.sleep(0.5)  # Update every 0.5 seconds
                
                progress_thread = threading.Thread(target=update_progress, daemon=True)
                progress_thread.start()
                
                try:
                    df = pd.read_csv(filename, engine='pyarrow')
                except Exception:
                    df = pd.read_csv(filename)
                finally:
                    # Stop progress updates
                    stop_progress.set()
                    progress_thread.join(timeout=1.0)
                    
                    # Set to 100%
                    pbar.n = 100
                    pbar.refresh()
                
                read_time = time.time() - start_read
        else:
            # No tqdm available, show periodic progress updates
            sys.stdout.flush()
            start_read = time.time()
            
            # Show progress updates while reading
            import threading
            stop_progress = threading.Event()
            
            def show_progress():
                """Show periodic progress updates."""
                while not stop_progress.is_set():
                    elapsed = time.time() - start_read
                    print(f"   ⏱️  Reading... {elapsed:.1f}s elapsed")
                    sys.stdout.flush()
                    time.sleep(5.0)  # Update every 5 seconds
            
            progress_thread = threading.Thread(target=show_progress, daemon=True)
            progress_thread.start()
            
            try:
                df = pd.read_csv(filename, engine='pyarrow')
            except Exception:
                df = pd.read_csv(filename)
            finally:
                stop_progress.set()
                progress_thread.join(timeout=1.0)
            
            read_time = time.time() - start_read
        
        actual_throughput = file_size_mb / read_time if read_time > 0 else 0
        
        # Record performance for future estimates
        _record_performance(file_type, file_size_mb, actual_throughput)
        
        print(f"✓ CSV read complete ({read_time:.2f}s, {actual_throughput:.1f} MB/s)")
        sys.stdout.flush()
        
        return df
    else:
        # No progress indication for small files
        try:
            df = pd.read_csv(filename, engine='pyarrow')
        except Exception:
            df = pd.read_csv(filename)
        return df


def load_modal_coordinates(filename: str) -> ModalData:
    """
    Load modal coordinate data from an MCF file.
    
    Shows progress indicator for large files (>100 MB).
    
    Args:
        filename: Path to the MCF file.
    
    Returns:
        ModalData object containing modal coordinates and time values.
    
    Raises:
        ValueError: If the file is invalid or cannot be loaded.
    """
    # Check file size for progress indication
    show_progress, file_size_mb = _should_show_progress(filename)
    
    # Start timing
    start_time = time.time()
    
    # Log start for large files
    if show_progress:
        _log_loading_start(filename, "Modal Coordinates", file_size_mb)
    
    # Validate first
    if show_progress:
        print("🔍 Validating file structure...")
        sys.stdout.flush()
    
    is_valid, error_msg = validate_mcf_file(filename)
    if not is_valid:
        raise ValueError(f"Invalid MCF file: {error_msg}")
    
    if show_progress:
        print("✓ Validation passed")
        sys.stdout.flush()
    
    # Unwrap the file
    if show_progress:
        print("📝 Unwrapping MCF file...")
        sys.stdout.flush()
    
    base, ext = os.path.splitext(filename)
    unwrapped_filename = base + "_unwrapped" + ext
    unwrap_mcf_file(filename, unwrapped_filename)
    
    try:
        # Process data
        if show_progress:
            print("⚙️  Processing data...")
            sys.stdout.flush()
        
        # Find start of data
        with open(unwrapped_filename, 'r') as file:
            start_index = next(i for i, line in enumerate(file) if 'Time' in line)
        
        # Load data
        df_val = pd.read_csv(unwrapped_filename, sep='\\s+', 
                             skiprows=start_index + 1, header=None)
        
        # Extract time values and modal coordinates
        time_values = df_val.iloc[:, 0].to_numpy()
        modal_coord = df_val.drop(columns=df_val.columns[0]).transpose().to_numpy()
        
        # Create result
        result = ModalData(modal_coord=modal_coord, time_values=time_values)
        
        # Log completion for large files
        if show_progress:
            elapsed = time.time() - start_time
            print(f"\n✅ Modal Coordinates file loaded successfully!")
            print(f"   Time: {elapsed:.2f}s")
            print(f"   Modes: {result.num_modes}")
            print(f"   Time Points: {result.num_time_points:,}")
            print(f"{'='*70}\n")
            sys.stdout.flush()
        
        return result
        
    finally:
        # Clean up temporary file
        if os.path.exists(unwrapped_filename):
            os.remove(unwrapped_filename)


def load_modal_coordinates_pch(filename: str) -> ModalData:
    """
    Load modal coordinate data from a NASTRAN punch file (.pch).
    
    Parses SDISPLACEMENT output from SOL 112 (Modal Transient) analysis.
    The punch file must contain '(SOLUTION SET)' sections which indicate
    modal/generalized displacements.
    
    Shows progress indicator for large files (>100 MB).
    
    Args:
        filename: Path to the NASTRAN punch file.
    
    Returns:
        ModalData object containing modal coordinates and time values.
        The modal_coord array has shape (num_modes, num_time_points).
    
    Raises:
        ValueError: If the file is invalid or cannot be loaded.
    
    Example:
        >>> modal_data = load_modal_coordinates_pch("analysis.pch")
        >>> print(f"Loaded {modal_data.num_modes} modes, {modal_data.num_time_points} time points")
    """
    # Check file size for progress indication
    show_progress, file_size_mb = _should_show_progress(filename)
    
    # Start timing
    start_time = time.time()
    
    # Log start for large files
    if show_progress:
        _log_loading_start(filename, "NASTRAN Punch File (Modal Coordinates)", file_size_mb)
    
    # Validate first
    if show_progress:
        print("🔍 Validating NASTRAN punch file structure...")
        sys.stdout.flush()
    
    is_valid, error_msg = validate_pch_file(filename)
    if not is_valid:
        raise ValueError(f"Invalid NASTRAN punch file: {error_msg}")
    
    if show_progress:
        print("✓ Validation passed")
        sys.stdout.flush()
    
    # Parse the punch file
    if show_progress:
        print("⚙️  Parsing modal coordinate data...")
        sys.stdout.flush()
    
    try:
        modal_coord, time_values, metadata = parse_nastran_pch_modal_coordinates(filename)
        
        # Convert to appropriate dtype
        modal_coord = modal_coord.astype(NP_DTYPE)
        time_values = time_values.astype(NP_DTYPE)
        
        # Create result
        result = ModalData(modal_coord=modal_coord, time_values=time_values)
        
        # Log completion for large files
        if show_progress:
            elapsed = time.time() - start_time
            print(f"\n✅ NASTRAN Punch file loaded successfully!")
            print(f"   Time: {elapsed:.2f}s")
            print(f"   Modes: {result.num_modes} (IDs: {metadata['mode_ids'][:5]}{'...' if len(metadata['mode_ids']) > 5 else ''})")
            print(f"   Time Points: {result.num_time_points:,}")
            print(f"   Time Range: {time_values[0]:.6f}s to {time_values[-1]:.6f}s")
            print(f"{'='*70}\n")
            sys.stdout.flush()
        
        return result
        
    except Exception as e:
        raise ValueError(f"Failed to parse NASTRAN punch file: {str(e)}") from e


def load_modal_stress(filename: str) -> ModalStressData:
    """
    Load modal stress data from a CSV file.
    
    Uses PyArrow engine for faster parsing of wide modal files with multi-threading.
    Shows progress indicator for large files (>100 MB). Falls back to default 
    pandas engine if PyArrow is unavailable.
    
    Args:
        filename: Path to the modal stress CSV file.
    
    Returns:
        ModalStressData object containing stress components and node information.
    
    Raises:
        ValueError: If the file is invalid or cannot be loaded.
    """
    # Check file size for progress indication
    show_progress, file_size_mb = _should_show_progress(filename)
    
    # Start timing
    start_time = time.time()
    
    # Log start for large files
    if show_progress:
        _log_loading_start(filename, "Modal Stress", file_size_mb)
    
    # Validate first (fast with nrows=10 optimization)
    if show_progress:
        print("🔍 Validating file structure...")
        sys.stdout.flush()
    
    is_valid, error_msg = validate_modal_stress_file(filename)
    if not is_valid:
        raise ValueError(f"Invalid modal stress file: {error_msg}")
    
    if show_progress:
        print("✓ Validation passed")
        sys.stdout.flush()
    
    # Load data with progress tracking
    df = _read_csv_with_progress(filename, file_size_mb, "stress")
    
    # Process data
    if show_progress:
        print("⚙️  Processing data...")
        sys.stdout.flush()
    
    # Drop duplicates, keeping the last entry
    df.drop_duplicates(subset=['NodeID'], keep='last', inplace=True)
    
    # Extract node IDs
    node_ids = df['NodeID'].to_numpy().flatten()
    
    # Extract coordinates if present
    node_coords = None
    if {'X', 'Y', 'Z'}.issubset(df.columns):
        node_coords = df[['X', 'Y', 'Z']].to_numpy()
    
    # Extract stress components
    modal_sx = df.filter(regex='(?i)sx_.*').to_numpy().astype(NP_DTYPE)
    modal_sy = df.filter(regex='(?i)sy_.*').to_numpy().astype(NP_DTYPE)
    modal_sz = df.filter(regex='(?i)sz_.*').to_numpy().astype(NP_DTYPE)
    modal_sxy = df.filter(regex='(?i)sxy_.*').to_numpy().astype(NP_DTYPE)
    modal_syz = df.filter(regex='(?i)syz_.*').to_numpy().astype(NP_DTYPE)
    modal_sxz = df.filter(regex='(?i)sxz_.*').to_numpy().astype(NP_DTYPE)
    
    # Create result
    result = ModalStressData(
        node_ids=node_ids,
        modal_sx=modal_sx,
        modal_sy=modal_sy,
        modal_sz=modal_sz,
        modal_sxy=modal_sxy,
        modal_syz=modal_syz,
        modal_sxz=modal_sxz,
        node_coords=node_coords
    )
    
    # Log completion for large files
    if show_progress:
        elapsed = time.time() - start_time
        _log_loading_complete("Modal Stress", elapsed, result.num_nodes, result.num_modes)
    
    return result


def load_modal_deformations(filename: str) -> DeformationData:
    """
    Load modal deformation data from a CSV file.
    
    Uses PyArrow engine for faster parsing of wide modal files with multi-threading.
    Shows progress indicator for large files (>100 MB). Falls back to default 
    pandas engine if PyArrow is unavailable.
    
    Args:
        filename: Path to the modal deformations CSV file.
    
    Returns:
        DeformationData object containing deformation components.
    
    Raises:
        ValueError: If the file is invalid or cannot be loaded.
    """
    # Check file size for progress indication
    show_progress, file_size_mb = _should_show_progress(filename)
    
    # Start timing
    start_time = time.time()
    
    # Log start for large files
    if show_progress:
        _log_loading_start(filename, "Modal Deformation", file_size_mb)
    
    # Validate first (fast with nrows=10 optimization)
    if show_progress:
        print("🔍 Validating file structure...")
        sys.stdout.flush()
    
    is_valid, error_msg = validate_deformation_file(filename)
    if not is_valid:
        raise ValueError(f"Invalid deformation file: {error_msg}")
    
    if show_progress:
        print("✓ Validation passed")
        sys.stdout.flush()
    
    # Load data with progress tracking
    df = _read_csv_with_progress(filename, file_size_mb, "deformation")
    
    # Process data
    if show_progress:
        print("⚙️  Processing data...")
        sys.stdout.flush()
    
    # Drop duplicates, keeping the last entry
    df.drop_duplicates(subset=['NodeID'], keep='last', inplace=True)

    # Extract node IDs
    node_ids = df['NodeID'].to_numpy().flatten()
    
    # Extract deformation components
    modal_ux = df.filter(regex='(?i)^ux_').to_numpy().astype(NP_DTYPE)
    modal_uy = df.filter(regex='(?i)^uy_').to_numpy().astype(NP_DTYPE)
    modal_uz = df.filter(regex='(?i)^uz_').to_numpy().astype(NP_DTYPE)
    
    # Create result
    result = DeformationData(
        node_ids=node_ids,
        modal_ux=modal_ux,
        modal_uy=modal_uy,
        modal_uz=modal_uz
    )
    
    # Log completion for large files
    if show_progress:
        elapsed = time.time() - start_time
        _log_loading_complete("Modal Deformation", elapsed, result.num_nodes, result.num_modes)
    
    return result


def load_steady_state_stress(filename: str) -> SteadyStateData:
    """
    Load steady-state stress data from a TXT file.
    
    Args:
        filename: Path to the steady-state stress file.
    
    Returns:
        SteadyStateData object containing steady-state stress components.
    
    Raises:
        ValueError: If the file is invalid or cannot be loaded.
    """
    # Validate first
    is_valid, error_msg = validate_steady_state_file(filename)
    if not is_valid:
        raise ValueError(f"Invalid steady-state stress file: {error_msg}")
    
    # Load data
    df = pd.read_csv(filename, delimiter='\t', header=0)
    
    # Extract data
    node_ids = df['Node Number'].to_numpy().reshape(-1, 1)
    steady_sx = df['SX (MPa)'].to_numpy().reshape(-1, 1).astype(NP_DTYPE)
    steady_sy = df['SY (MPa)'].to_numpy().reshape(-1, 1).astype(NP_DTYPE)
    steady_sz = df['SZ (MPa)'].to_numpy().reshape(-1, 1).astype(NP_DTYPE)
    steady_sxy = df['SXY (MPa)'].to_numpy().reshape(-1, 1).astype(NP_DTYPE)
    steady_syz = df['SYZ (MPa)'].to_numpy().reshape(-1, 1).astype(NP_DTYPE)
    steady_sxz = df['SXZ (MPa)'].to_numpy().reshape(-1, 1).astype(NP_DTYPE)
    
    return SteadyStateData(
        node_ids=node_ids,
        steady_sx=steady_sx,
        steady_sy=steady_sy,
        steady_sz=steady_sz,
        steady_sxy=steady_sxy,
        steady_syz=steady_syz,
        steady_sxz=steady_sxz
    )


def load_temperature_field(filename: str) -> TemperatureFieldData:
    """Load nodal temperature field data from a TXT file."""
    try:
        df = pd.read_csv(filename, sep='\t', engine='python')
        if df.shape[1] <= 1:
            df = pd.read_csv(filename, sep=r'\s+', engine='python')
    except Exception as exc:
        raise ValueError(f"Failed to parse temperature field file: {exc}") from exc

    if df.empty:
        raise ValueError("Temperature field file is empty.")

    df.columns = [col.strip() for col in df.columns]

    if 'Node Number' not in df.columns:
        raise ValueError("Temperature field file must contain a 'Node Number' column.")

    return TemperatureFieldData(dataframe=df)


def _build_material_profile_dataframe(section: dict, expected_columns) -> pd.DataFrame:
    if section is None:
        return pd.DataFrame(columns=expected_columns)

    columns = section.get("columns", expected_columns)
    data = section.get("data", [])

    df = pd.DataFrame(data, columns=columns)
    rename_map = {columns[i]: expected_columns[i] for i in range(min(len(columns), len(expected_columns)))}
    df = df.rename(columns=rename_map)

    missing = [col for col in expected_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {', '.join(missing)}")

    df = df[expected_columns]
    for column in expected_columns:
        if df[column].empty:
            continue
        df[column] = pd.to_numeric(df[column], errors='raise')
    return df


def load_material_profile(filename: str) -> MaterialProfileData:
    """
    Load a material profile JSON file into a MaterialProfileData object.

    Args:
        filename: Path to the material profile JSON file.

    Returns:
        MaterialProfileData populated with Young's modulus, Poisson's ratio,
        and plastic curve datasets.
    """
    if not os.path.exists(filename):
        raise ValueError("File does not exist.")

    try:
        with open(filename, "r", encoding="utf-8-sig") as fh:
            payload = json.load(fh)
    except Exception as exc:
        raise ValueError(f"Failed to read material profile: {exc}") from exc

    is_valid, error = validate_material_profile_payload(payload)
    if not is_valid:
        raise ValueError(f"Invalid material profile: {error}")

    youngs_df = _build_material_profile_dataframe(
        payload.get("youngs_modulus"),
        ["Temperature (°C)", "Young's Modulus [MPa]"],
    )
    poisson_df = _build_material_profile_dataframe(
        payload.get("poisson_ratio"),
        ["Temperature (°C)", "Poisson's Ratio"],
    )

    plastic_curves = {}
    for entry in payload.get("plastic_curves", []):
        temperature = float(entry.get("temperature"))
        curve_df = _build_material_profile_dataframe(
            entry,
            ["Plastic Strain", "True Stress [MPa]"],
        )
        plastic_curves[temperature] = curve_df

    return MaterialProfileData(
        youngs_modulus=youngs_df,
        poisson_ratio=poisson_df,
        plastic_curves=plastic_curves,
    )
