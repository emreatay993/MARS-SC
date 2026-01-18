"""
File utility helpers for MARS-SC (Solution Combination).

Provides helpers for manipulating modal coordinate files and related assets.
"""

import re
from typing import Tuple, Dict, List, Optional
import numpy as np


def parse_nastran_pch_modal_coordinates(filename: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Parse NASTRAN punch file (.pch) to extract modal coordinates from SOL 112.
    
    Looks for SDISPLACEMENT output sections marked with '$DISPLACEMENTS (SOLUTION SET)'.
    Each mode's time history is extracted and assembled into a modal coordinate matrix.
    
    Args:
        filename: Path to the NASTRAN punch file.
    
    Returns:
        Tuple of:
            - modal_coord: numpy array of shape (num_modes, num_time_points)
            - time_values: numpy array of time values
            - metadata: dict with 'num_modes', 'num_time_points', 'mode_ids'
    
    Raises:
        ValueError: If the file format is invalid or no modal coordinates found.
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Storage for parsed data
    mode_data: Dict[int, List[Tuple[float, float]]] = {}  # mode_id -> [(time, value), ...]
    current_mode_id: Optional[int] = None
    in_solution_set_section = False
    
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        
        # Check for SOLUTION SET section start (modal coordinates)
        if '$DISPLACEMENTS (SOLUTION SET)' in line:
            in_solution_set_section = True
            i += 1
            continue
        
        # Check for regular DISPLACEMENTS section (physical - skip these)
        if line.startswith('$DISPLACEMENTS') and '(SOLUTION SET)' not in line:
            in_solution_set_section = False
            i += 1
            continue
        
        # Check for VELOCITIES or ACCELERATIONS sections
        if line.startswith('$VELOCITIES') or line.startswith('$ACCELERATIONS'):
            in_solution_set_section = '(SOLUTION SET)' in line
            i += 1
            continue
        
        # Parse mode/point ID
        if in_solution_set_section and '$POINT ID =' in line:
            match = re.search(r'\$POINT ID\s*=\s*(\d+)', line)
            if match:
                current_mode_id = int(match.group(1))
                if current_mode_id not in mode_data:
                    mode_data[current_mode_id] = []
            i += 1
            continue
        
        # Parse data line (starts with time value and 'M' marker for modal)
        if in_solution_set_section and current_mode_id is not None:
            # Skip comment lines and headers
            if line.startswith('$') or line.startswith('-CONT-') or not line.strip():
                i += 1
                continue
            
            # Try to parse as data line: "TIME M VALUE1 VALUE2 VALUE3"
            # The 'M' marker indicates modal point data
            parts = line.split()
            if len(parts) >= 2:
                try:
                    # First field is time, second is 'M' marker (or 'G' for grid - skip)
                    time_val = float(parts[0])
                    
                    # Check for 'M' marker (modal point)
                    if len(parts) >= 3 and parts[1] == 'M':
                        # Third field is the modal coordinate value
                        modal_coord_val = float(parts[2])
                        mode_data[current_mode_id].append((time_val, modal_coord_val))
                except ValueError:
                    pass  # Not a data line, skip
        
        i += 1
    
    # Validate we found data
    if not mode_data:
        raise ValueError(
            "No modal coordinate data found in punch file. "
            "Ensure the file contains SDISPLACEMENT output with '(SOLUTION SET)' marker."
        )
    
    # Get sorted mode IDs
    mode_ids = sorted(mode_data.keys())
    num_modes = len(mode_ids)
    
    # Extract time values from first mode (should be same for all)
    first_mode_data = mode_data[mode_ids[0]]
    time_values = np.array([t for t, _ in first_mode_data])
    num_time_points = len(time_values)
    
    # Validate all modes have same number of time points
    for mode_id in mode_ids:
        if len(mode_data[mode_id]) != num_time_points:
            raise ValueError(
                f"Mode {mode_id} has {len(mode_data[mode_id])} time points, "
                f"expected {num_time_points}. Inconsistent punch file data."
            )
    
    # Build modal coordinate matrix (num_modes x num_time_points)
    modal_coord = np.zeros((num_modes, num_time_points), dtype=np.float64)
    
    for idx, mode_id in enumerate(mode_ids):
        modal_coord[idx, :] = np.array([v for _, v in mode_data[mode_id]])
    
    metadata = {
        'num_modes': num_modes,
        'num_time_points': num_time_points,
        'mode_ids': mode_ids,
    }
    
    return modal_coord, time_values, metadata


def unwrap_mcf_file(input_file, output_file):
    """
    Unwraps a Modal Coordinate File (MCF) that has wrapped data lines.
    
    After the header line (the one starting with "Number of Modes"), some records
    are wrapped across multiple lines. Additionally, there is a column header line
    (e.g. "Time          Coordinates...") in the data block that should remain
    separate.
    
    Algorithm:
    1. Keeps all lines up to and including the line that starts (after stripping)
       with "Number of Modes".
    2. For the remaining lines, if a line (after stripping) contains both "Time"
       and "Coordinates", it is treated as a header line and is preserved as its
       own record.
    3. For other lines, the minimum indentation among them is determined (the base
       indent). Lines with exactly that indentation start new records, while lines
       with extra indentation are treated as continuations (wrapped lines) and
       appended to the previous record.
    
    Args:
        input_file: Path to the input MCF file with wrapped lines.
        output_file: Path to save the unwrapped output file.
    
    Returns:
        list: List of unwrapped lines.
    """
    # Read all lines
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    # Separate header (everything up to and including "Number of Modes")
    header_end = None
    for i, line in enumerate(lines):
        if line.lstrip().startswith("Number of Modes"):
            header_end = i
            break
    
    if header_end is None:
        header_lines = []
        data_lines = lines
    else:
        header_lines = lines[:header_end + 1]
        data_lines = lines[header_end + 1:]
    
    # For base indentation calculation, skip header lines with "Time" and "Coordinates"
    data_non_header = []
    for line in data_lines:
        stripped = line.strip()
        if stripped and ("Time" in stripped and "Coordinates" in stripped):
            continue  # skip header lines for indent calculation
        if stripped:
            data_non_header.append(line)
    
    base_indent = None
    for line in data_non_header:
        indent = len(line) - len(line.lstrip(' '))
        if base_indent is None or indent < base_indent:
            base_indent = indent
    if base_indent is None:
        base_indent = 0
    
    # Process data lines
    unwrapped_data = []
    current_line = ""
    for line in data_lines:
        stripped = line.strip()
        if not stripped:
            continue  # skip empty lines
        
        # If this line is the special header (e.g., "Time          Coordinates...")
        if "Time" in stripped and "Coordinates" in stripped:
            if current_line:
                unwrapped_data.append(current_line)
                current_line = ""
            unwrapped_data.append(stripped)
            continue
        
        # Determine indentation of the current line
        indent = len(line) - len(line.lstrip(' '))
        if indent == base_indent:
            # New record
            if current_line:
                unwrapped_data.append(current_line)
            current_line = stripped
        else:
            # Wrapped (continuation) line
            current_line = current_line.rstrip('\n') + " " + stripped
    
    if current_line:
        unwrapped_data.append(current_line)
    
    # Combine header and unwrapped data
    final_lines = [h.rstrip('\n') for h in header_lines] + unwrapped_data
    
    # Write final result to output file
    with open(output_file, 'w') as f:
        for line in final_lines:
            f.write(line + "\n")
    
    return final_lines
