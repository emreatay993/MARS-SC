"""
Global constants, configuration settings, and UI styles for MARS-SC (Solution Combination).

Centralises configuration values and Qt styles used across the application.

Note: MARS-SC uses numpy and DPF for computations (no torch/GPU required).
"""

import os
import numpy as np

# ===== Data Type Configuration =====
# NumPy data type for numerical computations (double precision for accuracy).
NP_DTYPE = np.float64

# ===== Environment Configuration =====
# Set OpenBLAS to use all available CPU cores for NumPy operations.
os.environ["OPENBLAS_NUM_THREADS"] = str(os.cpu_count())

# ===== UI Configuration =====
# Note: All styling is now centralized in src/ui/styles/style_constants.py
# and applied directly to widgets using setStyleSheet().

# ===== UI Colors =====

WINDOW_BACKGROUND_COLOR = (230, 230, 230)
"""Light gray background color for main window (R, G, B)."""

THEME_BLUE = "#5b9bd5"
"""Primary theme color used throughout the UI."""

# ===== Display Tab Constants =====

DEFAULT_POINT_SIZE = 15
"""Default point size for 3D visualization."""

DEFAULT_BACKGROUND_COLOR = "#FFFFFF"
"""Default background color for PyVista plotter."""

# ----- User-facing messages (single source for repeated text) -----
MSG_NODAL_FORCES_ANSYS = (
    "Ensure 'Write element nodal forces' is enabled in ANSYS Output Controls."
)
