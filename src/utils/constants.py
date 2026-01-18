"""
Global constants, configuration settings, and UI styles for MARS-SC (Solution Combination).

Centralises configuration values and Qt styles used across the application.

Note: MARS-SC uses numpy and DPF for computations (no torch/GPU required).
"""

import os
import numpy as np

# ===== Solver Configuration =====
# These constants control the core behavior and precision of the solver.

RAM_PERCENT = 0.9
"""Default RAM allocation percentage based on available memory."""

DEFAULT_PRECISION = 'Double'
"""Precision for numerical computations: 'Single' or 'Double'."""

# ===== Data Type Configuration =====
# Dynamically set NumPy data types based on the selected precision.

if DEFAULT_PRECISION == 'Single':
    NP_DTYPE = np.float32
    RESULT_DTYPE = 'float32'
elif DEFAULT_PRECISION == 'Double':
    NP_DTYPE = np.float64
    RESULT_DTYPE = 'float64'
else:
    raise ValueError(f"Invalid precision: {DEFAULT_PRECISION}. Must be 'Single' or 'Double'.")

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

DEFAULT_ANIMATION_INTERVAL_MS = 100
"""Default animation frame interval in milliseconds."""
