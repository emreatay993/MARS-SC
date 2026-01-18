"""
Handles the application and management of advanced settings.

For MARS-SC (Solution Combination), settings are simplified as GPU acceleration
is not used (computations use numpy and DPF operators).
"""

import numpy as np

import utils.constants as constants


class SettingsHandler:
    """Manages applying advanced settings to the solver engine."""

    def __init__(self):
        """Initialize the settings handler."""
        pass

    def apply_advanced_settings(self, settings):
        """Apply advanced settings to global constants."""
        # Update global settings in constants module
        constants.RAM_PERCENT = settings.get("ram_percent", constants.RAM_PERCENT)
        constants.DEFAULT_PRECISION = settings.get("precision", constants.DEFAULT_PRECISION)

        # Update derived precision variables
        if constants.DEFAULT_PRECISION == 'Single':
            constants.NP_DTYPE = np.float32
            constants.RESULT_DTYPE = 'float32'
        elif constants.DEFAULT_PRECISION == 'Double':
            constants.NP_DTYPE = np.float64
            constants.RESULT_DTYPE = 'float64'

        print("\n--- Advanced settings updated ---")
        print(f"  RAM Allocation: {constants.RAM_PERCENT * 100:.0f}%")
        print(f"  Solver Precision: {constants.DEFAULT_PRECISION}")
        print("---------------------------------")