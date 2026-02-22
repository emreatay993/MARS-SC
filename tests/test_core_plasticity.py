"""
Tests for core plasticity input preparation helpers.
"""

import os
import sys

import numpy as np
import pandas as pd

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.data_models import MaterialProfileData
from core.plasticity import build_material_db_from_profile


def test_build_material_db_preserves_long_curve_tails():
    """
    Shared strain grid should preserve tails from all temperature curves.

    Regression: selecting the densest curve grid can truncate a longer tail from
    another curve that has fewer points.
    """
    youngs_modulus = pd.DataFrame(
        {
            "Temperature (°C)": [20.0, 100.0],
            "Young's Modulus [MPa]": [210000.0, 205000.0],
        }
    )
    poisson_ratio = pd.DataFrame(
        {
            "Temperature (°C)": [20.0, 100.0],
            "Poisson's Ratio": [0.30, 0.30],
        }
    )

    # Fewer points, but longer strain tail to 0.50
    curve_20 = pd.DataFrame(
        {
            "True Stress [MPa]": [350.0, 450.0, 550.0],
            "Plastic Strain": [0.0, 0.10, 0.50],
        }
    )
    # More points, shorter tail to 0.20
    curve_100 = pd.DataFrame(
        {
            "True Stress [MPa]": [330.0, 430.0, 530.0, 630.0],
            "Plastic Strain": [0.0, 0.05, 0.10, 0.20],
        }
    )

    profile = MaterialProfileData(
        youngs_modulus=youngs_modulus,
        poisson_ratio=poisson_ratio,
        plastic_curves={20.0: curve_20, 100.0: curve_100},
    )

    db = build_material_db_from_profile(profile)

    expected_grid = np.array([0.0, 0.05, 0.10, 0.20, 0.50])
    np.testing.assert_allclose(db.EPSP[0], expected_grid)
    np.testing.assert_allclose(db.EPSP[1], expected_grid)

    # The shorter 100C curve must be extrapolated to keep the shared 0.50 tail.
    assert db.SIG[1, -1] > 630.0
