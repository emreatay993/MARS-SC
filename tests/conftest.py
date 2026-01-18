"""
Pytest configuration and shared fixtures for MARS-SC tests.
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture
def sample_node_data():
    """Fixture providing sample node data."""
    n_nodes = 100
    return {
        'node_ids': np.arange(1, n_nodes + 1),
        'node_coords': np.random.rand(n_nodes, 3) * 100,
    }


@pytest.fixture
def sample_stress_tensor():
    """Fixture providing sample stress tensor data."""
    n_nodes = 50
    return {
        'sx': np.random.rand(n_nodes) * 200 - 50,
        'sy': np.random.rand(n_nodes) * 200 - 50,
        'sz': np.random.rand(n_nodes) * 200 - 50,
        'sxy': np.random.rand(n_nodes) * 100 - 25,
        'syz': np.random.rand(n_nodes) * 100 - 25,
        'sxz': np.random.rand(n_nodes) * 100 - 25,
    }


@pytest.fixture
def sample_combination_table():
    """Fixture providing sample combination table."""
    from core.data_models import CombinationTableData
    
    return CombinationTableData(
        combination_names=["Combo A", "Combo B", "Combo C"],
        combination_types=["Linear", "Linear", "Linear"],
        analysis1_coeffs=np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 0.0],
        ]),
        analysis2_coeffs=np.array([
            [0.5, 0.0],
            [0.0, 0.5],
            [0.25, 0.25],
        ]),
        analysis1_step_ids=[1, 2, 3],
        analysis2_step_ids=[1, 2],
    )


@pytest.fixture
def sample_material_db():
    """Fixture providing sample material database."""
    from solver.plasticity_engine import MaterialDB
    
    temp = np.array([20.0, 100.0, 200.0])
    e_tab = np.array([210000.0, 205000.0, 195000.0])
    sig = np.array([
        [350.0, 450.0, 550.0, 650.0],
        [330.0, 430.0, 530.0, 630.0],
        [300.0, 400.0, 500.0, 600.0],
    ])
    epsp = np.array([
        [0.0, 0.005, 0.02, 0.05],
        [0.0, 0.006, 0.025, 0.06],
        [0.0, 0.008, 0.03, 0.07],
    ])
    
    return MaterialDB.from_arrays(temp, e_tab, sig, epsp)


@pytest.fixture
def sample_csv_two_analysis(tmp_path):
    """Fixture creating a sample two-analysis CSV file."""
    csv_content = """Combination Name,Type,A1_Set_1,A1_Set_2,A2_Set_1,A2_Set_2
Load Case 1,Linear,1.0,0.0,0.0,0.0
Load Case 2,Linear,0.0,1.0,0.0,0.0
Combined 1,Linear,0.5,0.5,0.5,0.0
Combined 2,Linear,0.25,0.75,0.25,0.25
"""
    csv_path = tmp_path / "combinations.csv"
    csv_path.write_text(csv_content)
    return str(csv_path)


@pytest.fixture
def sample_envelope_result():
    """Fixture providing sample envelope result data."""
    from core.data_models import CombinationResult
    
    n_nodes = 20
    n_combos = 5
    
    all_results = np.random.rand(n_combos, n_nodes) * 500
    
    return CombinationResult(
        node_ids=np.arange(1, n_nodes + 1),
        node_coords=np.random.rand(n_nodes, 3),
        max_over_combo=np.max(all_results, axis=0),
        min_over_combo=np.min(all_results, axis=0),
        combo_of_max=np.argmax(all_results, axis=0),
        combo_of_min=np.argmin(all_results, axis=0),
        result_type="von_mises",
        all_combo_results=all_results,
    )


# Set random seed for reproducible tests
@pytest.fixture(autouse=True)
def seed_random():
    """Auto-used fixture to seed random for reproducibility."""
    np.random.seed(42)
