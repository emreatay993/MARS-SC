"""
Runtime integration tests for stress plasticity application in solve execution.
"""

import os
import sys

import numpy as np
import pandas as pd

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.data_models import (
    CombinationResult,
    MaterialProfileData,
    PlasticityConfig,
    SolverConfig,
    TemperatureFieldData,
)
from ui.handlers.solver_engine_factory import SolverEngineFactory
from ui.handlers.solver_run_execution_handler import SolverRunExecutionHandler


class _FakeConsole:
    def __init__(self):
        self.messages = []

    def append(self, message):
        self.messages.append(message)


class _FakeFileHandler:
    base_reader = None
    combine_reader = None


class _FakeTab:
    def __init__(self):
        self.console_textbox = _FakeConsole()
        self.file_handler = _FakeFileHandler()

    def get_nodal_scoping_for_selected_named_selection(self):
        return None

    def get_combination_table_data(self):
        return None


class _FakeStressEngineHistory:
    def __init__(self, combo_indices, stress_values):
        self._combo_indices = np.asarray(combo_indices)
        self._stress_values = np.asarray(stress_values, dtype=np.float64)

    def check_memory_available(
        self,
        raise_on_insufficient=False,
        calculate_scalar_plasticity=False,
    ):
        return True, {"available_bytes": 8_000_000_000}

    def compute_single_node_history_fast(self, node_id, stress_type, progress_callback=None):
        return self._combo_indices.copy(), self._stress_values.copy()


class _FakeStressEngineEnvelope:
    def __init__(self, result):
        self._result = result
        self.scoping = type("Scoping", (), {"ids": list(np.asarray(result.node_ids, dtype=int))})()

    def check_memory_available(
        self,
        raise_on_insufficient=False,
        calculate_scalar_plasticity=False,
    ):
        return True, {"available_bytes": 8_000_000_000}

    def compute_full_analysis_auto(
        self,
        stress_type,
        progress_callback=None,
        memory_threshold_gb=2.0,
        calculate_scalar_plasticity=False,
        chunk_results_transform=None,
    ):
        return self._result

    def compute_full_analysis_chunked(
        self,
        stress_type,
        progress_callback=None,
        calculate_scalar_plasticity=False,
        chunk_results_transform=None,
    ):
        return self._result


class _FakeStressEngineChunkedEnvelope:
    def __init__(self, elastic_all, node_ids):
        self._elastic_all = np.asarray(elastic_all, dtype=np.float64)
        self._node_ids = np.asarray(node_ids, dtype=int)
        self._node_coords = np.column_stack(
            (
                np.arange(self._node_ids.size, dtype=np.float64),
                np.zeros(self._node_ids.size, dtype=np.float64),
                np.zeros(self._node_ids.size, dtype=np.float64),
            )
        )
        self.scoping = type("Scoping", (), {"ids": list(self._node_ids)})()

    def check_memory_available(
        self,
        raise_on_insufficient=False,
        calculate_scalar_plasticity=False,
    ):
        return True, {"available_bytes": 8_000_000_000}

    def compute_full_analysis_chunked(
        self,
        stress_type,
        progress_callback=None,
        calculate_scalar_plasticity=False,
        chunk_results_transform=None,
    ):
        num_nodes = self._elastic_all.shape[1]
        max_env = np.full(num_nodes, -np.inf, dtype=np.float64)
        min_env = np.full(num_nodes, np.inf, dtype=np.float64)
        combo_of_max = np.zeros(num_nodes, dtype=np.int32)
        combo_of_min = np.zeros(num_nodes, dtype=np.int32)

        for start in range(0, num_nodes, 2):
            end = min(start + 2, num_nodes)
            chunk = self._elastic_all[:, start:end].copy()
            if chunk_results_transform is not None:
                chunk = chunk_results_transform(chunk, self._node_ids[start:end], start, end)

            max_env[start:end] = np.max(chunk, axis=0)
            min_env[start:end] = np.min(chunk, axis=0)
            combo_of_max[start:end] = np.argmax(chunk, axis=0)
            combo_of_min[start:end] = np.argmin(chunk, axis=0)

        return CombinationResult(
            node_ids=self._node_ids.copy(),
            node_coords=self._node_coords.copy(),
            max_over_combo=max_env,
            min_over_combo=min_env,
            combo_of_max=combo_of_max,
            combo_of_min=combo_of_min,
            result_type=stress_type,
            all_combo_results=None,
        )

    def compute_full_analysis_auto(
        self,
        stress_type,
        progress_callback=None,
        memory_threshold_gb=2.0,
        calculate_scalar_plasticity=False,
        chunk_results_transform=None,
    ):
        return self.compute_full_analysis_chunked(
            stress_type=stress_type,
            progress_callback=progress_callback,
            calculate_scalar_plasticity=calculate_scalar_plasticity,
            chunk_results_transform=chunk_results_transform,
        )


class _FakeStressEngineSingleCombination:
    def __init__(self, node_ids):
        self.scoping = type("Scoping", (), {"ids": list(np.asarray(node_ids, dtype=int))})()
        self._node_ids = np.asarray(node_ids, dtype=int)
        self._node_coords = np.column_stack(
            (
                np.arange(self._node_ids.size, dtype=np.float64),
                np.zeros(self._node_ids.size, dtype=np.float64),
                np.zeros(self._node_ids.size, dtype=np.float64),
            )
        )

    def compute_single_combination_chunked(
        self,
        combination_index,
        stress_type,
        progress_callback=None,
        calculate_scalar_plasticity=False,
        chunk_values_transform=None,
    ):
        base = np.array([10.0, 20.0, 30.0], dtype=np.float64)
        corrected = base
        if chunk_values_transform is not None:
            corrected = chunk_values_transform(base.copy(), self._node_ids, 0, self._node_ids.size)

        return CombinationResult(
            node_ids=self._node_ids.copy(),
            node_coords=self._node_coords.copy(),
            max_over_combo=corrected.copy(),
            combo_of_max=np.full(self._node_ids.size, combination_index, dtype=np.int32),
            result_type=stress_type,
            all_combo_results=corrected.reshape(1, -1),
        )


def _sample_material_profile():
    return MaterialProfileData(
        youngs_modulus=pd.DataFrame(
            {
                "Temperature": [20.0, 100.0],
                "Young's Modulus [MPa]": [210000.0, 205000.0],
            }
        ),
        poisson_ratio=pd.DataFrame(
            {
                "Temperature": [20.0, 100.0],
                "Poisson's Ratio": [0.30, 0.30],
            }
        ),
        plastic_curves={
            20.0: pd.DataFrame(
                {
                    "True Stress [MPa]": [350.0, 450.0, 550.0],
                    "Plastic Strain": [0.0, 0.01, 0.08],
                }
            ),
            100.0: pd.DataFrame(
                {
                    "True Stress [MPa]": [320.0, 420.0, 520.0],
                    "Plastic Strain": [0.0, 0.012, 0.10],
                }
            ),
        },
    )


def test_run_stress_history_applies_plasticity(monkeypatch):
    tab = _FakeTab()
    handler = SolverRunExecutionHandler(tab, SolverEngineFactory())

    combo_indices = np.array([0, 1, 2])
    elastic = np.array([300.0, 700.0, 950.0], dtype=np.float64)
    fake_engine = _FakeStressEngineHistory(combo_indices, elastic)

    monkeypatch.setattr(handler, "_create_stress_engine", lambda: fake_engine)
    monkeypatch.setattr(handler, "_should_use_chunked_processing", lambda _engine, **_kwargs: False)

    config = SolverConfig(
        calculate_von_mises=True,
        combination_history_mode=True,
        selected_node_id=101,
    )
    config.plasticity = PlasticityConfig(
        enabled=True,
        method="neuber",
        material_profile=_sample_material_profile(),
    )

    result = handler.run_stress_analysis(config, "von_mises", lambda *_: None)

    assert result.metadata is not None
    overlay = result.metadata.get("plasticity_overlay")
    assert overlay is not None
    np.testing.assert_allclose(overlay["elastic_vm"], elastic)
    assert not np.allclose(result.metadata["stress_values"], elastic)
    np.testing.assert_allclose(result.all_combo_results[:, 0], result.metadata["stress_values"])


def test_run_stress_envelope_applies_plasticity_to_full_combination_matrix(monkeypatch):
    tab = _FakeTab()
    handler = SolverRunExecutionHandler(tab, SolverEngineFactory())

    elastic_all = np.array(
        [
            [300.0, 820.0],
            [650.0, 500.0],
            [700.0, 950.0],
        ],
        dtype=np.float64,
    )
    node_ids = np.array([10, 20], dtype=int)
    elastic_result = CombinationResult(
        node_ids=node_ids,
        node_coords=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64),
        max_over_combo=np.max(elastic_all, axis=0),
        min_over_combo=np.min(elastic_all, axis=0),
        combo_of_max=np.argmax(elastic_all, axis=0),
        combo_of_min=np.argmin(elastic_all, axis=0),
        result_type="von_mises",
        all_combo_results=elastic_all.copy(),
    )
    fake_engine = _FakeStressEngineEnvelope(elastic_result)

    monkeypatch.setattr(handler, "_create_stress_engine", lambda: fake_engine)
    monkeypatch.setattr(handler, "_should_use_chunked_processing", lambda _engine, **_kwargs: False)

    temperature_field = TemperatureFieldData(
        dataframe=pd.DataFrame(
            {
                "Node Number": [10, 20],
                "Temperature [C]": [60.0, 80.0],
            }
        )
    )

    config = SolverConfig(calculate_von_mises=True, combination_history_mode=False)
    config.plasticity = PlasticityConfig(
        enabled=True,
        method="glinka",
        material_profile=_sample_material_profile(),
        temperature_field=temperature_field,
        extrapolation_mode="plateau",
    )

    result = handler.run_stress_analysis(config, "von_mises", lambda *_: None)

    assert result.metadata is not None
    assert "elastic_max_over_combo" in result.metadata
    np.testing.assert_allclose(result.metadata["elastic_max_over_combo"], np.max(elastic_all, axis=0))
    assert not np.allclose(result.all_combo_results, elastic_all)
    np.testing.assert_allclose(result.max_over_combo, np.max(result.all_combo_results, axis=0))
    np.testing.assert_array_equal(result.combo_of_max, np.argmax(result.all_combo_results, axis=0))


def test_run_stress_non_von_mises_skips_plasticity(monkeypatch):
    tab = _FakeTab()
    handler = SolverRunExecutionHandler(tab, SolverEngineFactory())

    combo_indices = np.array([0, 1, 2])
    elastic = np.array([100.0, 200.0, 300.0], dtype=np.float64)
    fake_engine = _FakeStressEngineHistory(combo_indices, elastic)

    monkeypatch.setattr(handler, "_create_stress_engine", lambda: fake_engine)
    monkeypatch.setattr(handler, "_should_use_chunked_processing", lambda _engine, **_kwargs: False)

    config = SolverConfig(
        calculate_von_mises=False,
        calculate_max_principal_stress=True,
        combination_history_mode=True,
        selected_node_id=42,
    )
    config.plasticity = PlasticityConfig(
        enabled=True,
        method="neuber",
        material_profile=_sample_material_profile(),
    )

    result = handler.run_stress_analysis(config, "max_principal", lambda *_: None)

    np.testing.assert_allclose(result.metadata["stress_values"], elastic)
    assert "plasticity_overlay" not in result.metadata
    assert any("only to Von Mises output" in msg for msg in tab.console_textbox.messages)


def test_run_stress_chunked_envelope_corrects_full_chunk_before_reduction(monkeypatch):
    tab = _FakeTab()
    handler = SolverRunExecutionHandler(tab, SolverEngineFactory())

    elastic_all = np.array(
        [
            [100.0, 200.0, 300.0],
            [150.0, 250.0, 350.0],
            [120.0, 180.0, 320.0],
        ],
        dtype=np.float64,
    )
    node_ids = np.array([10, 20, 30], dtype=int)
    fake_engine = _FakeStressEngineChunkedEnvelope(elastic_all, node_ids)

    monkeypatch.setattr(handler, "_create_stress_engine", lambda: fake_engine)
    monkeypatch.setattr(handler, "_should_use_chunked_processing", lambda _engine, **_kwargs: True)
    monkeypatch.setattr(
        handler,
        "_apply_scalar_plasticity",
        lambda stress_values, _temp_values, _ctx: (
            np.asarray(stress_values, dtype=np.float64) + 10.0,
            np.full_like(np.asarray(stress_values, dtype=np.float64), 0.25, dtype=np.float64),
        ),
    )

    config = SolverConfig(calculate_von_mises=True, combination_history_mode=False)
    config.plasticity = PlasticityConfig(
        enabled=True,
        method="neuber",
        material_profile=_sample_material_profile(),
    )

    result = handler.run_stress_analysis(config, "von_mises", lambda *_: None)

    np.testing.assert_allclose(result.max_over_combo, np.max(elastic_all + 10.0, axis=0))
    np.testing.assert_allclose(result.min_over_combo, np.min(elastic_all + 10.0, axis=0))
    np.testing.assert_allclose(result.metadata["elastic_max_over_combo"], np.max(elastic_all, axis=0))
    np.testing.assert_allclose(result.metadata["elastic_min_over_combo"], np.min(elastic_all, axis=0))
    assert result.metadata["plasticity"]["note"] == "Corrected from full combination matrix in chunked mode."


def test_run_single_combination_applies_scalar_plasticity_transform(monkeypatch):
    tab = _FakeTab()
    handler = SolverRunExecutionHandler(tab, SolverEngineFactory())

    fake_engine = _FakeStressEngineSingleCombination(node_ids=np.array([10, 20, 30], dtype=int))
    monkeypatch.setattr(handler, "_create_stress_engine", lambda *_args, **_kwargs: fake_engine)
    monkeypatch.setattr(
        handler,
        "_apply_scalar_plasticity",
        lambda stress_values, _temp_values, _ctx: (
            np.asarray(stress_values, dtype=np.float64) + 5.0,
            np.zeros_like(np.asarray(stress_values, dtype=np.float64)),
        ),
    )

    config = SolverConfig(calculate_von_mises=True, combination_history_mode=False)
    config.plasticity = PlasticityConfig(
        enabled=True,
        method="neuber",
        material_profile=_sample_material_profile(),
    )

    result = handler.run_stress_single_combination(
        config=config,
        stress_type="von_mises",
        combination_index=1,
        progress_callback=lambda *_: None,
        combo_table_override=None,
    )

    np.testing.assert_allclose(result.all_combo_results[0], np.array([15.0, 25.0, 35.0]))
    assert result.metadata["mode"] == "single_combination_recompute"
