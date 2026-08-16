"""Reproducible source-level performance evidence for MARS-SC."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path


DEFAULT_FIXTURE_A = Path(
    r"C:\Users\emre_\OneDrive\Desktop\J\ANSYS\Benchmark\MARS"
    r"\Benchmark_v1_files\dp0\SYS-1\MECH\file.rst"
)
DEFAULT_FIXTURE_B = Path(
    r"C:\Users\emre_\OneDrive\Desktop\J\ANSYS\Benchmark\MARS"
    r"\Benchmark_v1_files\dp0\SYS-29\MECH\file.rst"
)
SEED = 20260816


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    roots = parser.add_mutually_exclusive_group(required=False)
    roots.add_argument("--source-root", type=Path)
    roots.add_argument("--baseline-root", type=Path)
    parser.add_argument("--candidate-root", type=Path)
    parser.add_argument("--fixture-a", type=Path, default=DEFAULT_FIXTURE_A)
    parser.add_argument("--fixture-b", type=Path, default=DEFAULT_FIXTURE_B)
    parser.add_argument(
        "--mode",
        choices=("timing", "memory", "correctness", "plasticity"),
        default="timing",
    )
    parser.add_argument("--combinations", type=int, default=1000)
    parser.add_argument("--max-sets", type=int, default=0, help="0 uses every result set")
    parser.add_argument("--abba-blocks", type=int, default=3)
    parser.add_argument("--memory-trials", type=int, default=3)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def _coefficient_matrix(rng, rows: int, columns: int):
    import numpy as np

    values = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float64)
    matrix = rng.choice(values, size=(rows, columns), p=(0.2, 0.2, 0.2, 0.2, 0.2))
    nonzero = np.array([-1.0, -0.5, 0.5, 1.0], dtype=np.float64)
    matrix[0, :] = rng.choice(nonzero, size=columns)
    return np.ascontiguousarray(matrix)


def _build_table(analysis_a, analysis_b, combinations: int, max_sets: int):
    import numpy as np
    from core.data_models import CombinationTableData

    steps_a = list(analysis_a.load_step_ids)
    steps_b = list(analysis_b.load_step_ids)
    if max_sets > 0:
        steps_a = steps_a[:max_sets]
        steps_b = steps_b[:max_sets]
    rng = np.random.default_rng(SEED)
    coeff_a = _coefficient_matrix(rng, combinations, len(steps_a))
    coeff_b = _coefficient_matrix(rng, combinations, len(steps_b))
    digest = hashlib.sha256(coeff_a.tobytes() + coeff_b.tobytes()).hexdigest()
    table = CombinationTableData(
        combination_names=[f"Benchmark {index + 1}" for index in range(combinations)],
        combination_types=["Linear"] * combinations,
        analysis1_coeffs=coeff_a,
        analysis2_coeffs=coeff_b,
        analysis1_step_ids=steps_a,
        analysis2_step_ids=steps_b,
    )
    return table, digest


def _component_payload(result, prefix: str):
    import numpy as np

    payload = {}
    for component in ("x", "y", "z"):
        values = getattr(result, f"all_combo_{prefix}{component}")
        payload[f"max_{prefix}{component}"] = np.max(values, axis=0)
        payload[f"min_{prefix}{component}"] = np.min(values, axis=0)
        payload[f"combo_of_max_{prefix}{component}"] = np.argmax(values, axis=0)
        payload[f"combo_of_min_{prefix}{component}"] = np.argmin(values, axis=0)
    return payload


def _artifact_arrays(stress, forces, deformation, force_step_values, force_coefficients):
    return {
        "stress_node_ids": stress.node_ids,
        "stress_node_coords": stress.node_coords,
        "stress_max": stress.max_over_combo,
        "stress_min": stress.min_over_combo,
        "stress_combo_max": stress.combo_of_max,
        "stress_combo_min": stress.combo_of_min,
        "stress_all": stress.all_combo_results,
        "force_node_ids": forces.node_ids,
        "force_node_coords": forces.node_coords,
        "force_max": forces.max_magnitude_over_combo,
        "force_min": forces.min_magnitude_over_combo,
        "force_combo_max": forces.combo_of_max,
        "force_combo_min": forces.combo_of_min,
        "force_fx": forces.all_combo_fx,
        "force_fy": forces.all_combo_fy,
        "force_fz": forces.all_combo_fz,
        "deformation_node_ids": deformation.node_ids,
        "deformation_node_coords": deformation.node_coords,
        "deformation_max": deformation.max_magnitude_over_combo,
        "deformation_min": deformation.min_magnitude_over_combo,
        "deformation_combo_max": deformation.combo_of_max,
        "deformation_combo_min": deformation.combo_of_min,
        "deformation_ux": deformation.all_combo_ux,
        "deformation_uy": deformation.all_combo_uy,
        "deformation_uz": deformation.all_combo_uz,
        "reference_force_step_values": force_step_values,
        "reference_force_coefficients": force_coefficients,
    }


def _plasticity_worker(args: argparse.Namespace, source_root: Path) -> int:
    import numpy as np
    import numba
    from solver.plasticity_engine import (
        MaterialDB,
        apply_glinka_correction,
        apply_neuber_correction,
        solve_neuber_vector_core,
    )

    entries = args.combinations * 10528
    rng = np.random.default_rng(SEED)
    stress = rng.uniform(250.0, 950.0, entries)
    temperature = rng.uniform(20.0, 100.0, entries)
    material = MaterialDB.from_arrays(
        np.array([20.0, 100.0]),
        np.array([210000.0, 205000.0]),
        np.array([[350.0, 450.0, 550.0], [320.0, 420.0, 520.0]]),
        np.array([[0.0, 0.01, 0.08], [0.0, 0.012, 0.10]]),
    )

    methods = {}
    for name, function in (
        ("neuber", apply_neuber_correction),
        ("glinka", apply_glinka_correction),
    ):
        started = time.perf_counter()
        corrected, strain = function(stress, temperature, material, use_plateau=True)
        first_call = time.perf_counter() - started
        warm_trials = []
        for _ in range(3):
            started = time.perf_counter()
            corrected, strain = function(stress, temperature, material, use_plateau=True)
            warm_trials.append(time.perf_counter() - started)
        methods[name] = {
            "first_call_s": first_call,
            "warm_trials_s": warm_trials,
            "warm_median_s": statistics.median(warm_trials),
            "output_sha256": hashlib.sha256(
                corrected.tobytes() + strain.tobytes()
            ).hexdigest(),
        }

    payload = {
        "source_root": str(source_root),
        "mode": "plasticity",
        "entries": entries,
        "numba": numba.__version__,
        "numba_active": hasattr(solve_neuber_vector_core, "signatures"),
        "numba_cache_dir": os.environ.get("NUMBA_CACHE_DIR"),
        "methods": methods,
    }
    print(json.dumps(payload, separators=(",", ":")))
    return 0


def _force_reference_inputs(engine, table):
    import numpy as np

    a1_mask = np.any(table.analysis1_coeffs != 0.0, axis=0)
    a2_mask = np.any(table.analysis2_coeffs != 0.0, axis=0)
    keys = [
        (1, step_id)
        for step_id, active in zip(table.analysis1_step_ids, a1_mask)
        if active
    ] + [
        (2, step_id)
        for step_id, active in zip(table.analysis2_step_ids, a2_mask)
        if active
    ]
    coefficients = np.ascontiguousarray(
        np.concatenate(
            (table.analysis1_coeffs[:, a1_mask], table.analysis2_coeffs[:, a2_mask]),
            axis=1,
        ),
        dtype=np.float64,
    )
    step_values = getattr(engine, "_force_data", None)
    if not isinstance(step_values, np.ndarray):
        cache = engine._force_cache
        step_values = np.ascontiguousarray(
            np.stack([np.stack(cache[key][1:4], axis=0) for key in keys], axis=0)
        )
    return step_values, coefficients


def _worker(args: argparse.Namespace) -> int:
    source_root = args.source_root.resolve()
    sys.path.insert(0, str(source_root / "src"))

    if args.mode == "plasticity":
        return _plasticity_worker(args, source_root)

    import numpy as np
    import pandas as pd
    import psutil
    from ansys.dpf import core as dpf
    from file_io.dpf_reader import DPFAnalysisReader
    from file_io.exporters import (
        export_deformation_envelope,
        export_envelope_results,
        export_nodal_forces_envelope,
    )
    from solver.deformation_engine import DeformationCombinationEngine
    from solver.nodal_forces_engine import NodalForcesCombinationEngine
    from solver.stress_engine import StressCombinationEngine

    fixture_a = args.fixture_a.resolve()
    fixture_b = args.fixture_b.resolve()
    for fixture in (fixture_a, fixture_b):
        if not fixture.is_file():
            raise FileNotFoundError(fixture)

    started = time.perf_counter()
    stage_started = started
    reader_a = DPFAnalysisReader(str(fixture_a))
    reader_b = DPFAnalysisReader(str(fixture_b))
    analysis_a = reader_a.get_analysis_data(skip_substeps=False)
    analysis_b = reader_b.get_analysis_data(skip_substeps=False)
    scoping = reader_a.get_all_nodes_scoping()
    table, coefficient_hash = _build_table(
        analysis_a,
        analysis_b,
        args.combinations,
        args.max_sets,
    )
    timings = {"metadata_load_s": time.perf_counter() - stage_started}

    stress_engine = StressCombinationEngine(reader_a, reader_b, scoping, table)
    stage_started = time.perf_counter()
    stress_engine.preload_stress_data()
    timings["stress_read_s"] = time.perf_counter() - stage_started
    stage_started = time.perf_counter()
    stress = stress_engine.compute_full_analysis("von_mises")
    timings["stress_calculate_s"] = time.perf_counter() - stage_started

    force_engine = NodalForcesCombinationEngine(reader_a, reader_b, scoping, table)
    stage_started = time.perf_counter()
    force_valid, force_error = force_engine.validate_nodal_forces_availability()
    if not force_valid:
        raise RuntimeError(force_error)
    force_engine.preload_force_data()
    timings["force_read_s"] = time.perf_counter() - stage_started
    stage_started = time.perf_counter()
    forces = force_engine.compute_full_analysis(auto_cleanup=args.mode != "correctness")
    timings["force_calculate_s"] = time.perf_counter() - stage_started
    force_step_values = None
    force_coefficients = None
    if args.mode == "correctness":
        force_step_values, force_coefficients = _force_reference_inputs(force_engine, table)
        force_engine.clear_cache()

    deformation_engine = DeformationCombinationEngine(reader_a, reader_b, scoping, table)
    stage_started = time.perf_counter()
    deformation_valid, deformation_error = deformation_engine.validate_displacement_availability()
    if not deformation_valid:
        raise RuntimeError(deformation_error)
    deformation_engine.preload_displacement_data()
    timings["deformation_read_s"] = time.perf_counter() - stage_started
    stage_started = time.perf_counter()
    deformation = deformation_engine.compute_full_analysis()
    timings["deformation_calculate_s"] = time.perf_counter() - stage_started

    stage_started = time.perf_counter()
    result_contract = None
    with tempfile.TemporaryDirectory(prefix="mars_sc_benchmark_export_") as export_dir:
        export_root = Path(export_dir)
        stress_csv = export_root / "stress.csv"
        force_csv = export_root / "forces.csv"
        deformation_csv = export_root / "deformation.csv"
        export_envelope_results(
            str(stress_csv),
            stress.node_ids,
            stress.node_coords,
            stress.max_over_combo,
            stress.min_over_combo,
            stress.combo_of_max,
            stress.combo_of_min,
            stress.result_type,
            table.combination_names,
        )
        export_nodal_forces_envelope(
            str(force_csv),
            forces.node_ids,
            forces.node_coords,
            forces.max_magnitude_over_combo,
            forces.min_magnitude_over_combo,
            forces.combo_of_max,
            forces.combo_of_min,
            table.combination_names,
            forces.force_unit,
            forces.all_combo_fx,
            forces.all_combo_fy,
            forces.all_combo_fz,
            include_shear_variants=True,
            include_component_envelopes=True,
            include_component_combo_indices=True,
        )
        export_deformation_envelope(
            str(deformation_csv),
            deformation.node_ids,
            deformation.node_coords,
            deformation.max_magnitude_over_combo,
            deformation.min_magnitude_over_combo,
            deformation.combo_of_max,
            deformation.combo_of_min,
            table.combination_names,
            deformation.displacement_unit,
            _component_payload(deformation, "u"),
        )
        timings["export_s"] = time.perf_counter() - stage_started
        timings["total_s"] = time.perf_counter() - started
        if args.mode == "correctness":
            result_contract = {
                "combination_names": list(table.combination_names),
                "combination_types": list(table.combination_types),
                "units": {
                    "stress": "MPa",
                    "force": forces.force_unit,
                    "deformation": deformation.displacement_unit,
                },
                "csv": {
                    "stress": {
                        "columns": pd.read_csv(stress_csv, nrows=0).columns.tolist(),
                        "rows": int(stress.node_ids.size),
                    },
                    "force": {
                        "columns": pd.read_csv(force_csv, nrows=0).columns.tolist(),
                        "rows": int(forces.node_ids.size),
                    },
                    "deformation": {
                        "columns": pd.read_csv(deformation_csv, nrows=0).columns.tolist(),
                        "rows": int(deformation.node_ids.size),
                    },
                },
            }

    artifact_path = None
    if args.artifact_dir:
        args.artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = args.artifact_dir / "results.npz"
        np.savez(
            artifact_path,
            **_artifact_arrays(
                stress,
                forces,
                deformation,
                force_step_values,
                force_coefficients,
            ),
        )

    payload = {
        "source_root": str(source_root),
        "mode": args.mode,
        "timings": timings,
        "coefficient_sha256": coefficient_hash,
        "combinations": table.num_combinations,
        "sets_a": len(table.analysis1_step_ids),
        "sets_b": len(table.analysis2_step_ids),
        "nodes": int(stress.node_ids.size),
        "units": {
            "stress": "MPa",
            "force": forces.force_unit,
            "deformation": deformation.displacement_unit,
        },
        "artifact": str(artifact_path) if artifact_path else None,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "dpf": dpf.__version__,
            "psutil": psutil.__version__,
            "cpu": platform.processor(),
            "ram_bytes": psutil.virtual_memory().total,
        },
        "fixtures": [
            {"path": str(path), "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
            for path in (fixture_a, fixture_b)
        ],
    }
    if result_contract is not None:
        payload["result_contract"] = result_contract
    print(json.dumps(payload, separators=(",", ":")))
    return 0


def _worker_command(args: argparse.Namespace, source_root: Path, artifact_dir: Path | None = None):
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--source-root",
        str(source_root.resolve()),
        "--fixture-a",
        str(args.fixture_a.resolve()),
        "--fixture-b",
        str(args.fixture_b.resolve()),
        "--mode",
        args.mode,
        "--combinations",
        str(args.combinations),
        "--max-sets",
        str(args.max_sets),
    ]
    if artifact_dir:
        command.extend(("--artifact-dir", str(artifact_dir.resolve())))
    return command


def _decode_worker(completed: subprocess.CompletedProcess[str]):
    if completed.returncode:
        raise RuntimeError(
            f"benchmark worker failed ({completed.returncode})\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"benchmark worker produced no JSON\nstderr:\n{completed.stderr}")
    return json.loads(lines[-1])


def _run_timing(
    args: argparse.Namespace,
    source_root: Path,
    artifact_dir: Path | None = None,
    environment: dict | None = None,
):
    completed = subprocess.run(
        _worker_command(args, source_root, artifact_dir),
        text=True,
        capture_output=True,
        check=False,
        env=environment,
    )
    return _decode_worker(completed)


def _process_tree_rss(process) -> int:
    import psutil

    try:
        root = psutil.Process(process.pid)
        processes = [root, *root.children(recursive=True)]
        return sum(item.memory_info().rss for item in processes if item.is_running())
    except (psutil.Error, ProcessLookupError):
        return 0


def _run_memory(args: argparse.Namespace, source_root: Path):
    process = subprocess.Popen(
        _worker_command(args, source_root),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    peak_rss = 0
    samples = []
    while process.poll() is None:
        rss = _process_tree_rss(process)
        peak_rss = max(peak_rss, rss)
        samples.append(rss)
        time.sleep(0.05)
    stdout, stderr = process.communicate()
    completed = subprocess.CompletedProcess(process.args, process.returncode, stdout, stderr)
    payload = _decode_worker(completed)
    payload["peak_process_tree_rss_bytes"] = peak_rss
    payload["rss_sample_count"] = len(samples)
    return payload


def _compare_artifacts(path_a: Path, path_b: Path):
    import numpy as np

    exact_suffixes = ("node_ids", "combo_max", "combo_min")
    reference_keys = {"reference_force_step_values", "reference_force_coefficients"}
    dpf_sensitive_force_keys = {
        "force_max",
        "force_min",
        "force_combo_max",
        "force_combo_min",
        "force_fx",
        "force_fy",
        "force_fz",
    }
    comparison = {"passed": True, "dpf_sensitive_force_passed": True, "arrays": {}}
    with np.load(path_a, allow_pickle=False) as left, np.load(path_b, allow_pickle=False) as right:
        left_files = set(left.files) - reference_keys
        right_files = set(right.files) - reference_keys
        if left_files != right_files:
            return {"passed": False, "error": "artifact keys differ"}
        for key in left.files:
            if key in reference_keys:
                continue
            a = left[key]
            b = right[key]
            exact = key.endswith(exact_suffixes)
            gating = key not in dpf_sensitive_force_keys
            shape_equal = a.shape == b.shape
            dtype_equal = a.dtype == b.dtype
            values_equal = bool(np.array_equal(a, b, equal_nan=True)) if exact else bool(
                np.allclose(a, b, rtol=1e-12, atol=1e-12, equal_nan=True)
            )
            maximum = float(np.max(np.abs(a - b))) if a.size and not exact else 0.0
            passed = shape_equal and dtype_equal and values_equal
            comparison["arrays"][key] = {
                "passed": passed,
                "shape_equal": shape_equal,
                "dtype_equal": dtype_equal,
                "exact": exact,
                "gating": gating,
                "max_abs_difference": maximum,
            }
            if gating:
                comparison["passed"] = comparison["passed"] and passed
            else:
                comparison["dpf_sensitive_force_passed"] = (
                    comparison["dpf_sensitive_force_passed"] and passed
                )
    return comparison


def _validate_force_reference(path: Path):
    import numpy as np

    components = {}
    passed = True
    with np.load(path, allow_pickle=False) as artifact:
        coefficients = artifact["reference_force_coefficients"]
        step_values = artifact["reference_force_step_values"]
        for component_index, key in enumerate(("force_fx", "force_fy", "force_fz")):
            actual = artifact[key]
            expected = np.zeros_like(actual)
            for step_index in range(coefficients.shape[1]):
                expected += (
                    coefficients[:, step_index, None]
                    * step_values[step_index, component_index, None, :]
                )
            component_passed = bool(
                np.allclose(expected, actual, rtol=1e-12, atol=1e-12, equal_nan=True)
            )
            components[key] = {
                "passed": component_passed,
                "max_abs_difference": float(np.max(np.abs(expected - actual))),
            }
            passed = passed and component_passed

        fx = artifact["force_fx"]
        fy = artifact["force_fy"]
        fz = artifact["force_fz"]
        magnitude = np.sqrt(fx**2 + fy**2 + fz**2)
        derived = {
            "force_max": np.max(magnitude, axis=0),
            "force_min": np.min(magnitude, axis=0),
            "force_combo_max": np.argmax(magnitude, axis=0),
            "force_combo_min": np.argmin(magnitude, axis=0),
        }
        for key, expected in derived.items():
            actual = artifact[key]
            exact = key.startswith("force_combo_")
            item_passed = bool(np.array_equal(expected, actual)) if exact else bool(
                np.allclose(expected, actual, rtol=1e-12, atol=1e-12, equal_nan=True)
            )
            components[key] = {"passed": item_passed, "exact": exact}
            passed = passed and item_passed
    return {"passed": passed, "items": components}


def _compare_contracts(left: dict, right: dict):
    items = {}
    for key in ("combination_names", "combination_types", "units", "csv"):
        items[key] = {"passed": left.get(key) == right.get(key)}
    return {
        "passed": all(item["passed"] for item in items.values()),
        "items": items,
    }


def _summary(trials):
    totals = [trial["timings"]["total_s"] for trial in trials]
    return {
        "trials": len(trials),
        "median_total_s": statistics.median(totals),
        "min_total_s": min(totals),
        "max_total_s": max(totals),
        "stage_medians_s": {
            key: statistics.median(trial["timings"][key] for trial in trials)
            for key in trials[0]["timings"]
        },
    }


def _controller(args: argparse.Namespace) -> int:
    if args.baseline_root and not args.candidate_root:
        raise SystemExit("--candidate-root is required with --baseline-root")
    if args.candidate_root and not args.baseline_root:
        raise SystemExit("--baseline-root is required with --candidate-root")
    if not args.source_root and not args.baseline_root:
        raise SystemExit("provide --source-root or --baseline-root/--candidate-root")

    evidence = {"mode": args.mode, "seed": SEED, "trial_order": [], "trials": {}}
    if args.mode == "plasticity":
        roots = {"source": args.source_root} if args.source_root else {
            "baseline": args.baseline_root,
            "candidate": args.candidate_root,
        }
        for label, root in roots.items():
            with tempfile.TemporaryDirectory(prefix=f"mars_sc_numba_{label}_") as cache_dir:
                environment = os.environ.copy()
                environment["NUMBA_CACHE_DIR"] = cache_dir
                evidence["trials"][label] = [
                    _run_timing(args, root, environment=environment)
                ]
    elif args.source_root:
        if args.mode == "memory":
            evidence["trials"]["source"] = [
                _run_memory(args, args.source_root) for _ in range(args.memory_trials)
            ]
        else:
            evidence["trials"]["source"] = [_run_timing(args, args.source_root, args.artifact_dir)]
    elif args.mode == "correctness":
        with tempfile.TemporaryDirectory(prefix="mars_sc_correctness_") as temp_dir:
            temp_root = Path(temp_dir)
            baseline = _run_timing(args, args.baseline_root, temp_root / "baseline")
            candidate = _run_timing(args, args.candidate_root, temp_root / "candidate")
            evidence["trials"] = {"baseline": [baseline], "candidate": [candidate]}
            evidence["correctness"] = _compare_artifacts(
                Path(baseline["artifact"]), Path(candidate["artifact"])
            )
            evidence["result_contract"] = _compare_contracts(
                baseline["result_contract"],
                candidate["result_contract"],
            )
            evidence["force_reference"] = _validate_force_reference(
                Path(candidate["artifact"])
            )
            evidence["correctness"]["passed"] = (
                evidence["correctness"]["passed"]
                and evidence["result_contract"]["passed"]
                and evidence["force_reference"]["passed"]
            )
    elif args.mode == "timing":
        _run_timing(args, args.baseline_root)
        _run_timing(args, args.candidate_root)
        baseline_trials = []
        candidate_trials = []
        for _ in range(args.abba_blocks):
            for label, root in (
                ("baseline", args.baseline_root),
                ("candidate", args.candidate_root),
                ("candidate", args.candidate_root),
                ("baseline", args.baseline_root),
            ):
                evidence["trial_order"].append(label)
                trial = _run_timing(args, root)
                (baseline_trials if label == "baseline" else candidate_trials).append(trial)
        evidence["trials"] = {"baseline": baseline_trials, "candidate": candidate_trials}
        evidence["summary"] = {
            "baseline": _summary(baseline_trials),
            "candidate": _summary(candidate_trials),
        }
        evidence["summary"]["speedup"] = (
            evidence["summary"]["baseline"]["median_total_s"]
            / evidence["summary"]["candidate"]["median_total_s"]
        )
    else:
        baseline_trials = []
        candidate_trials = []
        for index in range(args.memory_trials * 2):
            label = "baseline" if index % 2 == 0 else "candidate"
            root = args.baseline_root if label == "baseline" else args.candidate_root
            evidence["trial_order"].append(label)
            trial = _run_memory(args, root)
            (baseline_trials if label == "baseline" else candidate_trials).append(trial)
        evidence["trials"] = {"baseline": baseline_trials, "candidate": candidate_trials}
        evidence["summary"] = {
            label: {
                "median_peak_process_tree_rss_bytes": statistics.median(
                    trial["peak_process_tree_rss_bytes"] for trial in trials
                )
            }
            for label, trials in evidence["trials"].items()
        }

    rendered = json.dumps(evidence, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


def main() -> int:
    args = _parser().parse_args()
    return _worker(args) if args.worker else _controller(args)


if __name__ == "__main__":
    raise SystemExit(main())
