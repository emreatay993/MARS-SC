# MARS-SC Performance Report

## Result

The optimized source runtime is **2.103x faster end to end** on the locked real-RST workload. Median elapsed time fell from **124.016 s to 58.979 s**, a **52.4% reduction**.

Median peak process-tree RSS fell from **1,401,552,896 bytes to 1,332,027,392 bytes**, saving **69,525,504 bytes (66.3 MiB)**. This is a 5.0% proportional reduction and passes the predeclared noticeable-RAM threshold because it exceeds 50 MiB.

All acceptance gates passed:

- Headline speedup: 2.103x, required at least 1.50x.
- RAM: candidate median peak is lower and saves more than 50 MiB.
- Stage guard: no median phase regressed by more than 5%.
- Correctness: all 25 raw baseline/candidate arrays passed in the final run; node IDs and governing indices were exact, floats passed `rtol=1e-12`, `atol=1e-12`, `equal_nan=True`, and the independent same-input legacy force reference passed.
- Result contract: combination names/types, units, expected CSV row counts, and exact CSV column order matched.
- Regression suite: 216 passed, 11 existing optional real-RST tests skipped, zero failures or collection errors.

## Follow-up: metadata-only RST startup

Initial file discovery now uses `result_info.available_results` for result availability
and units instead of evaluating unscoped first-set stress, ENF, and displacement fields.
One warm-up per root followed by three ABBA blocks produced six fresh-process trials
per root while loading both locked RST files:

| Startup metric | Baseline | Candidate | Change |
|---|---:|---:|---:|
| Median two-RST load | 1.869 s | 0.167 s | **11.21x faster** |
| Median post-load process-tree RSS increment | 113,625,088 B | 79,067,136 B | **33.0 MiB lower** |

Set counts and stress/force/displacement units remained exact on SYS-1 and SYS-29.
When nodal forces are selected, MARS-SC replays the historical DPF probe order once
at solve preparation because DPF 0.10.1 ENF reduction is cancellation-sensitive.
The full 401-set, 1,000-combination correctness rerun passed all 25 arrays, result/CSV
contracts, the raw force diagnostic, and the same-input legacy force reference.

Further cost checks did not justify more code. DPF's scoped-mesh coordinate operator
took about 1.67 s on SYS-1 versus less than 1 ms for the existing cached/vectorized
coordinate path, so it was rejected. The duplicated ENF validation/preload sequence
also remains unchanged because altering it previously changed the sensitive node-1 result.

## Follow-up: 5.1 GB static RST

The original SYS-1 and SYS-29 fixtures each contain 10,528 nodes and 1,692 elements.
To test a materially larger result file, the follow-up replaced SYS-29 with static
SYS-30: 5,104,992,256 bytes, 2,001 result sets, 10,528 nodes, and 1,692 elements.
This is a file/result-set scaling check, not a larger-mesh check.

The reduced timing workload used 100 combinations, the first 40 sets from each RST,
one warm-up per root, and two ABBA blocks (four measured fresh processes per root).
Peak RSS was measured separately with two alternating sampled trials per root:

| Large-RST metric | Baseline | Candidate | Change |
|---|---:|---:|---:|
| Metadata-only startup | 2.026 s | 0.383 s | **5.29x faster** |
| Reduced end-to-end workflow | 16.279 s | 13.295 s | **1.22x faster; 18.3% lower** |
| Median peak process-tree RSS | 678,535,168 B | 679,141,376 B | **0.58 MiB higher; effectively flat** |

The smaller combination count makes this lane read-dominated, so its end-to-end
speedup is intentionally lower than the 1,000-combination headline workload. A separate
20-combination, 10-set correctness run passed the result/CSV contracts and the exact
same-input legacy force reference. Five raw cross-process ENF arrays differed because
of the documented DPF 0.10.1 reduction-order sensitivity; they are retained as a
non-gating diagnostic rather than presented as code differences.

## Locked workload

- Baseline: commit `9f2ca01a44c9226cadf1d4085212198c2bca8295` in a detached worktree.
- Fixtures: `SYS-1/MECH/file.rst` (200 sets, 449,249,280 bytes) and `SYS-29/MECH/file.rst` (201 sets, 512,688,128 bytes) from the user's OneDrive Desktop `J/ANSYS/Benchmark/MARS` tree.
- Scope: all 10,528 nodes, all 401 result sets active, 1,000 deterministic combinations from seed `20260816`.
- Workflow: von Mises stress envelope, global nodal-force components/magnitude, Cartesian deformation components/magnitude, and all three CSV exports.
- Runtime: Python 3.11.6, NumPy 2.4.1, pandas 2.2.3, DPF 0.10.1, psutil 7.2.1.
- Cache statement: fresh subprocesses with a warm filesystem cache. This report makes no cold-disk claim.

The measured virtual environment differs from two repository pins: `requirements.txt` specifies NumPy 2.4.0 and pandas 2.3.3. The evidence therefore applies to the versions above; a fresh requirements-only environment should be rebenchmarked before treating these exact timings as portable.

## Timing evidence

One unmeasured warm-up was run per root, followed by three ABBA blocks. This produced six measured fresh-process trials per root without RSS sampling.

| Phase | Baseline median (s) | Candidate median (s) | Change | Speedup |
|---|---:|---:|---:|---:|
| Metadata load | 1.832 | 1.844 | +0.6% | 0.99x |
| Stress read | 18.423 | 18.475 | +0.3% | 1.00x |
| Stress calculation | 31.636 | 1.443 | -95.4% | 21.93x |
| Force read | 30.200 | 31.063 | +2.9% | 0.97x |
| Force calculation | 15.106 | 0.714 | -95.3% | 21.15x |
| Deformation read | 6.629 | 2.632 | -60.3% | 2.52x |
| Deformation calculation | 16.751 | 0.451 | -97.3% | 37.14x |
| Export | 1.766 | 1.759 | -0.4% | 1.00x |
| **End to end** | **124.016** | **58.979** | **-52.4%** | **2.103x** |

Baseline total trials were 123.037, 120.769, 117.161, 124.994, 137.877, and 157.888 seconds. Candidate trials were 57.697, 55.199, 60.260, 55.754, 61.049, and 67.546 seconds.

## RAM evidence

Three alternating subprocess trials per root sampled aggregate worker-plus-DPF-descendant RSS every 50 ms. Sampling affects wall time, so these trial times were excluded from the speed gate.

| Root | Peak RSS trials (bytes) | Median (bytes) |
|---|---|---:|
| Baseline | 1,420,664,832; 1,400,586,240; 1,401,552,896 | 1,401,552,896 |
| Candidate | 1,332,027,392; 1,343,684,608; 1,293,459,456 | 1,332,027,392 |

The measured reduction is 69,525,504 bytes, or 66.3 MiB.

## Correctness evidence

The full workload was recomputed in the frozen baseline and candidate, then serialized and compared outside timed regions. All 25 raw arrays passed in the final run. This covers:

- node IDs, ordering, coordinates, shapes, dtypes, units, and NaN/Inf behavior;
- complete stress, force-component, and deformation-component combination matrices;
- maximum/minimum envelopes;
- exact governing-combination indices.
- exact combination names/types, units, expected CSV row counts, and CSV column order.

The largest absolute float difference was `3.2741809263825417e-11`. It occurs at a large-magnitude value and passes the locked combined relative/absolute tolerance. The force preload deliberately retains the legacy DPF read/retain sequence before packing because a direct-preallocation experiment changed one cancellation-sensitive node in the full real-RST gate.

DPF 0.10.1's parallel element-nodal-force reduction can vary across fresh processes at cancellation-sensitive node ID 1. The raw cross-root force comparison is preserved as a diagnostic. To keep the code regression gate independent of that DPF noise, the candidate's extracted step forces are also accumulated in the legacy step order outside timed regions. FX/FY/FZ pass at `1e-12`, and the derived magnitude governing indices are exact. This reference check does not affect the headline timing or RSS trials.

## What changed

- Combination coefficients and per-step results use contiguous packed NumPy layouts.
- Stress, force, and deformation calculations use RAM-bounded matrix multiplication instead of Python combination loops.
- Principal stress uses batched `numpy.linalg.eigvalsh`.
- Force/deformation magnitude envelopes are reduced in node blocks instead of retaining another full combination matrix.
- Mesh coordinates and displacement availability are cached; coordinate alignment is vectorized.
- Force shear planes are reduced sequentially, avoiding three simultaneous full shear matrices.
- Plasticity correction works in 64 MiB combination blocks without a full temperature tile or full plastic-strain matrix.
- Numba 0.66.0 and llvmlite 0.48.0 are pinned while the existing no-Numba fallback remains available.

Stress batching, metadata-only force preflight, metadata-first stress-unit lookup, and direct force-cache preallocation were measured and withdrawn because the real DPF force reduction is sensitive to operator/read pacing. Exact engineering outputs took priority over additional I/O gains.

## Plasticity evidence

Plasticity is reported separately and does not inflate the headline speedup. With a unique empty Numba cache and 2,105,600 entries:

| Method | First call (s) | Warm median (s) |
|---|---:|---:|
| Neuber | 3.546 | 0.124 |
| Glinka | 3.009 | 0.136 |

Numba was confirmed active. The first call includes compilation; warm medians use three subsequent calls in the same process.

## Evidence files

- `evidence/mars_sc_final.json`: acceptance summary and SHA-256 manifest.
- `evidence/mars_sc_timing.json`: all timing trials and environment/fixture metadata.
- `evidence/mars_sc_memory.json`: all RSS trials and sample counts.
- `evidence/mars_sc_correctness.json`: per-array comparison results.
- `evidence/mars_sc_plasticity.json`: first-call and warmed Numba trials.
- `evidence/mars_sc_startup.json`: six-trial-per-root ABBA startup timing/RSS evidence.
- `evidence/mars_sc_startup_correctness.json`: full post-change correctness rerun.
- `evidence/mars_sc_large_rst_startup.json`: four-trial-per-root large-RST startup evidence.
- `evidence/mars_sc_large_rst_timing.json`: reduced large-RST timing trials.
- `evidence/mars_sc_large_rst_memory.json`: two sampled large-RST RSS trials per root.
- `evidence/mars_sc_large_rst_correctness.json`: reduced large-RST correctness comparison.
- `scripts/benchmark_mars_sc.py`: reproducible controller/worker benchmark.
