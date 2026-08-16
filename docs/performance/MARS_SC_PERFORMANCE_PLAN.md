# MARS-SC Performance Optimization Plan

## Objective

Make the source application measurably faster and more RAM-efficient across RST metadata loading, DPF result reads, stress/force/deformation combination calculations, CSV export, and Neuber/Glinka plasticity without changing engineering-result contracts.

The performance baseline is commit `9f2ca01a44c9226cadf1d4085212198c2bca8295`. That commit already reuses the background loader's RST reader, so that earlier improvement is not part of this plan's claimed gain.

## Locked Workload and Acceptance Contract

- Source runtime only: Python 3.11.6, NumPy 2.4.1, pandas 2.2.3, DPF 0.10.1, psutil 7.2.1.
- Fixtures:
  - `C:\Users\emre_\OneDrive\Desktop\J\ANSYS\Benchmark\MARS\Benchmark_v1_files\dp0\SYS-1\MECH\file.rst` (449,249,280 bytes, 200 sets).
  - `C:\Users\emre_\OneDrive\Desktop\J\ANSYS\Benchmark\MARS\Benchmark_v1_files\dp0\SYS-29\MECH\file.rst` (512,688,128 bytes, 201 sets).
- All 10,528 nodes, all 401 result-set columns active, and 1,000 deterministic combinations generated from seed `20260816`.
- Headline workflow: von Mises stress envelope, global nodal-force components/magnitude, Cartesian deformation components/magnitude, then the three exports.
- Optional plasticity is measured separately so it cannot inflate the headline speedup.
- Required headline speed: `baseline median / candidate median >= 1.50`.
- Candidate median process-tree peak RSS must be lower. It is called a noticeable RAM gain only at 15% or 50 MiB reduction.
- No stage may regress by more than 5%.
- Exact equality: node IDs/order, combination names/order, units, result shapes/dtypes, NaN/Inf locations, CSV columns/order, and governing-combination indices.
- Floating values: `rtol=1e-12`, `atol=1e-12`, `equal_nan=True`.
- Evidence is fresh-process but warm-filesystem-cache evidence; no cold-disk claim is allowed.

Measured acceptance exception: DPF 0.10.1's parallel element-nodal-force reduction
can vary across fresh processes at cancellation-sensitive node ID 1. The raw cross-root
force comparison remains recorded. The force code gate additionally compares the
candidate components against legacy step-order accumulation using that same run's
extracted force inputs, then verifies the magnitude envelopes and governing indices.
This reference work is outside all timing and RSS measurements.

## Execution Order

### T01 Bootstrap and Continuity

Maintain this document and `MARS_SC_PERFORMANCE_TASK_LEDGER.md`. One implementation owner edits sequentially; no concurrent editing agents. Preserve the unrelated untracked `docs/cdb_import_ui_mockups.html`.

### T02 Test Isolation and Benchmark Foundation

- Remove module-level PyQt `sys.modules` pollution from progress tests and keep Qt substitutions local to their test module.
- Add `scripts/benchmark_mars_sc.py` with controller/worker operation, selectable source root, correctness/timing/memory/plasticity modes, stage timing, and worker-plus-DPF-descendant RSS sampling.
- Generate the coefficient matrix in memory. Exports go to a non-OneDrive temporary directory.
- Benchmark baseline through a detached worktree at `9f2ca01`; the controller itself must not import production modules.
- Correctness artifacts are written outside timed sections. Timing and RSS passes are separate.

### T03 DPF Reader Boundary

- Use one private alignment implementation for stress, force, displacement, and coordinates: exact-order return, verified `searchsorted` for sorted IDs, zero-fill missing IDs, dictionary fallback only for unsorted/incompatible cases.
- Cache mesh IDs/coordinates/sortedness and availability/unit metadata.
- Measure metadata-first result lookup and retain real-field validation wherever the full correctness gate requires its DPF operator sequence.
- Read stress in fixed batches of 16 with single-set fallback for a failed batch, preserving set/node order and step-specific errors.
- Use the same batching inside the node-chunk path.

Measured implementation exception: 16-set stress batching was bitwise-equivalent for
stress itself, but changing the preceding DPF operator sequence changed the legacy
element-nodal-force reduction at one cancellation-sensitive node. Full-workload proof
also required the legacy stress-unit field probe, all-set force validation, force
dictionary alignment, and the force preload read/retain pacing before a one-time pack.
Because exact force values are locked above, those paths retain their legacy order;
vectorized coordinates, cached displacement preflight, final packed numerical caches,
and the calculation/export optimizations remain enabled.

Post-acceptance follow-up: initial file discovery is metadata-only. If nodal forces are
selected, the legacy unscoped probe order is replayed once immediately before analysis
execution, preserving the force contract while avoiding those reads for users who do
not request forces.

### T04 Packed Numerical Engines

- Add `CombinationTableData.get_active_step_matrix()` returning ordered `(analysis, set)` keys and a contiguous `float64` coefficient matrix.
- Replace per-step dictionaries containing repeated node IDs with one node-ID array and packed component caches.
- Use blocked matrix multiplication with a 64 MiB temporary-work target.
- Batch principal-stress eigensolves across nodes.
- Keep full force/deformation component arrays, but stream magnitude envelopes instead of retaining `magnitude_all`.
- Preserve first-index tie behavior with strict streamed comparisons.
- Update standard, chunked, single-combination, and single-node paths plus memory estimates.
- If BLAS changes governing indices, use step-major vectorized accumulation for the affected kernel to preserve the original summation order.

### T05 Export RAM

Reduce XY, XZ, and YZ force-shear variants sequentially, retaining only their one-dimensional reductions. Preserve the existing CSV column order and Pandas writer.

### T06 Plasticity

- Add `numba==0.66.0` and `llvmlite==0.48.0`; retain the no-Numba fallback.
- Correct Neuber/Glinka results in combination blocks and remove full-table temperature tiles and plastic-strain matrices.
- Preserve equations, tolerances, plateau behavior, interpolation, elastic metadata, and governing plastic strain.
- Do not warm JIT kernels at application startup. Measure isolated first-call and warmed behavior with unique `NUMBA_CACHE_DIR` values.

### T07 Verification and Evidence

- Focused checks first, then full pytest with zero failures/collection errors. Existing real-RST skips must be listed.
- Compare complete baseline/candidate arrays and result/CSV contracts outside timed regions; validate candidate force calculations against same-input legacy accumulation to isolate DPF cross-process reduction noise.
- Timing: one unmeasured warm-up per root, then three ABBA blocks (six measured subprocess trials per root) without RSS sampling.
- Memory: three alternating subprocess trials per root, sampling the whole process tree every 50 ms. Memory-trial wall times do not enter the speed gate.
- Publish a consolidated evidence manifest to `docs/performance/evidence/mars_sc_final.json`, the raw lane files beside it, and the readable report to `docs/performance/MARS_SC_PERFORMANCE_REPORT.md`.
- If the 1.50x gate fails, profile the candidate. Add 16-set force/displacement batch readers only when that phase is the largest remaining owner and an isolated real-RST probe improves it by at least 20% without increasing RSS. Otherwise report the failed gate honestly.

### T08 Independent Review

A separate read-only reviewer checks the full diff, ledger, focused/full tests, real-RST equivalence, raw timing/RSS trials, calculations, and report claims. Remediation returns to the single implementation owner; only invalidated checks are rerun before re-review.

## Non-goals

- GPU, multiprocessing, memmaps, disk cache services, new result schemas, or UI redesign.
- PyInstaller/frozen-runtime work.
- IBG or cylindrical-history optimization.
- Batched force/displacement reads unless the measured contingency triggers.
- Cold-cache claims.

## Mandatory Re-entry Rule

After any context compaction, before resuming or delegating, the orchestrator must:

1. Re-read this plan completely.
2. Re-read `MARS_SC_PERFORMANCE_TASK_LEDGER.md` completely.
3. Inspect `git rev-parse HEAD`, `git status --short`, and `git diff --stat`.
4. Inspect the active task's complete diff.
5. Reconcile repository state with the ledger before assigning or changing work.
