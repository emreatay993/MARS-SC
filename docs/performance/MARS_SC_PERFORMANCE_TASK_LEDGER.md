# MARS-SC Performance Task Ledger

Baseline: `9f2ca01a44c9226cadf1d4085212198c2bca8295`

Unrelated preserved path: `docs/cdb_import_ui_mockups.html`

| Task | Owner | Dependencies | Status | Allowed paths | Proving check | Evidence |
|---|---|---|---|---|---|---|
| T01 Bootstrap and continuity | root | none | complete | `docs/performance/` | plan and ledger exist; baseline/status recorded | baseline HEAD, dirty path, runtime, and fixture metadata recorded in plan/ledger |
| T02 Test isolation and benchmark | root | T01 | complete | focused progress/loader tests, `scripts/benchmark_mars_sc.py`, performance docs | full collection succeeds; benchmark smoke and baseline run | 204 passed, 11 skipped; heavy baseline 120.347 s and 1,409,691,648-byte peak RSS |
| T03 DPF reader boundary | root | T02 | complete | `src/file_io/dpf_reader.py`, three solver engines, focused reader tests | exact real-RST boundary equivalence | vectorized coordinates, cached displacement preflight, and shared fast alignment shipped; stress batching, metadata-first stress units, metadata-only force validation, force fast alignment, and direct force preallocation were withdrawn after full-workload force-equivalence failures at one cancellation-sensitive node |
| T04 Packed numerical engines | root | T03 | complete | `src/core/data_models.py`, three solver engines, focused tests | legacy equivalence, tie, block, chunk, low-memory checks | 43 focused packed/chunk tests passed; 20-combination/36-set baseline-candidate artifact comparison passed at 1e-12 with exact indices; preliminary full candidate 58.654 s versus 120.347 s baseline; full suite 212 passed, 11 skipped |
| T05 Export RAM | root | T04 | complete | `src/file_io/exporters.py`, exporter tests | semantic CSV equality and scaled timing/RSS | shear planes are now reduced sequentially with only 1-D statistics retained; 16 exporter tests pass with exact values, indices, names, and column order; exploratory 200 x 50,000 evidence was 380 MiB to 230 MiB |
| T06 Plasticity | root | T04 | complete | `requirements.txt`, plasticity/executor code and tests | compiled/fallback equivalence; first/warm timing | Numba 0.66.0/llvmlite 0.48.0 pinned and installed; executor corrects in 64 MiB combo blocks without full temperature/strain matrices and preserves first NaN/Inf governing behavior; 40 focused tests pass; 2,105,600-entry first/warm timings: Neuber 3.546/0.124 s, Glinka 3.009/0.136 s |
| T07 Acceptance evidence | root | T02-T06 | complete | performance evidence/report; measured contingency only | full tests, real-RST comparison, >=1.50x gate, lower RSS | 215 passed/11 skipped; all 25 raw arrays, result/CSV contracts, and same-input legacy force reference pass; 124.016 s to 58.979 s median (2.103x); median peak RSS 1,401,552,896 to 1,332,027,392 bytes (66.3 MiB saved) |
| T08 Independent review | read-only reviewer | T07 | complete | none | no unresolved correctness/evidence/claim findings | independent recheck confirmed 215 passed/11 skipped, raw hashes and arithmetic, NaN/Inf semantics, result contracts, DPF exception, and acceptance claims; no blocking findings remain |
| T09 Metadata-only RST startup | root | T08 | complete | DPF reader, solve orchestration, benchmark/tests, performance evidence | metadata-only load speed/RSS; locked real-RST equivalence | two-RST median 1.869 s to 0.167 s (11.21x), 33.0 MiB lower post-load RSS increment; all 25 arrays, contracts, force references, and 216-test suite pass |

## Re-entry Checklist

After context compaction: read the complete plan and this ledger, inspect HEAD/status/diff-stat and the active task diff, then update this table before resuming or delegating.
