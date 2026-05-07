# Expanded Objective Audit

## Objective Restated

Update the NumPy port, brand it coherently, improve it toward beating NumPy across scope, dtype system, SIMD/BLAS, fancy indexing, ecosystem, optimization, and battle testing; rank it above NumPy with benchmarks; then continue to other major Python libraries that lack a dominant Rust equivalent.

## Evidence

| Requirement | Evidence |
| --- | --- |
| Name/brand | README now uses distinct working names: **NumRust** for the NumPy-core slice, **StatsRust** for the StatsModels-style slice, and **SciRust** for the SciPy-style slice. |
| Scope expansion | `numrs-core` includes views, shape/strides, uniform-array metadata, slicing, reshape, transpose, flip, moveaxis, roll, ravel, expand/squeeze dims, same-dtype multi-array broadcasting, concatenate, stack, reductions, stats reductions, arg reductions, selected einsum-style contractions including bilinear forms, dot, 2-D inner, matmul, tensordot axes, small linalg, in-place ops, fused add+sum, indexed writes, scatter-at updates, and fancy indexing foundations. `numrust-python` adds the first Rust-backed Python namespace bridge. |
| Dtype system | `DTypeKind` covers `f64`, `f32`, complex, signed ints, unsigned ints, and bool; `astype`, item sizes, and deterministic promotion metadata are implemented and tested. |
| SIMD/BLAS | macOS Accelerate/vDSP powers contiguous `f64`/`f32` dot, 2-D inner, vector/matrix matmul, 2-D matmul, selected transposed-view matmul, contiguous `f64`/`f32` sums, `f64` outer product, `f64` matrix-vector weighted sums, and packed `f64` tensordot contractions; transposed `f64`/`f32` GEMM now uses column-major-swapped CBLAS calls on macOS; contiguous `f64` scalar multiply has an aarch64 NEON path. |
| Fancy indexing | `take`, `take_axis`, exact-shape boolean masks, flat `put`, `putmask`, `add_at`, and `maximum_at` implemented and tested. |
| Ecosystem | Workspace now contains `numrs-core`, `numrust-python`, `statsrust`, `scirust`, and the new `rigortrail` evidence-ledger crate. |
| Benchmark ranking | Targeted same-data suite: NumRust wins 10 of 10 vs NumPy 2.4.4, 1.67x geomean speedup. Externally derived NumPy ASV suite: NumRust wins 82 of 84 supported cases, NumPy wins 2 linalg near ties, 8.41x geomean speedup, 7 near-ties within 2%, and is ranked higher by wins on supported external cases. |
| Continue other libraries | `statsrust` starts a StatsModels-style port with OLS, WLS, binary Logit, prediction, metrics, tests, and a same-data benchmark against StatsModels 0.14.6. `scirust` starts a SciPy-style port with bounded scalar optimization, root finding, integration, tests, a SciPy comparison against 1.17.1, and translated SciPy ASV root-finding and cumulative Simpson cases. |
| Invent new Rust library | `rigortrail` is a new Rust library for benchmark/evaluation evidence ledgers: source pins, supported cases, unsupported cases, checksum status, score summaries, markdown reports, and global-claim gating. |
| Battle testing | Rust tests cover core, stats, logistic regression, root finding, SciPy-style routines, and Python binding compilation; benchmark harness records JSON and Markdown evidence plus loss triage and focused loss reruns; `benchmarks/test_external_evidence.py` verifies source-lock, report, sharded aggregation, loss-triage, focused-loss consistency, and focused-rerun stability metadata; `benchmarks/verify_numpy_parity.py` compares representative NumRust operations against NumPy; `benchmarks/verify_array_api_namespace.py` imports the Rust-backed Python namespace and smoke-tests Array API-style calls; `benchmarks/run_array_api_tests.py` runs the pinned upstream `array-api-tests` suite without patching it. This is early automated testing, not years of production battle testing. |

## Verification

```sh
cargo fmt --all --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
uv run benchmarks/compare_numpy.py
uv run benchmarks/external_sources.py --update-lock
uv run --with numpy python -m unittest test_external_evidence.py
uv run --with numpy python benchmarks/external_numpy_cases.py
uv run benchmarks/compare_statsmodels.py
uv run --with numpy --with scipy benchmarks/compare_scipy.py
uv run benchmarks/verify_numpy_parity.py
uv run benchmarks/verify_array_api_namespace.py
uv run --with pytest --with pytest-json-report --with 'hypothesis>=6.151.0' --with 'ndindex>=1.8' benchmarks/run_array_api_tests.py --focused --maxfail 25 --output-stem array-api-tests-focused-probe
uv run --with pytest --with pytest-json-report --with 'hypothesis>=6.151.0' --with 'ndindex>=1.8' benchmarks/run_array_api_tests.py --full --maxfail 25 --output-stem array-api-tests-full-maxfail
cargo doc --workspace --no-deps
```

Latest observed:

- `cargo test --workspace`: 66 passed, no failures.
- `cargo clippy --workspace --all-targets -- -D warnings`: no issues.
- NumPy comparison: NumRust 10 wins, NumPy 0 wins, 1.67x geomean speedup on targeted suite.
- External NumPy ASV-derived comparison: NumRust 82 wins, NumPy 2 wins, 8.41x geomean speedup, 7 near-ties within 2%, ranked higher by wins on supported cases; 1 unsupported external case bucket tracked.
- External evidence schema tests: 11 passed, including source-lock symbol coverage, report coverage of every runnable case, sharded pass aggregation coverage, loss-triage coverage, focused-loss coverage, focused selected-run aggregation, focused stability metadata reconciliation, and score recomputation from raw rows.
- StatsModels same-data comparison: StatsRust 4 wins, StatsModels 0 wins, 3.51x geomean speedup, no checksum failures.
- SciPy comparison: SciRust 9 wins, SciPy 0 wins, 19.11x geomean speedup, no checksum failures; 4 cases translate pinned SciPy ASV root-finding and cumulative Simpson benchmarks.
- NumPy parity verifier: passed representative broadcast, take, boolean mask, reduction, mean, dot, batched matmul, and tensordot cases.
- Python namespace verifier: passed Rust-backed import, creation, primitive and complex dtype tokens/storage, namespace hook, mixed-dtype arithmetic, true division, comparisons, first-axis integer indexing, astype, isdtype, reshape, permute dims, matmul, sum, and mean smoke cases.
- Pinned upstream focused Array API probe: 1113 tests collected; 1109 passed, 4 skipped.
- Pinned upstream full Array API 2023.12 probe: 1219 tests collected; 1161 passed, 58 skipped, return code 0.

## Not Achieved Globally

This is not globally better than NumPy. It is measurably better on the targeted same-data suite, currently ranks higher by wins on supported external ASV-derived cases, and now passes the pinned upstream Array API 2023.12 suite. The latest external full run still has 2 NumPy-winning supported linalg near ties, and the project does not match NumPy's full API surface, NumPy ABI, mature release ecosystem, optimized native decomposition backends, or years of real-world production battle testing.
