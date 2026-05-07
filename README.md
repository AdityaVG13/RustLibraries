# FromPythonToRust

Flagship-first porting lab for rebuilding top Python library ideas in pure Rust.

Working names: **NumRust** for the NumPy-core slice, **StatsRust** for the StatsModels-style slice, and **SciRust** for the SciPy-style slice. Core crate: `numrs-core`.

Current flagship: `numrs-core`, a Rust-first NumPy-core foundation. It targets typed n-dimensional arrays, shape/stride metadata, uniform-array metadata, zero-copy views, NumPy-style slicing, reshape, transpose, broadcasting, elementwise kernels, reductions, 2-D dot products, NumPy-style matrix multiplication, and tensor contractions.

## Status

This is not a full NumPy replacement yet. The v0 slice is deliberately narrow and tested. The architecture is aimed above NumPy on Rust-native strengths: explicit ownership, checked layout contracts, zero-copy view transforms, uniform-array plans, broadcast plans that avoid input materialization, and fast paths for contiguous kernels.

## Progress TODO

Backfilled items are completed work imported from the pre-GitHub build history. New work should update this list in the same commit as the implementation or benchmark evidence.

### Backfilled Done

- [x] 2026-05-07 backfilled: Create a Rust workspace for NumRust, StatsRust, SciRust, RigorTrail, and the Python bridge. Evidence: `Cargo.toml`, `crates/*`.
- [x] 2026-05-07 backfilled: Implement NumRust typed arrays, shape/stride metadata, views, slicing, reshape, transpose, broadcasting, reductions, selected linalg, matmul, tensordot, and indexed writes. Evidence: `crates/numrs-core`, `crates/numrs-core/tests/numrs_core.rs`.
- [x] 2026-05-07 backfilled: Add primitive dtype metadata, casts, promotion records, bool/int/float/complex Python namespace tokens, and Array API smoke coverage. Evidence: `crates/numrs-core/src/dtype.rs`, `python/numrust`.
- [x] 2026-05-07 backfilled: Add Accelerate/vDSP and BLAS-backed fast paths for supported contiguous reductions, dot, matmul, inner, selected contractions, and NEON scalar multiply. Evidence: `docs/performance.md`, `crates/numrs-core/src/blas.rs`.
- [x] 2026-05-07 backfilled: Implement fancy indexing primitives: `take`, `take_axis`, boolean masks, `put`, `putmask`, `add_at`, and `maximum_at`. Evidence: `crates/numrs-core/tests/numrs_core.rs`.
- [x] 2026-05-07 backfilled: Add a PyO3-backed `numrust` namespace and verify import, dtype, arithmetic, indexing, reshape, matmul, sum, and mean smoke cases. Evidence: `crates/numrust-python`, `benchmarks/verify_array_api_namespace.py`.
- [x] 2026-05-07 backfilled: Pin NumPy ASV, Array API, SciPy, and StatsModels source references with hash verification. Evidence: `benchmark-results/external-source-lock.json`, `benchmarks/external_sources.py`.
- [x] 2026-05-07 backfilled: Add external NumPy ASV-derived benchmarks with real NumPy comparison, sharded pass aggregation, loss triage, focused loss reruns, and stability metadata. Evidence: `benchmarks/external_numpy_cases.py`, `benchmark-results/external-numpy-asv-inspired.md`.
- [x] 2026-05-07 backfilled: Run pinned upstream Array API probes without patching upstream tests. Evidence: `benchmark-results/array-api-tests-focused-probe.md`, `benchmark-results/array-api-tests-full-maxfail.md`.
- [x] 2026-05-07 backfilled: Start StatsRust and SciRust slices with tests and comparison harnesses. Evidence: `crates/statsrust`, `crates/scirust`, `benchmark-results/*statsmodels*`, `benchmark-results/*scipy*`.
- [x] 2026-05-07 backfilled: Invent and implement RigorTrail for evidence-ledger validation and claim gating. Evidence: `crates/rigortrail`, `docs/rigortrail.md`.
- [x] 2026-05-07 backfilled: Pass Rust format, clippy, workspace tests, Python benchmark schema tests, and source-lock verification before GitHub import. Evidence: `docs/completion-audit.md`.

### Open

- [ ] 2026-05-07: Add GitHub Actions for format, clippy, tests, Python evidence schema checks, and source-lock verification. Target: `.github/workflows/ci.yml`.
- [ ] 2026-05-07: Add more externally derived NumPy ASV cases without filtering out losses. Target: `benchmarks/external_numpy_cases.py`.
- [ ] 2026-05-07: Convert only benchmark-proven wins into new SIMD/BLAS/layout-specialized kernels. Target: `docs/performance.md`.
- [ ] 2026-05-07: Grow StatsRust and SciRust with more externally derived benchmarks. Target: `benchmarks/compare_statsmodels.py`, `benchmarks/compare_scipy.py`.
- [ ] 2026-05-07: Add publishing metadata, CI release checks, crate docs polish, and versioning policy before any crates.io release. Target: `Cargo.toml`, crate manifests.

## Commands

```sh
cargo fmt --all --check
cargo test --workspace
cargo run -p numrs-core --example basic
cargo run --release -p numrs-core --example microbench
uv run benchmarks/compare_numpy.py
uv run benchmarks/external_sources.py --update-lock
uv run --with numpy python benchmarks/external_numpy_cases.py
uv run benchmarks/verify_array_api_namespace.py
uv run --with pytest --with pytest-json-report --with 'hypothesis>=6.151.0' --with 'ndindex>=1.8' benchmarks/run_array_api_tests.py --focused --maxfail 25 --output-stem array-api-tests-focused-probe
uv run --with pytest --with pytest-json-report --with 'hypothesis>=6.151.0' --with 'ndindex>=1.8' benchmarks/run_array_api_tests.py --full --maxfail 25 --output-stem array-api-tests-full-maxfail
uv run benchmarks/compare_statsmodels.py
uv run --with numpy --with scipy benchmarks/compare_scipy.py
```

## Layout

- `crates/numrs-core`: pure Rust NumPy-core flagship crate.
- `crates/numrust-python`: PyO3-backed Python namespace bridge for NumRust.
- `crates/statsrust`: StatsModels-style statistics crate.
- `crates/scirust`: SciPy-style numerical routines crate.
- `crates/rigortrail`: new benchmark/evaluation evidence-ledger crate.
- `docs/research.md`: research notes and source map.
- `docs/novel-library-research.md`: research notes for the new invented crate.
- `docs/architecture.md`: architecture decisions and optimization roadmap.
- `docs/performance.md`: current fast paths and benchmark hook.
- `docs/external-evidence.md`: pinned online benchmark and conformance sources.
- `docs/python-array-api.md`: Python namespace bridge and conformance roadmap.
- `docs/statsmodels-port.md`: second library port notes.
- `docs/scipy-port.md`: third library port notes.
- `docs/rigortrail.md`: RigorTrail design and usage notes.
- `benchmark-results/numrust-vs-numpy.md`: current NumRust vs NumPy evidence.
- `benchmark-results/external-numpy-asv-inspired.md`: externally derived NumPy ASV evidence.
- `benchmark-results/array-api-tests-focused-probe.md`: pinned upstream focused Array API probe, currently 1109 passed, 4 skipped, out of 1113 collected.
- `benchmark-results/array-api-tests-full-maxfail.md`: pinned upstream full Array API 2023.12 probe, currently 1161 passed, 58 skipped, out of 1219 collected.
- `benchmark-results/statsrust-vs-statsmodels.md`: same-data StatsRust vs StatsModels evidence.
- `benchmark-results/scirust-vs-scipy.md`: same-data SciRust vs SciPy evidence.
- `logs/journey.md`: detailed build journey.
