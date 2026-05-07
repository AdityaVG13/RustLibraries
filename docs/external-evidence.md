# External Evidence

The in-repo `compare_numpy.py`, `compare_statsmodels.py`, and `compare_scipy.py` suites are useful for regression work, but self-selected cases are not enough evidence for broad replacement claims.

Use [`benchmark-dashboard.md`](benchmark-dashboard.md) for the table-first scorecard across all implemented library slices. This file documents where the external evidence comes from and how the pinned sources are verified.

## Evidence Tiers

| Tier | Source | Current use | Claim strength |
| --- | --- | --- | --- |
| 1 | Pinned upstream benchmark or conformance source | NumPy ASV-derived cases, Array API tests, selected SciPy ASV translations | Strongest current evidence |
| 2 | Same-data Rust-vs-Python harness | StatsRust, SciRust local cases, FrameRust, GraphRust, MediaExtractRust, ValidateRust, ImageRust, TextRust | Good slice-regression evidence |
| 3 | Source/API pin without benchmark translation | StatsModels and remaining SciPy source pins | Research support only |
| 4 | Self-authored smoke cases | Narrow NumRust comparison harnesses and examples | Useful for development, not broad claims |

This repo separates evidence tiers:

1. Self-authored or same-data slice benchmarks: `benchmarks/compare_numpy.py`, `benchmarks/compare_statsmodels.py`, and `benchmarks/compare_scipy.py`.
2. Externally derived NumPy ASV cases: `benchmarks/external_numpy_cases.py`.
3. Translated SciPy ASV root-finding and cumulative Simpson cases inside `benchmarks/compare_scipy.py`.
4. Pinned upstream API/source references for StatsModels and SciPy cases that are not yet translated from benchmark suites.
5. External conformance target: `array-api-tests`; the Rust-backed Python namespace now passes the pinned upstream 2023.12 suite, with upstream skips preserved.

## Sources

Run:

```sh
uv run benchmarks/external_sources.py --update-lock
```

This writes `benchmark-results/external-source-lock.json` with pinned commits, raw file URLs, and SHA-256 hashes for:

- NumPy ASV benchmark files under `numpy/numpy/benchmarks/benchmarks`, including ufunc, reduction, itemselection, linalg/einsum, and manipulation files.
- The `data-apis/array-api-tests` README that documents the conformance workflow.
- SciPy ASV benchmark/source files for `Zeros.time_zeros`, `f2`, root solver lists, and `CumulativeSimpson`.
- StatsModels source files for the implemented OLS, WLS, and Logit APIs.

The lock is intentionally committed as evidence. Re-run with `--verify-pinned` to fetch the pinned raw URLs and confirm the hashes still match.

## Benchmark

Run:

```sh
uv run --with numpy python benchmarks/external_numpy_cases.py
```

This writes:

- `benchmark-results/external-numpy-asv-inspired.json`
- `benchmark-results/external-numpy-asv-inspired.md`
- `benchmark-results/external-numpy-loss-triage.json`
- `benchmark-results/external-numpy-loss-triage.md`
- `benchmark-results/external-numpy-loss-focused.json` when focused loss reruns are requested
- `benchmark-results/external-numpy-loss-focused.md` when focused loss reruns are requested

The NumPy harness directly translates supported cases from pinned NumPy ASV files. It uses the same operation, shape, dtype, and setup values where those values are literal in the ASV source. Tiny ASV cases are repeated equally for both engines because ASV normally auto-calibrates repetitions. Each case gets one untimed warmup on both engines before median timing.
The latest harness run uses 5 full benchmark passes per engine, alternates engine order, and reports the per-case median across those full passes.
For long local sessions, the same run can be sharded into one-pass artifacts and aggregated without changing the scoring path:

```sh
uv run --with numpy python benchmarks/external_numpy_cases.py --pass-index 0 --pass-out benchmark-results/external-numpy-pass-0.json
uv run --with numpy python benchmarks/external_numpy_cases.py --pass-index 1 --pass-out benchmark-results/external-numpy-pass-1.json
uv run --with numpy python benchmarks/external_numpy_cases.py --pass-index 2 --pass-out benchmark-results/external-numpy-pass-2.json
uv run --with numpy python benchmarks/external_numpy_cases.py --pass-index 3 --pass-out benchmark-results/external-numpy-pass-3.json
uv run --with numpy python benchmarks/external_numpy_cases.py --pass-index 4 --pass-out benchmark-results/external-numpy-pass-4.json
uv run --with numpy python benchmarks/external_numpy_cases.py --aggregate-passes benchmark-results/external-numpy-pass-0.json benchmark-results/external-numpy-pass-1.json benchmark-results/external-numpy-pass-2.json benchmark-results/external-numpy-pass-3.json benchmark-results/external-numpy-pass-4.json
```

`benchmarks/verify_array_api_namespace.py` builds the `numrust-python` PyO3 extension, imports the `numrust` Python package, and smoke-tests Rust-backed Array API-style calls.

`benchmarks/run_array_api_tests.py` clones the pinned upstream `array-api-tests` repository, checks out commit `55fcc60179efa2680ddd6cd926ddf17b83530e2b`, initializes the spec submodule, imports the Rust-backed `numrust` package, and runs pytest without patching the suite. Latest observed:

- Focused probe: 1113 collected, 1109 passed, 4 skipped.
- Full 2023.12 suite probe: 1219 collected, 1161 passed, 58 skipped, return code 0.

The SciPy harness includes translated SciPy ASV `Zeros.time_zeros` cases for `f2` with `bisect` and `brentq`, plus translated `CumulativeSimpson.time_1d` and `CumulativeSimpson.time_multid` cases. Its remaining integrate/minimize cases are same-data local cases, not upstream ASV translations.

StatsModels has source/API pins in this repo, but no benchmark/asv tree was found in the upstream repository at the pinned commit. The current StatsRust comparison is therefore same-data local evidence, not externally derived benchmark evidence.

Unsupported external cases are counted in the report. They are not silently omitted from the project status.

`benchmarks/test_external_evidence.py` guards the evidence plumbing: runnable cases must carry source metadata, their root upstream symbols must be present in the pinned source lock, the generated JSON report must cover every runnable case, sharded pass aggregation must preserve raw per-pass samples, loss triage must match the generated NumPy-winning rows, focused rerun stability metadata must match raw focused rows, and the published score must match raw comparison rows.
The loss-triage artifacts contain only NumPy-winning rows from the generated report, sorted by worst `speedup_vs_numpy` first, and preserve source metadata plus per-pass samples for optimization work.
Use `uv run --with numpy python benchmarks/external_numpy_cases.py --rerun-losses --loss-passes 3` to rerun just those NumPy-winning rows into focused loss artifacts after a backend experiment. The focused rerun uses alternating engine order and median aggregation when `--loss-passes` is greater than 1, records source-loss winner flips, and keeps the full external report as the authoritative score source.

Latest observed supported-case result:

- Supported external cases: 68
- NumRust wins: 67
- NumPy wins: 1
- Geomean speedup vs NumPy: 8.67x
- Near-tie cases within 2%: 2
- Ranked higher by wins: true
- Unsupported external case buckets tracked: 1
- Current NumPy win: `asv_linalg_matmul_trans_atc_a_f64_400x150_150x400` at 0.993x in the authoritative full report.

Latest focused rerun of those NumPy-winning rows:

- Focused cases: 1
- Focused passes per engine: 3
- NumRust wins: 1
- NumPy wins: 0
- Near ties within 2%: 1
- Checksum failures: 0
- Report: `benchmark-results/external-numpy-loss-focused.md`

## Current Limitation

This is stronger than a purely self-authored suite, but it is still not a full neutral certification. The pinned 2023.12 `array-api-tests` suite now passes for the Rust-backed Python namespace, but NumPy replacement still requires much more than Array API conformance. The global claim remains:

> NumRust is not a full NumPy replacement.
