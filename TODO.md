# TODO

This file is intentionally committed and should be updated in normal git commits as work progresses.

## Benchmark Integrity

- Keep full external NumPy ASV-derived reports separate from focused loss reruns.
- Re-run `uv run benchmarks/external_sources.py --verify-pinned` before publishing benchmark claims.
- Preserve raw pass samples in JSON artifacts for any benchmark result cited in docs.

## NumRust

- Expand supported NumPy ASV cases without filtering losses.
- Add more dtype coverage for integer, boolean, and complex Array API behavior.
- Add portable SIMD backends where stable Rust support is strong enough.
- Continue replacing generic stride paths with layout-specialized kernels only when real benchmarks improve.

## Ecosystem

- Grow StatsRust and SciRust with externally derived benchmarks where upstream suites exist.
- Use RigorTrail to gate broad replacement claims on source pins, checksums, unsupported cases, and benchmark scope.

## Release Readiness

- [x] Add CI for format, clippy, tests, Python benchmark schema checks, and source-lock verification.
- Add crate-level publishing metadata before any crates.io release.
- Keep TODO changes committed with the implementation commits they describe.
