# Python Array API Namespace

`crates/numrust-python` exposes a first Python namespace for `numrs-core`.

The pinned upstream `array-api-tests` 2023.12 suite now passes against this namespace. This is Array API conformance evidence for the tested surface, not a claim of full NumPy replacement.

Run:

```sh
uv run benchmarks/verify_array_api_namespace.py
uv run --with pytest --with pytest-json-report --with 'hypothesis>=6.151.0' --with 'ndindex>=1.8' benchmarks/run_array_api_tests.py --focused --maxfail 25 --output-stem array-api-tests-focused-probe
uv run --with pytest --with pytest-json-report --with 'hypothesis>=6.151.0' --with 'ndindex>=1.8' benchmarks/run_array_api_tests.py --full --maxfail 25 --output-stem array-api-tests-full-maxfail
```

The smoke verifier builds the PyO3 extension with Cargo, copies it into a temporary Python package tree, imports `numrust`, and checks that Python calls execute through `numrs-core`. The Array API runner clones the pinned upstream `array-api-tests` commit, initializes its spec submodule, and runs pytest without patching the suite.

Current supported smoke surface includes:

- `asarray`
- `zeros`
- `ones`
- `full`
- `arange`
- primitive and complex dtype objects: `bool`, signed ints, unsigned ints, `float32`, `float64`, `complex64`, `complex128`
- `__array_api_version__`
- `Array.__array_namespace__`
- `__array_namespace_info__` with CPU-only metadata
- `Array.shape`, `Array.ndim`, `Array.size`, `Array.dtype`, `Array.tolist`
- `empty`, `zeros`, `ones`, `full`, and `arange`
- `+`, `-`, `*`, and `/`
- `add`, `subtract`, `multiply`, `divide`
- `equal`, `not_equal`, `less`, `less_equal`, `greater`, `greater_equal`
- Mixed `int64`/`float64` arithmetic promotion
- `astype`
- `isdtype`
- First-axis integer indexing
- `reshape`
- `permute_dims` for full axis reversal
- `matmul`
- reductions, statistics, sorting/searching/set functions, FFT, and linalg coverage exercised by the pinned upstream suite

Current upstream `array-api-tests` evidence:

- Focused probe: 1113 collected, 1109 passed, 4 skipped against pinned commit `55fcc60179efa2680ddd6cd926ddf17b83530e2b`.
- Full 2023.12 suite probe: 1219 collected, 1161 passed, 58 skipped, return code 0.

Remaining after the pinned suite:

- Broader NumPy API parity beyond the Array API standard.
- NumPy ABI compatibility and package ecosystem integration.
- Production-grade linalg backend work; the current Python bridge has pure deterministic fallback algorithms for conformance coverage, while `numrs-core` still needs optimized Rust-native decompositions.
- PyPI-style packaging with maturin or equivalent.
