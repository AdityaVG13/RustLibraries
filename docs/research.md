# Research Notes

## Source Map

- NumPy ndarray reference, v2.4 manual: https://numpy.org/doc/stable/reference/arrays.ndarray.html
- NumPy broadcasting guide, v2.4 manual: https://numpy.org/doc/stable/user/basics.broadcasting.html
- Python Array API standard indexing spec: https://data-apis.org/array-api/latest/API_specification/indexing.html
- Python Array API standard broadcasting spec: https://data-apis.org/array-api/latest/API_specification/broadcasting.html
- Rust `ndarray` crate docs: https://docs.rs/ndarray/latest/ndarray/
- Rust `ndarray` for NumPy users: https://docs.rs/ndarray/latest/ndarray/doc/ndarray_for_numpy_users/
- NumPy benchmark docs: https://numpy.org/doc/1.25/benchmarking.html
- NumPy ASV benchmark tree: https://github.com/numpy/numpy/tree/main/benchmarks/benchmarks
- Array API conformance suite: https://data-apis.org/array-api-tests/
- Array API conformance repo: https://github.com/data-apis/array-api-tests
- Apple Accelerate BLAS docs: https://developer.apple.com/documentation/accelerate/blas-library
- Apple Accelerate `vDSP_sveD` docs: https://developer.apple.com/documentation/accelerate/vdsp_sved
- StatsModels OLS docs: https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html
- StatsModels regression docs: https://www.statsmodels.org/stable/regression.html

## Findings

NumPy's center of gravity is the `ndarray`: a fixed-size multidimensional container with one dtype, a shape tuple, and indexing/slicing methods. The reference docs also make views fundamental: multiple arrays may share the same data while presenting different metadata.

Broadcasting compares trailing dimensions first. Dimensions are compatible when equal or when one dimension is `1`; missing leading dimensions behave like `1`. Good implementations should broadcast by metadata, not by copying the smaller operand.

The Array API standard is useful as a portability boundary, especially for indexing. It documents where behavior is specified and where advanced indexing remains implementation-defined. For this v0, `numrs-core` implements orthogonal integer/slice indexing and intentionally omits integer-array and boolean-array indexing.

Rust's mature `ndarray` crate confirms the right primitives: owned arrays, views, arbitrary strides, row-major defaults, cheap slicing, elementwise arithmetic, and matrix multiplication. It also exposes useful Rust-specific tradeoffs: owned/view/mutable-view distinctions follow Rust aliasing rules instead of NumPy's unified mutable object model.

## Design Consequences

- Store layout as `{shape, strides, offset}`.
- Treat reshape, transpose, slice, and broadcast as metadata transformations.
- Keep owned results contiguous unless an operation explicitly returns a view.
- Start with `f64`, `i64`, and `bool` through a tiny dtype trait, then expand.
- Prefer explicit `Result` errors over NumPy's dynamic exceptions and warnings.
- Build optimized kernel dispatch around three tiers: contiguous equal-shape, broadcast strided, then future SIMD/tiled kernels.
- On macOS, use Accelerate BLAS/vDSP for matrix multiply and contiguous reductions rather than pretending a young generic kernel will beat platform math libraries.
- Treat online benchmark evidence as a separate tier. Self-authored microbenchmarks are useful for regressions, but pinned NumPy ASV cases and `array-api-tests` are the right external anchors.
- For the second library line, StatsModels is a useful target because the official regression module centers OLS/WLS/GLS-style models and result objects; StatsRust starts with OLS fit/predict/diagnostics and binary Logit.
