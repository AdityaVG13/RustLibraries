# SciPy Port

**SciRust** starts the third library line after NumRust crossed the targeted NumPy benchmark gate.

## Why SciPy

SciPy is a top Python numerical library. Rust has strong individual crates for linear algebra, optimization, and statistics, but no single SciPy-scale, batteries-included numerical standard library.

## Current Slice

- `optimize::minimize_scalar_bounded`: bounded scalar minimization using golden-section search.
- `root::bisect`: bracketed bisection root finding.
- `root::brentq`: bracketed Brent-style root finding.
- `integrate::trapezoid`: trapezoidal integration with uniform or explicit coordinates.
- `integrate::simpson_uniform`: Simpson integration for uniformly spaced odd-length samples.
- `integrate::cumulative_simpson_uniform`: cumulative Simpson integration for uniformly spaced samples.
- `integrate::cumulative_simpson_uniform_axis_last`: batched cumulative Simpson integration over the last axis.

## Benchmark Evidence

Run:

```sh
uv run --with numpy --with scipy benchmarks/compare_scipy.py
```

Latest benchmark against SciPy 1.17.1:

- SciRust wins: 9 of 9.
- SciPy wins: 0 of 9.
- Geomean speedup vs SciPy: 19.11x.
- Checksum failures: 0.
- Global SciPy replacement claim: false.

Four cases are translations of pinned SciPy ASV benchmarks: `Zeros.time_zeros` for `f2` with `bisect` and `brentq`, plus `CumulativeSimpson.time_1d` and `CumulativeSimpson.time_multid`. The remaining integration, root-finding, and bounded-minimization cases are same-data local cases.

Recent optimization: Simpson integration now applies the composite rule in two-interval segments, removing the per-sample parity branch from the hot loop.

## Next Scope

- Multivariate optimization.
- Interpolation.
- Sparse matrices.
- Signal processing.
- ODE solvers.
