# Performance Notes

`numrs-core` exposes safe Rust kernels, uses explicit layout dispatch, and calls optimized native backends when a layout maps cleanly to BLAS or vDSP.

For the table-first benchmark view, use [`benchmark-dashboard.md`](benchmark-dashboard.md). The dashboard separates full external scores, focused reruns, same-data slice benchmarks, checksum status, rerun commands, and remaining Python-winning rows.

## Current Benchmark Summary

| Suite | Cases | Rust wins | Python wins | Speedup summary | Checksum failures |
| --- | ---: | ---: | ---: | ---: | ---: |
| NumRust vs NumPy, targeted same-data | 10 | 10 | 0 | 1.67x geomean | 0 |
| NumRust vs NumPy, external ASV-derived | 53 | 51 | 2 | 8.87x geomean | 0 |
| NumRust vs NumPy, focused current-loss rerun | 2 | 1 | 1 | 1.07x win / 0.99x near-tie loss | 0 |
| StatsRust vs StatsModels | 4 | 4 | 0 | 3.51x geomean | 0 |
| SciRust vs SciPy | 9 | 9 | 0 | 19.11x geomean | 0 |
| FrameRust vs Pandas | 1 | 1 | 0 | 2.14x | 0 |
| GraphRust vs NetworkX | 1 | 1 | 0 | 27.15x | 0 |
| MediaExtractRust vs Python extraction libraries | 2 | 2 | 0 | 81.21x to 122.93x | 0 |
| ValidateRust vs Pydantic | 1 | 1 | 0 | 35.69x | 0 |
| ImageRust vs Pillow | 1 | 1 | 0 | 16.53x | 0 |
| TextRust vs NLTK | 1 | 1 | 0 | 23.92x | 0 |
| LearnRust vs scikit-learn | 3 | 3 | 0 | 6.53x geomean | 0 |

## Implemented Fast Paths

- Contiguous equal-shape elementwise operations zip input slices directly.
- `full` and `zeros` arrays carry uniform-value metadata; reductions and all-true/all-false `putmask` use that metadata without scanning payloads.
- Broadcasted operations compute zero-stride views and write one contiguous output.
- Reductions over contiguous `f64`/`f32` views use Accelerate/vDSP on macOS.
- Contiguous `f64` and `f32` 2-D dot use Accelerate BLAS on macOS.
- `f64` add+sum has a fused temporary-free kernel.
- `f64` column-vector plus row-vector outer broadcast has a direct 2-D kernel.
- `f64` matrix-minus-row-vector broadcast has a direct contiguous row kernel.
- 2-D contiguous `take_axis` has axis-specialized gather paths.
- Contiguous `argmax`/`argmin` scan direct slices instead of generic stride iterators.
- Primitive `astype` casts contiguous arrays with direct slice iteration.
- `putmask`, `put`, `add_at`, and `maximum_at` cover NumPy ASV itemselection and ufunc-at patterns.
- `nonzero` and broadcasted `where_select` add NumPy-style selection scope for boolean masks. Contiguous `nonzero` has 1-D and 2-D direct paths; dtype-specialized `where_select_f64` fast paths remove the previous targeted-suite selection loss.
- `min_all`, `max_all`, `prod_all`, `var_all`, and `std_all` cover NumPy ASV stats-reduction patterns.
- `norm_l2`, `det`, and `solve` cover NumPy ASV small-linalg patterns without calling LAPACK.
- `outer_product`, `mul_scalar`, repeated trailing-tile multiply, `weighted_axis1_sum`, and `bilinear_form` cover selected NumPy ASV einsum-style contraction patterns; contiguous `f64` scalar multiply and repeated trailing-tile multiply have aarch64 NEON paths.
- `matmul` covers vector, matrix, and broadcasted batched matrix multiplication semantics; contiguous array-level `f64`/`f32` vector-vector, matrix-vector, vector-matrix, 2-D matrix-matrix, and selected transposed-view matrix-matrix paths dispatch straight to vDSP/BLAS. On macOS, transposed `f64`/`f32` 2-D GEMM uses column-major-swapped CBLAS calls to avoid the slower row-major transpose path.
- `inner2d` covers the 2-D NumPy `inner` slice and dispatches contiguous `f64`/`f32` cases to transpose-B BLAS.
- `tensordot_axes` covers explicit-axis tensor contractions and uses packed BLAS for contiguous `f64` contraction plans, including direct-pack elision when an operand already matches the GEMM layout.
- Contiguous `outer_product`, `weighted_axis1_sum`, and `bilinear_form` use Accelerate BLAS rank-1, matrix-vector, and dot kernels on macOS.
- Shape transforms do not copy data.

## Benchmark Hook

Run:

```sh
cargo run --release -p numrs-core --example microbench
uv run --with numpy benchmarks/compare_numpy.py
uv run benchmarks/external_sources.py --update-lock
uv run --with numpy python benchmarks/external_numpy_cases.py
uv run benchmarks/compare_statsmodels.py
uv run --with numpy --with scipy benchmarks/compare_scipy.py
```

The benchmark harnesses run one untimed warmup for each case on both engines, then report median timed samples. They are smoke tests for kernel path regressions, not replacements for Criterion or hardware-counter profiling.

`benchmarks/compare_numpy.py` is the NumPy comparison harness. Current evidence is written to `benchmark-results/numrust-vs-numpy.md`; the latest run ranks NumRust higher on 10 of 10 targeted cases, with 1.67x geometric-mean speedup and 0 checksum failures. This is still a targeted implemented-slice result, not a full NumPy replacement claim.

`benchmarks/external_numpy_cases.py` is the externally derived harness. It pins NumPy ASV and Array API test sources in `benchmark-results/external-source-lock.json`, then translates supported NumPy ASV cases without filtering out losses. Latest run: NumRust wins 51 of 53 supported external cases, NumPy wins 2, with 8.87x geometric-mean speedup. The remaining NumPy wins are copied-transpose linalg near ties: `asv_linalg_dot_trans_a_atc_f64_150x400_400x150` at 0.98x and `asv_linalg_matmul_trans_atc_a_f64_400x150_150x400` at 0.99x. The harness uses 5 full passes per engine, alternates engine order, aggregates each case by median across passes, supports sharded one-pass artifacts for long runs, writes pass samples for NumPy-winning rows, emits loss-triage artifacts sorted by worst NumPy win, and can rerun focused NumPy-winning rows after backend experiments with explicit stability metadata. This ranks NumRust higher on supported external cases only and does not prove global NumPy replacement status.

The latest 3-pass focused rerun of the two NumPy-winning rows reports NumRust wins 1 of 2, NumPy wins 1 of 2, with the dot row at 1.07x and the copied-transpose matmul row still a NumPy near-tie at 0.99x. There are no checksum failures. This is a triage signal for timing sensitivity, not a replacement for the full 53-case score.

`benchmarks/compare_statsmodels.py` is the same-data StatsModels comparison harness for the implemented StatsRust slice. Latest run: StatsRust wins 4 of 4 cases against StatsModels 0.14.6, StatsModels wins 0, with 3.51x geometric-mean speedup and no checksum failures. This does not prove full StatsModels replacement status.

`benchmarks/compare_scipy.py` is the SciPy comparison harness for the implemented SciRust slice. Latest run: SciRust wins 9 of 9 cases against SciPy 1.17.1, SciPy wins 0, with 19.11x geometric-mean speedup and no checksum failures. Four cases translate pinned SciPy ASV `Zeros.time_zeros` and `CumulativeSimpson` cases; the remaining implemented-slice cases are same-data local benchmarks. This does not prove full SciPy replacement status.

## Next Optimizations

- Portable SIMD for contiguous `f64` and `i64` kernels.
- Axis-specialized reductions for innermost contiguous dimensions.
- Tiled 2-D dot with cache-aware blocking.
- Optional BLAS backend.
- Rayon feature for large elementwise and reduction workloads.
- Small-rank inline layout storage to remove most shape/stride heap allocations.
