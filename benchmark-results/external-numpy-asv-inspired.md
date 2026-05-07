# External NumPy Evidence

This report uses pinned online sources. It is not a self-authored-only benchmark.

## Source Lock

- NumPy ASV: `80b1a07494964733f7d4571781608238f500e2dd`
- Array API tests: `55fcc60179efa2680ddd6cd926ddf17b83530e2b`
- NumPy version measured locally: `2.4.4`
- Python: `3.12.13`
- Full benchmark passes per engine: 5
- Pass aggregation: per-case median across alternating engine-order full passes.

## Array API Conformance

- Source: `array-api-tests` commit `55fcc60179efa2680ddd6cd926ddf17b83530e2b`
- API version: `2023.12`
- Full suite: True
- Status: passed
- Return code: 0
- Summary: 1161 passed, 58 skipped, 1219 collected
- Report: `benchmark-results/array-api-tests-full-maxfail.json`

## Supported External Cases

| Case | Source | Reps | NumRust ms | NumPy ms | Speedup vs NumPy | Winner | Checksum |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| `asv_ufunc_broadcast_sub_f64` | `benchmarks/benchmarks/bench_ufunc.py::Broadcast.time_broadcast` | 1 | 0.866 | 1.523 | 1.76x | numrust | ok |
| `asv_ufunc_astype_i32_to_f64_100x100` | `benchmarks/benchmarks/bench_ufunc.py::NDArrayAsType.time_astype(typeconv=('int32', 'float64'))` | 5000 | 5.106 | 13.809 | 2.70x | numrust | ok |
| `asv_ufunc_add_at_f64_10000000` | `benchmarks/benchmarks/bench_ufunc.py::At.time_sum_at` | 1 | 3.806 | 5.822 | 1.53x | numrust | ok |
| `asv_ufunc_maximum_at_f64_10000000` | `benchmarks/benchmarks/bench_ufunc.py::At.time_maximum_at` | 1 | 3.720 | 5.975 | 1.61x | numrust | ok |
| `asv_reduce_small_sum_f32_100` | `benchmarks/benchmarks/bench_reduce.py::SmallReduction.time_small` | 200000 | 2.661 | 130.881 | 49.19x | numrust | ok |
| `asv_reduce_stats_min_f64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_min(dtype=float64)` | 100000 | 0.332 | 65.985 | 198.50x | numrust | ok |
| `asv_reduce_stats_max_f64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_max(dtype=float64)` | 100000 | 0.333 | 65.748 | 197.74x | numrust | ok |
| `asv_reduce_stats_mean_f64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_mean(dtype=float64)` | 100000 | 0.112 | 100.886 | 903.13x | numrust | ok |
| `asv_reduce_stats_std_f64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_std(dtype=float64)` | 100000 | 0.107 | 332.608 | 3103.64x | numrust | ok |
| `asv_reduce_stats_prod_f64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_prod(dtype=float64)` | 100000 | 8.728 | 75.866 | 8.69x | numrust | ok |
| `asv_reduce_stats_var_f64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_var(dtype=float64)` | 100000 | 0.104 | 308.785 | 2957.23x | numrust | ok |
| `asv_reduce_argmax_i64_200000` | `benchmarks/benchmarks/bench_reduce.py::ArgMax.time_argmax(dtype=int64)` | 20000 | 0.025 | 483.333 | 19205.80x | numrust | ok |
| `asv_reduce_argmin_i64_200000` | `benchmarks/benchmarks/bench_reduce.py::ArgMin.time_argmin(dtype=int64)` | 20000 | 0.026 | 517.483 | 20227.59x | numrust | ok |
| `asv_itemselection_take_i64_1000x1` | `benchmarks/benchmarks/bench_itemselection.py::Take.time_contiguous(shape=(1000, 1), mode='raise', dtype='int64')` | 2000 | 0.868 | 3.774 | 4.35x | numrust | ok |
| `asv_itemselection_putmask_dense_scalar_f64_1000` | `benchmarks/benchmarks/bench_itemselection.py::PutMask.time_dense(values_is_scalar=True, dtype=float64)` | 10000 | 0.665 | 4.919 | 7.39x | numrust | ok |
| `asv_itemselection_putmask_sparse_scalar_f64_1000` | `benchmarks/benchmarks/bench_itemselection.py::PutMask.time_sparse(values_is_scalar=True, dtype=float64)` | 10000 | 0.026 | 4.686 | 182.56x | numrust | ok |
| `asv_itemselection_put_ordered_f64_1000` | `benchmarks/benchmarks/bench_itemselection.py::Put.time_ordered(values_is_scalar=False, dtype=float64)` | 10000 | 7.030 | 19.787 | 2.81x | numrust | ok |
| `asv_manipulate_broadcast_arrays_f64_16x32` | `benchmarks/benchmarks/bench_manipulate.py::BroadcastArrays.time_broadcast_arrays(shape=(16, 32), ndtype=float64)` | 200000 | 58.156 | 1207.166 | 20.76x | numrust | ok |
| `asv_manipulate_broadcast_arrays_f64_128x256` | `benchmarks/benchmarks/bench_manipulate.py::BroadcastArrays.time_broadcast_arrays(shape=(128, 256), ndtype=float64)` | 100000 | 29.365 | 596.353 | 20.31x | numrust | ok |
| `asv_manipulate_broadcast_arrays_f32_128x256` | `benchmarks/benchmarks/bench_manipulate.py::BroadcastArrays.time_broadcast_arrays(shape=(128, 256), ndtype=float32)` | 100000 | 29.230 | 599.360 | 20.50x | numrust | ok |
| `asv_manipulate_broadcast_arrays_i32_128x256` | `benchmarks/benchmarks/bench_manipulate.py::BroadcastArrays.time_broadcast_arrays(shape=(128, 256), ndtype=int32)` | 100000 | 29.119 | 597.920 | 20.53x | numrust | ok |
| `asv_manipulate_broadcast_arrays_f64_512x1024` | `benchmarks/benchmarks/bench_manipulate.py::BroadcastArrays.time_broadcast_arrays(shape=(512, 1024), ndtype=float64)` | 100000 | 29.092 | 615.463 | 21.16x | numrust | ok |
| `asv_manipulate_broadcast_to_f64_16` | `benchmarks/benchmarks/bench_manipulate.py::BroadcastArraysTo.time_broadcast_to(size=16, ndtype=float64)` | 200000 | 10.881 | 179.372 | 16.48x | numrust | ok |
| `asv_manipulate_broadcast_to_f64_64` | `benchmarks/benchmarks/bench_manipulate.py::BroadcastArraysTo.time_broadcast_to(size=64, ndtype=float64)` | 200000 | 10.884 | 178.287 | 16.38x | numrust | ok |
| `asv_manipulate_broadcast_to_f32_64` | `benchmarks/benchmarks/bench_manipulate.py::BroadcastArraysTo.time_broadcast_to(size=64, ndtype=float32)` | 200000 | 10.820 | 177.524 | 16.41x | numrust | ok |
| `asv_manipulate_broadcast_to_i32_64` | `benchmarks/benchmarks/bench_manipulate.py::BroadcastArraysTo.time_broadcast_to(size=64, ndtype=int32)` | 200000 | 10.820 | 179.393 | 16.58x | numrust | ok |
| `asv_manipulate_broadcast_to_f64_512` | `benchmarks/benchmarks/bench_manipulate.py::BroadcastArraysTo.time_broadcast_to(size=512, ndtype=float64)` | 200000 | 10.867 | 183.950 | 16.93x | numrust | ok |
| `asv_manipulate_concatenate_ax0_f64_32x64_n5` | `benchmarks/benchmarks/bench_manipulate.py::ConcatenateStackArrays.time_concatenate_ax0(shape=(32, 64), narrays=5, ndtype=float64)` | 2000 | 1.734 | 3.594 | 2.07x | numrust | ok |
| `asv_manipulate_concatenate_ax1_f64_32x64_n5` | `benchmarks/benchmarks/bench_manipulate.py::ConcatenateStackArrays.time_concatenate_ax1(shape=(32, 64), narrays=5, ndtype=float64)` | 2000 | 2.426 | 4.727 | 1.95x | numrust | ok |
| `asv_manipulate_stack_ax0_f64_32x64_n5` | `benchmarks/benchmarks/bench_manipulate.py::ConcatenateStackArrays.time_stack_ax0(shape=(32, 64), narrays=5, ndtype=float64)` | 2000 | 1.840 | 5.180 | 2.81x | numrust | ok |
| `asv_manipulate_stack_ax1_f64_32x64_n5` | `benchmarks/benchmarks/bench_manipulate.py::ConcatenateStackArrays.time_stack_ax1(shape=(32, 64), narrays=5, ndtype=float64)` | 2000 | 2.339 | 6.507 | 2.78x | numrust | ok |
| `asv_manipulate_expand_dims_f64_5x2x3x1_axis1` | `benchmarks/benchmarks/bench_manipulate.py::DimsManipulations.time_expand_dims(shape=(5, 2, 3, 1))` | 200000 | 19.800 | 131.612 | 6.65x | numrust | ok |
| `asv_manipulate_expand_dims_neg_f64_5x2x3x1` | `benchmarks/benchmarks/bench_manipulate.py::DimsManipulations.time_expand_dims_neg(shape=(5, 2, 3, 1))` | 200000 | 18.886 | 132.183 | 7.00x | numrust | ok |
| `asv_manipulate_squeeze_dims_f64_5x2x3x1` | `benchmarks/benchmarks/bench_manipulate.py::DimsManipulations.time_squeeze_dims(shape=(5, 2, 3, 1))` | 200000 | 9.768 | 37.103 | 3.80x | numrust | ok |
| `asv_manipulate_flip_all_f64_5x2x3x1` | `benchmarks/benchmarks/bench_manipulate.py::DimsManipulations.time_flip_all(shape=(5, 2, 3, 1))` | 200000 | 29.498 | 517.608 | 17.55x | numrust | ok |
| `asv_manipulate_flip_one_f64_5x2x3x1_axis1` | `benchmarks/benchmarks/bench_manipulate.py::DimsManipulations.time_flip_one(shape=(5, 2, 3, 1))` | 200000 | 29.233 | 596.068 | 20.39x | numrust | ok |
| `asv_manipulate_flip_neg_f64_5x2x3x1_axis_neg1` | `benchmarks/benchmarks/bench_manipulate.py::DimsManipulations.time_flip_neg(shape=(5, 2, 3, 1))` | 200000 | 28.855 | 594.137 | 20.59x | numrust | ok |
| `asv_manipulate_moveaxis_f64_5x2x3x1` | `benchmarks/benchmarks/bench_manipulate.py::DimsManipulations.time_moveaxis(shape=(5, 2, 3, 1))` | 200000 | 39.281 | 689.910 | 17.56x | numrust | ok |
| `asv_manipulate_roll_f64_5x2x3x1_shift3` | `benchmarks/benchmarks/bench_manipulate.py::DimsManipulations.time_roll(shape=(5, 2, 3, 1))` | 100000 | 22.063 | 455.087 | 20.63x | numrust | ok |
| `asv_manipulate_reshape_f64_5x2x3x1_to_1x5x2x3` | `benchmarks/benchmarks/bench_manipulate.py::DimsManipulations.time_reshape(shape=(5, 2, 3, 1))` | 200000 | 20.408 | 63.908 | 3.13x | numrust | ok |
| `asv_linalg_dot_a_b_f64_150x400_400x600` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_dot_a_b` | 1000 | 144.212 | 146.901 | 1.02x | numrust | ok |
| `asv_linalg_matmul_a_b_f64_150x400_400x600` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_matmul_a_b` | 1000 | 144.262 | 145.908 | 1.01x | numrust | ok |
| `asv_linalg_matmul_d_matmul_b_c_f64` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_matmul_d_matmul_b_c` | 1000 | 4.058 | 5.023 | 1.24x | numrust | ok |
| `asv_linalg_dot_d_dot_b_c_f64` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_dot_d_dot_b_c` | 1000 | 4.006 | 4.827 | 1.20x | numrust | ok |
| `asv_linalg_dot_trans_a_at_f64_150x400_400x150` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_dot_trans_a_at` | 1000 | 30.687 | 34.105 | 1.11x | numrust | ok |
| `asv_linalg_dot_trans_a_atc_f64_150x400_400x150` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_dot_trans_a_atc` | 1000 | 42.567 | 43.223 | 1.02x | numrust | ok |
| `asv_linalg_dot_trans_at_a_f64_400x150_150x400` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_dot_trans_at_a` | 1000 | 78.196 | 95.905 | 1.23x | numrust | ok |
| `asv_linalg_dot_trans_atc_a_f64_400x150_150x400` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_dot_trans_atc_a` | 1000 | 96.030 | 100.291 | 1.04x | numrust | ok |
| `asv_linalg_inner_a_a_f64_150x400_150x400` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_inner_trans_a_a` | 1000 | 29.206 | 34.245 | 1.17x | numrust | ok |
| `asv_linalg_inner_a_ac_f64_150x400_150x400` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_inner_trans_a_ac` | 1000 | 48.925 | 50.446 | 1.03x | numrust | ok |
| `asv_linalg_matmul_trans_a_at_f64_150x400_400x150` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_matmul_trans_a_at` | 1000 | 30.528 | 30.550 | 1.00x | numrust | ok |
| `asv_linalg_matmul_trans_a_atc_f64_150x400_400x150` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_matmul_trans_a_atc` | 1000 | 42.561 | 42.897 | 1.01x | numrust | ok |
| `asv_linalg_matmul_trans_at_a_f64_400x150_150x400` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_matmul_trans_at_a` | 1000 | 77.828 | 79.441 | 1.02x | numrust | ok |
| `asv_linalg_matmul_trans_atc_a_f64_400x150_150x400` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_matmul_trans_atc_a` | 1000 | 96.129 | 96.755 | 1.01x | numrust | ok |
| `asv_linalg_tensordot_a3_b3_axes_10_01` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_tensordot_a_b_axes_1_0_0_1` | 10 | 2.778 | 4.484 | 1.61x | numrust | ok |
| `asv_linalg_norm_small_array_f64_5` | `benchmarks/benchmarks/bench_linalg.py::LinalgSmallArrays.time_norm_small_array` | 100000 | 4.672 | 41.223 | 8.82x | numrust | ok |
| `asv_linalg_det_small_array_f64_5x5` | `benchmarks/benchmarks/bench_linalg.py::LinalgSmallArrays.time_det_small_array` | 100000 | 8.394 | 91.097 | 10.85x | numrust | ok |
| `asv_linalg_det_3x3_f64` | `benchmarks/benchmarks/bench_linalg.py::LinalgSmallArrays.time_det_3x3` | 100000 | 7.055 | 87.405 | 12.39x | numrust | ok |
| `asv_linalg_solve_3x3_f64` | `benchmarks/benchmarks/bench_linalg.py::LinalgSmallArrays.time_solve_3x3` | 100000 | 14.165 | 172.900 | 12.21x | numrust | ok |
| `asv_linalg_lstsq_square_f64_100x100` | `benchmarks/benchmarks/bench_linalg.py::Lstsq.time_numpy_linalg_lstsq_a__b_float64` | 100 | 5.158 | 63.676 | 12.35x | numrust | ok |
| `asv_linalg_einsum_outer_f64_3000` | `benchmarks/benchmarks/bench_linalg.py::Einsum.time_einsum_outer(dtype=float64)` | 1 | 0.661 | 1.700 | 2.57x | numrust | ok |
| `asv_linalg_einsum_i_ij_j_f64_400_400x600_600` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_einsum_i_ij_j` | 1000 | 4.217 | 253.231 | 60.06x | numrust | ok |
| `asv_linalg_einsum_ij_jk_f64_150x400_400x600` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_einsum_ij_jk_a_b` | 1000 | 143.856 | 4152.564 | 28.87x | numrust | ok |
| `asv_linalg_einsum_multiply_f64_30x40_20x30x40` | `benchmarks/benchmarks/bench_linalg.py::Einsum.time_einsum_multiply(dtype=float64)` | 1 | 0.003 | 0.015 | 4.67x | numrust | ok |
| `asv_linalg_einsum_sum_mul_f64_scalar_10x100x10` | `benchmarks/benchmarks/bench_linalg.py::Einsum.time_einsum_sum_mul(dtype=float64)` | 100 | 0.049 | 0.655 | 13.27x | numrust | ok |
| `asv_linalg_einsum_sum_mul2_f64_10x100x10_scalar` | `benchmarks/benchmarks/bench_linalg.py::Einsum.time_einsum_sum_mul2(dtype=float64)` | 100 | 0.050 | 0.648 | 12.91x | numrust | ok |
| `asv_linalg_einsum_scalar_mul_f64_480000` | `benchmarks/benchmarks/bench_linalg.py::Einsum.time_einsum_mul(dtype=float64)` | 100 | 6.266 | 6.737 | 1.08x | numrust | ok |
| `asv_linalg_einsum_sum_f64_480000` | `benchmarks/benchmarks/bench_linalg.py::Einsum.time_einsum_contig_outstride0(dtype=float64)` | 100 | 3.349 | 4.132 | 1.23x | numrust | ok |
| `asv_linalg_einsum_weighted_sum_f64_400x600` | `benchmarks/benchmarks/bench_linalg.py::Einsum.time_einsum_contig_contig(dtype=float64)` | 100 | 0.404 | 4.377 | 10.83x | numrust | ok |
| `asv_linalg_einsum_noncon_outer_f64_2000` | `benchmarks/benchmarks/bench_linalg.py::Einsum.time_einsum_noncon_outer(dtype=float64)` | 1 | 0.395 | 1.564 | 3.96x | numrust | ok |
| `asv_linalg_einsum_noncon_multiply_f64_30x40_20x30x40` | `benchmarks/benchmarks/bench_linalg.py::Einsum.time_einsum_noncon_multiply(dtype=float64)` | 1 | 0.003 | 0.014 | 4.39x | numrust | ok |
| `asv_linalg_einsum_noncon_sum_mul_f64_scalar_20x30x40` | `benchmarks/benchmarks/bench_linalg.py::Einsum.time_einsum_noncon_sum_mul(dtype=float64)` | 100 | 0.156 | 0.773 | 4.97x | numrust | ok |
| `asv_linalg_einsum_noncon_sum_mul2_f64_20x30x40_scalar` | `benchmarks/benchmarks/bench_linalg.py::Einsum.time_einsum_noncon_sum_mul2(dtype=float64)` | 100 | 0.155 | 0.776 | 5.00x | numrust | ok |
| `asv_linalg_einsum_noncon_scalar_mul_f64_2000` | `benchmarks/benchmarks/bench_linalg.py::Einsum.time_einsum_noncon_mul(dtype=float64)` | 100 | 0.023 | 0.419 | 18.16x | numrust | ok |
| `asv_linalg_einsum_noncon_weighted_sum_f64_30x40` | `benchmarks/benchmarks/bench_linalg.py::Einsum.time_einsum_noncon_contig_contig(dtype=float64)` | 100 | 0.048 | 0.483 | 10.13x | numrust | ok |
| `asv_linalg_einsum_noncon_sum_f64_2000` | `benchmarks/benchmarks/bench_linalg.py::Einsum.time_einsum_noncon_contig_outstride0(dtype=float64)` | 100 | 0.012 | 0.250 | 20.46x | numrust | ok |

## Score

- Supported external cases: 76
- Unsupported external cases tracked: 1
- NumRust wins: 76
- NumPy wins: 0
- Geomean speedup vs NumPy: 9.46x
- Near-tie relative margin: 2%
- Near-tie cases: 6
- Ranked higher on supported external cases: True
- Global NumPy replacement claim: false

## NumPy-Winning Cases

- None in this run.

## Unsupported External Cases

| Source | Case | Reason |
| --- | --- | --- |
| `bench_linalg.py` | `eig/remaining LAPACK, remaining strided/batched matmul, and full linalg/einsum grammar` | NumRust has dot, matmul, tensordot axes, norm, det, solve, square full-rank lstsq-equivalent ASV coverage, selected transposed-view and copied-transpose matmul coverage, and both contiguous and NumPy-ASV noncon einsum-style contractions, but not eig, the remaining LAPACK-style routines, every strided/batched matmul case, or the full tensor expression surface. |

## Neutrality Controls

- Benchmark cases come from pinned NumPy ASV files; conformance evidence comes from the pinned Array API suite.
- The supported-case table includes losses and checksum failures.
- Unsupported external cases are counted outside the speed score instead of omitted silently.
- Tiny ASV cases are repeated equally for both engines because ASV normally auto-calibrates repetitions.
- This still does not prove full NumPy replacement status.
