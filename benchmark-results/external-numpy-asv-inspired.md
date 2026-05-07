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
| `asv_ufunc_broadcast_sub_f64` | `benchmarks/benchmarks/bench_ufunc.py::Broadcast.time_broadcast` | 1 | 0.814 | 1.560 | 1.92x | numrust | ok |
| `asv_ufunc_astype_i32_to_f64_100x100` | `benchmarks/benchmarks/bench_ufunc.py::NDArrayAsType.time_astype(typeconv=('int32', 'float64'))` | 5000 | 4.914 | 13.767 | 2.80x | numrust | ok |
| `asv_ufunc_add_at_f64_10000000` | `benchmarks/benchmarks/bench_ufunc.py::At.time_sum_at` | 1 | 3.985 | 5.958 | 1.50x | numrust | ok |
| `asv_ufunc_maximum_at_f64_10000000` | `benchmarks/benchmarks/bench_ufunc.py::At.time_maximum_at` | 1 | 3.944 | 6.117 | 1.55x | numrust | ok |
| `asv_reduce_small_sum_f32_100` | `benchmarks/benchmarks/bench_reduce.py::SmallReduction.time_small` | 200000 | 2.760 | 134.879 | 48.87x | numrust | ok |
| `asv_reduce_stats_min_f64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_min(dtype=float64)` | 100000 | 0.334 | 65.894 | 197.07x | numrust | ok |
| `asv_reduce_stats_max_f64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_max(dtype=float64)` | 100000 | 0.333 | 65.580 | 196.71x | numrust | ok |
| `asv_reduce_stats_mean_f64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_mean(dtype=float64)` | 100000 | 0.110 | 100.973 | 915.16x | numrust | ok |
| `asv_reduce_stats_std_f64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_std(dtype=float64)` | 100000 | 0.106 | 329.206 | 3100.84x | numrust | ok |
| `asv_reduce_stats_prod_f64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_prod(dtype=float64)` | 100000 | 8.687 | 75.087 | 8.64x | numrust | ok |
| `asv_reduce_stats_var_f64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_var(dtype=float64)` | 100000 | 0.107 | 306.559 | 2868.39x | numrust | ok |
| `asv_reduce_argmax_i64_200000` | `benchmarks/benchmarks/bench_reduce.py::ArgMax.time_argmax(dtype=int64)` | 20000 | 0.025 | 486.678 | 19179.42x | numrust | ok |
| `asv_reduce_argmin_i64_200000` | `benchmarks/benchmarks/bench_reduce.py::ArgMin.time_argmin(dtype=int64)` | 20000 | 0.026 | 522.755 | 20301.17x | numrust | ok |
| `asv_itemselection_take_i64_1000x1` | `benchmarks/benchmarks/bench_itemselection.py::Take.time_contiguous(shape=(1000, 1), mode='raise', dtype='int64')` | 2000 | 0.888 | 3.871 | 4.36x | numrust | ok |
| `asv_itemselection_putmask_dense_scalar_f64_1000` | `benchmarks/benchmarks/bench_itemselection.py::PutMask.time_dense(values_is_scalar=True, dtype=float64)` | 10000 | 0.632 | 5.363 | 8.48x | numrust | ok |
| `asv_itemselection_putmask_sparse_scalar_f64_1000` | `benchmarks/benchmarks/bench_itemselection.py::PutMask.time_sparse(values_is_scalar=True, dtype=float64)` | 10000 | 0.026 | 4.666 | 182.07x | numrust | ok |
| `asv_itemselection_put_ordered_f64_1000` | `benchmarks/benchmarks/bench_itemselection.py::Put.time_ordered(values_is_scalar=False, dtype=float64)` | 10000 | 7.007 | 19.766 | 2.82x | numrust | ok |
| `asv_manipulate_broadcast_arrays_f64_16x32` | `benchmarks/benchmarks/bench_manipulate.py::BroadcastArrays.time_broadcast_arrays(shape=(16, 32), ndtype=float64)` | 200000 | 57.724 | 1193.269 | 20.67x | numrust | ok |
| `asv_manipulate_broadcast_arrays_f64_128x256` | `benchmarks/benchmarks/bench_manipulate.py::BroadcastArrays.time_broadcast_arrays(shape=(128, 256), ndtype=float64)` | 100000 | 29.096 | 593.468 | 20.40x | numrust | ok |
| `asv_manipulate_broadcast_arrays_f64_512x1024` | `benchmarks/benchmarks/bench_manipulate.py::BroadcastArrays.time_broadcast_arrays(shape=(512, 1024), ndtype=float64)` | 100000 | 28.811 | 610.655 | 21.20x | numrust | ok |
| `asv_manipulate_broadcast_to_f64_16` | `benchmarks/benchmarks/bench_manipulate.py::BroadcastArraysTo.time_broadcast_to(size=16, ndtype=float64)` | 200000 | 10.865 | 179.025 | 16.48x | numrust | ok |
| `asv_manipulate_broadcast_to_f64_64` | `benchmarks/benchmarks/bench_manipulate.py::BroadcastArraysTo.time_broadcast_to(size=64, ndtype=float64)` | 200000 | 10.759 | 178.223 | 16.56x | numrust | ok |
| `asv_manipulate_broadcast_to_f64_512` | `benchmarks/benchmarks/bench_manipulate.py::BroadcastArraysTo.time_broadcast_to(size=512, ndtype=float64)` | 200000 | 10.802 | 183.108 | 16.95x | numrust | ok |
| `asv_manipulate_concatenate_ax0_f64_32x64_n5` | `benchmarks/benchmarks/bench_manipulate.py::ConcatenateStackArrays.time_concatenate_ax0(shape=(32, 64), narrays=5, ndtype=float64)` | 2000 | 1.746 | 3.555 | 2.04x | numrust | ok |
| `asv_manipulate_concatenate_ax1_f64_32x64_n5` | `benchmarks/benchmarks/bench_manipulate.py::ConcatenateStackArrays.time_concatenate_ax1(shape=(32, 64), narrays=5, ndtype=float64)` | 2000 | 2.444 | 4.715 | 1.93x | numrust | ok |
| `asv_manipulate_stack_ax0_f64_32x64_n5` | `benchmarks/benchmarks/bench_manipulate.py::ConcatenateStackArrays.time_stack_ax0(shape=(32, 64), narrays=5, ndtype=float64)` | 2000 | 1.812 | 5.155 | 2.85x | numrust | ok |
| `asv_manipulate_stack_ax1_f64_32x64_n5` | `benchmarks/benchmarks/bench_manipulate.py::ConcatenateStackArrays.time_stack_ax1(shape=(32, 64), narrays=5, ndtype=float64)` | 2000 | 2.397 | 6.452 | 2.69x | numrust | ok |
| `asv_manipulate_expand_dims_f64_5x2x3x1_axis1` | `benchmarks/benchmarks/bench_manipulate.py::DimsManipulations.time_expand_dims(shape=(5, 2, 3, 1))` | 200000 | 19.250 | 130.444 | 6.78x | numrust | ok |
| `asv_manipulate_expand_dims_neg_f64_5x2x3x1` | `benchmarks/benchmarks/bench_manipulate.py::DimsManipulations.time_expand_dims_neg(shape=(5, 2, 3, 1))` | 200000 | 18.478 | 130.318 | 7.05x | numrust | ok |
| `asv_manipulate_squeeze_dims_f64_5x2x3x1` | `benchmarks/benchmarks/bench_manipulate.py::DimsManipulations.time_squeeze_dims(shape=(5, 2, 3, 1))` | 200000 | 9.787 | 39.073 | 3.99x | numrust | ok |
| `asv_manipulate_flip_all_f64_5x2x3x1` | `benchmarks/benchmarks/bench_manipulate.py::DimsManipulations.time_flip_all(shape=(5, 2, 3, 1))` | 200000 | 28.957 | 508.648 | 17.57x | numrust | ok |
| `asv_manipulate_flip_one_f64_5x2x3x1_axis1` | `benchmarks/benchmarks/bench_manipulate.py::DimsManipulations.time_flip_one(shape=(5, 2, 3, 1))` | 200000 | 28.432 | 586.934 | 20.64x | numrust | ok |
| `asv_manipulate_flip_neg_f64_5x2x3x1_axis_neg1` | `benchmarks/benchmarks/bench_manipulate.py::DimsManipulations.time_flip_neg(shape=(5, 2, 3, 1))` | 200000 | 28.547 | 586.978 | 20.56x | numrust | ok |
| `asv_manipulate_moveaxis_f64_5x2x3x1` | `benchmarks/benchmarks/bench_manipulate.py::DimsManipulations.time_moveaxis(shape=(5, 2, 3, 1))` | 200000 | 38.729 | 686.844 | 17.73x | numrust | ok |
| `asv_manipulate_roll_f64_5x2x3x1_shift3` | `benchmarks/benchmarks/bench_manipulate.py::DimsManipulations.time_roll(shape=(5, 2, 3, 1))` | 100000 | 22.194 | 444.714 | 20.04x | numrust | ok |
| `asv_manipulate_reshape_f64_5x2x3x1_to_1x5x2x3` | `benchmarks/benchmarks/bench_manipulate.py::DimsManipulations.time_reshape(shape=(5, 2, 3, 1))` | 200000 | 20.319 | 61.135 | 3.01x | numrust | ok |
| `asv_linalg_dot_a_b_f64_150x400_400x600` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_dot_a_b` | 1000 | 144.010 | 146.880 | 1.02x | numrust | ok |
| `asv_linalg_matmul_a_b_f64_150x400_400x600` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_matmul_a_b` | 1000 | 143.885 | 144.447 | 1.00x | numrust | ok |
| `asv_linalg_matmul_d_matmul_b_c_f64` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_matmul_d_matmul_b_c` | 1000 | 4.044 | 4.654 | 1.15x | numrust | ok |
| `asv_linalg_dot_d_dot_b_c_f64` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_dot_d_dot_b_c` | 1000 | 4.068 | 4.647 | 1.14x | numrust | ok |
| `asv_linalg_dot_trans_a_at_f64_150x400_400x150` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_dot_trans_a_at` | 1000 | 29.926 | 34.152 | 1.14x | numrust | ok |
| `asv_linalg_dot_trans_a_atc_f64_150x400_400x150` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_dot_trans_a_atc` | 1000 | 41.575 | 43.223 | 1.04x | numrust | ok |
| `asv_linalg_dot_trans_at_a_f64_400x150_150x400` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_dot_trans_at_a` | 1000 | 78.549 | 95.949 | 1.22x | numrust | ok |
| `asv_linalg_dot_trans_atc_a_f64_400x150_150x400` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_dot_trans_atc_a` | 1000 | 94.820 | 100.756 | 1.06x | numrust | ok |
| `asv_linalg_inner_a_a_f64_150x400_150x400` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_inner_trans_a_a` | 1000 | 29.532 | 34.552 | 1.17x | numrust | ok |
| `asv_linalg_inner_a_ac_f64_150x400_150x400` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_inner_trans_a_ac` | 1000 | 47.583 | 50.185 | 1.05x | numrust | ok |
| `asv_linalg_matmul_trans_a_at_f64_150x400_400x150` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_matmul_trans_a_at` | 1000 | 30.725 | 30.725 | 1.00x | numrust | ok |
| `asv_linalg_matmul_trans_a_atc_f64_150x400_400x150` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_matmul_trans_a_atc` | 1000 | 42.526 | 42.493 | 1.00x | numpy | ok |
| `asv_linalg_matmul_trans_at_a_f64_400x150_150x400` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_matmul_trans_at_a` | 1000 | 78.403 | 79.600 | 1.02x | numrust | ok |
| `asv_linalg_matmul_trans_atc_a_f64_400x150_150x400` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_matmul_trans_atc_a` | 1000 | 96.061 | 96.414 | 1.00x | numrust | ok |
| `asv_linalg_tensordot_a3_b3_axes_10_01` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_tensordot_a_b_axes_1_0_0_1` | 10 | 2.908 | 4.673 | 1.61x | numrust | ok |
| `asv_linalg_norm_small_array_f64_5` | `benchmarks/benchmarks/bench_linalg.py::LinalgSmallArrays.time_norm_small_array` | 100000 | 4.942 | 40.848 | 8.27x | numrust | ok |
| `asv_linalg_det_small_array_f64_5x5` | `benchmarks/benchmarks/bench_linalg.py::LinalgSmallArrays.time_det_small_array` | 100000 | 8.333 | 90.810 | 10.90x | numrust | ok |
| `asv_linalg_det_3x3_f64` | `benchmarks/benchmarks/bench_linalg.py::LinalgSmallArrays.time_det_3x3` | 100000 | 6.925 | 86.915 | 12.55x | numrust | ok |
| `asv_linalg_solve_3x3_f64` | `benchmarks/benchmarks/bench_linalg.py::LinalgSmallArrays.time_solve_3x3` | 100000 | 14.176 | 172.553 | 12.17x | numrust | ok |
| `asv_linalg_lstsq_square_f64_100x100` | `benchmarks/benchmarks/bench_linalg.py::Lstsq.time_numpy_linalg_lstsq_a__b_float64` | 100 | 5.151 | 63.610 | 12.35x | numrust | ok |
| `asv_linalg_einsum_outer_f64_3000` | `benchmarks/benchmarks/bench_linalg.py::Einsum.time_einsum_outer(dtype=float64)` | 1 | 0.655 | 1.720 | 2.63x | numrust | ok |
| `asv_linalg_einsum_i_ij_j_f64_400_400x600_600` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_einsum_i_ij_j` | 1000 | 4.088 | 252.379 | 61.73x | numrust | ok |
| `asv_linalg_einsum_ij_jk_f64_150x400_400x600` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_einsum_ij_jk_a_b` | 1000 | 143.690 | 4149.942 | 28.88x | numrust | ok |
| `asv_linalg_einsum_multiply_f64_30x40_20x30x40` | `benchmarks/benchmarks/bench_linalg.py::Einsum.time_einsum_multiply(dtype=float64)` | 1 | 0.003 | 0.014 | 4.46x | numrust | ok |
| `asv_linalg_einsum_sum_mul_f64_scalar_10x100x10` | `benchmarks/benchmarks/bench_linalg.py::Einsum.time_einsum_sum_mul(dtype=float64)` | 100 | 0.050 | 0.651 | 13.13x | numrust | ok |
| `asv_linalg_einsum_sum_mul2_f64_10x100x10_scalar` | `benchmarks/benchmarks/bench_linalg.py::Einsum.time_einsum_sum_mul2(dtype=float64)` | 100 | 0.052 | 0.659 | 12.57x | numrust | ok |
| `asv_linalg_einsum_scalar_mul_f64_480000` | `benchmarks/benchmarks/bench_linalg.py::Einsum.time_einsum_mul(dtype=float64)` | 100 | 6.260 | 6.767 | 1.08x | numrust | ok |
| `asv_linalg_einsum_sum_f64_480000` | `benchmarks/benchmarks/bench_linalg.py::Einsum.time_einsum_contig_outstride0(dtype=float64)` | 100 | 3.325 | 4.123 | 1.24x | numrust | ok |
| `asv_linalg_einsum_weighted_sum_f64_400x600` | `benchmarks/benchmarks/bench_linalg.py::Einsum.time_einsum_contig_contig(dtype=float64)` | 100 | 0.399 | 4.371 | 10.96x | numrust | ok |
| `asv_linalg_einsum_noncon_outer_f64_2000` | `benchmarks/benchmarks/bench_linalg.py::Einsum.time_einsum_noncon_outer(dtype=float64)` | 1 | 0.584 | 1.496 | 2.56x | numrust | ok |
| `asv_linalg_einsum_noncon_multiply_f64_30x40_20x30x40` | `benchmarks/benchmarks/bench_linalg.py::Einsum.time_einsum_noncon_multiply(dtype=float64)` | 1 | 0.003 | 0.014 | 4.35x | numrust | ok |
| `asv_linalg_einsum_noncon_sum_mul_f64_scalar_20x30x40` | `benchmarks/benchmarks/bench_linalg.py::Einsum.time_einsum_noncon_sum_mul(dtype=float64)` | 100 | 0.157 | 0.780 | 4.97x | numrust | ok |
| `asv_linalg_einsum_noncon_sum_mul2_f64_20x30x40_scalar` | `benchmarks/benchmarks/bench_linalg.py::Einsum.time_einsum_noncon_sum_mul2(dtype=float64)` | 100 | 0.156 | 0.790 | 5.07x | numrust | ok |
| `asv_linalg_einsum_noncon_scalar_mul_f64_2000` | `benchmarks/benchmarks/bench_linalg.py::Einsum.time_einsum_noncon_mul(dtype=float64)` | 100 | 0.023 | 0.432 | 18.90x | numrust | ok |
| `asv_linalg_einsum_noncon_weighted_sum_f64_30x40` | `benchmarks/benchmarks/bench_linalg.py::Einsum.time_einsum_noncon_contig_contig(dtype=float64)` | 100 | 0.048 | 0.487 | 10.11x | numrust | ok |
| `asv_linalg_einsum_noncon_sum_f64_2000` | `benchmarks/benchmarks/bench_linalg.py::Einsum.time_einsum_noncon_contig_outstride0(dtype=float64)` | 100 | 0.013 | 0.258 | 20.09x | numrust | ok |

## Score

- Supported external cases: 72
- Unsupported external cases tracked: 1
- NumRust wins: 71
- NumPy wins: 1
- Geomean speedup vs NumPy: 9.07x
- Near-tie relative margin: 2%
- Near-tie cases: 6
- Ranked higher on supported external cases: True
- Global NumPy replacement claim: false

## NumPy-Winning Cases

| Case | NumRust ms | NumPy ms | NumRust pass ms | NumPy pass ms |
| --- | ---: | ---: | --- | --- |
| `asv_linalg_matmul_trans_a_atc_f64_150x400_400x150` | 42.526 | 42.493 | 41.179, 42.860, 40.543, 42.708, 42.526 | 41.305, 43.102, 45.355, 42.493, 41.204 |

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
