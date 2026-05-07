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
| `asv_ufunc_broadcast_sub_f64` | `benchmarks/benchmarks/bench_ufunc.py::Broadcast.time_broadcast` | 1 | 0.863 | 1.493 | 1.73x | numrust | ok |
| `asv_ufunc_astype_i32_to_f64_100x100` | `benchmarks/benchmarks/bench_ufunc.py::NDArrayAsType.time_astype(typeconv=('int32', 'float64'))` | 5000 | 4.803 | 13.783 | 2.87x | numrust | ok |
| `asv_ufunc_add_at_f64_10000000` | `benchmarks/benchmarks/bench_ufunc.py::At.time_sum_at` | 1 | 3.976 | 5.697 | 1.43x | numrust | ok |
| `asv_ufunc_maximum_at_f64_10000000` | `benchmarks/benchmarks/bench_ufunc.py::At.time_maximum_at` | 1 | 4.775 | 5.862 | 1.23x | numrust | ok |
| `asv_reduce_small_sum_f32_100` | `benchmarks/benchmarks/bench_reduce.py::SmallReduction.time_small` | 200000 | 2.755 | 132.606 | 48.13x | numrust | ok |
| `asv_reduce_stats_min_f64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_min(dtype=float64)` | 100000 | 0.335 | 66.346 | 197.90x | numrust | ok |
| `asv_reduce_stats_min_f32_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_min(dtype=float32)` | 100000 | 0.332 | 65.197 | 196.11x | numrust | ok |
| `asv_reduce_stats_min_i64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_min(dtype=int64)` | 100000 | 0.109 | 65.763 | 601.72x | numrust | ok |
| `asv_reduce_stats_min_u64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_min(dtype=uint64)` | 100000 | 0.118 | 66.230 | 563.26x | numrust | ok |
| `asv_reduce_stats_max_f64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_max(dtype=float64)` | 100000 | 0.334 | 65.410 | 195.67x | numrust | ok |
| `asv_reduce_stats_max_f32_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_max(dtype=float32)` | 100000 | 0.334 | 64.955 | 194.31x | numrust | ok |
| `asv_reduce_stats_max_i64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_max(dtype=int64)` | 100000 | 0.109 | 65.551 | 603.00x | numrust | ok |
| `asv_reduce_stats_max_u64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_max(dtype=uint64)` | 100000 | 0.108 | 65.187 | 602.89x | numrust | ok |
| `asv_reduce_stats_mean_f64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_mean(dtype=float64)` | 100000 | 0.112 | 100.455 | 896.25x | numrust | ok |
| `asv_reduce_stats_mean_f32_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_mean(dtype=float32)` | 100000 | 0.113 | 167.513 | 1481.32x | numrust | ok |
| `asv_reduce_stats_mean_i64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_mean(dtype=int64)` | 100000 | 0.121 | 116.618 | 966.45x | numrust | ok |
| `asv_reduce_stats_mean_u64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_mean(dtype=uint64)` | 100000 | 0.119 | 115.972 | 974.21x | numrust | ok |
| `asv_reduce_stats_std_f64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_std(dtype=float64)` | 100000 | 0.106 | 330.981 | 3109.03x | numrust | ok |
| `asv_reduce_stats_std_f32_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_std(dtype=float32)` | 100000 | 0.106 | 411.902 | 3901.21x | numrust | ok |
| `asv_reduce_stats_std_i64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_std(dtype=int64)` | 100000 | 0.108 | 369.700 | 3423.15x | numrust | ok |
| `asv_reduce_stats_std_u64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_std(dtype=uint64)` | 100000 | 0.107 | 370.968 | 3477.85x | numrust | ok |
| `asv_reduce_stats_prod_f64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_prod(dtype=float64)` | 100000 | 8.714 | 75.619 | 8.68x | numrust | ok |
| `asv_reduce_stats_prod_f32_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_prod(dtype=float32)` | 100000 | 8.751 | 75.000 | 8.57x | numrust | ok |
| `asv_reduce_stats_prod_i64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_prod(dtype=int64)` | 100000 | 3.005 | 66.266 | 22.05x | numrust | ok |
| `asv_reduce_stats_prod_u64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_prod(dtype=uint64)` | 100000 | 3.003 | 66.677 | 22.20x | numrust | ok |
| `asv_reduce_stats_var_f64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_var(dtype=float64)` | 100000 | 0.106 | 306.617 | 2896.03x | numrust | ok |
| `asv_reduce_stats_var_f32_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_var(dtype=float32)` | 100000 | 0.106 | 381.936 | 3607.43x | numrust | ok |
| `asv_reduce_stats_var_i64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_var(dtype=int64)` | 100000 | 0.107 | 346.297 | 3233.91x | numrust | ok |
| `asv_reduce_stats_var_u64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_var(dtype=uint64)` | 100000 | 0.106 | 349.348 | 3290.55x | numrust | ok |
| `asv_reduce_argmax_i64_200000` | `benchmarks/benchmarks/bench_reduce.py::ArgMax.time_argmax(dtype=int64)` | 20000 | 0.026 | 482.234 | 18697.77x | numrust | ok |
| `asv_reduce_argmin_i64_200000` | `benchmarks/benchmarks/bench_reduce.py::ArgMin.time_argmin(dtype=int64)` | 20000 | 0.026 | 516.453 | 19991.97x | numrust | ok |
| `asv_itemselection_take_i64_1000x1` | `benchmarks/benchmarks/bench_itemselection.py::Take.time_contiguous(shape=(1000, 1), mode='raise', dtype='int64')` | 2000 | 0.888 | 3.826 | 4.31x | numrust | ok |
| `asv_itemselection_putmask_dense_scalar_f64_1000` | `benchmarks/benchmarks/bench_itemselection.py::PutMask.time_dense(values_is_scalar=True, dtype=float64)` | 10000 | 0.684 | 4.965 | 7.26x | numrust | ok |
| `asv_itemselection_putmask_sparse_scalar_f64_1000` | `benchmarks/benchmarks/bench_itemselection.py::PutMask.time_sparse(values_is_scalar=True, dtype=float64)` | 10000 | 0.026 | 4.785 | 185.83x | numrust | ok |
| `asv_itemselection_put_ordered_f64_1000` | `benchmarks/benchmarks/bench_itemselection.py::Put.time_ordered(values_is_scalar=False, dtype=float64)` | 10000 | 6.992 | 19.696 | 2.82x | numrust | ok |
| `asv_manipulate_broadcast_arrays_f64_16x32` | `benchmarks/benchmarks/bench_manipulate.py::BroadcastArrays.time_broadcast_arrays(shape=(16, 32), ndtype=float64)` | 200000 | 58.506 | 1203.069 | 20.56x | numrust | ok |
| `asv_manipulate_broadcast_arrays_f64_128x256` | `benchmarks/benchmarks/bench_manipulate.py::BroadcastArrays.time_broadcast_arrays(shape=(128, 256), ndtype=float64)` | 100000 | 29.127 | 593.722 | 20.38x | numrust | ok |
| `asv_manipulate_broadcast_arrays_f32_128x256` | `benchmarks/benchmarks/bench_manipulate.py::BroadcastArrays.time_broadcast_arrays(shape=(128, 256), ndtype=float32)` | 100000 | 29.173 | 597.931 | 20.50x | numrust | ok |
| `asv_manipulate_broadcast_arrays_i32_128x256` | `benchmarks/benchmarks/bench_manipulate.py::BroadcastArrays.time_broadcast_arrays(shape=(128, 256), ndtype=int32)` | 100000 | 29.244 | 596.725 | 20.40x | numrust | ok |
| `asv_manipulate_broadcast_arrays_f64_512x1024` | `benchmarks/benchmarks/bench_manipulate.py::BroadcastArrays.time_broadcast_arrays(shape=(512, 1024), ndtype=float64)` | 100000 | 29.124 | 613.866 | 21.08x | numrust | ok |
| `asv_manipulate_broadcast_to_f64_16` | `benchmarks/benchmarks/bench_manipulate.py::BroadcastArraysTo.time_broadcast_to(size=16, ndtype=float64)` | 200000 | 10.976 | 176.548 | 16.08x | numrust | ok |
| `asv_manipulate_broadcast_to_f64_64` | `benchmarks/benchmarks/bench_manipulate.py::BroadcastArraysTo.time_broadcast_to(size=64, ndtype=float64)` | 200000 | 10.961 | 176.358 | 16.09x | numrust | ok |
| `asv_manipulate_broadcast_to_f32_64` | `benchmarks/benchmarks/bench_manipulate.py::BroadcastArraysTo.time_broadcast_to(size=64, ndtype=float32)` | 200000 | 10.973 | 175.846 | 16.03x | numrust | ok |
| `asv_manipulate_broadcast_to_i32_64` | `benchmarks/benchmarks/bench_manipulate.py::BroadcastArraysTo.time_broadcast_to(size=64, ndtype=int32)` | 200000 | 10.905 | 176.034 | 16.14x | numrust | ok |
| `asv_manipulate_broadcast_to_f64_512` | `benchmarks/benchmarks/bench_manipulate.py::BroadcastArraysTo.time_broadcast_to(size=512, ndtype=float64)` | 200000 | 10.857 | 181.006 | 16.67x | numrust | ok |
| `asv_manipulate_concatenate_ax0_f64_32x64_n5` | `benchmarks/benchmarks/bench_manipulate.py::ConcatenateStackArrays.time_concatenate_ax0(shape=(32, 64), narrays=5, ndtype=float64)` | 2000 | 1.737 | 3.599 | 2.07x | numrust | ok |
| `asv_manipulate_concatenate_ax0_f32_32x64_n5` | `benchmarks/benchmarks/bench_manipulate.py::ConcatenateStackArrays.time_concatenate_ax0(shape=(32, 64), narrays=5, ndtype=float32)` | 2000 | 1.173 | 2.828 | 2.41x | numrust | ok |
| `asv_manipulate_concatenate_ax0_i32_32x64_n5` | `benchmarks/benchmarks/bench_manipulate.py::ConcatenateStackArrays.time_concatenate_ax0(shape=(32, 64), narrays=5, ndtype=int32)` | 2000 | 1.191 | 2.785 | 2.34x | numrust | ok |
| `asv_manipulate_concatenate_ax1_f64_32x64_n5` | `benchmarks/benchmarks/bench_manipulate.py::ConcatenateStackArrays.time_concatenate_ax1(shape=(32, 64), narrays=5, ndtype=float64)` | 2000 | 2.406 | 4.683 | 1.95x | numrust | ok |
| `asv_manipulate_concatenate_ax1_f32_32x64_n5` | `benchmarks/benchmarks/bench_manipulate.py::ConcatenateStackArrays.time_concatenate_ax1(shape=(32, 64), narrays=5, ndtype=float32)` | 2000 | 1.581 | 2.923 | 1.85x | numrust | ok |
| `asv_manipulate_concatenate_ax1_i32_32x64_n5` | `benchmarks/benchmarks/bench_manipulate.py::ConcatenateStackArrays.time_concatenate_ax1(shape=(32, 64), narrays=5, ndtype=int32)` | 2000 | 1.615 | 2.968 | 1.84x | numrust | ok |
| `asv_manipulate_stack_ax0_f64_32x64_n5` | `benchmarks/benchmarks/bench_manipulate.py::ConcatenateStackArrays.time_stack_ax0(shape=(32, 64), narrays=5, ndtype=float64)` | 2000 | 1.834 | 5.211 | 2.84x | numrust | ok |
| `asv_manipulate_stack_ax0_f32_32x64_n5` | `benchmarks/benchmarks/bench_manipulate.py::ConcatenateStackArrays.time_stack_ax0(shape=(32, 64), narrays=5, ndtype=float32)` | 2000 | 1.255 | 4.520 | 3.60x | numrust | ok |
| `asv_manipulate_stack_ax0_i32_32x64_n5` | `benchmarks/benchmarks/bench_manipulate.py::ConcatenateStackArrays.time_stack_ax0(shape=(32, 64), narrays=5, ndtype=int32)` | 2000 | 1.267 | 4.581 | 3.62x | numrust | ok |
| `asv_manipulate_stack_ax1_f64_32x64_n5` | `benchmarks/benchmarks/bench_manipulate.py::ConcatenateStackArrays.time_stack_ax1(shape=(32, 64), narrays=5, ndtype=float64)` | 2000 | 2.195 | 6.501 | 2.96x | numrust | ok |
| `asv_manipulate_stack_ax1_f32_32x64_n5` | `benchmarks/benchmarks/bench_manipulate.py::ConcatenateStackArrays.time_stack_ax1(shape=(32, 64), narrays=5, ndtype=float32)` | 2000 | 1.475 | 4.643 | 3.15x | numrust | ok |
| `asv_manipulate_stack_ax1_i32_32x64_n5` | `benchmarks/benchmarks/bench_manipulate.py::ConcatenateStackArrays.time_stack_ax1(shape=(32, 64), narrays=5, ndtype=int32)` | 2000 | 1.507 | 4.715 | 3.13x | numrust | ok |
| `asv_manipulate_expand_dims_f64_5x2x3x1_axis1` | `benchmarks/benchmarks/bench_manipulate.py::DimsManipulations.time_expand_dims(shape=(5, 2, 3, 1))` | 200000 | 19.736 | 135.269 | 6.85x | numrust | ok |
| `asv_manipulate_expand_dims_neg_f64_5x2x3x1` | `benchmarks/benchmarks/bench_manipulate.py::DimsManipulations.time_expand_dims_neg(shape=(5, 2, 3, 1))` | 200000 | 18.706 | 130.892 | 7.00x | numrust | ok |
| `asv_manipulate_squeeze_dims_f64_5x2x3x1` | `benchmarks/benchmarks/bench_manipulate.py::DimsManipulations.time_squeeze_dims(shape=(5, 2, 3, 1))` | 200000 | 9.587 | 38.216 | 3.99x | numrust | ok |
| `asv_manipulate_flip_all_f64_5x2x3x1` | `benchmarks/benchmarks/bench_manipulate.py::DimsManipulations.time_flip_all(shape=(5, 2, 3, 1))` | 200000 | 29.261 | 517.720 | 17.69x | numrust | ok |
| `asv_manipulate_flip_one_f64_5x2x3x1_axis1` | `benchmarks/benchmarks/bench_manipulate.py::DimsManipulations.time_flip_one(shape=(5, 2, 3, 1))` | 200000 | 28.879 | 596.552 | 20.66x | numrust | ok |
| `asv_manipulate_flip_neg_f64_5x2x3x1_axis_neg1` | `benchmarks/benchmarks/bench_manipulate.py::DimsManipulations.time_flip_neg(shape=(5, 2, 3, 1))` | 200000 | 28.541 | 594.901 | 20.84x | numrust | ok |
| `asv_manipulate_moveaxis_f64_5x2x3x1` | `benchmarks/benchmarks/bench_manipulate.py::DimsManipulations.time_moveaxis(shape=(5, 2, 3, 1))` | 200000 | 38.305 | 691.080 | 18.04x | numrust | ok |
| `asv_manipulate_roll_f64_5x2x3x1_shift3` | `benchmarks/benchmarks/bench_manipulate.py::DimsManipulations.time_roll(shape=(5, 2, 3, 1))` | 100000 | 21.444 | 449.555 | 20.96x | numrust | ok |
| `asv_manipulate_reshape_f64_5x2x3x1_to_1x5x2x3` | `benchmarks/benchmarks/bench_manipulate.py::DimsManipulations.time_reshape(shape=(5, 2, 3, 1))` | 200000 | 20.359 | 61.994 | 3.05x | numrust | ok |
| `asv_linalg_dot_a_b_f64_150x400_400x600` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_dot_a_b` | 1000 | 144.079 | 146.578 | 1.02x | numrust | ok |
| `asv_linalg_matmul_a_b_f64_150x400_400x600` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_matmul_a_b` | 1000 | 143.956 | 144.661 | 1.00x | numrust | ok |
| `asv_linalg_matmul_d_matmul_b_c_f64` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_matmul_d_matmul_b_c` | 1000 | 4.037 | 4.781 | 1.18x | numrust | ok |
| `asv_linalg_dot_d_dot_b_c_f64` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_dot_d_dot_b_c` | 1000 | 4.099 | 4.678 | 1.14x | numrust | ok |
| `asv_linalg_dot_trans_a_at_f64_150x400_400x150` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_dot_trans_a_at` | 1000 | 30.664 | 33.098 | 1.08x | numrust | ok |
| `asv_linalg_dot_trans_a_atc_f64_150x400_400x150` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_dot_trans_a_atc` | 1000 | 42.444 | 42.920 | 1.01x | numrust | ok |
| `asv_linalg_dot_trans_at_a_f64_400x150_150x400` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_dot_trans_at_a` | 1000 | 78.464 | 95.820 | 1.22x | numrust | ok |
| `asv_linalg_dot_trans_atc_a_f64_400x150_150x400` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_dot_trans_atc_a` | 1000 | 95.546 | 99.709 | 1.04x | numrust | ok |
| `asv_linalg_inner_a_a_f64_150x400_150x400` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_inner_trans_a_a` | 1000 | 30.701 | 34.475 | 1.12x | numrust | ok |
| `asv_linalg_inner_a_ac_f64_150x400_150x400` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_inner_trans_a_ac` | 1000 | 49.426 | 50.161 | 1.01x | numrust | ok |
| `asv_linalg_matmul_trans_a_at_f64_150x400_400x150` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_matmul_trans_a_at` | 1000 | 30.736 | 29.217 | 0.95x | numpy | ok |
| `asv_linalg_matmul_trans_a_atc_f64_150x400_400x150` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_matmul_trans_a_atc` | 1000 | 40.917 | 41.496 | 1.01x | numrust | ok |
| `asv_linalg_matmul_trans_at_a_f64_400x150_150x400` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_matmul_trans_at_a` | 1000 | 78.117 | 78.639 | 1.01x | numrust | ok |
| `asv_linalg_matmul_trans_atc_a_f64_400x150_150x400` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_matmul_trans_atc_a` | 1000 | 95.348 | 96.477 | 1.01x | numrust | ok |
| `asv_linalg_tensordot_a3_b3_axes_10_01` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_tensordot_a_b_axes_1_0_0_1` | 10 | 2.794 | 4.446 | 1.59x | numrust | ok |
| `asv_linalg_norm_small_array_f64_5` | `benchmarks/benchmarks/bench_linalg.py::LinalgSmallArrays.time_norm_small_array` | 100000 | 4.696 | 40.486 | 8.62x | numrust | ok |
| `asv_linalg_det_small_array_f64_5x5` | `benchmarks/benchmarks/bench_linalg.py::LinalgSmallArrays.time_det_small_array` | 100000 | 8.340 | 90.864 | 10.90x | numrust | ok |
| `asv_linalg_det_3x3_f64` | `benchmarks/benchmarks/bench_linalg.py::LinalgSmallArrays.time_det_3x3` | 100000 | 6.881 | 87.032 | 12.65x | numrust | ok |
| `asv_linalg_solve_3x3_f64` | `benchmarks/benchmarks/bench_linalg.py::LinalgSmallArrays.time_solve_3x3` | 100000 | 13.910 | 175.049 | 12.58x | numrust | ok |
| `asv_linalg_lstsq_square_f64_100x100` | `benchmarks/benchmarks/bench_linalg.py::Lstsq.time_numpy_linalg_lstsq_a__b_float64` | 100 | 5.152 | 63.583 | 12.34x | numrust | ok |
| `asv_linalg_einsum_outer_f64_3000` | `benchmarks/benchmarks/bench_linalg.py::Einsum.time_einsum_outer(dtype=float64)` | 1 | 0.654 | 1.695 | 2.59x | numrust | ok |
| `asv_linalg_einsum_i_ij_j_f64_400_400x600_600` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_einsum_i_ij_j` | 1000 | 4.179 | 251.368 | 60.15x | numrust | ok |
| `asv_linalg_einsum_ij_jk_f64_150x400_400x600` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_einsum_ij_jk_a_b` | 1000 | 143.850 | 4144.354 | 28.81x | numrust | ok |
| `asv_linalg_einsum_multiply_f64_30x40_20x30x40` | `benchmarks/benchmarks/bench_linalg.py::Einsum.time_einsum_multiply(dtype=float64)` | 1 | 0.003 | 0.014 | 4.57x | numrust | ok |
| `asv_linalg_einsum_sum_mul_f64_scalar_10x100x10` | `benchmarks/benchmarks/bench_linalg.py::Einsum.time_einsum_sum_mul(dtype=float64)` | 100 | 0.052 | 0.675 | 12.91x | numrust | ok |
| `asv_linalg_einsum_sum_mul2_f64_10x100x10_scalar` | `benchmarks/benchmarks/bench_linalg.py::Einsum.time_einsum_sum_mul2(dtype=float64)` | 100 | 0.052 | 0.657 | 12.64x | numrust | ok |
| `asv_linalg_einsum_scalar_mul_f64_480000` | `benchmarks/benchmarks/bench_linalg.py::Einsum.time_einsum_mul(dtype=float64)` | 100 | 6.250 | 6.715 | 1.07x | numrust | ok |
| `asv_linalg_einsum_sum_f64_480000` | `benchmarks/benchmarks/bench_linalg.py::Einsum.time_einsum_contig_outstride0(dtype=float64)` | 100 | 3.339 | 4.126 | 1.24x | numrust | ok |
| `asv_linalg_einsum_weighted_sum_f64_400x600` | `benchmarks/benchmarks/bench_linalg.py::Einsum.time_einsum_contig_contig(dtype=float64)` | 100 | 0.431 | 4.349 | 10.09x | numrust | ok |
| `asv_linalg_einsum_noncon_outer_f64_2000` | `benchmarks/benchmarks/bench_linalg.py::Einsum.time_einsum_noncon_outer(dtype=float64)` | 1 | 0.597 | 1.485 | 2.49x | numrust | ok |
| `asv_linalg_einsum_noncon_multiply_f64_30x40_20x30x40` | `benchmarks/benchmarks/bench_linalg.py::Einsum.time_einsum_noncon_multiply(dtype=float64)` | 1 | 0.003 | 0.014 | 4.30x | numrust | ok |
| `asv_linalg_einsum_noncon_sum_mul_f64_scalar_20x30x40` | `benchmarks/benchmarks/bench_linalg.py::Einsum.time_einsum_noncon_sum_mul(dtype=float64)` | 100 | 0.155 | 0.782 | 5.03x | numrust | ok |
| `asv_linalg_einsum_noncon_sum_mul2_f64_20x30x40_scalar` | `benchmarks/benchmarks/bench_linalg.py::Einsum.time_einsum_noncon_sum_mul2(dtype=float64)` | 100 | 0.157 | 0.785 | 5.01x | numrust | ok |
| `asv_linalg_einsum_noncon_scalar_mul_f64_2000` | `benchmarks/benchmarks/bench_linalg.py::Einsum.time_einsum_noncon_mul(dtype=float64)` | 100 | 0.023 | 0.420 | 18.14x | numrust | ok |
| `asv_linalg_einsum_noncon_weighted_sum_f64_30x40` | `benchmarks/benchmarks/bench_linalg.py::Einsum.time_einsum_noncon_contig_contig(dtype=float64)` | 100 | 0.051 | 0.486 | 9.54x | numrust | ok |
| `asv_linalg_einsum_noncon_sum_f64_2000` | `benchmarks/benchmarks/bench_linalg.py::Einsum.time_einsum_noncon_contig_outstride0(dtype=float64)` | 100 | 0.013 | 0.252 | 19.36x | numrust | ok |

## Score

- Supported external cases: 102
- Unsupported external cases tracked: 1
- NumRust wins: 101
- NumPy wins: 1
- Geomean speedup vs NumPy: 17.47x
- Near-tie relative margin: 2%
- Near-tie cases: 7
- Ranked higher on supported external cases: True
- Global NumPy replacement claim: false

## NumPy-Winning Cases

| Case | NumRust ms | NumPy ms | NumRust pass ms | NumPy pass ms |
| --- | ---: | ---: | --- | --- |
| `asv_linalg_matmul_trans_a_at_f64_150x400_400x150` | 30.736 | 29.217 | 30.002, 30.792, 30.778, 30.671, 30.736 | 31.193, 30.825, 29.023, 29.094, 29.217 |

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
