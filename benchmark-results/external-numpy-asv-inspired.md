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
| `asv_ufunc_broadcast_sub_f64` | `benchmarks/benchmarks/bench_ufunc.py::Broadcast.time_broadcast` | 1 | 0.822 | 1.522 | 1.85x | numrust | ok |
| `asv_ufunc_astype_i32_to_f64_100x100` | `benchmarks/benchmarks/bench_ufunc.py::NDArrayAsType.time_astype(typeconv=('int32', 'float64'))` | 5000 | 4.846 | 13.751 | 2.84x | numrust | ok |
| `asv_ufunc_add_at_f64_10000000` | `benchmarks/benchmarks/bench_ufunc.py::At.time_sum_at` | 1 | 3.959 | 5.727 | 1.45x | numrust | ok |
| `asv_ufunc_maximum_at_f64_10000000` | `benchmarks/benchmarks/bench_ufunc.py::At.time_maximum_at` | 1 | 4.793 | 5.857 | 1.22x | numrust | ok |
| `asv_reduce_small_sum_f32_100` | `benchmarks/benchmarks/bench_reduce.py::SmallReduction.time_small` | 200000 | 2.759 | 131.471 | 47.65x | numrust | ok |
| `asv_reduce_stats_min_f64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_min(dtype=float64)` | 100000 | 0.334 | 65.285 | 195.76x | numrust | ok |
| `asv_reduce_stats_min_f32_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_min(dtype=float32)` | 100000 | 0.338 | 64.432 | 190.65x | numrust | ok |
| `asv_reduce_stats_min_i64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_min(dtype=int64)` | 100000 | 0.109 | 65.232 | 599.84x | numrust | ok |
| `asv_reduce_stats_min_u64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_min(dtype=uint64)` | 100000 | 0.109 | 65.368 | 601.55x | numrust | ok |
| `asv_reduce_stats_min_bool_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_min(dtype=bool_)` | 100000 | 0.105 | 69.188 | 658.41x | numrust | ok |
| `asv_reduce_stats_max_f64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_max(dtype=float64)` | 100000 | 0.334 | 65.197 | 195.01x | numrust | ok |
| `asv_reduce_stats_max_f32_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_max(dtype=float32)` | 100000 | 0.336 | 64.976 | 193.48x | numrust | ok |
| `asv_reduce_stats_max_i64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_max(dtype=int64)` | 100000 | 0.109 | 65.224 | 600.68x | numrust | ok |
| `asv_reduce_stats_max_u64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_max(dtype=uint64)` | 100000 | 0.109 | 65.352 | 600.94x | numrust | ok |
| `asv_reduce_stats_max_bool_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_max(dtype=bool_)` | 100000 | 0.105 | 68.710 | 653.87x | numrust | ok |
| `asv_reduce_stats_mean_f64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_mean(dtype=float64)` | 100000 | 0.111 | 99.653 | 894.41x | numrust | ok |
| `asv_reduce_stats_mean_f32_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_mean(dtype=float32)` | 100000 | 0.120 | 167.720 | 1392.34x | numrust | ok |
| `asv_reduce_stats_mean_i64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_mean(dtype=int64)` | 100000 | 0.111 | 116.591 | 1050.37x | numrust | ok |
| `asv_reduce_stats_mean_u64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_mean(dtype=uint64)` | 100000 | 0.113 | 116.227 | 1028.56x | numrust | ok |
| `asv_reduce_stats_mean_bool_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_mean(dtype=bool_)` | 100000 | 0.117 | 116.574 | 992.82x | numrust | ok |
| `asv_reduce_stats_std_f64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_std(dtype=float64)` | 100000 | 0.107 | 325.829 | 3055.84x | numrust | ok |
| `asv_reduce_stats_std_f32_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_std(dtype=float32)` | 100000 | 0.106 | 414.162 | 3910.25x | numrust | ok |
| `asv_reduce_stats_std_i64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_std(dtype=int64)` | 100000 | 0.107 | 366.569 | 3423.22x | numrust | ok |
| `asv_reduce_stats_std_u64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_std(dtype=uint64)` | 100000 | 0.117 | 366.259 | 3137.12x | numrust | ok |
| `asv_reduce_stats_std_bool_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_std(dtype=bool_)` | 100000 | 0.106 | 395.052 | 3738.65x | numrust | ok |
| `asv_reduce_stats_prod_f64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_prod(dtype=float64)` | 100000 | 8.714 | 74.957 | 8.60x | numrust | ok |
| `asv_reduce_stats_prod_f32_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_prod(dtype=float32)` | 100000 | 8.745 | 74.926 | 8.57x | numrust | ok |
| `asv_reduce_stats_prod_i64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_prod(dtype=int64)` | 100000 | 3.007 | 66.117 | 21.99x | numrust | ok |
| `asv_reduce_stats_prod_u64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_prod(dtype=uint64)` | 100000 | 3.016 | 66.171 | 21.94x | numrust | ok |
| `asv_reduce_stats_prod_bool_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_prod(dtype=bool_)` | 100000 | 0.105 | 74.281 | 706.87x | numrust | ok |
| `asv_reduce_stats_var_f64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_var(dtype=float64)` | 100000 | 0.107 | 305.099 | 2860.29x | numrust | ok |
| `asv_reduce_stats_var_f32_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_var(dtype=float32)` | 100000 | 0.106 | 381.498 | 3608.98x | numrust | ok |
| `asv_reduce_stats_var_i64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_var(dtype=int64)` | 100000 | 0.107 | 345.427 | 3227.06x | numrust | ok |
| `asv_reduce_stats_var_u64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_var(dtype=uint64)` | 100000 | 0.106 | 346.104 | 3256.20x | numrust | ok |
| `asv_reduce_stats_var_bool_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_var(dtype=bool_)` | 100000 | 0.105 | 371.778 | 3528.14x | numrust | ok |
| `asv_reduce_argmax_i64_200000` | `benchmarks/benchmarks/bench_reduce.py::ArgMax.time_argmax(dtype=int64)` | 20000 | 0.026 | 482.437 | 18857.71x | numrust | ok |
| `asv_reduce_argmin_i64_200000` | `benchmarks/benchmarks/bench_reduce.py::ArgMin.time_argmin(dtype=int64)` | 20000 | 0.026 | 516.179 | 20143.55x | numrust | ok |
| `asv_itemselection_take_i64_1000x1` | `benchmarks/benchmarks/bench_itemselection.py::Take.time_contiguous(shape=(1000, 1), mode='raise', dtype='int64')` | 2000 | 0.881 | 3.384 | 3.84x | numrust | ok |
| `asv_itemselection_putmask_dense_scalar_f64_1000` | `benchmarks/benchmarks/bench_itemselection.py::PutMask.time_dense(values_is_scalar=True, dtype=float64)` | 10000 | 1.108 | 4.965 | 4.48x | numrust | ok |
| `asv_itemselection_putmask_sparse_scalar_f64_1000` | `benchmarks/benchmarks/bench_itemselection.py::PutMask.time_sparse(values_is_scalar=True, dtype=float64)` | 10000 | 0.026 | 4.678 | 182.26x | numrust | ok |
| `asv_itemselection_put_ordered_f64_1000` | `benchmarks/benchmarks/bench_itemselection.py::Put.time_ordered(values_is_scalar=False, dtype=float64)` | 10000 | 7.004 | 19.725 | 2.82x | numrust | ok |
| `asv_manipulate_broadcast_arrays_f64_16x32` | `benchmarks/benchmarks/bench_manipulate.py::BroadcastArrays.time_broadcast_arrays(shape=(16, 32), ndtype=float64)` | 200000 | 57.397 | 1199.863 | 20.90x | numrust | ok |
| `asv_manipulate_broadcast_arrays_f64_128x256` | `benchmarks/benchmarks/bench_manipulate.py::BroadcastArrays.time_broadcast_arrays(shape=(128, 256), ndtype=float64)` | 100000 | 28.917 | 594.745 | 20.57x | numrust | ok |
| `asv_manipulate_broadcast_arrays_f32_128x256` | `benchmarks/benchmarks/bench_manipulate.py::BroadcastArrays.time_broadcast_arrays(shape=(128, 256), ndtype=float32)` | 100000 | 28.914 | 599.340 | 20.73x | numrust | ok |
| `asv_manipulate_broadcast_arrays_i32_128x256` | `benchmarks/benchmarks/bench_manipulate.py::BroadcastArrays.time_broadcast_arrays(shape=(128, 256), ndtype=int32)` | 100000 | 28.995 | 596.786 | 20.58x | numrust | ok |
| `asv_manipulate_broadcast_arrays_f64_512x1024` | `benchmarks/benchmarks/bench_manipulate.py::BroadcastArrays.time_broadcast_arrays(shape=(512, 1024), ndtype=float64)` | 100000 | 28.708 | 617.343 | 21.50x | numrust | ok |
| `asv_manipulate_broadcast_to_f64_16` | `benchmarks/benchmarks/bench_manipulate.py::BroadcastArraysTo.time_broadcast_to(size=16, ndtype=float64)` | 200000 | 10.831 | 181.436 | 16.75x | numrust | ok |
| `asv_manipulate_broadcast_to_f64_64` | `benchmarks/benchmarks/bench_manipulate.py::BroadcastArraysTo.time_broadcast_to(size=64, ndtype=float64)` | 200000 | 10.824 | 180.692 | 16.69x | numrust | ok |
| `asv_manipulate_broadcast_to_f32_64` | `benchmarks/benchmarks/bench_manipulate.py::BroadcastArraysTo.time_broadcast_to(size=64, ndtype=float32)` | 200000 | 10.761 | 181.781 | 16.89x | numrust | ok |
| `asv_manipulate_broadcast_to_i32_64` | `benchmarks/benchmarks/bench_manipulate.py::BroadcastArraysTo.time_broadcast_to(size=64, ndtype=int32)` | 200000 | 10.719 | 181.598 | 16.94x | numrust | ok |
| `asv_manipulate_broadcast_to_f64_512` | `benchmarks/benchmarks/bench_manipulate.py::BroadcastArraysTo.time_broadcast_to(size=512, ndtype=float64)` | 200000 | 10.744 | 184.161 | 17.14x | numrust | ok |
| `asv_manipulate_concatenate_ax0_f64_32x64_n5` | `benchmarks/benchmarks/bench_manipulate.py::ConcatenateStackArrays.time_concatenate_ax0(shape=(32, 64), narrays=5, ndtype=float64)` | 2000 | 1.724 | 3.540 | 2.05x | numrust | ok |
| `asv_manipulate_concatenate_ax0_f32_32x64_n5` | `benchmarks/benchmarks/bench_manipulate.py::ConcatenateStackArrays.time_concatenate_ax0(shape=(32, 64), narrays=5, ndtype=float32)` | 2000 | 1.108 | 2.736 | 2.47x | numrust | ok |
| `asv_manipulate_concatenate_ax0_i32_32x64_n5` | `benchmarks/benchmarks/bench_manipulate.py::ConcatenateStackArrays.time_concatenate_ax0(shape=(32, 64), narrays=5, ndtype=int32)` | 2000 | 1.091 | 2.815 | 2.58x | numrust | ok |
| `asv_manipulate_concatenate_ax1_f64_32x64_n5` | `benchmarks/benchmarks/bench_manipulate.py::ConcatenateStackArrays.time_concatenate_ax1(shape=(32, 64), narrays=5, ndtype=float64)` | 2000 | 2.393 | 4.698 | 1.96x | numrust | ok |
| `asv_manipulate_concatenate_ax1_f32_32x64_n5` | `benchmarks/benchmarks/bench_manipulate.py::ConcatenateStackArrays.time_concatenate_ax1(shape=(32, 64), narrays=5, ndtype=float32)` | 2000 | 1.577 | 2.910 | 1.85x | numrust | ok |
| `asv_manipulate_concatenate_ax1_i32_32x64_n5` | `benchmarks/benchmarks/bench_manipulate.py::ConcatenateStackArrays.time_concatenate_ax1(shape=(32, 64), narrays=5, ndtype=int32)` | 2000 | 1.484 | 2.964 | 2.00x | numrust | ok |
| `asv_manipulate_stack_ax0_f64_32x64_n5` | `benchmarks/benchmarks/bench_manipulate.py::ConcatenateStackArrays.time_stack_ax0(shape=(32, 64), narrays=5, ndtype=float64)` | 2000 | 1.783 | 5.132 | 2.88x | numrust | ok |
| `asv_manipulate_stack_ax0_f32_32x64_n5` | `benchmarks/benchmarks/bench_manipulate.py::ConcatenateStackArrays.time_stack_ax0(shape=(32, 64), narrays=5, ndtype=float32)` | 2000 | 1.263 | 4.440 | 3.51x | numrust | ok |
| `asv_manipulate_stack_ax0_i32_32x64_n5` | `benchmarks/benchmarks/bench_manipulate.py::ConcatenateStackArrays.time_stack_ax0(shape=(32, 64), narrays=5, ndtype=int32)` | 2000 | 1.315 | 4.490 | 3.41x | numrust | ok |
| `asv_manipulate_stack_ax1_f64_32x64_n5` | `benchmarks/benchmarks/bench_manipulate.py::ConcatenateStackArrays.time_stack_ax1(shape=(32, 64), narrays=5, ndtype=float64)` | 2000 | 2.186 | 6.437 | 2.94x | numrust | ok |
| `asv_manipulate_stack_ax1_f32_32x64_n5` | `benchmarks/benchmarks/bench_manipulate.py::ConcatenateStackArrays.time_stack_ax1(shape=(32, 64), narrays=5, ndtype=float32)` | 2000 | 1.460 | 4.601 | 3.15x | numrust | ok |
| `asv_manipulate_stack_ax1_i32_32x64_n5` | `benchmarks/benchmarks/bench_manipulate.py::ConcatenateStackArrays.time_stack_ax1(shape=(32, 64), narrays=5, ndtype=int32)` | 2000 | 1.622 | 4.709 | 2.90x | numrust | ok |
| `asv_manipulate_expand_dims_f64_5x2x3x1_axis1` | `benchmarks/benchmarks/bench_manipulate.py::DimsManipulations.time_expand_dims(shape=(5, 2, 3, 1))` | 200000 | 19.392 | 131.813 | 6.80x | numrust | ok |
| `asv_manipulate_expand_dims_neg_f64_5x2x3x1` | `benchmarks/benchmarks/bench_manipulate.py::DimsManipulations.time_expand_dims_neg(shape=(5, 2, 3, 1))` | 200000 | 18.763 | 130.984 | 6.98x | numrust | ok |
| `asv_manipulate_squeeze_dims_f64_5x2x3x1` | `benchmarks/benchmarks/bench_manipulate.py::DimsManipulations.time_squeeze_dims(shape=(5, 2, 3, 1))` | 200000 | 9.614 | 37.462 | 3.90x | numrust | ok |
| `asv_manipulate_flip_all_f64_5x2x3x1` | `benchmarks/benchmarks/bench_manipulate.py::DimsManipulations.time_flip_all(shape=(5, 2, 3, 1))` | 200000 | 29.153 | 506.764 | 17.38x | numrust | ok |
| `asv_manipulate_flip_one_f64_5x2x3x1_axis1` | `benchmarks/benchmarks/bench_manipulate.py::DimsManipulations.time_flip_one(shape=(5, 2, 3, 1))` | 200000 | 28.641 | 590.694 | 20.62x | numrust | ok |
| `asv_manipulate_flip_neg_f64_5x2x3x1_axis_neg1` | `benchmarks/benchmarks/bench_manipulate.py::DimsManipulations.time_flip_neg(shape=(5, 2, 3, 1))` | 200000 | 28.769 | 590.552 | 20.53x | numrust | ok |
| `asv_manipulate_moveaxis_f64_5x2x3x1` | `benchmarks/benchmarks/bench_manipulate.py::DimsManipulations.time_moveaxis(shape=(5, 2, 3, 1))` | 200000 | 38.805 | 683.889 | 17.62x | numrust | ok |
| `asv_manipulate_roll_f64_5x2x3x1_shift3` | `benchmarks/benchmarks/bench_manipulate.py::DimsManipulations.time_roll(shape=(5, 2, 3, 1))` | 100000 | 21.792 | 446.151 | 20.47x | numrust | ok |
| `asv_manipulate_reshape_f64_5x2x3x1_to_1x5x2x3` | `benchmarks/benchmarks/bench_manipulate.py::DimsManipulations.time_reshape(shape=(5, 2, 3, 1))` | 200000 | 20.350 | 61.048 | 3.00x | numrust | ok |
| `asv_linalg_dot_a_b_f64_150x400_400x600` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_dot_a_b` | 1000 | 144.111 | 146.613 | 1.02x | numrust | ok |
| `asv_linalg_matmul_a_b_f64_150x400_400x600` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_matmul_a_b` | 1000 | 144.018 | 144.635 | 1.00x | numrust | ok |
| `asv_linalg_matmul_d_matmul_b_c_f64` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_matmul_d_matmul_b_c` | 1000 | 4.362 | 4.732 | 1.08x | numrust | ok |
| `asv_linalg_dot_d_dot_b_c_f64` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_dot_d_dot_b_c` | 1000 | 4.079 | 4.712 | 1.16x | numrust | ok |
| `asv_linalg_dot_trans_a_at_f64_150x400_400x150` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_dot_trans_a_at` | 1000 | 29.712 | 33.344 | 1.12x | numrust | ok |
| `asv_linalg_dot_trans_a_atc_f64_150x400_400x150` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_dot_trans_a_atc` | 1000 | 40.904 | 41.843 | 1.02x | numrust | ok |
| `asv_linalg_dot_trans_at_a_f64_400x150_150x400` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_dot_trans_at_a` | 1000 | 78.378 | 95.986 | 1.22x | numrust | ok |
| `asv_linalg_dot_trans_atc_a_f64_400x150_150x400` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_dot_trans_atc_a` | 1000 | 96.090 | 100.071 | 1.04x | numrust | ok |
| `asv_linalg_inner_a_a_f64_150x400_150x400` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_inner_trans_a_a` | 1000 | 30.229 | 33.126 | 1.10x | numrust | ok |
| `asv_linalg_inner_a_ac_f64_150x400_150x400` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_inner_trans_a_ac` | 1000 | 49.355 | 50.663 | 1.03x | numrust | ok |
| `asv_linalg_matmul_trans_a_at_f64_150x400_400x150` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_matmul_trans_a_at` | 1000 | 30.644 | 30.921 | 1.01x | numrust | ok |
| `asv_linalg_matmul_trans_a_atc_f64_150x400_400x150` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_matmul_trans_a_atc` | 1000 | 42.661 | 43.226 | 1.01x | numrust | ok |
| `asv_linalg_matmul_trans_at_a_f64_400x150_150x400` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_matmul_trans_at_a` | 1000 | 77.901 | 79.037 | 1.01x | numrust | ok |
| `asv_linalg_matmul_trans_atc_a_f64_400x150_150x400` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_matmul_trans_atc_a` | 1000 | 95.969 | 96.454 | 1.01x | numrust | ok |
| `asv_linalg_tensordot_a3_b3_axes_10_01` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_tensordot_a_b_axes_1_0_0_1` | 10 | 2.936 | 4.427 | 1.51x | numrust | ok |
| `asv_linalg_norm_small_array_f64_5` | `benchmarks/benchmarks/bench_linalg.py::LinalgSmallArrays.time_norm_small_array` | 100000 | 4.651 | 40.456 | 8.70x | numrust | ok |
| `asv_linalg_det_small_array_f64_5x5` | `benchmarks/benchmarks/bench_linalg.py::LinalgSmallArrays.time_det_small_array` | 100000 | 8.344 | 90.864 | 10.89x | numrust | ok |
| `asv_linalg_det_3x3_f64` | `benchmarks/benchmarks/bench_linalg.py::LinalgSmallArrays.time_det_3x3` | 100000 | 6.968 | 87.198 | 12.51x | numrust | ok |
| `asv_linalg_solve_3x3_f64` | `benchmarks/benchmarks/bench_linalg.py::LinalgSmallArrays.time_solve_3x3` | 100000 | 14.041 | 172.950 | 12.32x | numrust | ok |
| `asv_linalg_lstsq_square_f64_100x100` | `benchmarks/benchmarks/bench_linalg.py::Lstsq.time_numpy_linalg_lstsq_a__b_float64` | 100 | 5.144 | 63.713 | 12.39x | numrust | ok |
| `asv_linalg_einsum_outer_f64_3000` | `benchmarks/benchmarks/bench_linalg.py::Einsum.time_einsum_outer(dtype=float64)` | 1 | 0.667 | 1.695 | 2.54x | numrust | ok |
| `asv_linalg_einsum_i_ij_j_f64_400_400x600_600` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_einsum_i_ij_j` | 1000 | 4.203 | 250.988 | 59.71x | numrust | ok |
| `asv_linalg_einsum_ij_jk_f64_150x400_400x600` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_einsum_ij_jk_a_b` | 1000 | 143.759 | 4140.033 | 28.80x | numrust | ok |
| `asv_linalg_einsum_multiply_f64_30x40_20x30x40` | `benchmarks/benchmarks/bench_linalg.py::Einsum.time_einsum_multiply(dtype=float64)` | 1 | 0.003 | 0.015 | 4.65x | numrust | ok |
| `asv_linalg_einsum_sum_mul_f64_scalar_10x100x10` | `benchmarks/benchmarks/bench_linalg.py::Einsum.time_einsum_sum_mul(dtype=float64)` | 100 | 0.051 | 0.655 | 12.72x | numrust | ok |
| `asv_linalg_einsum_sum_mul2_f64_10x100x10_scalar` | `benchmarks/benchmarks/bench_linalg.py::Einsum.time_einsum_sum_mul2(dtype=float64)` | 100 | 0.052 | 0.652 | 12.45x | numrust | ok |
| `asv_linalg_einsum_scalar_mul_f64_480000` | `benchmarks/benchmarks/bench_linalg.py::Einsum.time_einsum_mul(dtype=float64)` | 100 | 6.267 | 6.751 | 1.08x | numrust | ok |
| `asv_linalg_einsum_sum_f64_480000` | `benchmarks/benchmarks/bench_linalg.py::Einsum.time_einsum_contig_outstride0(dtype=float64)` | 100 | 3.344 | 4.110 | 1.23x | numrust | ok |
| `asv_linalg_einsum_weighted_sum_f64_400x600` | `benchmarks/benchmarks/bench_linalg.py::Einsum.time_einsum_contig_contig(dtype=float64)` | 100 | 0.427 | 4.380 | 10.26x | numrust | ok |
| `asv_linalg_einsum_noncon_outer_f64_2000` | `benchmarks/benchmarks/bench_linalg.py::Einsum.time_einsum_noncon_outer(dtype=float64)` | 1 | 0.577 | 1.497 | 2.59x | numrust | ok |
| `asv_linalg_einsum_noncon_multiply_f64_30x40_20x30x40` | `benchmarks/benchmarks/bench_linalg.py::Einsum.time_einsum_noncon_multiply(dtype=float64)` | 1 | 0.003 | 0.014 | 4.27x | numrust | ok |
| `asv_linalg_einsum_noncon_sum_mul_f64_scalar_20x30x40` | `benchmarks/benchmarks/bench_linalg.py::Einsum.time_einsum_noncon_sum_mul(dtype=float64)` | 100 | 0.155 | 0.777 | 5.02x | numrust | ok |
| `asv_linalg_einsum_noncon_sum_mul2_f64_20x30x40_scalar` | `benchmarks/benchmarks/bench_linalg.py::Einsum.time_einsum_noncon_sum_mul2(dtype=float64)` | 100 | 0.157 | 0.774 | 4.94x | numrust | ok |
| `asv_linalg_einsum_noncon_scalar_mul_f64_2000` | `benchmarks/benchmarks/bench_linalg.py::Einsum.time_einsum_noncon_mul(dtype=float64)` | 100 | 0.024 | 0.423 | 17.75x | numrust | ok |
| `asv_linalg_einsum_noncon_weighted_sum_f64_30x40` | `benchmarks/benchmarks/bench_linalg.py::Einsum.time_einsum_noncon_contig_contig(dtype=float64)` | 100 | 0.048 | 0.483 | 10.00x | numrust | ok |
| `asv_linalg_einsum_noncon_sum_f64_2000` | `benchmarks/benchmarks/bench_linalg.py::Einsum.time_einsum_noncon_contig_outstride0(dtype=float64)` | 100 | 0.012 | 0.255 | 21.23x | numrust | ok |

## Score

- Supported external cases: 108
- Unsupported external cases tracked: 1
- NumRust wins: 108
- NumPy wins: 0
- Geomean speedup vs NumPy: 22.07x
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
