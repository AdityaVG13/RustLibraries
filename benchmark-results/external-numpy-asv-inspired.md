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
| `asv_ufunc_broadcast_sub_f64` | `benchmarks/benchmarks/bench_ufunc.py::Broadcast.time_broadcast` | 1 | 0.844 | 1.488 | 1.76x | numrust | ok |
| `asv_ufunc_astype_i32_to_f64_100x100` | `benchmarks/benchmarks/bench_ufunc.py::NDArrayAsType.time_astype(typeconv=('int32', 'float64'))` | 5000 | 4.810 | 13.778 | 2.86x | numrust | ok |
| `asv_ufunc_add_at_f64_10000000` | `benchmarks/benchmarks/bench_ufunc.py::At.time_sum_at` | 1 | 3.972 | 5.919 | 1.49x | numrust | ok |
| `asv_ufunc_maximum_at_f64_10000000` | `benchmarks/benchmarks/bench_ufunc.py::At.time_maximum_at` | 1 | 3.974 | 6.084 | 1.53x | numrust | ok |
| `asv_reduce_small_sum_f32_100` | `benchmarks/benchmarks/bench_reduce.py::SmallReduction.time_small` | 200000 | 2.760 | 131.686 | 47.71x | numrust | ok |
| `asv_reduce_stats_min_f64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_min(dtype=float64)` | 100000 | 0.113 | 65.940 | 582.47x | numrust | ok |
| `asv_reduce_stats_min_f32_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_min(dtype=float32)` | 100000 | 0.111 | 65.648 | 590.53x | numrust | ok |
| `asv_reduce_stats_min_i64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_min(dtype=int64)` | 100000 | 0.110 | 65.987 | 600.34x | numrust | ok |
| `asv_reduce_stats_min_u64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_min(dtype=uint64)` | 100000 | 0.109 | 65.782 | 604.66x | numrust | ok |
| `asv_reduce_stats_min_bool_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_min(dtype=bool_)` | 100000 | 0.105 | 70.009 | 666.23x | numrust | ok |
| `asv_reduce_stats_min_c64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_min(dtype=complex64)` | 100000 | 0.114 | 110.527 | 970.25x | numrust | ok |
| `asv_reduce_stats_max_f64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_max(dtype=float64)` | 100000 | 0.111 | 65.802 | 591.04x | numrust | ok |
| `asv_reduce_stats_max_f32_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_max(dtype=float32)` | 100000 | 0.111 | 65.011 | 585.47x | numrust | ok |
| `asv_reduce_stats_max_i64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_max(dtype=int64)` | 100000 | 0.109 | 65.862 | 605.63x | numrust | ok |
| `asv_reduce_stats_max_u64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_max(dtype=uint64)` | 100000 | 0.110 | 65.890 | 600.82x | numrust | ok |
| `asv_reduce_stats_max_bool_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_max(dtype=bool_)` | 100000 | 0.105 | 69.756 | 661.98x | numrust | ok |
| `asv_reduce_stats_max_c64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_max(dtype=complex64)` | 100000 | 0.114 | 110.328 | 965.32x | numrust | ok |
| `asv_reduce_stats_mean_f64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_mean(dtype=float64)` | 100000 | 0.111 | 101.823 | 913.89x | numrust | ok |
| `asv_reduce_stats_mean_f32_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_mean(dtype=float32)` | 100000 | 0.111 | 168.996 | 1517.36x | numrust | ok |
| `asv_reduce_stats_mean_i64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_mean(dtype=int64)` | 100000 | 0.116 | 117.606 | 1016.03x | numrust | ok |
| `asv_reduce_stats_mean_u64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_mean(dtype=uint64)` | 100000 | 0.112 | 118.272 | 1059.16x | numrust | ok |
| `asv_reduce_stats_mean_bool_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_mean(dtype=bool_)` | 100000 | 0.118 | 118.587 | 1005.33x | numrust | ok |
| `asv_reduce_stats_mean_c64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_mean(dtype=complex64)` | 100000 | 0.111 | 181.078 | 1624.62x | numrust | ok |
| `asv_reduce_stats_std_f64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_std(dtype=float64)` | 100000 | 0.106 | 332.704 | 3135.02x | numrust | ok |
| `asv_reduce_stats_std_f32_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_std(dtype=float32)` | 100000 | 0.106 | 418.177 | 3952.82x | numrust | ok |
| `asv_reduce_stats_std_i64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_std(dtype=int64)` | 100000 | 0.107 | 375.036 | 3517.34x | numrust | ok |
| `asv_reduce_stats_std_u64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_std(dtype=uint64)` | 100000 | 0.108 | 375.586 | 3489.76x | numrust | ok |
| `asv_reduce_stats_std_bool_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_std(dtype=bool_)` | 100000 | 0.105 | 405.618 | 3849.28x | numrust | ok |
| `asv_reduce_stats_std_c64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_std(dtype=complex64)` | 100000 | 0.105 | 515.305 | 4894.10x | numrust | ok |
| `asv_reduce_stats_prod_f64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_prod(dtype=float64)` | 100000 | 8.676 | 75.680 | 8.72x | numrust | ok |
| `asv_reduce_stats_prod_f32_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_prod(dtype=float32)` | 100000 | 8.771 | 75.765 | 8.64x | numrust | ok |
| `asv_reduce_stats_prod_i64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_prod(dtype=int64)` | 100000 | 3.013 | 66.933 | 22.22x | numrust | ok |
| `asv_reduce_stats_prod_u64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_prod(dtype=uint64)` | 100000 | 3.027 | 67.019 | 22.14x | numrust | ok |
| `asv_reduce_stats_prod_bool_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_prod(dtype=bool_)` | 100000 | 0.104 | 74.701 | 715.13x | numrust | ok |
| `asv_reduce_stats_prod_c64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_prod(dtype=complex64)` | 100000 | 0.688 | 150.109 | 218.05x | numrust | ok |
| `asv_reduce_stats_var_f64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_var(dtype=float64)` | 100000 | 0.106 | 313.225 | 2953.81x | numrust | ok |
| `asv_reduce_stats_var_f32_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_var(dtype=float32)` | 100000 | 0.105 | 386.416 | 3664.17x | numrust | ok |
| `asv_reduce_stats_var_i64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_var(dtype=int64)` | 100000 | 0.107 | 353.404 | 3297.70x | numrust | ok |
| `asv_reduce_stats_var_u64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_var(dtype=uint64)` | 100000 | 0.117 | 354.298 | 3021.73x | numrust | ok |
| `asv_reduce_stats_var_bool_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_var(dtype=bool_)` | 100000 | 0.106 | 381.797 | 3606.11x | numrust | ok |
| `asv_reduce_stats_var_c64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_var(dtype=complex64)` | 100000 | 0.106 | 481.579 | 4550.37x | numrust | ok |
| `asv_reduce_fmin_f32_20000` | `benchmarks/benchmarks/bench_reduce.py::FMinMax.time_min(dtype=float32)` | 20000 | 0.022 | 15.308 | 688.00x | numrust | ok |
| `asv_reduce_fmax_f32_20000` | `benchmarks/benchmarks/bench_reduce.py::FMinMax.time_max(dtype=float32)` | 20000 | 0.022 | 15.407 | 693.78x | numrust | ok |
| `asv_reduce_fmin_f64_20000` | `benchmarks/benchmarks/bench_reduce.py::FMinMax.time_min(dtype=float64)` | 20000 | 0.022 | 31.588 | 1422.33x | numrust | ok |
| `asv_reduce_fmax_f64_20000` | `benchmarks/benchmarks/bench_reduce.py::FMinMax.time_max(dtype=float64)` | 20000 | 0.022 | 31.567 | 1413.40x | numrust | ok |
| `asv_reduce_argmax_i64_200000` | `benchmarks/benchmarks/bench_reduce.py::ArgMax.time_argmax(dtype=int64)` | 20000 | 0.026 | 482.564 | 18740.34x | numrust | ok |
| `asv_reduce_argmin_i64_200000` | `benchmarks/benchmarks/bench_reduce.py::ArgMin.time_argmin(dtype=int64)` | 20000 | 0.026 | 517.756 | 20238.27x | numrust | ok |
| `asv_itemselection_take_i64_1000x1` | `benchmarks/benchmarks/bench_itemselection.py::Take.time_contiguous(shape=(1000, 1), mode='raise', dtype='int64')` | 2000 | 0.885 | 3.390 | 3.83x | numrust | ok |
| `asv_itemselection_take_i64_2x1000x1` | `benchmarks/benchmarks/bench_itemselection.py::Take.time_contiguous(shape=(2, 1000, 1), mode='raise', dtype='int64')` | 2000 | 1.038 | 6.280 | 6.05x | numrust | ok |
| `asv_itemselection_take_i64_1000x3` | `benchmarks/benchmarks/bench_itemselection.py::Take.time_contiguous(shape=(1000, 3), mode='raise', dtype='int64')` | 2000 | 1.300 | 4.434 | 3.41x | numrust | ok |
| `asv_itemselection_putmask_dense_scalar_f64_1000` | `benchmarks/benchmarks/bench_itemselection.py::PutMask.time_dense(values_is_scalar=True, dtype=float64)` | 10000 | 0.902 | 4.970 | 5.51x | numrust | ok |
| `asv_itemselection_putmask_sparse_scalar_f64_1000` | `benchmarks/benchmarks/bench_itemselection.py::PutMask.time_sparse(values_is_scalar=True, dtype=float64)` | 10000 | 0.026 | 4.658 | 178.02x | numrust | ok |
| `asv_itemselection_put_ordered_f64_1000` | `benchmarks/benchmarks/bench_itemselection.py::Put.time_ordered(values_is_scalar=False, dtype=float64)` | 10000 | 7.043 | 19.839 | 2.82x | numrust | ok |
| `asv_manipulate_broadcast_arrays_f64_16x32` | `benchmarks/benchmarks/bench_manipulate.py::BroadcastArrays.time_broadcast_arrays(shape=(16, 32), ndtype=float64)` | 200000 | 58.201 | 1201.416 | 20.64x | numrust | ok |
| `asv_manipulate_broadcast_arrays_f64_128x256` | `benchmarks/benchmarks/bench_manipulate.py::BroadcastArrays.time_broadcast_arrays(shape=(128, 256), ndtype=float64)` | 100000 | 28.854 | 595.310 | 20.63x | numrust | ok |
| `asv_manipulate_broadcast_arrays_f32_128x256` | `benchmarks/benchmarks/bench_manipulate.py::BroadcastArrays.time_broadcast_arrays(shape=(128, 256), ndtype=float32)` | 100000 | 28.730 | 596.992 | 20.78x | numrust | ok |
| `asv_manipulate_broadcast_arrays_i32_128x256` | `benchmarks/benchmarks/bench_manipulate.py::BroadcastArrays.time_broadcast_arrays(shape=(128, 256), ndtype=int32)` | 100000 | 28.749 | 595.956 | 20.73x | numrust | ok |
| `asv_manipulate_broadcast_arrays_f64_512x1024` | `benchmarks/benchmarks/bench_manipulate.py::BroadcastArrays.time_broadcast_arrays(shape=(512, 1024), ndtype=float64)` | 100000 | 28.712 | 613.539 | 21.37x | numrust | ok |
| `asv_manipulate_broadcast_to_f64_16` | `benchmarks/benchmarks/bench_manipulate.py::BroadcastArraysTo.time_broadcast_to(size=16, ndtype=float64)` | 200000 | 10.747 | 176.410 | 16.42x | numrust | ok |
| `asv_manipulate_broadcast_to_f64_64` | `benchmarks/benchmarks/bench_manipulate.py::BroadcastArraysTo.time_broadcast_to(size=64, ndtype=float64)` | 200000 | 10.786 | 176.128 | 16.33x | numrust | ok |
| `asv_manipulate_broadcast_to_f32_64` | `benchmarks/benchmarks/bench_manipulate.py::BroadcastArraysTo.time_broadcast_to(size=64, ndtype=float32)` | 200000 | 10.842 | 175.835 | 16.22x | numrust | ok |
| `asv_manipulate_broadcast_to_i32_64` | `benchmarks/benchmarks/bench_manipulate.py::BroadcastArraysTo.time_broadcast_to(size=64, ndtype=int32)` | 200000 | 10.893 | 175.899 | 16.15x | numrust | ok |
| `asv_manipulate_broadcast_to_f64_512` | `benchmarks/benchmarks/bench_manipulate.py::BroadcastArraysTo.time_broadcast_to(size=512, ndtype=float64)` | 200000 | 10.788 | 179.950 | 16.68x | numrust | ok |
| `asv_manipulate_concatenate_ax0_f64_32x64_n5` | `benchmarks/benchmarks/bench_manipulate.py::ConcatenateStackArrays.time_concatenate_ax0(shape=(32, 64), narrays=5, ndtype=float64)` | 2000 | 1.735 | 3.571 | 2.06x | numrust | ok |
| `asv_manipulate_concatenate_ax0_f32_32x64_n5` | `benchmarks/benchmarks/bench_manipulate.py::ConcatenateStackArrays.time_concatenate_ax0(shape=(32, 64), narrays=5, ndtype=float32)` | 2000 | 1.071 | 2.866 | 2.68x | numrust | ok |
| `asv_manipulate_concatenate_ax0_i32_32x64_n5` | `benchmarks/benchmarks/bench_manipulate.py::ConcatenateStackArrays.time_concatenate_ax0(shape=(32, 64), narrays=5, ndtype=int32)` | 2000 | 1.171 | 2.817 | 2.41x | numrust | ok |
| `asv_manipulate_concatenate_ax1_f64_32x64_n5` | `benchmarks/benchmarks/bench_manipulate.py::ConcatenateStackArrays.time_concatenate_ax1(shape=(32, 64), narrays=5, ndtype=float64)` | 2000 | 2.418 | 4.689 | 1.94x | numrust | ok |
| `asv_manipulate_concatenate_ax1_f32_32x64_n5` | `benchmarks/benchmarks/bench_manipulate.py::ConcatenateStackArrays.time_concatenate_ax1(shape=(32, 64), narrays=5, ndtype=float32)` | 2000 | 1.542 | 2.929 | 1.90x | numrust | ok |
| `asv_manipulate_concatenate_ax1_i32_32x64_n5` | `benchmarks/benchmarks/bench_manipulate.py::ConcatenateStackArrays.time_concatenate_ax1(shape=(32, 64), narrays=5, ndtype=int32)` | 2000 | 1.562 | 3.041 | 1.95x | numrust | ok |
| `asv_manipulate_stack_ax0_f64_32x64_n5` | `benchmarks/benchmarks/bench_manipulate.py::ConcatenateStackArrays.time_stack_ax0(shape=(32, 64), narrays=5, ndtype=float64)` | 2000 | 1.794 | 5.151 | 2.87x | numrust | ok |
| `asv_manipulate_stack_ax0_f32_32x64_n5` | `benchmarks/benchmarks/bench_manipulate.py::ConcatenateStackArrays.time_stack_ax0(shape=(32, 64), narrays=5, ndtype=float32)` | 2000 | 1.251 | 4.549 | 3.64x | numrust | ok |
| `asv_manipulate_stack_ax0_i32_32x64_n5` | `benchmarks/benchmarks/bench_manipulate.py::ConcatenateStackArrays.time_stack_ax0(shape=(32, 64), narrays=5, ndtype=int32)` | 2000 | 1.201 | 4.603 | 3.83x | numrust | ok |
| `asv_manipulate_stack_ax1_f64_32x64_n5` | `benchmarks/benchmarks/bench_manipulate.py::ConcatenateStackArrays.time_stack_ax1(shape=(32, 64), narrays=5, ndtype=float64)` | 2000 | 2.135 | 6.476 | 3.03x | numrust | ok |
| `asv_manipulate_stack_ax1_f32_32x64_n5` | `benchmarks/benchmarks/bench_manipulate.py::ConcatenateStackArrays.time_stack_ax1(shape=(32, 64), narrays=5, ndtype=float32)` | 2000 | 1.464 | 4.672 | 3.19x | numrust | ok |
| `asv_manipulate_stack_ax1_i32_32x64_n5` | `benchmarks/benchmarks/bench_manipulate.py::ConcatenateStackArrays.time_stack_ax1(shape=(32, 64), narrays=5, ndtype=int32)` | 2000 | 1.454 | 4.716 | 3.24x | numrust | ok |
| `asv_manipulate_expand_dims_f64_5x2x3x1_axis1` | `benchmarks/benchmarks/bench_manipulate.py::DimsManipulations.time_expand_dims(shape=(5, 2, 3, 1))` | 200000 | 19.422 | 129.845 | 6.69x | numrust | ok |
| `asv_manipulate_expand_dims_neg_f64_5x2x3x1` | `benchmarks/benchmarks/bench_manipulate.py::DimsManipulations.time_expand_dims_neg(shape=(5, 2, 3, 1))` | 200000 | 18.954 | 130.273 | 6.87x | numrust | ok |
| `asv_manipulate_squeeze_dims_f64_5x2x3x1` | `benchmarks/benchmarks/bench_manipulate.py::DimsManipulations.time_squeeze_dims(shape=(5, 2, 3, 1))` | 200000 | 9.710 | 37.152 | 3.83x | numrust | ok |
| `asv_manipulate_flip_all_f64_5x2x3x1` | `benchmarks/benchmarks/bench_manipulate.py::DimsManipulations.time_flip_all(shape=(5, 2, 3, 1))` | 200000 | 29.570 | 509.329 | 17.22x | numrust | ok |
| `asv_manipulate_flip_one_f64_5x2x3x1_axis1` | `benchmarks/benchmarks/bench_manipulate.py::DimsManipulations.time_flip_one(shape=(5, 2, 3, 1))` | 200000 | 28.923 | 595.382 | 20.59x | numrust | ok |
| `asv_manipulate_flip_neg_f64_5x2x3x1_axis_neg1` | `benchmarks/benchmarks/bench_manipulate.py::DimsManipulations.time_flip_neg(shape=(5, 2, 3, 1))` | 200000 | 29.325 | 593.777 | 20.25x | numrust | ok |
| `asv_manipulate_moveaxis_f64_5x2x3x1` | `benchmarks/benchmarks/bench_manipulate.py::DimsManipulations.time_moveaxis(shape=(5, 2, 3, 1))` | 200000 | 38.846 | 692.475 | 17.83x | numrust | ok |
| `asv_manipulate_roll_f64_5x2x3x1_shift3` | `benchmarks/benchmarks/bench_manipulate.py::DimsManipulations.time_roll(shape=(5, 2, 3, 1))` | 100000 | 22.322 | 453.008 | 20.29x | numrust | ok |
| `asv_manipulate_reshape_f64_5x2x3x1_to_1x5x2x3` | `benchmarks/benchmarks/bench_manipulate.py::DimsManipulations.time_reshape(shape=(5, 2, 3, 1))` | 200000 | 20.248 | 61.614 | 3.04x | numrust | ok |
| `asv_linalg_dot_a_b_f64_150x400_400x600` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_dot_a_b` | 1000 | 143.990 | 146.686 | 1.02x | numrust | ok |
| `asv_linalg_matmul_a_b_f64_150x400_400x600` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_matmul_a_b` | 1000 | 144.186 | 144.762 | 1.00x | numrust | ok |
| `asv_linalg_matmul_d_matmul_b_c_f64` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_matmul_d_matmul_b_c` | 1000 | 4.166 | 4.913 | 1.18x | numrust | ok |
| `asv_linalg_dot_d_dot_b_c_f64` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_dot_d_dot_b_c` | 1000 | 4.364 | 4.661 | 1.07x | numrust | ok |
| `asv_linalg_dot_trans_a_at_f64_150x400_400x150` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_dot_trans_a_at` | 1000 | 29.828 | 33.076 | 1.11x | numrust | ok |
| `asv_linalg_dot_trans_a_atc_f64_150x400_400x150` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_dot_trans_a_atc` | 1000 | 41.017 | 41.877 | 1.02x | numrust | ok |
| `asv_linalg_dot_trans_at_a_f64_400x150_150x400` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_dot_trans_at_a` | 1000 | 77.947 | 96.131 | 1.23x | numrust | ok |
| `asv_linalg_dot_trans_atc_a_f64_400x150_150x400` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_dot_trans_atc_a` | 1000 | 95.802 | 100.382 | 1.05x | numrust | ok |
| `asv_linalg_inner_a_a_f64_150x400_150x400` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_inner_trans_a_a` | 1000 | 30.451 | 33.617 | 1.10x | numrust | ok |
| `asv_linalg_inner_a_ac_f64_150x400_150x400` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_inner_trans_a_ac` | 1000 | 47.845 | 50.226 | 1.05x | numrust | ok |
| `asv_linalg_matmul_trans_a_at_f64_150x400_400x150` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_matmul_trans_a_at` | 1000 | 30.252 | 30.660 | 1.01x | numrust | ok |
| `asv_linalg_matmul_trans_a_atc_f64_150x400_400x150` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_matmul_trans_a_atc` | 1000 | 41.895 | 42.921 | 1.02x | numrust | ok |
| `asv_linalg_matmul_trans_at_a_f64_400x150_150x400` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_matmul_trans_at_a` | 1000 | 78.213 | 79.465 | 1.02x | numrust | ok |
| `asv_linalg_matmul_trans_atc_a_f64_400x150_150x400` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_matmul_trans_atc_a` | 1000 | 95.380 | 96.516 | 1.01x | numrust | ok |
| `asv_linalg_tensordot_a3_b3_axes_10_01` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_tensordot_a_b_axes_1_0_0_1` | 10 | 3.147 | 4.465 | 1.42x | numrust | ok |
| `asv_linalg_norm_small_array_f64_5` | `benchmarks/benchmarks/bench_linalg.py::LinalgSmallArrays.time_norm_small_array` | 100000 | 4.617 | 41.723 | 9.04x | numrust | ok |
| `asv_linalg_det_small_array_f64_5x5` | `benchmarks/benchmarks/bench_linalg.py::LinalgSmallArrays.time_det_small_array` | 100000 | 8.138 | 91.754 | 11.27x | numrust | ok |
| `asv_linalg_det_3x3_f64` | `benchmarks/benchmarks/bench_linalg.py::LinalgSmallArrays.time_det_3x3` | 100000 | 6.862 | 87.828 | 12.80x | numrust | ok |
| `asv_linalg_solve_3x3_f64` | `benchmarks/benchmarks/bench_linalg.py::LinalgSmallArrays.time_solve_3x3` | 100000 | 14.036 | 174.965 | 12.47x | numrust | ok |
| `asv_linalg_lstsq_square_f64_100x100` | `benchmarks/benchmarks/bench_linalg.py::Lstsq.time_numpy_linalg_lstsq_a__b_float64` | 100 | 5.156 | 63.817 | 12.38x | numrust | ok |
| `asv_linalg_einsum_outer_f64_3000` | `benchmarks/benchmarks/bench_linalg.py::Einsum.time_einsum_outer(dtype=float64)` | 1 | 0.663 | 1.705 | 2.57x | numrust | ok |
| `asv_linalg_einsum_i_ij_j_f64_400_400x600_600` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_einsum_i_ij_j` | 1000 | 4.293 | 252.772 | 58.88x | numrust | ok |
| `asv_linalg_einsum_ij_jk_f64_150x400_400x600` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_einsum_ij_jk_a_b` | 1000 | 143.881 | 4150.862 | 28.85x | numrust | ok |
| `asv_linalg_einsum_multiply_f64_30x40_20x30x40` | `benchmarks/benchmarks/bench_linalg.py::Einsum.time_einsum_multiply(dtype=float64)` | 1 | 0.003 | 0.014 | 4.61x | numrust | ok |
| `asv_linalg_einsum_sum_mul_f64_scalar_10x100x10` | `benchmarks/benchmarks/bench_linalg.py::Einsum.time_einsum_sum_mul(dtype=float64)` | 100 | 0.049 | 0.659 | 13.34x | numrust | ok |
| `asv_linalg_einsum_sum_mul2_f64_10x100x10_scalar` | `benchmarks/benchmarks/bench_linalg.py::Einsum.time_einsum_sum_mul2(dtype=float64)` | 100 | 0.052 | 0.656 | 12.59x | numrust | ok |
| `asv_linalg_einsum_scalar_mul_f64_480000` | `benchmarks/benchmarks/bench_linalg.py::Einsum.time_einsum_mul(dtype=float64)` | 100 | 6.291 | 6.770 | 1.08x | numrust | ok |
| `asv_linalg_einsum_sum_f64_480000` | `benchmarks/benchmarks/bench_linalg.py::Einsum.time_einsum_contig_outstride0(dtype=float64)` | 100 | 3.368 | 4.130 | 1.23x | numrust | ok |
| `asv_linalg_einsum_weighted_sum_f64_400x600` | `benchmarks/benchmarks/bench_linalg.py::Einsum.time_einsum_contig_contig(dtype=float64)` | 100 | 0.395 | 4.379 | 11.08x | numrust | ok |
| `asv_linalg_einsum_noncon_outer_f64_2000` | `benchmarks/benchmarks/bench_linalg.py::Einsum.time_einsum_noncon_outer(dtype=float64)` | 1 | 0.519 | 1.526 | 2.94x | numrust | ok |
| `asv_linalg_einsum_noncon_multiply_f64_30x40_20x30x40` | `benchmarks/benchmarks/bench_linalg.py::Einsum.time_einsum_noncon_multiply(dtype=float64)` | 1 | 0.003 | 0.014 | 4.32x | numrust | ok |
| `asv_linalg_einsum_noncon_sum_mul_f64_scalar_20x30x40` | `benchmarks/benchmarks/bench_linalg.py::Einsum.time_einsum_noncon_sum_mul(dtype=float64)` | 100 | 0.157 | 0.788 | 5.04x | numrust | ok |
| `asv_linalg_einsum_noncon_sum_mul2_f64_20x30x40_scalar` | `benchmarks/benchmarks/bench_linalg.py::Einsum.time_einsum_noncon_sum_mul2(dtype=float64)` | 100 | 0.157 | 0.773 | 4.92x | numrust | ok |
| `asv_linalg_einsum_noncon_scalar_mul_f64_2000` | `benchmarks/benchmarks/bench_linalg.py::Einsum.time_einsum_noncon_mul(dtype=float64)` | 100 | 0.023 | 0.417 | 18.32x | numrust | ok |
| `asv_linalg_einsum_noncon_weighted_sum_f64_30x40` | `benchmarks/benchmarks/bench_linalg.py::Einsum.time_einsum_noncon_contig_contig(dtype=float64)` | 100 | 0.048 | 0.485 | 10.13x | numrust | ok |
| `asv_linalg_einsum_noncon_sum_f64_2000` | `benchmarks/benchmarks/bench_linalg.py::Einsum.time_einsum_noncon_contig_outstride0(dtype=float64)` | 100 | 0.013 | 0.254 | 19.44x | numrust | ok |

## Score

- Supported external cases: 120
- Unsupported external cases tracked: 1
- NumRust wins: 120
- NumPy wins: 0
- Geomean speedup vs NumPy: 31.48x
- Near-tie relative margin: 2%
- Near-tie cases: 5
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
