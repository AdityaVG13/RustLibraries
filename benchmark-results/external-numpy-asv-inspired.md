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
| `asv_ufunc_broadcast_sub_f64` | `benchmarks/benchmarks/bench_ufunc.py::Broadcast.time_broadcast` | 1 | 0.896 | 1.520 | 1.70x | numrust | ok |
| `asv_ufunc_astype_i32_to_f64_100x100` | `benchmarks/benchmarks/bench_ufunc.py::NDArrayAsType.time_astype(typeconv=('int32', 'float64'))` | 5000 | 5.109 | 13.733 | 2.69x | numrust | ok |
| `asv_ufunc_add_at_f64_10000000` | `benchmarks/benchmarks/bench_ufunc.py::At.time_sum_at` | 1 | 3.719 | 5.715 | 1.54x | numrust | ok |
| `asv_ufunc_maximum_at_f64_10000000` | `benchmarks/benchmarks/bench_ufunc.py::At.time_maximum_at` | 1 | 4.843 | 5.881 | 1.21x | numrust | ok |
| `asv_reduce_small_sum_f32_100` | `benchmarks/benchmarks/bench_reduce.py::SmallReduction.time_small` | 200000 | 2.764 | 132.525 | 47.95x | numrust | ok |
| `asv_reduce_stats_min_f64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_min(dtype=float64)` | 100000 | 0.112 | 66.026 | 592.16x | numrust | ok |
| `asv_reduce_stats_min_f32_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_min(dtype=float32)` | 100000 | 0.115 | 65.324 | 566.81x | numrust | ok |
| `asv_reduce_stats_min_i64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_min(dtype=int64)` | 100000 | 0.108 | 66.065 | 611.23x | numrust | ok |
| `asv_reduce_stats_min_u64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_min(dtype=uint64)` | 100000 | 0.108 | 65.989 | 609.60x | numrust | ok |
| `asv_reduce_stats_min_bool_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_min(dtype=bool_)` | 100000 | 0.105 | 70.076 | 666.07x | numrust | ok |
| `asv_reduce_stats_min_c64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_min(dtype=complex64)` | 100000 | 0.112 | 110.166 | 982.53x | numrust | ok |
| `asv_reduce_stats_max_f64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_max(dtype=float64)` | 100000 | 0.111 | 66.000 | 595.04x | numrust | ok |
| `asv_reduce_stats_max_f32_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_max(dtype=float32)` | 100000 | 0.111 | 65.862 | 593.79x | numrust | ok |
| `asv_reduce_stats_max_i64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_max(dtype=int64)` | 100000 | 0.109 | 66.385 | 611.61x | numrust | ok |
| `asv_reduce_stats_max_u64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_max(dtype=uint64)` | 100000 | 0.109 | 66.100 | 607.59x | numrust | ok |
| `asv_reduce_stats_max_bool_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_max(dtype=bool_)` | 100000 | 0.105 | 70.000 | 665.87x | numrust | ok |
| `asv_reduce_stats_max_c64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_max(dtype=complex64)` | 100000 | 0.113 | 110.329 | 976.72x | numrust | ok |
| `asv_reduce_stats_mean_f64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_mean(dtype=float64)` | 100000 | 0.112 | 101.110 | 904.78x | numrust | ok |
| `asv_reduce_stats_mean_f32_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_mean(dtype=float32)` | 100000 | 0.111 | 169.326 | 1526.61x | numrust | ok |
| `asv_reduce_stats_mean_i64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_mean(dtype=int64)` | 100000 | 0.113 | 115.054 | 1014.43x | numrust | ok |
| `asv_reduce_stats_mean_u64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_mean(dtype=uint64)` | 100000 | 0.112 | 114.858 | 1029.73x | numrust | ok |
| `asv_reduce_stats_mean_bool_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_mean(dtype=bool_)` | 100000 | 0.118 | 115.167 | 978.76x | numrust | ok |
| `asv_reduce_stats_mean_c64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_mean(dtype=complex64)` | 100000 | 0.114 | 178.928 | 1573.56x | numrust | ok |
| `asv_reduce_stats_std_f64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_std(dtype=float64)` | 100000 | 0.106 | 325.404 | 3073.47x | numrust | ok |
| `asv_reduce_stats_std_f32_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_std(dtype=float32)` | 100000 | 0.106 | 417.732 | 3939.31x | numrust | ok |
| `asv_reduce_stats_std_i64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_std(dtype=int64)` | 100000 | 0.107 | 369.636 | 3447.84x | numrust | ok |
| `asv_reduce_stats_std_u64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_std(dtype=uint64)` | 100000 | 0.106 | 366.533 | 3456.49x | numrust | ok |
| `asv_reduce_stats_std_bool_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_std(dtype=bool_)` | 100000 | 0.106 | 395.435 | 3729.08x | numrust | ok |
| `asv_reduce_stats_std_c64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_std(dtype=complex64)` | 100000 | 0.105 | 509.134 | 4837.38x | numrust | ok |
| `asv_reduce_stats_prod_f64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_prod(dtype=float64)` | 100000 | 8.668 | 75.471 | 8.71x | numrust | ok |
| `asv_reduce_stats_prod_f32_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_prod(dtype=float32)` | 100000 | 8.727 | 75.599 | 8.66x | numrust | ok |
| `asv_reduce_stats_prod_i64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_prod(dtype=int64)` | 100000 | 2.995 | 66.910 | 22.34x | numrust | ok |
| `asv_reduce_stats_prod_u64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_prod(dtype=uint64)` | 100000 | 3.007 | 67.243 | 22.36x | numrust | ok |
| `asv_reduce_stats_prod_bool_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_prod(dtype=bool_)` | 100000 | 0.104 | 74.752 | 719.35x | numrust | ok |
| `asv_reduce_stats_prod_c64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_prod(dtype=complex64)` | 100000 | 0.689 | 149.247 | 216.73x | numrust | ok |
| `asv_reduce_stats_var_f64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_var(dtype=float64)` | 100000 | 0.106 | 305.043 | 2881.16x | numrust | ok |
| `asv_reduce_stats_var_f32_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_var(dtype=float32)` | 100000 | 0.106 | 385.212 | 3646.98x | numrust | ok |
| `asv_reduce_stats_var_i64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_var(dtype=int64)` | 100000 | 0.107 | 346.209 | 3248.26x | numrust | ok |
| `asv_reduce_stats_var_u64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_var(dtype=uint64)` | 100000 | 0.106 | 344.364 | 3234.74x | numrust | ok |
| `asv_reduce_stats_var_bool_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_var(dtype=bool_)` | 100000 | 0.105 | 370.704 | 3516.55x | numrust | ok |
| `asv_reduce_stats_var_c64_200` | `benchmarks/benchmarks/bench_reduce.py::StatsReductions.time_var(dtype=complex64)` | 100000 | 0.106 | 476.598 | 4497.95x | numrust | ok |
| `asv_reduce_argmax_i64_200000` | `benchmarks/benchmarks/bench_reduce.py::ArgMax.time_argmax(dtype=int64)` | 20000 | 0.025 | 485.558 | 19104.44x | numrust | ok |
| `asv_reduce_argmin_i64_200000` | `benchmarks/benchmarks/bench_reduce.py::ArgMin.time_argmin(dtype=int64)` | 20000 | 0.026 | 543.366 | 21239.33x | numrust | ok |
| `asv_itemselection_take_i64_1000x1` | `benchmarks/benchmarks/bench_itemselection.py::Take.time_contiguous(shape=(1000, 1), mode='raise', dtype='int64')` | 2000 | 0.906 | 3.994 | 4.41x | numrust | ok |
| `asv_itemselection_putmask_dense_scalar_f64_1000` | `benchmarks/benchmarks/bench_itemselection.py::PutMask.time_dense(values_is_scalar=True, dtype=float64)` | 10000 | 1.263 | 5.269 | 4.17x | numrust | ok |
| `asv_itemselection_putmask_sparse_scalar_f64_1000` | `benchmarks/benchmarks/bench_itemselection.py::PutMask.time_sparse(values_is_scalar=True, dtype=float64)` | 10000 | 0.023 | 4.664 | 198.80x | numrust | ok |
| `asv_itemselection_put_ordered_f64_1000` | `benchmarks/benchmarks/bench_itemselection.py::Put.time_ordered(values_is_scalar=False, dtype=float64)` | 10000 | 6.998 | 19.837 | 2.83x | numrust | ok |
| `asv_manipulate_broadcast_arrays_f64_16x32` | `benchmarks/benchmarks/bench_manipulate.py::BroadcastArrays.time_broadcast_arrays(shape=(16, 32), ndtype=float64)` | 200000 | 57.268 | 1228.518 | 21.45x | numrust | ok |
| `asv_manipulate_broadcast_arrays_f64_128x256` | `benchmarks/benchmarks/bench_manipulate.py::BroadcastArrays.time_broadcast_arrays(shape=(128, 256), ndtype=float64)` | 100000 | 28.927 | 597.844 | 20.67x | numrust | ok |
| `asv_manipulate_broadcast_arrays_f32_128x256` | `benchmarks/benchmarks/bench_manipulate.py::BroadcastArrays.time_broadcast_arrays(shape=(128, 256), ndtype=float32)` | 100000 | 28.811 | 620.329 | 21.53x | numrust | ok |
| `asv_manipulate_broadcast_arrays_i32_128x256` | `benchmarks/benchmarks/bench_manipulate.py::BroadcastArrays.time_broadcast_arrays(shape=(128, 256), ndtype=int32)` | 100000 | 28.993 | 610.458 | 21.06x | numrust | ok |
| `asv_manipulate_broadcast_arrays_f64_512x1024` | `benchmarks/benchmarks/bench_manipulate.py::BroadcastArrays.time_broadcast_arrays(shape=(512, 1024), ndtype=float64)` | 100000 | 28.723 | 633.872 | 22.07x | numrust | ok |
| `asv_manipulate_broadcast_to_f64_16` | `benchmarks/benchmarks/bench_manipulate.py::BroadcastArraysTo.time_broadcast_to(size=16, ndtype=float64)` | 200000 | 11.028 | 193.091 | 17.51x | numrust | ok |
| `asv_manipulate_broadcast_to_f64_64` | `benchmarks/benchmarks/bench_manipulate.py::BroadcastArraysTo.time_broadcast_to(size=64, ndtype=float64)` | 200000 | 11.174 | 186.000 | 16.65x | numrust | ok |
| `asv_manipulate_broadcast_to_f32_64` | `benchmarks/benchmarks/bench_manipulate.py::BroadcastArraysTo.time_broadcast_to(size=64, ndtype=float32)` | 200000 | 10.851 | 184.711 | 17.02x | numrust | ok |
| `asv_manipulate_broadcast_to_i32_64` | `benchmarks/benchmarks/bench_manipulate.py::BroadcastArraysTo.time_broadcast_to(size=64, ndtype=int32)` | 200000 | 10.805 | 189.210 | 17.51x | numrust | ok |
| `asv_manipulate_broadcast_to_f64_512` | `benchmarks/benchmarks/bench_manipulate.py::BroadcastArraysTo.time_broadcast_to(size=512, ndtype=float64)` | 200000 | 11.139 | 194.667 | 17.48x | numrust | ok |
| `asv_manipulate_concatenate_ax0_f64_32x64_n5` | `benchmarks/benchmarks/bench_manipulate.py::ConcatenateStackArrays.time_concatenate_ax0(shape=(32, 64), narrays=5, ndtype=float64)` | 2000 | 1.731 | 3.567 | 2.06x | numrust | ok |
| `asv_manipulate_concatenate_ax0_f32_32x64_n5` | `benchmarks/benchmarks/bench_manipulate.py::ConcatenateStackArrays.time_concatenate_ax0(shape=(32, 64), narrays=5, ndtype=float32)` | 2000 | 1.112 | 2.961 | 2.66x | numrust | ok |
| `asv_manipulate_concatenate_ax0_i32_32x64_n5` | `benchmarks/benchmarks/bench_manipulate.py::ConcatenateStackArrays.time_concatenate_ax0(shape=(32, 64), narrays=5, ndtype=int32)` | 2000 | 1.155 | 2.954 | 2.56x | numrust | ok |
| `asv_manipulate_concatenate_ax1_f64_32x64_n5` | `benchmarks/benchmarks/bench_manipulate.py::ConcatenateStackArrays.time_concatenate_ax1(shape=(32, 64), narrays=5, ndtype=float64)` | 2000 | 2.415 | 4.648 | 1.92x | numrust | ok |
| `asv_manipulate_concatenate_ax1_f32_32x64_n5` | `benchmarks/benchmarks/bench_manipulate.py::ConcatenateStackArrays.time_concatenate_ax1(shape=(32, 64), narrays=5, ndtype=float32)` | 2000 | 1.515 | 3.310 | 2.19x | numrust | ok |
| `asv_manipulate_concatenate_ax1_i32_32x64_n5` | `benchmarks/benchmarks/bench_manipulate.py::ConcatenateStackArrays.time_concatenate_ax1(shape=(32, 64), narrays=5, ndtype=int32)` | 2000 | 1.552 | 3.034 | 1.96x | numrust | ok |
| `asv_manipulate_stack_ax0_f64_32x64_n5` | `benchmarks/benchmarks/bench_manipulate.py::ConcatenateStackArrays.time_stack_ax0(shape=(32, 64), narrays=5, ndtype=float64)` | 2000 | 2.072 | 5.153 | 2.49x | numrust | ok |
| `asv_manipulate_stack_ax0_f32_32x64_n5` | `benchmarks/benchmarks/bench_manipulate.py::ConcatenateStackArrays.time_stack_ax0(shape=(32, 64), narrays=5, ndtype=float32)` | 2000 | 1.133 | 4.556 | 4.02x | numrust | ok |
| `asv_manipulate_stack_ax0_i32_32x64_n5` | `benchmarks/benchmarks/bench_manipulate.py::ConcatenateStackArrays.time_stack_ax0(shape=(32, 64), narrays=5, ndtype=int32)` | 2000 | 1.370 | 4.575 | 3.34x | numrust | ok |
| `asv_manipulate_stack_ax1_f64_32x64_n5` | `benchmarks/benchmarks/bench_manipulate.py::ConcatenateStackArrays.time_stack_ax1(shape=(32, 64), narrays=5, ndtype=float64)` | 2000 | 2.182 | 6.781 | 3.11x | numrust | ok |
| `asv_manipulate_stack_ax1_f32_32x64_n5` | `benchmarks/benchmarks/bench_manipulate.py::ConcatenateStackArrays.time_stack_ax1(shape=(32, 64), narrays=5, ndtype=float32)` | 2000 | 1.555 | 4.832 | 3.11x | numrust | ok |
| `asv_manipulate_stack_ax1_i32_32x64_n5` | `benchmarks/benchmarks/bench_manipulate.py::ConcatenateStackArrays.time_stack_ax1(shape=(32, 64), narrays=5, ndtype=int32)` | 2000 | 1.711 | 4.878 | 2.85x | numrust | ok |
| `asv_manipulate_expand_dims_f64_5x2x3x1_axis1` | `benchmarks/benchmarks/bench_manipulate.py::DimsManipulations.time_expand_dims(shape=(5, 2, 3, 1))` | 200000 | 19.250 | 135.901 | 7.06x | numrust | ok |
| `asv_manipulate_expand_dims_neg_f64_5x2x3x1` | `benchmarks/benchmarks/bench_manipulate.py::DimsManipulations.time_expand_dims_neg(shape=(5, 2, 3, 1))` | 200000 | 18.686 | 137.170 | 7.34x | numrust | ok |
| `asv_manipulate_squeeze_dims_f64_5x2x3x1` | `benchmarks/benchmarks/bench_manipulate.py::DimsManipulations.time_squeeze_dims(shape=(5, 2, 3, 1))` | 200000 | 9.683 | 40.911 | 4.23x | numrust | ok |
| `asv_manipulate_flip_all_f64_5x2x3x1` | `benchmarks/benchmarks/bench_manipulate.py::DimsManipulations.time_flip_all(shape=(5, 2, 3, 1))` | 200000 | 29.002 | 525.753 | 18.13x | numrust | ok |
| `asv_manipulate_flip_one_f64_5x2x3x1_axis1` | `benchmarks/benchmarks/bench_manipulate.py::DimsManipulations.time_flip_one(shape=(5, 2, 3, 1))` | 200000 | 28.466 | 615.286 | 21.62x | numrust | ok |
| `asv_manipulate_flip_neg_f64_5x2x3x1_axis_neg1` | `benchmarks/benchmarks/bench_manipulate.py::DimsManipulations.time_flip_neg(shape=(5, 2, 3, 1))` | 200000 | 28.050 | 593.045 | 21.14x | numrust | ok |
| `asv_manipulate_moveaxis_f64_5x2x3x1` | `benchmarks/benchmarks/bench_manipulate.py::DimsManipulations.time_moveaxis(shape=(5, 2, 3, 1))` | 200000 | 37.911 | 691.307 | 18.24x | numrust | ok |
| `asv_manipulate_roll_f64_5x2x3x1_shift3` | `benchmarks/benchmarks/bench_manipulate.py::DimsManipulations.time_roll(shape=(5, 2, 3, 1))` | 100000 | 21.560 | 447.767 | 20.77x | numrust | ok |
| `asv_manipulate_reshape_f64_5x2x3x1_to_1x5x2x3` | `benchmarks/benchmarks/bench_manipulate.py::DimsManipulations.time_reshape(shape=(5, 2, 3, 1))` | 200000 | 19.660 | 61.780 | 3.14x | numrust | ok |
| `asv_linalg_dot_a_b_f64_150x400_400x600` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_dot_a_b` | 1000 | 143.642 | 146.731 | 1.02x | numrust | ok |
| `asv_linalg_matmul_a_b_f64_150x400_400x600` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_matmul_a_b` | 1000 | 144.071 | 144.594 | 1.00x | numrust | ok |
| `asv_linalg_matmul_d_matmul_b_c_f64` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_matmul_d_matmul_b_c` | 1000 | 4.108 | 4.740 | 1.15x | numrust | ok |
| `asv_linalg_dot_d_dot_b_c_f64` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_dot_d_dot_b_c` | 1000 | 4.170 | 4.682 | 1.12x | numrust | ok |
| `asv_linalg_dot_trans_a_at_f64_150x400_400x150` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_dot_trans_a_at` | 1000 | 30.880 | 34.582 | 1.12x | numrust | ok |
| `asv_linalg_dot_trans_a_atc_f64_150x400_400x150` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_dot_trans_a_atc` | 1000 | 42.850 | 43.678 | 1.02x | numrust | ok |
| `asv_linalg_dot_trans_at_a_f64_400x150_150x400` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_dot_trans_at_a` | 1000 | 79.145 | 95.774 | 1.21x | numrust | ok |
| `asv_linalg_dot_trans_atc_a_f64_400x150_150x400` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_dot_trans_atc_a` | 1000 | 96.193 | 100.414 | 1.04x | numrust | ok |
| `asv_linalg_inner_a_a_f64_150x400_150x400` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_inner_trans_a_a` | 1000 | 30.769 | 33.336 | 1.08x | numrust | ok |
| `asv_linalg_inner_a_ac_f64_150x400_150x400` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_inner_trans_a_ac` | 1000 | 49.482 | 50.425 | 1.02x | numrust | ok |
| `asv_linalg_matmul_trans_a_at_f64_150x400_400x150` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_matmul_trans_a_at` | 1000 | 29.050 | 30.821 | 1.06x | numrust | ok |
| `asv_linalg_matmul_trans_a_atc_f64_150x400_400x150` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_matmul_trans_a_atc` | 1000 | 42.158 | 43.104 | 1.02x | numrust | ok |
| `asv_linalg_matmul_trans_at_a_f64_400x150_150x400` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_matmul_trans_at_a` | 1000 | 79.193 | 79.567 | 1.00x | numrust | ok |
| `asv_linalg_matmul_trans_atc_a_f64_400x150_150x400` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_matmul_trans_atc_a` | 1000 | 96.338 | 95.829 | 0.99x | numpy | ok |
| `asv_linalg_tensordot_a3_b3_axes_10_01` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_tensordot_a_b_axes_1_0_0_1` | 10 | 3.526 | 4.374 | 1.24x | numrust | ok |
| `asv_linalg_norm_small_array_f64_5` | `benchmarks/benchmarks/bench_linalg.py::LinalgSmallArrays.time_norm_small_array` | 100000 | 4.702 | 40.249 | 8.56x | numrust | ok |
| `asv_linalg_det_small_array_f64_5x5` | `benchmarks/benchmarks/bench_linalg.py::LinalgSmallArrays.time_det_small_array` | 100000 | 8.683 | 89.480 | 10.30x | numrust | ok |
| `asv_linalg_det_3x3_f64` | `benchmarks/benchmarks/bench_linalg.py::LinalgSmallArrays.time_det_3x3` | 100000 | 7.528 | 85.961 | 11.42x | numrust | ok |
| `asv_linalg_solve_3x3_f64` | `benchmarks/benchmarks/bench_linalg.py::LinalgSmallArrays.time_solve_3x3` | 100000 | 14.678 | 170.069 | 11.59x | numrust | ok |
| `asv_linalg_lstsq_square_f64_100x100` | `benchmarks/benchmarks/bench_linalg.py::Lstsq.time_numpy_linalg_lstsq_a__b_float64` | 100 | 5.353 | 63.441 | 11.85x | numrust | ok |
| `asv_linalg_einsum_outer_f64_3000` | `benchmarks/benchmarks/bench_linalg.py::Einsum.time_einsum_outer(dtype=float64)` | 1 | 0.690 | 1.688 | 2.45x | numrust | ok |
| `asv_linalg_einsum_i_ij_j_f64_400_400x600_600` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_einsum_i_ij_j` | 1000 | 4.248 | 251.370 | 59.17x | numrust | ok |
| `asv_linalg_einsum_ij_jk_f64_150x400_400x600` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_einsum_ij_jk_a_b` | 1000 | 144.001 | 4140.594 | 28.75x | numrust | ok |
| `asv_linalg_einsum_multiply_f64_30x40_20x30x40` | `benchmarks/benchmarks/bench_linalg.py::Einsum.time_einsum_multiply(dtype=float64)` | 1 | 0.003 | 0.015 | 4.54x | numrust | ok |
| `asv_linalg_einsum_sum_mul_f64_scalar_10x100x10` | `benchmarks/benchmarks/bench_linalg.py::Einsum.time_einsum_sum_mul(dtype=float64)` | 100 | 0.049 | 0.656 | 13.29x | numrust | ok |
| `asv_linalg_einsum_sum_mul2_f64_10x100x10_scalar` | `benchmarks/benchmarks/bench_linalg.py::Einsum.time_einsum_sum_mul2(dtype=float64)` | 100 | 0.055 | 0.656 | 11.92x | numrust | ok |
| `asv_linalg_einsum_scalar_mul_f64_480000` | `benchmarks/benchmarks/bench_linalg.py::Einsum.time_einsum_mul(dtype=float64)` | 100 | 6.609 | 6.720 | 1.02x | numrust | ok |
| `asv_linalg_einsum_sum_f64_480000` | `benchmarks/benchmarks/bench_linalg.py::Einsum.time_einsum_contig_outstride0(dtype=float64)` | 100 | 3.566 | 4.079 | 1.14x | numrust | ok |
| `asv_linalg_einsum_weighted_sum_f64_400x600` | `benchmarks/benchmarks/bench_linalg.py::Einsum.time_einsum_contig_contig(dtype=float64)` | 100 | 0.436 | 4.360 | 10.00x | numrust | ok |
| `asv_linalg_einsum_noncon_outer_f64_2000` | `benchmarks/benchmarks/bench_linalg.py::Einsum.time_einsum_noncon_outer(dtype=float64)` | 1 | 0.594 | 1.507 | 2.54x | numrust | ok |
| `asv_linalg_einsum_noncon_multiply_f64_30x40_20x30x40` | `benchmarks/benchmarks/bench_linalg.py::Einsum.time_einsum_noncon_multiply(dtype=float64)` | 1 | 0.003 | 0.014 | 4.15x | numrust | ok |
| `asv_linalg_einsum_noncon_sum_mul_f64_scalar_20x30x40` | `benchmarks/benchmarks/bench_linalg.py::Einsum.time_einsum_noncon_sum_mul(dtype=float64)` | 100 | 0.161 | 0.776 | 4.81x | numrust | ok |
| `asv_linalg_einsum_noncon_sum_mul2_f64_20x30x40_scalar` | `benchmarks/benchmarks/bench_linalg.py::Einsum.time_einsum_noncon_sum_mul2(dtype=float64)` | 100 | 0.160 | 0.772 | 4.83x | numrust | ok |
| `asv_linalg_einsum_noncon_scalar_mul_f64_2000` | `benchmarks/benchmarks/bench_linalg.py::Einsum.time_einsum_noncon_mul(dtype=float64)` | 100 | 0.023 | 0.419 | 18.17x | numrust | ok |
| `asv_linalg_einsum_noncon_weighted_sum_f64_30x40` | `benchmarks/benchmarks/bench_linalg.py::Einsum.time_einsum_noncon_contig_contig(dtype=float64)` | 100 | 0.051 | 0.475 | 9.33x | numrust | ok |
| `asv_linalg_einsum_noncon_sum_f64_2000` | `benchmarks/benchmarks/bench_linalg.py::Einsum.time_einsum_noncon_contig_outstride0(dtype=float64)` | 100 | 0.013 | 0.251 | 19.40x | numrust | ok |

## Score

- Supported external cases: 114
- Unsupported external cases tracked: 1
- NumRust wins: 113
- NumPy wins: 1
- Geomean speedup vs NumPy: 28.63x
- Near-tie relative margin: 2%
- Near-tie cases: 6
- Ranked higher on supported external cases: True
- Global NumPy replacement claim: false

## NumPy-Winning Cases

| Case | NumRust ms | NumPy ms | NumRust pass ms | NumPy pass ms |
| --- | ---: | ---: | --- | --- |
| `asv_linalg_matmul_trans_atc_a_f64_400x150_150x400` | 96.338 | 95.829 | 95.402, 96.042, 96.338, 107.884, 101.342 | 95.555, 95.298, 96.645, 95.829, 96.574 |

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
