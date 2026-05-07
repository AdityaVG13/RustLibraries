# /// script
# dependencies = ["numpy"]
# ///

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "benchmark-results"
LOCK_IN = RESULT_DIR / "external-source-lock.json"
JSON_OUT = RESULT_DIR / "external-numpy-asv-inspired.json"
MD_OUT = RESULT_DIR / "external-numpy-asv-inspired.md"
LOSS_TRIAGE_JSON_OUT = RESULT_DIR / "external-numpy-loss-triage.json"
LOSS_TRIAGE_MD_OUT = RESULT_DIR / "external-numpy-loss-triage.md"
LOSS_FOCUSED_JSON_OUT = RESULT_DIR / "external-numpy-loss-focused.json"
LOSS_FOCUSED_MD_OUT = RESULT_DIR / "external-numpy-loss-focused.md"
ARRAY_API_REPORT = RESULT_DIR / "array-api-tests-full-maxfail.json"
LSTSQ_FIXTURE = RESULT_DIR / "numpy-asv-lstsq-f64.bin"
NEAR_TIE_RELATIVE_MARGIN = 0.02


@dataclass(frozen=True)
class CaseSpec:
    name: str
    source_id: str
    source_path: str
    source_symbol: str
    translation: str
    repetitions: int


RUNNABLE_CASES: list[CaseSpec] = [
    CaseSpec(
        name="asv_ufunc_broadcast_sub_f64",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_ufunc.py",
        source_symbol="Broadcast.time_broadcast",
        translation="direct setup and operation: ones((50000, 100), f64) - ones((100,), f64)",
        repetitions=1,
    ),
    CaseSpec(
        name="asv_ufunc_astype_i32_to_f64_100x100",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_ufunc.py",
        source_symbol="NDArrayAsType.time_astype(typeconv=('int32', 'float64'))",
        translation="same operation, dtype pair, and 100x100 get_squares_ shape; deterministic arange values",
        repetitions=5_000,
    ),
    CaseSpec(
        name="asv_ufunc_add_at_f64_10000000",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_ufunc.py",
        source_symbol="At.time_sum_at",
        translation="same operation and sizes; deterministic values and indices instead of RNG setup",
        repetitions=1,
    ),
    CaseSpec(
        name="asv_ufunc_maximum_at_f64_10000000",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_ufunc.py",
        source_symbol="At.time_maximum_at",
        translation="same operation and sizes; deterministic values and indices instead of RNG setup",
        repetitions=1,
    ),
    CaseSpec(
        name="asv_reduce_small_sum_f32_100",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_reduce.py",
        source_symbol="SmallReduction.time_small",
        translation="direct setup and operation, repeated because ASV auto-calibrates tiny timings",
        repetitions=200_000,
    ),
    CaseSpec(
        name="asv_reduce_stats_min_f64_200",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_reduce.py",
        source_symbol="StatsReductions.time_min(dtype=float64)",
        translation="direct setup and operation: min(ones(200, float64)), repeated because ASV auto-calibrates tiny timings",
        repetitions=100_000,
    ),
    CaseSpec(
        name="asv_reduce_stats_min_f32_200",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_reduce.py",
        source_symbol="StatsReductions.time_min(dtype=float32)",
        translation="direct setup and operation: min(ones(200, float32)), repeated because ASV auto-calibrates tiny timings",
        repetitions=100_000,
    ),
    CaseSpec(
        name="asv_reduce_stats_min_i64_200",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_reduce.py",
        source_symbol="StatsReductions.time_min(dtype=int64)",
        translation="direct setup and operation: min(ones(200, int64)), repeated because ASV auto-calibrates tiny timings",
        repetitions=100_000,
    ),
    CaseSpec(
        name="asv_reduce_stats_min_u64_200",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_reduce.py",
        source_symbol="StatsReductions.time_min(dtype=uint64)",
        translation="direct setup and operation: min(ones(200, uint64)), repeated because ASV auto-calibrates tiny timings",
        repetitions=100_000,
    ),
    CaseSpec(
        name="asv_reduce_stats_min_bool_200",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_reduce.py",
        source_symbol="StatsReductions.time_min(dtype=bool_)",
        translation="direct setup and operation: min(ones(200, bool_)), repeated because ASV auto-calibrates tiny timings",
        repetitions=100_000,
    ),
    CaseSpec(
        name="asv_reduce_stats_min_c64_200",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_reduce.py",
        source_symbol="StatsReductions.time_min(dtype=complex64)",
        translation="direct setup and operation: min(ones(200, complex64) * 1j), repeated because ASV auto-calibrates tiny timings",
        repetitions=100_000,
    ),
    CaseSpec(
        name="asv_reduce_stats_max_f64_200",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_reduce.py",
        source_symbol="StatsReductions.time_max(dtype=float64)",
        translation="direct setup and operation: max(ones(200, float64)), repeated because ASV auto-calibrates tiny timings",
        repetitions=100_000,
    ),
    CaseSpec(
        name="asv_reduce_stats_max_f32_200",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_reduce.py",
        source_symbol="StatsReductions.time_max(dtype=float32)",
        translation="direct setup and operation: max(ones(200, float32)), repeated because ASV auto-calibrates tiny timings",
        repetitions=100_000,
    ),
    CaseSpec(
        name="asv_reduce_stats_max_i64_200",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_reduce.py",
        source_symbol="StatsReductions.time_max(dtype=int64)",
        translation="direct setup and operation: max(ones(200, int64)), repeated because ASV auto-calibrates tiny timings",
        repetitions=100_000,
    ),
    CaseSpec(
        name="asv_reduce_stats_max_u64_200",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_reduce.py",
        source_symbol="StatsReductions.time_max(dtype=uint64)",
        translation="direct setup and operation: max(ones(200, uint64)), repeated because ASV auto-calibrates tiny timings",
        repetitions=100_000,
    ),
    CaseSpec(
        name="asv_reduce_stats_max_bool_200",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_reduce.py",
        source_symbol="StatsReductions.time_max(dtype=bool_)",
        translation="direct setup and operation: max(ones(200, bool_)), repeated because ASV auto-calibrates tiny timings",
        repetitions=100_000,
    ),
    CaseSpec(
        name="asv_reduce_stats_max_c64_200",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_reduce.py",
        source_symbol="StatsReductions.time_max(dtype=complex64)",
        translation="direct setup and operation: max(ones(200, complex64) * 1j), repeated because ASV auto-calibrates tiny timings",
        repetitions=100_000,
    ),
    CaseSpec(
        name="asv_reduce_stats_mean_f64_200",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_reduce.py",
        source_symbol="StatsReductions.time_mean(dtype=float64)",
        translation="direct setup and operation: mean(ones(200, float64)), repeated because ASV auto-calibrates tiny timings",
        repetitions=100_000,
    ),
    CaseSpec(
        name="asv_reduce_stats_mean_f32_200",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_reduce.py",
        source_symbol="StatsReductions.time_mean(dtype=float32)",
        translation="direct setup and operation: mean(ones(200, float32)), repeated because ASV auto-calibrates tiny timings",
        repetitions=100_000,
    ),
    CaseSpec(
        name="asv_reduce_stats_mean_i64_200",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_reduce.py",
        source_symbol="StatsReductions.time_mean(dtype=int64)",
        translation="direct setup and operation: mean(ones(200, int64)), repeated because ASV auto-calibrates tiny timings",
        repetitions=100_000,
    ),
    CaseSpec(
        name="asv_reduce_stats_mean_u64_200",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_reduce.py",
        source_symbol="StatsReductions.time_mean(dtype=uint64)",
        translation="direct setup and operation: mean(ones(200, uint64)), repeated because ASV auto-calibrates tiny timings",
        repetitions=100_000,
    ),
    CaseSpec(
        name="asv_reduce_stats_mean_bool_200",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_reduce.py",
        source_symbol="StatsReductions.time_mean(dtype=bool_)",
        translation="direct setup and operation: mean(ones(200, bool_)), repeated because ASV auto-calibrates tiny timings",
        repetitions=100_000,
    ),
    CaseSpec(
        name="asv_reduce_stats_mean_c64_200",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_reduce.py",
        source_symbol="StatsReductions.time_mean(dtype=complex64)",
        translation="direct setup and operation: mean(ones(200, complex64) * 1j), repeated because ASV auto-calibrates tiny timings",
        repetitions=100_000,
    ),
    CaseSpec(
        name="asv_reduce_stats_std_f64_200",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_reduce.py",
        source_symbol="StatsReductions.time_std(dtype=float64)",
        translation="direct setup and operation: std(ones(200, float64)), repeated because ASV auto-calibrates tiny timings",
        repetitions=100_000,
    ),
    CaseSpec(
        name="asv_reduce_stats_std_f32_200",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_reduce.py",
        source_symbol="StatsReductions.time_std(dtype=float32)",
        translation="direct setup and operation: std(ones(200, float32)), repeated because ASV auto-calibrates tiny timings",
        repetitions=100_000,
    ),
    CaseSpec(
        name="asv_reduce_stats_std_i64_200",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_reduce.py",
        source_symbol="StatsReductions.time_std(dtype=int64)",
        translation="direct setup and operation: std(ones(200, int64)), repeated because ASV auto-calibrates tiny timings",
        repetitions=100_000,
    ),
    CaseSpec(
        name="asv_reduce_stats_std_u64_200",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_reduce.py",
        source_symbol="StatsReductions.time_std(dtype=uint64)",
        translation="direct setup and operation: std(ones(200, uint64)), repeated because ASV auto-calibrates tiny timings",
        repetitions=100_000,
    ),
    CaseSpec(
        name="asv_reduce_stats_std_bool_200",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_reduce.py",
        source_symbol="StatsReductions.time_std(dtype=bool_)",
        translation="direct setup and operation: std(ones(200, bool_)), repeated because ASV auto-calibrates tiny timings",
        repetitions=100_000,
    ),
    CaseSpec(
        name="asv_reduce_stats_std_c64_200",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_reduce.py",
        source_symbol="StatsReductions.time_std(dtype=complex64)",
        translation="direct setup and operation: std(ones(200, complex64) * 1j), repeated because ASV auto-calibrates tiny timings",
        repetitions=100_000,
    ),
    CaseSpec(
        name="asv_reduce_stats_prod_f64_200",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_reduce.py",
        source_symbol="StatsReductions.time_prod(dtype=float64)",
        translation="direct setup and operation: prod(ones(200, float64)), repeated because ASV auto-calibrates tiny timings",
        repetitions=100_000,
    ),
    CaseSpec(
        name="asv_reduce_stats_prod_f32_200",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_reduce.py",
        source_symbol="StatsReductions.time_prod(dtype=float32)",
        translation="direct setup and operation: prod(ones(200, float32)), repeated because ASV auto-calibrates tiny timings",
        repetitions=100_000,
    ),
    CaseSpec(
        name="asv_reduce_stats_prod_i64_200",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_reduce.py",
        source_symbol="StatsReductions.time_prod(dtype=int64)",
        translation="direct setup and operation: prod(ones(200, int64)), repeated because ASV auto-calibrates tiny timings",
        repetitions=100_000,
    ),
    CaseSpec(
        name="asv_reduce_stats_prod_u64_200",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_reduce.py",
        source_symbol="StatsReductions.time_prod(dtype=uint64)",
        translation="direct setup and operation: prod(ones(200, uint64)), repeated because ASV auto-calibrates tiny timings",
        repetitions=100_000,
    ),
    CaseSpec(
        name="asv_reduce_stats_prod_bool_200",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_reduce.py",
        source_symbol="StatsReductions.time_prod(dtype=bool_)",
        translation="direct setup and operation: prod(ones(200, bool_)), repeated because ASV auto-calibrates tiny timings",
        repetitions=100_000,
    ),
    CaseSpec(
        name="asv_reduce_stats_prod_c64_200",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_reduce.py",
        source_symbol="StatsReductions.time_prod(dtype=complex64)",
        translation="direct setup and operation: prod(ones(200, complex64) * 1j), repeated because ASV auto-calibrates tiny timings",
        repetitions=100_000,
    ),
    CaseSpec(
        name="asv_reduce_stats_var_f64_200",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_reduce.py",
        source_symbol="StatsReductions.time_var(dtype=float64)",
        translation="direct setup and operation: var(ones(200, float64)), repeated because ASV auto-calibrates tiny timings",
        repetitions=100_000,
    ),
    CaseSpec(
        name="asv_reduce_stats_var_f32_200",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_reduce.py",
        source_symbol="StatsReductions.time_var(dtype=float32)",
        translation="direct setup and operation: var(ones(200, float32)), repeated because ASV auto-calibrates tiny timings",
        repetitions=100_000,
    ),
    CaseSpec(
        name="asv_reduce_stats_var_i64_200",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_reduce.py",
        source_symbol="StatsReductions.time_var(dtype=int64)",
        translation="direct setup and operation: var(ones(200, int64)), repeated because ASV auto-calibrates tiny timings",
        repetitions=100_000,
    ),
    CaseSpec(
        name="asv_reduce_stats_var_u64_200",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_reduce.py",
        source_symbol="StatsReductions.time_var(dtype=uint64)",
        translation="direct setup and operation: var(ones(200, uint64)), repeated because ASV auto-calibrates tiny timings",
        repetitions=100_000,
    ),
    CaseSpec(
        name="asv_reduce_stats_var_bool_200",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_reduce.py",
        source_symbol="StatsReductions.time_var(dtype=bool_)",
        translation="direct setup and operation: var(ones(200, bool_)), repeated because ASV auto-calibrates tiny timings",
        repetitions=100_000,
    ),
    CaseSpec(
        name="asv_reduce_stats_var_c64_200",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_reduce.py",
        source_symbol="StatsReductions.time_var(dtype=complex64)",
        translation="direct setup and operation: var(ones(200, complex64) * 1j), repeated because ASV auto-calibrates tiny timings",
        repetitions=100_000,
    ),
    CaseSpec(
        name="asv_reduce_argmax_i64_200000",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_reduce.py",
        source_symbol="ArgMax.time_argmax(dtype=int64)",
        translation="direct setup and operation: zeros(200000, int64).argmax(), repeated because ASV auto-calibrates tiny timings",
        repetitions=20_000,
    ),
    CaseSpec(
        name="asv_reduce_argmin_i64_200000",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_reduce.py",
        source_symbol="ArgMin.time_argmin(dtype=int64)",
        translation="direct setup and operation: ones(200000, int64).argmin(), repeated because ASV auto-calibrates tiny timings",
        repetitions=20_000,
    ),
    CaseSpec(
        name="asv_itemselection_take_i64_1000x1",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_itemselection.py",
        source_symbol="Take.time_contiguous(shape=(1000, 1), mode='raise', dtype='int64')",
        translation="direct setup and operation, repeated because ASV auto-calibrates tiny timings",
        repetitions=2_000,
    ),
    CaseSpec(
        name="asv_itemselection_putmask_dense_scalar_f64_1000",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_itemselection.py",
        source_symbol="PutMask.time_dense(values_is_scalar=True, dtype=float64)",
        translation="direct setup and operation: putmask(ones(1000), dense_bool_mask, scalar_one), repeated because ASV auto-calibrates tiny timings",
        repetitions=10_000,
    ),
    CaseSpec(
        name="asv_itemselection_putmask_sparse_scalar_f64_1000",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_itemselection.py",
        source_symbol="PutMask.time_sparse(values_is_scalar=True, dtype=float64)",
        translation="direct setup and operation: putmask(ones(1000), sparse_bool_mask, scalar_one), repeated because ASV auto-calibrates tiny timings",
        repetitions=10_000,
    ),
    CaseSpec(
        name="asv_itemselection_put_ordered_f64_1000",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_itemselection.py",
        source_symbol="Put.time_ordered(values_is_scalar=False, dtype=float64)",
        translation="direct setup and operation: put(ones(1000), arange(1000), ones(1000)), repeated because ASV auto-calibrates tiny timings",
        repetitions=10_000,
    ),
    CaseSpec(
        name="asv_manipulate_broadcast_arrays_f64_16x32",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_manipulate.py",
        source_symbol="BroadcastArrays.time_broadcast_arrays(shape=(16, 32), ndtype=float64)",
        translation="same operation, shape, and dtype; deterministic arange values instead of RNG setup, repeated because ASV auto-calibrates metadata timings",
        repetitions=200_000,
    ),
    CaseSpec(
        name="asv_manipulate_broadcast_arrays_f64_128x256",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_manipulate.py",
        source_symbol="BroadcastArrays.time_broadcast_arrays(shape=(128, 256), ndtype=float64)",
        translation="same operation, shape, and dtype; deterministic arange values instead of RNG setup, repeated because ASV auto-calibrates metadata timings",
        repetitions=100_000,
    ),
    CaseSpec(
        name="asv_manipulate_broadcast_arrays_f32_128x256",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_manipulate.py",
        source_symbol="BroadcastArrays.time_broadcast_arrays(shape=(128, 256), ndtype=float32)",
        translation="same operation, shape, and dtype; deterministic arange values instead of RNG setup, repeated because ASV auto-calibrates metadata timings",
        repetitions=100_000,
    ),
    CaseSpec(
        name="asv_manipulate_broadcast_arrays_i32_128x256",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_manipulate.py",
        source_symbol="BroadcastArrays.time_broadcast_arrays(shape=(128, 256), ndtype=int32)",
        translation="same operation, shape, and dtype; deterministic arange values instead of RNG setup, repeated because ASV auto-calibrates metadata timings",
        repetitions=100_000,
    ),
    CaseSpec(
        name="asv_manipulate_broadcast_arrays_f64_512x1024",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_manipulate.py",
        source_symbol="BroadcastArrays.time_broadcast_arrays(shape=(512, 1024), ndtype=float64)",
        translation="same operation, shape, and dtype; deterministic arange values instead of RNG setup, repeated because ASV auto-calibrates metadata timings",
        repetitions=100_000,
    ),
    CaseSpec(
        name="asv_manipulate_broadcast_to_f64_16",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_manipulate.py",
        source_symbol="BroadcastArraysTo.time_broadcast_to(size=16, ndtype=float64)",
        translation="same operation, shape, and dtype; deterministic arange values instead of RNG setup, repeated because ASV auto-calibrates metadata timings",
        repetitions=200_000,
    ),
    CaseSpec(
        name="asv_manipulate_broadcast_to_f64_64",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_manipulate.py",
        source_symbol="BroadcastArraysTo.time_broadcast_to(size=64, ndtype=float64)",
        translation="same operation, shape, and dtype; deterministic arange values instead of RNG setup, repeated because ASV auto-calibrates metadata timings",
        repetitions=200_000,
    ),
    CaseSpec(
        name="asv_manipulate_broadcast_to_f32_64",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_manipulate.py",
        source_symbol="BroadcastArraysTo.time_broadcast_to(size=64, ndtype=float32)",
        translation="same operation, shape, and dtype; deterministic arange values instead of RNG setup, repeated because ASV auto-calibrates metadata timings",
        repetitions=200_000,
    ),
    CaseSpec(
        name="asv_manipulate_broadcast_to_i32_64",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_manipulate.py",
        source_symbol="BroadcastArraysTo.time_broadcast_to(size=64, ndtype=int32)",
        translation="same operation, shape, and dtype; deterministic arange values instead of RNG setup, repeated because ASV auto-calibrates metadata timings",
        repetitions=200_000,
    ),
    CaseSpec(
        name="asv_manipulate_broadcast_to_f64_512",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_manipulate.py",
        source_symbol="BroadcastArraysTo.time_broadcast_to(size=512, ndtype=float64)",
        translation="same operation, shape, and dtype; deterministic arange values instead of RNG setup, repeated because ASV auto-calibrates metadata timings",
        repetitions=200_000,
    ),
    CaseSpec(
        name="asv_manipulate_concatenate_ax0_f64_32x64_n5",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_manipulate.py",
        source_symbol="ConcatenateStackArrays.time_concatenate_ax0(shape=(32, 64), narrays=5, ndtype=float64)",
        translation="same operation, shape, array count, and dtype; deterministic arange values instead of RNG setup, repeated equally",
        repetitions=2_000,
    ),
    CaseSpec(
        name="asv_manipulate_concatenate_ax0_f32_32x64_n5",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_manipulate.py",
        source_symbol="ConcatenateStackArrays.time_concatenate_ax0(shape=(32, 64), narrays=5, ndtype=float32)",
        translation="same operation, shape, array count, and dtype; deterministic arange values instead of RNG setup, repeated equally",
        repetitions=2_000,
    ),
    CaseSpec(
        name="asv_manipulate_concatenate_ax0_i32_32x64_n5",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_manipulate.py",
        source_symbol="ConcatenateStackArrays.time_concatenate_ax0(shape=(32, 64), narrays=5, ndtype=int32)",
        translation="same operation, shape, array count, and dtype; deterministic arange values instead of RNG setup, repeated equally",
        repetitions=2_000,
    ),
    CaseSpec(
        name="asv_manipulate_concatenate_ax1_f64_32x64_n5",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_manipulate.py",
        source_symbol="ConcatenateStackArrays.time_concatenate_ax1(shape=(32, 64), narrays=5, ndtype=float64)",
        translation="same operation, shape, array count, and dtype; deterministic arange values instead of RNG setup, repeated equally",
        repetitions=2_000,
    ),
    CaseSpec(
        name="asv_manipulate_concatenate_ax1_f32_32x64_n5",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_manipulate.py",
        source_symbol="ConcatenateStackArrays.time_concatenate_ax1(shape=(32, 64), narrays=5, ndtype=float32)",
        translation="same operation, shape, array count, and dtype; deterministic arange values instead of RNG setup, repeated equally",
        repetitions=2_000,
    ),
    CaseSpec(
        name="asv_manipulate_concatenate_ax1_i32_32x64_n5",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_manipulate.py",
        source_symbol="ConcatenateStackArrays.time_concatenate_ax1(shape=(32, 64), narrays=5, ndtype=int32)",
        translation="same operation, shape, array count, and dtype; deterministic arange values instead of RNG setup, repeated equally",
        repetitions=2_000,
    ),
    CaseSpec(
        name="asv_manipulate_stack_ax0_f64_32x64_n5",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_manipulate.py",
        source_symbol="ConcatenateStackArrays.time_stack_ax0(shape=(32, 64), narrays=5, ndtype=float64)",
        translation="same operation, shape, array count, and dtype; deterministic arange values instead of RNG setup, repeated equally",
        repetitions=2_000,
    ),
    CaseSpec(
        name="asv_manipulate_stack_ax0_f32_32x64_n5",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_manipulate.py",
        source_symbol="ConcatenateStackArrays.time_stack_ax0(shape=(32, 64), narrays=5, ndtype=float32)",
        translation="same operation, shape, array count, and dtype; deterministic arange values instead of RNG setup, repeated equally",
        repetitions=2_000,
    ),
    CaseSpec(
        name="asv_manipulate_stack_ax0_i32_32x64_n5",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_manipulate.py",
        source_symbol="ConcatenateStackArrays.time_stack_ax0(shape=(32, 64), narrays=5, ndtype=int32)",
        translation="same operation, shape, array count, and dtype; deterministic arange values instead of RNG setup, repeated equally",
        repetitions=2_000,
    ),
    CaseSpec(
        name="asv_manipulate_stack_ax1_f64_32x64_n5",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_manipulate.py",
        source_symbol="ConcatenateStackArrays.time_stack_ax1(shape=(32, 64), narrays=5, ndtype=float64)",
        translation="same operation, shape, array count, and dtype; deterministic arange values instead of RNG setup, repeated equally",
        repetitions=2_000,
    ),
    CaseSpec(
        name="asv_manipulate_stack_ax1_f32_32x64_n5",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_manipulate.py",
        source_symbol="ConcatenateStackArrays.time_stack_ax1(shape=(32, 64), narrays=5, ndtype=float32)",
        translation="same operation, shape, array count, and dtype; deterministic arange values instead of RNG setup, repeated equally",
        repetitions=2_000,
    ),
    CaseSpec(
        name="asv_manipulate_stack_ax1_i32_32x64_n5",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_manipulate.py",
        source_symbol="ConcatenateStackArrays.time_stack_ax1(shape=(32, 64), narrays=5, ndtype=int32)",
        translation="same operation, shape, array count, and dtype; deterministic arange values instead of RNG setup, repeated equally",
        repetitions=2_000,
    ),
    CaseSpec(
        name="asv_manipulate_expand_dims_f64_5x2x3x1_axis1",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_manipulate.py",
        source_symbol="DimsManipulations.time_expand_dims(shape=(5, 2, 3, 1))",
        translation="direct setup and operation: expand_dims(ones((5,2,3,1)), axis=1), repeated because ASV auto-calibrates metadata timings",
        repetitions=200_000,
    ),
    CaseSpec(
        name="asv_manipulate_expand_dims_neg_f64_5x2x3x1",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_manipulate.py",
        source_symbol="DimsManipulations.time_expand_dims_neg(shape=(5, 2, 3, 1))",
        translation="direct setup and operation: expand_dims(ones((5,2,3,1)), axis=-1), repeated because ASV auto-calibrates metadata timings",
        repetitions=200_000,
    ),
    CaseSpec(
        name="asv_manipulate_squeeze_dims_f64_5x2x3x1",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_manipulate.py",
        source_symbol="DimsManipulations.time_squeeze_dims(shape=(5, 2, 3, 1))",
        translation="direct setup and operation: squeeze(ones((5,2,3,1))), repeated because ASV auto-calibrates metadata timings",
        repetitions=200_000,
    ),
    CaseSpec(
        name="asv_manipulate_flip_all_f64_5x2x3x1",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_manipulate.py",
        source_symbol="DimsManipulations.time_flip_all(shape=(5, 2, 3, 1))",
        translation="same operation and shape with deterministic arange values instead of ASV ones, repeated because ASV auto-calibrates metadata timings",
        repetitions=200_000,
    ),
    CaseSpec(
        name="asv_manipulate_flip_one_f64_5x2x3x1_axis1",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_manipulate.py",
        source_symbol="DimsManipulations.time_flip_one(shape=(5, 2, 3, 1))",
        translation="same operation and shape with deterministic arange values instead of ASV ones: flip(axis=1), repeated because ASV auto-calibrates metadata timings",
        repetitions=200_000,
    ),
    CaseSpec(
        name="asv_manipulate_flip_neg_f64_5x2x3x1_axis_neg1",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_manipulate.py",
        source_symbol="DimsManipulations.time_flip_neg(shape=(5, 2, 3, 1))",
        translation="same operation and shape with deterministic arange values instead of ASV ones: flip(axis=-1), repeated because ASV auto-calibrates metadata timings",
        repetitions=200_000,
    ),
    CaseSpec(
        name="asv_manipulate_moveaxis_f64_5x2x3x1",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_manipulate.py",
        source_symbol="DimsManipulations.time_moveaxis(shape=(5, 2, 3, 1))",
        translation="same operation and shape with deterministic arange values instead of ASV ones: moveaxis([0,1],[-1,-2]), repeated because ASV auto-calibrates metadata timings",
        repetitions=200_000,
    ),
    CaseSpec(
        name="asv_manipulate_roll_f64_5x2x3x1_shift3",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_manipulate.py",
        source_symbol="DimsManipulations.time_roll(shape=(5, 2, 3, 1))",
        translation="same operation and shape with deterministic arange values instead of ASV ones: roll(shift=3), repeated because ASV auto-calibrates small-array timings",
        repetitions=100_000,
    ),
    CaseSpec(
        name="asv_manipulate_reshape_f64_5x2x3x1_to_1x5x2x3",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_manipulate.py",
        source_symbol="DimsManipulations.time_reshape(shape=(5, 2, 3, 1))",
        translation="direct setup and operation: reshape(ones((5,2,3,1)), deque-rotated shape (1,5,2,3)), repeated because ASV auto-calibrates metadata timings",
        repetitions=200_000,
    ),
    CaseSpec(
        name="asv_linalg_dot_a_b_f64_150x400_400x600",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_linalg.py",
        source_symbol="Eindot.time_dot_a_b",
        translation="direct setup and operation: dot(arange(60000).reshape(150,400), arange(240000).reshape(400,600)), repeated equally",
        repetitions=1000,
    ),
    CaseSpec(
        name="asv_linalg_matmul_a_b_f64_150x400_400x600",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_linalg.py",
        source_symbol="Eindot.time_matmul_a_b",
        translation="direct setup and operation: matmul(arange(60000).reshape(150,400), arange(240000).reshape(400,600)), repeated equally",
        repetitions=1000,
    ),
    CaseSpec(
        name="asv_linalg_matmul_d_matmul_b_c_f64",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_linalg.py",
        source_symbol="Eindot.time_matmul_d_matmul_b_c",
        translation="direct setup and operation: matmul(arange(400), matmul(arange(240000).reshape(400,600), arange(600))), repeated equally",
        repetitions=1000,
    ),
    CaseSpec(
        name="asv_linalg_dot_d_dot_b_c_f64",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_linalg.py",
        source_symbol="Eindot.time_dot_d_dot_b_c",
        translation="direct setup and operation: dot(arange(400), dot(arange(240000).reshape(400,600), arange(600))); equivalent vector/matrix matmul path in Rust, repeated equally",
        repetitions=1000,
    ),
    CaseSpec(
        name="asv_linalg_dot_trans_a_at_f64_150x400_400x150",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_linalg.py",
        source_symbol="Eindot.time_dot_trans_a_at",
        translation="direct setup and operation: dot(arange(60000).reshape(150,400), arange(60000).reshape(150,400).T), repeated equally",
        repetitions=1000,
    ),
    CaseSpec(
        name="asv_linalg_dot_trans_a_atc_f64_150x400_400x150",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_linalg.py",
        source_symbol="Eindot.time_dot_trans_a_atc",
        translation="direct setup and operation: dot(arange(60000).reshape(150,400), arange(60000).reshape(150,400).T.copy()), repeated equally",
        repetitions=1000,
    ),
    CaseSpec(
        name="asv_linalg_dot_trans_at_a_f64_400x150_150x400",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_linalg.py",
        source_symbol="Eindot.time_dot_trans_at_a",
        translation="direct setup and operation: dot(arange(60000).reshape(150,400).T, arange(60000).reshape(150,400)), repeated equally",
        repetitions=1000,
    ),
    CaseSpec(
        name="asv_linalg_dot_trans_atc_a_f64_400x150_150x400",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_linalg.py",
        source_symbol="Eindot.time_dot_trans_atc_a",
        translation="direct setup and operation: dot(arange(60000).reshape(150,400).T.copy(), arange(60000).reshape(150,400)), repeated equally",
        repetitions=1000,
    ),
    CaseSpec(
        name="asv_linalg_inner_a_a_f64_150x400_150x400",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_linalg.py",
        source_symbol="Eindot.time_inner_trans_a_a",
        translation="direct setup and operation: inner(arange(60000).reshape(150,400), same array), repeated equally",
        repetitions=1000,
    ),
    CaseSpec(
        name="asv_linalg_inner_a_ac_f64_150x400_150x400",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_linalg.py",
        source_symbol="Eindot.time_inner_trans_a_ac",
        translation="direct setup and operation: inner(arange(60000).reshape(150,400), copied array), repeated equally",
        repetitions=1000,
    ),
    CaseSpec(
        name="asv_linalg_matmul_trans_a_at_f64_150x400_400x150",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_linalg.py",
        source_symbol="Eindot.time_matmul_trans_a_at",
        translation="direct setup and operation: matmul(arange(60000).reshape(150,400), arange(60000).reshape(150,400).T), repeated equally",
        repetitions=1000,
    ),
    CaseSpec(
        name="asv_linalg_matmul_trans_a_atc_f64_150x400_400x150",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_linalg.py",
        source_symbol="Eindot.time_matmul_trans_a_atc",
        translation="direct setup and operation: matmul(arange(60000).reshape(150,400), arange(60000).reshape(150,400).T.copy()), repeated equally",
        repetitions=1000,
    ),
    CaseSpec(
        name="asv_linalg_matmul_trans_at_a_f64_400x150_150x400",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_linalg.py",
        source_symbol="Eindot.time_matmul_trans_at_a",
        translation="direct setup and operation: matmul(arange(60000).reshape(150,400).T, arange(60000).reshape(150,400)), repeated equally",
        repetitions=1000,
    ),
    CaseSpec(
        name="asv_linalg_matmul_trans_atc_a_f64_400x150_150x400",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_linalg.py",
        source_symbol="Eindot.time_matmul_trans_atc_a",
        translation="direct setup and operation: matmul(arange(60000).reshape(150,400).T.copy(), arange(60000).reshape(150,400)), repeated equally",
        repetitions=1000,
    ),
    CaseSpec(
        name="asv_linalg_tensordot_a3_b3_axes_10_01",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_linalg.py",
        source_symbol="Eindot.time_tensordot_a_b_axes_1_0_0_1",
        translation="direct setup and operation: tensordot(arange(480000).reshape(60,80,100), arange(192000).reshape(80,60,40), axes=([1,0],[0,1])), repeated equally",
        repetitions=10,
    ),
    CaseSpec(
        name="asv_linalg_norm_small_array_f64_5",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_linalg.py",
        source_symbol="LinalgSmallArrays.time_norm_small_array",
        translation="direct setup and operation: norm(arange(5.0)), repeated because ASV auto-calibrates tiny timings",
        repetitions=100_000,
    ),
    CaseSpec(
        name="asv_linalg_det_small_array_f64_5x5",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_linalg.py",
        source_symbol="LinalgSmallArrays.time_det_small_array",
        translation="direct setup and operation: det(arange(25.0).reshape(5,5)), repeated because ASV auto-calibrates tiny timings",
        repetitions=100_000,
    ),
    CaseSpec(
        name="asv_linalg_det_3x3_f64",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_linalg.py",
        source_symbol="LinalgSmallArrays.time_det_3x3",
        translation="direct setup and operation: det(eye(3) + arange(9.0).reshape(3,3)), repeated because ASV auto-calibrates tiny timings",
        repetitions=100_000,
    ),
    CaseSpec(
        name="asv_linalg_solve_3x3_f64",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_linalg.py",
        source_symbol="LinalgSmallArrays.time_solve_3x3",
        translation="direct setup and operation: solve(eye(3) + arange(9.0).reshape(3,3), arange(3.0)), repeated because ASV auto-calibrates tiny timings",
        repetitions=100_000,
    ),
    CaseSpec(
        name="asv_linalg_lstsq_square_f64_100x100",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_linalg.py",
        source_symbol="Lstsq.time_numpy_linalg_lstsq_a__b_float64",
        translation="direct setup from common.py get_squares_()['float64'] and get_indexes_rand()[:100]; square full-rank lstsq compared to equivalent square solve, repeated equally",
        repetitions=100,
    ),
    CaseSpec(
        name="asv_linalg_einsum_outer_f64_3000",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_linalg.py",
        source_symbol="Einsum.time_einsum_outer(dtype=float64)",
        translation='direct setup and operation: einsum("i,j", arange(3000), arange(3000))',
        repetitions=1,
    ),
    CaseSpec(
        name="asv_linalg_einsum_i_ij_j_f64_400_400x600_600",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_linalg.py",
        source_symbol="Eindot.time_einsum_i_ij_j",
        translation="direct setup and operation: einsum('i,ij,j', arange(400), arange(240000).reshape(400,600), arange(600)), repeated equally",
        repetitions=1000,
    ),
    CaseSpec(
        name="asv_linalg_einsum_ij_jk_f64_150x400_400x600",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_linalg.py",
        source_symbol="Eindot.time_einsum_ij_jk_a_b",
        translation="direct setup and operation: einsum('ij,jk', arange(60000).reshape(150,400), arange(240000).reshape(400,600)), repeated equally; equivalent Rust matrix-contraction path",
        repetitions=1000,
    ),
    CaseSpec(
        name="asv_linalg_einsum_multiply_f64_30x40_20x30x40",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_linalg.py",
        source_symbol="Einsum.time_einsum_multiply(dtype=float64)",
        translation='direct setup and operation: einsum("..., ...", arange(1200).reshape(30,40), arange(24000).reshape(20,30,40))',
        repetitions=1,
    ),
    CaseSpec(
        name="asv_linalg_einsum_sum_mul_f64_scalar_10x100x10",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_linalg.py",
        source_symbol="Einsum.time_einsum_sum_mul(dtype=float64)",
        translation='direct setup and operation: einsum(",i...->", 300, arange(10000).reshape(10,100,10)), repeated equally',
        repetitions=100,
    ),
    CaseSpec(
        name="asv_linalg_einsum_sum_mul2_f64_10x100x10_scalar",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_linalg.py",
        source_symbol="Einsum.time_einsum_sum_mul2(dtype=float64)",
        translation='direct setup and operation: einsum("i...,->", arange(10000).reshape(10,100,10), 300), repeated equally',
        repetitions=100,
    ),
    CaseSpec(
        name="asv_linalg_einsum_scalar_mul_f64_480000",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_linalg.py",
        source_symbol="Einsum.time_einsum_mul(dtype=float64)",
        translation='direct setup and operation: einsum("i,->i", arange(480000), 300), repeated equally',
        repetitions=100,
    ),
    CaseSpec(
        name="asv_linalg_einsum_sum_f64_480000",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_linalg.py",
        source_symbol="Einsum.time_einsum_contig_outstride0(dtype=float64)",
        translation='direct setup and operation: einsum("i->", arange(480000)), repeated equally',
        repetitions=100,
    ),
    CaseSpec(
        name="asv_linalg_einsum_weighted_sum_f64_400x600",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_linalg.py",
        source_symbol="Einsum.time_einsum_contig_contig(dtype=float64)",
        translation='direct setup and operation: einsum("ji,i->", arange(240000).reshape(400,600), arange(600)), repeated equally',
        repetitions=100,
    ),
    CaseSpec(
        name="asv_linalg_einsum_noncon_outer_f64_2000",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_linalg.py",
        source_symbol="Einsum.time_einsum_noncon_outer(dtype=float64)",
        translation='direct setup and operation: einsum("i,j", arange(1,4000,2), arange(1,4000,2))',
        repetitions=1,
    ),
    CaseSpec(
        name="asv_linalg_einsum_noncon_multiply_f64_30x40_20x30x40",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_linalg.py",
        source_symbol="Einsum.time_einsum_noncon_multiply(dtype=float64)",
        translation='direct setup and operation: einsum("..., ...", arange(1,2400,2).reshape(30,40), arange(1,48000,2).reshape(20,30,40))',
        repetitions=1,
    ),
    CaseSpec(
        name="asv_linalg_einsum_noncon_sum_mul_f64_scalar_20x30x40",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_linalg.py",
        source_symbol="Einsum.time_einsum_noncon_sum_mul(dtype=float64)",
        translation='direct setup and operation: einsum(",i...->", 300, arange(1,48000,2).reshape(20,30,40)), repeated equally',
        repetitions=100,
    ),
    CaseSpec(
        name="asv_linalg_einsum_noncon_sum_mul2_f64_20x30x40_scalar",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_linalg.py",
        source_symbol="Einsum.time_einsum_noncon_sum_mul2(dtype=float64)",
        translation='direct setup and operation: einsum("i...,->", arange(1,48000,2).reshape(20,30,40), 300), repeated equally',
        repetitions=100,
    ),
    CaseSpec(
        name="asv_linalg_einsum_noncon_scalar_mul_f64_2000",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_linalg.py",
        source_symbol="Einsum.time_einsum_noncon_mul(dtype=float64)",
        translation='direct setup and operation: einsum("i,->i", arange(1,4000,2), 300), repeated equally',
        repetitions=100,
    ),
    CaseSpec(
        name="asv_linalg_einsum_noncon_weighted_sum_f64_30x40",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_linalg.py",
        source_symbol="Einsum.time_einsum_noncon_contig_contig(dtype=float64)",
        translation='direct setup and operation: einsum("ji,i->", arange(1,2400,2).reshape(30,40), arange(1,80,2)), repeated equally',
        repetitions=100,
    ),
    CaseSpec(
        name="asv_linalg_einsum_noncon_sum_f64_2000",
        source_id="numpy-asv",
        source_path="benchmarks/benchmarks/bench_linalg.py",
        source_symbol="Einsum.time_einsum_noncon_contig_outstride0(dtype=float64)",
        translation='direct setup and operation: einsum("i->", arange(1,4000,2)), repeated equally',
        repetitions=100,
    ),
]


UNSUPPORTED_EXTERNAL_CASES = [
    {
        "source": "bench_linalg.py",
        "case": "eig/remaining LAPACK, remaining strided/batched matmul, and full linalg/einsum grammar",
        "reason": "NumRust has dot, matmul, tensordot axes, norm, det, solve, square full-rank lstsq-equivalent ASV coverage, selected transposed-view and copied-transpose matmul coverage, and both contiguous and NumPy-ASV noncon einsum-style contractions, but not eig, the remaining LAPACK-style routines, every strided/batched matmul case, or the full tensor expression surface.",
    },
]


def median_ms(fn: Callable[[], float], rounds: int = 7) -> tuple[float, float]:
    samples = []
    checksum = fn()
    for _ in range(rounds):
        start = time.perf_counter()
        checksum = fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples), checksum


def edge_checksum(array: np.ndarray) -> float:
    flat = array.ravel()
    return float(flat[0]) + float(flat[-1])


def complex_scalar_checksum(value: np.generic) -> float:
    item = complex(value)
    return float(item.real + 7.0 * item.imag)


def shape_checksum(array: np.ndarray) -> float:
    return float(sum(array.shape) + array.ndim)


def sample_checksum(array: np.ndarray) -> float:
    total = shape_checksum(array)
    if array.size == 0:
        return total
    raw_indices = [0, array.size // 3, (array.size * 2) // 3, array.size - 1]
    indices = []
    for index in raw_indices:
        if index not in indices:
            indices.append(index)
    for weight, flat_index in enumerate(indices, start=1):
        logical_index = np.unravel_index(flat_index, array.shape)
        total += weight * float(array[logical_index])
    return float(total)


def paired_broadcast_checksum(left: np.ndarray, right: np.ndarray) -> float:
    return (
        sample_checksum(left)
        + sample_checksum(right)
        + shape_checksum(left)
        + shape_checksum(right)
    )


def lstsq_fixture() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(1804169117)
    values = np.tile(rng.uniform(0, 100, size=1000 * 1000 // 10), 10)
    a = values.astype(np.float64).reshape((1000, 1000))[:100, :100].copy()

    indexes = list(range(1000))
    indexes.pop(5)
    indexes.pop(95)
    import random

    rnd = random.Random(1)
    rnd.shuffle(indexes)
    b = np.array(indexes[:100], dtype=np.float64)
    return a, b


def write_lstsq_fixture() -> None:
    a, b = lstsq_fixture()
    data = np.concatenate((a.ravel(), b))
    data.astype("<f8", copy=False).tofile(LSTSQ_FIXTURE)


def bench_numpy() -> dict:
    cases = []

    d = np.ones((50_000, 100), dtype=np.float64)
    e = np.ones((100,), dtype=np.float64)
    millis, checksum = median_ms(lambda: edge_checksum(d - e), rounds=7)
    cases.append({"name": "asv_ufunc_broadcast_sub_f64", "millis": millis, "checksum": checksum})

    astype_source = np.arange(10_000, dtype=np.int32).reshape(100, 100)

    def astype_i32_to_f64() -> float:
        checksum = 0.0
        for _ in range(5_000):
            checksum += edge_checksum(astype_source.astype(np.float64))
        return checksum

    millis, checksum = median_ms(astype_i32_to_f64, rounds=7)
    cases.append(
        {
            "name": "asv_ufunc_astype_i32_to_f64_100x100",
            "millis": millis,
            "checksum": checksum,
        }
    )

    at_count = 10_000_000
    at_vals = ((np.arange(at_count, dtype=np.float64) * 17.0) % 1000.0) / 1000.0
    at_idx = ((np.arange(at_count, dtype=np.int64) * 37) % 1000).astype(np.intp)

    def add_at() -> float:
        result = np.zeros(1000, dtype=np.float64)
        np.add.at(result, at_idx, at_vals)
        return edge_checksum(result)

    millis, checksum = median_ms(add_at, rounds=7)
    cases.append({"name": "asv_ufunc_add_at_f64_10000000", "millis": millis, "checksum": checksum})

    def maximum_at() -> float:
        result = np.zeros(1000, dtype=np.float64)
        np.maximum.at(result, at_idx, at_vals)
        return edge_checksum(result)

    millis, checksum = median_ms(maximum_at, rounds=7)
    cases.append(
        {"name": "asv_ufunc_maximum_at_f64_10000000", "millis": millis, "checksum": checksum}
    )

    small = np.ones(100, dtype=np.float32)

    def small_sum() -> float:
        checksum = 0.0
        for _ in range(200_000):
            checksum += float(np.sum(small))
        return checksum

    millis, checksum = median_ms(small_sum, rounds=7)
    cases.append({"name": "asv_reduce_small_sum_f32_100", "millis": millis, "checksum": checksum})

    stats_data = np.ones(200, dtype=np.float64)

    def stats_min() -> float:
        checksum = 0.0
        for _ in range(100_000):
            checksum += float(np.min(stats_data))
        return checksum

    millis, checksum = median_ms(stats_min, rounds=7)
    cases.append({"name": "asv_reduce_stats_min_f64_200", "millis": millis, "checksum": checksum})

    def stats_max() -> float:
        checksum = 0.0
        for _ in range(100_000):
            checksum += float(np.max(stats_data))
        return checksum

    millis, checksum = median_ms(stats_max, rounds=7)
    cases.append({"name": "asv_reduce_stats_max_f64_200", "millis": millis, "checksum": checksum})

    def stats_mean() -> float:
        checksum = 0.0
        for _ in range(100_000):
            checksum += float(np.mean(stats_data))
        return checksum

    millis, checksum = median_ms(stats_mean, rounds=7)
    cases.append({"name": "asv_reduce_stats_mean_f64_200", "millis": millis, "checksum": checksum})

    def stats_std() -> float:
        checksum = 0.0
        for _ in range(100_000):
            checksum += float(np.std(stats_data))
        return checksum

    millis, checksum = median_ms(stats_std, rounds=7)
    cases.append({"name": "asv_reduce_stats_std_f64_200", "millis": millis, "checksum": checksum})

    def stats_prod() -> float:
        checksum = 0.0
        for _ in range(100_000):
            checksum += float(np.prod(stats_data))
        return checksum

    millis, checksum = median_ms(stats_prod, rounds=7)
    cases.append({"name": "asv_reduce_stats_prod_f64_200", "millis": millis, "checksum": checksum})

    def stats_var() -> float:
        checksum = 0.0
        for _ in range(100_000):
            checksum += float(np.var(stats_data))
        return checksum

    millis, checksum = median_ms(stats_var, rounds=7)
    cases.append({"name": "asv_reduce_stats_var_f64_200", "millis": millis, "checksum": checksum})

    stats_data_f32 = np.ones(200, dtype=np.float32)

    def stats_min_f32() -> float:
        checksum = 0.0
        for _ in range(100_000):
            checksum += float(np.min(stats_data_f32))
        return checksum

    millis, checksum = median_ms(stats_min_f32, rounds=7)
    cases.append({"name": "asv_reduce_stats_min_f32_200", "millis": millis, "checksum": checksum})

    def stats_max_f32() -> float:
        checksum = 0.0
        for _ in range(100_000):
            checksum += float(np.max(stats_data_f32))
        return checksum

    millis, checksum = median_ms(stats_max_f32, rounds=7)
    cases.append({"name": "asv_reduce_stats_max_f32_200", "millis": millis, "checksum": checksum})

    def stats_mean_f32() -> float:
        checksum = 0.0
        for _ in range(100_000):
            checksum += float(np.mean(stats_data_f32))
        return checksum

    millis, checksum = median_ms(stats_mean_f32, rounds=7)
    cases.append({"name": "asv_reduce_stats_mean_f32_200", "millis": millis, "checksum": checksum})

    def stats_std_f32() -> float:
        checksum = 0.0
        for _ in range(100_000):
            checksum += float(np.std(stats_data_f32))
        return checksum

    millis, checksum = median_ms(stats_std_f32, rounds=7)
    cases.append({"name": "asv_reduce_stats_std_f32_200", "millis": millis, "checksum": checksum})

    def stats_prod_f32() -> float:
        checksum = 0.0
        for _ in range(100_000):
            checksum += float(np.prod(stats_data_f32))
        return checksum

    millis, checksum = median_ms(stats_prod_f32, rounds=7)
    cases.append({"name": "asv_reduce_stats_prod_f32_200", "millis": millis, "checksum": checksum})

    def stats_var_f32() -> float:
        checksum = 0.0
        for _ in range(100_000):
            checksum += float(np.var(stats_data_f32))
        return checksum

    millis, checksum = median_ms(stats_var_f32, rounds=7)
    cases.append({"name": "asv_reduce_stats_var_f32_200", "millis": millis, "checksum": checksum})

    stats_data_i64 = np.ones(200, dtype=np.int64)

    def stats_min_i64() -> float:
        checksum = 0.0
        for _ in range(100_000):
            checksum += float(np.min(stats_data_i64))
        return checksum

    millis, checksum = median_ms(stats_min_i64, rounds=7)
    cases.append({"name": "asv_reduce_stats_min_i64_200", "millis": millis, "checksum": checksum})

    def stats_max_i64() -> float:
        checksum = 0.0
        for _ in range(100_000):
            checksum += float(np.max(stats_data_i64))
        return checksum

    millis, checksum = median_ms(stats_max_i64, rounds=7)
    cases.append({"name": "asv_reduce_stats_max_i64_200", "millis": millis, "checksum": checksum})

    def stats_mean_i64() -> float:
        checksum = 0.0
        for _ in range(100_000):
            checksum += float(np.mean(stats_data_i64))
        return checksum

    millis, checksum = median_ms(stats_mean_i64, rounds=7)
    cases.append({"name": "asv_reduce_stats_mean_i64_200", "millis": millis, "checksum": checksum})

    def stats_std_i64() -> float:
        checksum = 0.0
        for _ in range(100_000):
            checksum += float(np.std(stats_data_i64))
        return checksum

    millis, checksum = median_ms(stats_std_i64, rounds=7)
    cases.append({"name": "asv_reduce_stats_std_i64_200", "millis": millis, "checksum": checksum})

    def stats_prod_i64() -> float:
        checksum = 0.0
        for _ in range(100_000):
            checksum += float(np.prod(stats_data_i64))
        return checksum

    millis, checksum = median_ms(stats_prod_i64, rounds=7)
    cases.append({"name": "asv_reduce_stats_prod_i64_200", "millis": millis, "checksum": checksum})

    def stats_var_i64() -> float:
        checksum = 0.0
        for _ in range(100_000):
            checksum += float(np.var(stats_data_i64))
        return checksum

    millis, checksum = median_ms(stats_var_i64, rounds=7)
    cases.append({"name": "asv_reduce_stats_var_i64_200", "millis": millis, "checksum": checksum})

    stats_data_u64 = np.ones(200, dtype=np.uint64)

    def stats_min_u64() -> float:
        checksum = 0.0
        for _ in range(100_000):
            checksum += float(np.min(stats_data_u64))
        return checksum

    millis, checksum = median_ms(stats_min_u64, rounds=7)
    cases.append({"name": "asv_reduce_stats_min_u64_200", "millis": millis, "checksum": checksum})

    def stats_max_u64() -> float:
        checksum = 0.0
        for _ in range(100_000):
            checksum += float(np.max(stats_data_u64))
        return checksum

    millis, checksum = median_ms(stats_max_u64, rounds=7)
    cases.append({"name": "asv_reduce_stats_max_u64_200", "millis": millis, "checksum": checksum})

    def stats_mean_u64() -> float:
        checksum = 0.0
        for _ in range(100_000):
            checksum += float(np.mean(stats_data_u64))
        return checksum

    millis, checksum = median_ms(stats_mean_u64, rounds=7)
    cases.append({"name": "asv_reduce_stats_mean_u64_200", "millis": millis, "checksum": checksum})

    def stats_std_u64() -> float:
        checksum = 0.0
        for _ in range(100_000):
            checksum += float(np.std(stats_data_u64))
        return checksum

    millis, checksum = median_ms(stats_std_u64, rounds=7)
    cases.append({"name": "asv_reduce_stats_std_u64_200", "millis": millis, "checksum": checksum})

    def stats_prod_u64() -> float:
        checksum = 0.0
        for _ in range(100_000):
            checksum += float(np.prod(stats_data_u64))
        return checksum

    millis, checksum = median_ms(stats_prod_u64, rounds=7)
    cases.append({"name": "asv_reduce_stats_prod_u64_200", "millis": millis, "checksum": checksum})

    def stats_var_u64() -> float:
        checksum = 0.0
        for _ in range(100_000):
            checksum += float(np.var(stats_data_u64))
        return checksum

    millis, checksum = median_ms(stats_var_u64, rounds=7)
    cases.append({"name": "asv_reduce_stats_var_u64_200", "millis": millis, "checksum": checksum})

    stats_data_bool = np.ones(200, dtype=np.bool_)

    def stats_min_bool() -> float:
        checksum = 0.0
        for _ in range(100_000):
            checksum += float(np.min(stats_data_bool))
        return checksum

    millis, checksum = median_ms(stats_min_bool, rounds=7)
    cases.append({"name": "asv_reduce_stats_min_bool_200", "millis": millis, "checksum": checksum})

    def stats_max_bool() -> float:
        checksum = 0.0
        for _ in range(100_000):
            checksum += float(np.max(stats_data_bool))
        return checksum

    millis, checksum = median_ms(stats_max_bool, rounds=7)
    cases.append({"name": "asv_reduce_stats_max_bool_200", "millis": millis, "checksum": checksum})

    def stats_mean_bool() -> float:
        checksum = 0.0
        for _ in range(100_000):
            checksum += float(np.mean(stats_data_bool))
        return checksum

    millis, checksum = median_ms(stats_mean_bool, rounds=7)
    cases.append({"name": "asv_reduce_stats_mean_bool_200", "millis": millis, "checksum": checksum})

    def stats_std_bool() -> float:
        checksum = 0.0
        for _ in range(100_000):
            checksum += float(np.std(stats_data_bool))
        return checksum

    millis, checksum = median_ms(stats_std_bool, rounds=7)
    cases.append({"name": "asv_reduce_stats_std_bool_200", "millis": millis, "checksum": checksum})

    def stats_prod_bool() -> float:
        checksum = 0.0
        for _ in range(100_000):
            checksum += float(np.prod(stats_data_bool))
        return checksum

    millis, checksum = median_ms(stats_prod_bool, rounds=7)
    cases.append({"name": "asv_reduce_stats_prod_bool_200", "millis": millis, "checksum": checksum})

    def stats_var_bool() -> float:
        checksum = 0.0
        for _ in range(100_000):
            checksum += float(np.var(stats_data_bool))
        return checksum

    millis, checksum = median_ms(stats_var_bool, rounds=7)
    cases.append({"name": "asv_reduce_stats_var_bool_200", "millis": millis, "checksum": checksum})

    stats_data_c64 = np.ones(200, dtype=np.complex64)
    stats_data_c64 = stats_data_c64 * stats_data_c64.T * 1j

    def stats_min_c64() -> float:
        checksum = 0.0
        for _ in range(100_000):
            checksum += complex_scalar_checksum(np.min(stats_data_c64))
        return checksum

    millis, checksum = median_ms(stats_min_c64, rounds=7)
    cases.append({"name": "asv_reduce_stats_min_c64_200", "millis": millis, "checksum": checksum})

    def stats_max_c64() -> float:
        checksum = 0.0
        for _ in range(100_000):
            checksum += complex_scalar_checksum(np.max(stats_data_c64))
        return checksum

    millis, checksum = median_ms(stats_max_c64, rounds=7)
    cases.append({"name": "asv_reduce_stats_max_c64_200", "millis": millis, "checksum": checksum})

    def stats_mean_c64() -> float:
        checksum = 0.0
        for _ in range(100_000):
            checksum += complex_scalar_checksum(np.mean(stats_data_c64))
        return checksum

    millis, checksum = median_ms(stats_mean_c64, rounds=7)
    cases.append({"name": "asv_reduce_stats_mean_c64_200", "millis": millis, "checksum": checksum})

    def stats_std_c64() -> float:
        checksum = 0.0
        for _ in range(100_000):
            checksum += float(np.std(stats_data_c64))
        return checksum

    millis, checksum = median_ms(stats_std_c64, rounds=7)
    cases.append({"name": "asv_reduce_stats_std_c64_200", "millis": millis, "checksum": checksum})

    def stats_prod_c64() -> float:
        checksum = 0.0
        for _ in range(100_000):
            checksum += complex_scalar_checksum(np.prod(stats_data_c64))
        return checksum

    millis, checksum = median_ms(stats_prod_c64, rounds=7)
    cases.append({"name": "asv_reduce_stats_prod_c64_200", "millis": millis, "checksum": checksum})

    def stats_var_c64() -> float:
        checksum = 0.0
        for _ in range(100_000):
            checksum += float(np.var(stats_data_c64))
        return checksum

    millis, checksum = median_ms(stats_var_c64, rounds=7)
    cases.append({"name": "asv_reduce_stats_var_c64_200", "millis": millis, "checksum": checksum})

    argmax_data = np.zeros(200_000, dtype=np.int64)

    def argmax_i64() -> float:
        checksum = 0.0
        for _ in range(20_000):
            checksum += float(np.argmax(argmax_data))
        return checksum

    millis, checksum = median_ms(argmax_i64, rounds=7)
    cases.append({"name": "asv_reduce_argmax_i64_200000", "millis": millis, "checksum": checksum})

    argmin_data = np.ones(200_000, dtype=np.int64)

    def argmin_i64() -> float:
        checksum = 0.0
        for _ in range(20_000):
            checksum += float(np.argmin(argmin_data))
        return checksum

    millis, checksum = median_ms(argmin_i64, rounds=7)
    cases.append({"name": "asv_reduce_argmin_i64_200000", "millis": millis, "checksum": checksum})

    take_arr = np.ones((1000, 1), dtype=np.int64)
    take_indices = np.arange(1000)

    def take_contiguous() -> float:
        checksum = 0.0
        for _ in range(2_000):
            checksum += edge_checksum(take_arr.take(take_indices, axis=-2, mode="raise"))
        return checksum

    millis, checksum = median_ms(take_contiguous, rounds=7)
    cases.append({"name": "asv_itemselection_take_i64_1000x1", "millis": millis, "checksum": checksum})

    putmask_vals = np.array(1.0, dtype=np.float64)
    dense_mask = np.ones(1000, dtype=bool)
    sparse_mask = np.zeros(1000, dtype=bool)

    def putmask_dense() -> float:
        arr = np.ones(1000, dtype=np.float64)
        checksum = 0.0
        for _ in range(10_000):
            np.putmask(arr, dense_mask, putmask_vals)
            checksum += edge_checksum(arr)
        return checksum

    millis, checksum = median_ms(putmask_dense, rounds=7)
    cases.append(
        {
            "name": "asv_itemselection_putmask_dense_scalar_f64_1000",
            "millis": millis,
            "checksum": checksum,
        }
    )

    def putmask_sparse() -> float:
        arr = np.ones(1000, dtype=np.float64)
        checksum = 0.0
        for _ in range(10_000):
            np.putmask(arr, sparse_mask, putmask_vals)
            checksum += edge_checksum(arr)
        return checksum

    millis, checksum = median_ms(putmask_sparse, rounds=7)
    cases.append(
        {
            "name": "asv_itemselection_putmask_sparse_scalar_f64_1000",
            "millis": millis,
            "checksum": checksum,
        }
    )

    put_indices = np.arange(1000, dtype=np.intp)
    put_values = np.ones(1000, dtype=np.float64)

    def put_ordered() -> float:
        arr = np.ones(1000, dtype=np.float64)
        checksum = 0.0
        for _ in range(10_000):
            np.put(arr, put_indices, put_values)
            checksum += edge_checksum(arr)
        return checksum

    millis, checksum = median_ms(put_ordered, rounds=7)
    cases.append(
        {
            "name": "asv_itemselection_put_ordered_f64_1000",
            "millis": millis,
            "checksum": checksum,
        }
    )

    def append_broadcast_arrays(dtype_name: str, rows: int, cols: int, repetitions: int) -> None:
        dtype = np.dtype(dtype_name)
        source = np.arange(rows * cols, dtype=dtype).reshape(rows, cols)
        scalar_one = np.ones(1, dtype=dtype)

        def broadcast_arrays() -> float:
            checksum = 0.0
            for _ in range(repetitions):
                left, right = np.broadcast_arrays(source, scalar_one)
                checksum += paired_broadcast_checksum(left, right)
            return checksum

        millis, checksum = median_ms(broadcast_arrays, rounds=7)
        cases.append(
            {
                "name": f"asv_manipulate_broadcast_arrays_{dtype_name.replace('float', 'f').replace('int', 'i')}_{rows}x{cols}",
                "millis": millis,
                "checksum": checksum,
            }
        )

    append_broadcast_arrays("float64", 16, 32, 200_000)
    append_broadcast_arrays("float64", 128, 256, 100_000)
    append_broadcast_arrays("float32", 128, 256, 100_000)
    append_broadcast_arrays("int32", 128, 256, 100_000)
    append_broadcast_arrays("float64", 512, 1024, 100_000)

    def append_broadcast_to(dtype_name: str, size: int) -> None:
        dtype = np.dtype(dtype_name)
        source = np.arange(size, dtype=dtype)

        def broadcast_to() -> float:
            checksum = 0.0
            for _ in range(200_000):
                checksum += shape_checksum(np.broadcast_to(source, (size, size)))
            return checksum

        millis, checksum = median_ms(broadcast_to, rounds=7)
        cases.append(
            {
                "name": f"asv_manipulate_broadcast_to_{dtype_name.replace('float', 'f').replace('int', 'i')}_{size}",
                "millis": millis,
                "checksum": checksum,
            }
        )

    append_broadcast_to("float64", 16)
    append_broadcast_to("float64", 64)
    append_broadcast_to("float32", 64)
    append_broadcast_to("int32", 64)
    append_broadcast_to("float64", 512)

    concat_arrays = [
        (np.arange(32 * 64, dtype=np.float64) + idx * 32 * 64).reshape(32, 64)
        for idx in range(5)
    ]

    def concatenate_ax0() -> float:
        checksum = 0.0
        for _ in range(2_000):
            checksum += edge_checksum(np.concatenate(concat_arrays, axis=0))
        return checksum

    millis, checksum = median_ms(concatenate_ax0, rounds=7)
    cases.append(
        {
            "name": "asv_manipulate_concatenate_ax0_f64_32x64_n5",
            "millis": millis,
            "checksum": checksum,
        }
    )

    def concatenate_ax1() -> float:
        checksum = 0.0
        for _ in range(2_000):
            checksum += edge_checksum(np.concatenate(concat_arrays, axis=1))
        return checksum

    millis, checksum = median_ms(concatenate_ax1, rounds=7)
    cases.append(
        {
            "name": "asv_manipulate_concatenate_ax1_f64_32x64_n5",
            "millis": millis,
            "checksum": checksum,
        }
    )

    def stack_ax0() -> float:
        checksum = 0.0
        for _ in range(2_000):
            checksum += edge_checksum(np.stack(concat_arrays, axis=0))
        return checksum

    millis, checksum = median_ms(stack_ax0, rounds=7)
    cases.append(
        {
            "name": "asv_manipulate_stack_ax0_f64_32x64_n5",
            "millis": millis,
            "checksum": checksum,
        }
    )

    def stack_ax1() -> float:
        checksum = 0.0
        for _ in range(2_000):
            checksum += edge_checksum(np.stack(concat_arrays, axis=1))
        return checksum

    millis, checksum = median_ms(stack_ax1, rounds=7)
    cases.append(
        {
            "name": "asv_manipulate_stack_ax1_f64_32x64_n5",
            "millis": millis,
            "checksum": checksum,
        }
    )

    def append_concat_stack_dtype(dtype_name: str) -> None:
        dtype = np.dtype(dtype_name)
        arrays = [
            (np.arange(32 * 64, dtype=dtype) + idx * 32 * 64).reshape(32, 64)
            for idx in range(5)
        ]
        suffix = dtype_name.replace("float", "f").replace("int", "i")

        def concatenate_ax0_dtype() -> float:
            checksum = 0.0
            for _ in range(2_000):
                checksum += edge_checksum(np.concatenate(arrays, axis=0))
            return checksum

        millis, checksum = median_ms(concatenate_ax0_dtype, rounds=7)
        cases.append(
            {
                "name": f"asv_manipulate_concatenate_ax0_{suffix}_32x64_n5",
                "millis": millis,
                "checksum": checksum,
            }
        )

        def concatenate_ax1_dtype() -> float:
            checksum = 0.0
            for _ in range(2_000):
                checksum += edge_checksum(np.concatenate(arrays, axis=1))
            return checksum

        millis, checksum = median_ms(concatenate_ax1_dtype, rounds=7)
        cases.append(
            {
                "name": f"asv_manipulate_concatenate_ax1_{suffix}_32x64_n5",
                "millis": millis,
                "checksum": checksum,
            }
        )

        def stack_ax0_dtype() -> float:
            checksum = 0.0
            for _ in range(2_000):
                checksum += edge_checksum(np.stack(arrays, axis=0))
            return checksum

        millis, checksum = median_ms(stack_ax0_dtype, rounds=7)
        cases.append(
            {
                "name": f"asv_manipulate_stack_ax0_{suffix}_32x64_n5",
                "millis": millis,
                "checksum": checksum,
            }
        )

        def stack_ax1_dtype() -> float:
            checksum = 0.0
            for _ in range(2_000):
                checksum += edge_checksum(np.stack(arrays, axis=1))
            return checksum

        millis, checksum = median_ms(stack_ax1_dtype, rounds=7)
        cases.append(
            {
                "name": f"asv_manipulate_stack_ax1_{suffix}_32x64_n5",
                "millis": millis,
                "checksum": checksum,
            }
        )

    append_concat_stack_dtype("float32")
    append_concat_stack_dtype("int32")

    dims_source = np.ones((5, 2, 3, 1), dtype=np.float64)

    def expand_dims_axis1() -> float:
        checksum = 0.0
        for _ in range(200_000):
            checksum += shape_checksum(np.expand_dims(dims_source, axis=1))
        return checksum

    millis, checksum = median_ms(expand_dims_axis1, rounds=7)
    cases.append(
        {
            "name": "asv_manipulate_expand_dims_f64_5x2x3x1_axis1",
            "millis": millis,
            "checksum": checksum,
        }
    )

    def expand_dims_neg() -> float:
        checksum = 0.0
        for _ in range(200_000):
            checksum += shape_checksum(np.expand_dims(dims_source, axis=-1))
        return checksum

    millis, checksum = median_ms(expand_dims_neg, rounds=7)
    cases.append(
        {
            "name": "asv_manipulate_expand_dims_neg_f64_5x2x3x1",
            "millis": millis,
            "checksum": checksum,
        }
    )

    def squeeze_dims() -> float:
        checksum = 0.0
        for _ in range(200_000):
            checksum += shape_checksum(np.squeeze(dims_source))
        return checksum

    millis, checksum = median_ms(squeeze_dims, rounds=7)
    cases.append(
        {
            "name": "asv_manipulate_squeeze_dims_f64_5x2x3x1",
            "millis": millis,
            "checksum": checksum,
        }
    )

    dims_values = np.arange(5 * 2 * 3 * 1, dtype=np.float64).reshape(5, 2, 3, 1)

    def flip_all() -> float:
        checksum = 0.0
        for _ in range(200_000):
            checksum += sample_checksum(np.flip(dims_values, axis=None))
        return checksum

    millis, checksum = median_ms(flip_all, rounds=7)
    cases.append(
        {
            "name": "asv_manipulate_flip_all_f64_5x2x3x1",
            "millis": millis,
            "checksum": checksum,
        }
    )

    def flip_one() -> float:
        checksum = 0.0
        for _ in range(200_000):
            checksum += sample_checksum(np.flip(dims_values, axis=1))
        return checksum

    millis, checksum = median_ms(flip_one, rounds=7)
    cases.append(
        {
            "name": "asv_manipulate_flip_one_f64_5x2x3x1_axis1",
            "millis": millis,
            "checksum": checksum,
        }
    )

    def flip_neg() -> float:
        checksum = 0.0
        for _ in range(200_000):
            checksum += sample_checksum(np.flip(dims_values, axis=-1))
        return checksum

    millis, checksum = median_ms(flip_neg, rounds=7)
    cases.append(
        {
            "name": "asv_manipulate_flip_neg_f64_5x2x3x1_axis_neg1",
            "millis": millis,
            "checksum": checksum,
        }
    )

    def moveaxis() -> float:
        checksum = 0.0
        for _ in range(200_000):
            checksum += sample_checksum(np.moveaxis(dims_values, [0, 1], [-1, -2]))
        return checksum

    millis, checksum = median_ms(moveaxis, rounds=7)
    cases.append(
        {
            "name": "asv_manipulate_moveaxis_f64_5x2x3x1",
            "millis": millis,
            "checksum": checksum,
        }
    )

    def roll() -> float:
        checksum = 0.0
        for _ in range(100_000):
            checksum += sample_checksum(np.roll(dims_values, 3))
        return checksum

    millis, checksum = median_ms(roll, rounds=7)
    cases.append(
        {
            "name": "asv_manipulate_roll_f64_5x2x3x1_shift3",
            "millis": millis,
            "checksum": checksum,
        }
    )

    def reshape_dims() -> float:
        checksum = 0.0
        for _ in range(200_000):
            checksum += shape_checksum(np.reshape(dims_source, (1, 5, 2, 3)))
        return checksum

    millis, checksum = median_ms(reshape_dims, rounds=7)
    cases.append(
        {
            "name": "asv_manipulate_reshape_f64_5x2x3x1_to_1x5x2x3",
            "millis": millis,
            "checksum": checksum,
        }
    )

    a = np.arange(60_000.0).reshape(150, 400)
    b = np.arange(240_000.0).reshape(400, 600)

    def dot_a_b() -> float:
        checksum = 0.0
        for _ in range(1000):
            checksum += edge_checksum(np.dot(a, b))
        return checksum

    millis, checksum = median_ms(dot_a_b, rounds=7)
    cases.append(
        {
            "name": "asv_linalg_dot_a_b_f64_150x400_400x600",
            "millis": millis,
            "checksum": checksum,
        }
    )

    def matmul_a_b() -> float:
        checksum = 0.0
        for _ in range(1000):
            checksum += edge_checksum(np.matmul(a, b))
        return checksum

    millis, checksum = median_ms(matmul_a_b, rounds=7)
    cases.append(
        {
            "name": "asv_linalg_matmul_a_b_f64_150x400_400x600",
            "millis": millis,
            "checksum": checksum,
        }
    )

    c = np.arange(600.0)
    d = np.arange(400.0)

    def matmul_d_matmul_b_c() -> float:
        checksum = 0.0
        for _ in range(1000):
            checksum += edge_checksum(np.matmul(d, np.matmul(b, c)))
        return checksum

    millis, checksum = median_ms(matmul_d_matmul_b_c, rounds=7)
    cases.append(
        {
            "name": "asv_linalg_matmul_d_matmul_b_c_f64",
            "millis": millis,
            "checksum": checksum,
        }
    )

    def dot_d_dot_b_c() -> float:
        checksum = 0.0
        for _ in range(1000):
            checksum += edge_checksum(np.dot(d, np.dot(b, c)))
        return checksum

    millis, checksum = median_ms(dot_d_dot_b_c, rounds=7)
    cases.append(
        {
            "name": "asv_linalg_dot_d_dot_b_c_f64",
            "millis": millis,
            "checksum": checksum,
        }
    )

    at = a.T
    atc = a.T.copy()

    def dot_trans_a_at() -> float:
        checksum = 0.0
        for _ in range(1000):
            checksum += edge_checksum(np.dot(a, at))
        return checksum

    millis, checksum = median_ms(dot_trans_a_at, rounds=7)
    cases.append(
        {
            "name": "asv_linalg_dot_trans_a_at_f64_150x400_400x150",
            "millis": millis,
            "checksum": checksum,
        }
    )

    def dot_trans_a_atc() -> float:
        checksum = 0.0
        for _ in range(1000):
            checksum += edge_checksum(np.dot(a, atc))
        return checksum

    millis, checksum = median_ms(dot_trans_a_atc, rounds=7)
    cases.append(
        {
            "name": "asv_linalg_dot_trans_a_atc_f64_150x400_400x150",
            "millis": millis,
            "checksum": checksum,
        }
    )

    def dot_trans_at_a() -> float:
        checksum = 0.0
        for _ in range(1000):
            checksum += edge_checksum(np.dot(at, a))
        return checksum

    millis, checksum = median_ms(dot_trans_at_a, rounds=7)
    cases.append(
        {
            "name": "asv_linalg_dot_trans_at_a_f64_400x150_150x400",
            "millis": millis,
            "checksum": checksum,
        }
    )

    def dot_trans_atc_a() -> float:
        checksum = 0.0
        for _ in range(1000):
            checksum += edge_checksum(np.dot(atc, a))
        return checksum

    millis, checksum = median_ms(dot_trans_atc_a, rounds=7)
    cases.append(
        {
            "name": "asv_linalg_dot_trans_atc_a_f64_400x150_150x400",
            "millis": millis,
            "checksum": checksum,
        }
    )

    ac = a.copy()

    def inner_a_a() -> float:
        checksum = 0.0
        for _ in range(1000):
            checksum += edge_checksum(np.inner(a, a))
        return checksum

    millis, checksum = median_ms(inner_a_a, rounds=7)
    cases.append(
        {
            "name": "asv_linalg_inner_a_a_f64_150x400_150x400",
            "millis": millis,
            "checksum": checksum,
        }
    )

    def inner_a_ac() -> float:
        checksum = 0.0
        for _ in range(1000):
            checksum += edge_checksum(np.inner(a, ac))
        return checksum

    millis, checksum = median_ms(inner_a_ac, rounds=7)
    cases.append(
        {
            "name": "asv_linalg_inner_a_ac_f64_150x400_150x400",
            "millis": millis,
            "checksum": checksum,
        }
    )

    def matmul_trans_a_at() -> float:
        checksum = 0.0
        for _ in range(1000):
            checksum += edge_checksum(np.matmul(a, at))
        return checksum

    millis, checksum = median_ms(matmul_trans_a_at, rounds=7)
    cases.append(
        {
            "name": "asv_linalg_matmul_trans_a_at_f64_150x400_400x150",
            "millis": millis,
            "checksum": checksum,
        }
    )

    def matmul_trans_a_atc() -> float:
        checksum = 0.0
        for _ in range(1000):
            checksum += edge_checksum(np.matmul(a, atc))
        return checksum

    millis, checksum = median_ms(matmul_trans_a_atc, rounds=7)
    cases.append(
        {
            "name": "asv_linalg_matmul_trans_a_atc_f64_150x400_400x150",
            "millis": millis,
            "checksum": checksum,
        }
    )

    def matmul_trans_at_a() -> float:
        checksum = 0.0
        for _ in range(1000):
            checksum += edge_checksum(np.matmul(at, a))
        return checksum

    millis, checksum = median_ms(matmul_trans_at_a, rounds=7)
    cases.append(
        {
            "name": "asv_linalg_matmul_trans_at_a_f64_400x150_150x400",
            "millis": millis,
            "checksum": checksum,
        }
    )

    def matmul_trans_atc_a() -> float:
        checksum = 0.0
        for _ in range(1000):
            checksum += edge_checksum(np.matmul(atc, a))
        return checksum

    millis, checksum = median_ms(matmul_trans_atc_a, rounds=7)
    cases.append(
        {
            "name": "asv_linalg_matmul_trans_atc_a_f64_400x150_150x400",
            "millis": millis,
            "checksum": checksum,
        }
    )

    a3 = np.arange(480_000.0).reshape(60, 80, 100)
    b3 = np.arange(192_000.0).reshape(80, 60, 40)

    def tensordot_a3_b3() -> float:
        checksum = 0.0
        for _ in range(10):
            checksum += edge_checksum(np.tensordot(a3, b3, axes=([1, 0], [0, 1])))
        return checksum

    millis, checksum = median_ms(tensordot_a3_b3, rounds=7)
    cases.append(
        {
            "name": "asv_linalg_tensordot_a3_b3_axes_10_01",
            "millis": millis,
            "checksum": checksum,
        }
    )

    def einsum_i_ij_j() -> float:
        checksum = 0.0
        for _ in range(1000):
            checksum += float(np.einsum("i,ij,j", d, b, c))
        return checksum

    millis, checksum = median_ms(einsum_i_ij_j, rounds=7)
    cases.append(
        {
            "name": "asv_linalg_einsum_i_ij_j_f64_400_400x600_600",
            "millis": millis,
            "checksum": checksum,
        }
    )

    def einsum_ij_jk() -> float:
        checksum = 0.0
        for _ in range(1000):
            checksum += edge_checksum(np.einsum("ij,jk", a, b))
        return checksum

    millis, checksum = median_ms(einsum_ij_jk, rounds=7)
    cases.append(
        {
            "name": "asv_linalg_einsum_ij_jk_f64_150x400_400x600",
            "millis": millis,
            "checksum": checksum,
        }
    )

    array_5 = np.arange(5.0)

    def norm_small() -> float:
        checksum = 0.0
        for _ in range(100_000):
            checksum += float(np.linalg.norm(array_5))
        return checksum

    millis, checksum = median_ms(norm_small, rounds=7)
    cases.append(
        {"name": "asv_linalg_norm_small_array_f64_5", "millis": millis, "checksum": checksum}
    )

    array_5_5 = np.arange(25.0).reshape(5, 5)

    def det_small() -> float:
        checksum = 0.0
        for _ in range(100_000):
            checksum += float(np.linalg.det(array_5_5))
        return checksum

    millis, checksum = median_ms(det_small, rounds=7)
    cases.append(
        {"name": "asv_linalg_det_small_array_f64_5x5", "millis": millis, "checksum": checksum}
    )

    array_3_3 = np.eye(3) + np.arange(9.0).reshape(3, 3)

    def det_3x3() -> float:
        checksum = 0.0
        for _ in range(100_000):
            checksum += float(np.linalg.det(array_3_3))
        return checksum

    millis, checksum = median_ms(det_3x3, rounds=7)
    cases.append({"name": "asv_linalg_det_3x3_f64", "millis": millis, "checksum": checksum})

    array_3 = np.arange(3.0)

    def solve_3x3() -> float:
        checksum = 0.0
        for _ in range(100_000):
            checksum += edge_checksum(np.linalg.solve(array_3_3, array_3))
        return checksum

    millis, checksum = median_ms(solve_3x3, rounds=7)
    cases.append({"name": "asv_linalg_solve_3x3_f64", "millis": millis, "checksum": checksum})

    lstsq_a, lstsq_b = lstsq_fixture()

    def lstsq_square() -> float:
        checksum = 0.0
        for _ in range(100):
            result = np.linalg.lstsq(lstsq_a, lstsq_b, rcond=-1)[0]
            checksum += edge_checksum(result)
        return checksum

    millis, checksum = median_ms(lstsq_square, rounds=7)
    cases.append(
        {
            "name": "asv_linalg_lstsq_square_f64_100x100",
            "millis": millis,
            "checksum": checksum,
        }
    )

    one_dim = np.arange(3000, dtype=np.float64)
    millis, checksum = median_ms(lambda: edge_checksum(np.einsum("i,j", one_dim, one_dim)), rounds=7)
    cases.append(
        {"name": "asv_linalg_einsum_outer_f64_3000", "millis": millis, "checksum": checksum}
    )

    two_dim_small = np.arange(1200, dtype=np.float64).reshape(30, 40)
    three_dim = np.arange(24_000, dtype=np.float64).reshape(20, 30, 40)
    millis, checksum = median_ms(
        lambda: edge_checksum(np.einsum("..., ...", two_dim_small, three_dim, optimize=True)),
        rounds=7,
    )
    cases.append(
        {
            "name": "asv_linalg_einsum_multiply_f64_30x40_20x30x40",
            "millis": millis,
            "checksum": checksum,
        }
    )

    three_dim_small = np.arange(10_000, dtype=np.float64).reshape(10, 100, 10)

    def einsum_sum_mul() -> float:
        checksum = 0.0
        for _ in range(100):
            checksum += float(np.einsum(",i...->", 300, three_dim_small, optimize=True))
        return checksum

    millis, checksum = median_ms(einsum_sum_mul, rounds=7)
    cases.append(
        {
            "name": "asv_linalg_einsum_sum_mul_f64_scalar_10x100x10",
            "millis": millis,
            "checksum": checksum,
        }
    )

    def einsum_sum_mul2() -> float:
        checksum = 0.0
        for _ in range(100):
            checksum += float(np.einsum("i...,->", three_dim_small, 300, optimize=True))
        return checksum

    millis, checksum = median_ms(einsum_sum_mul2, rounds=7)
    cases.append(
        {
            "name": "asv_linalg_einsum_sum_mul2_f64_10x100x10_scalar",
            "millis": millis,
            "checksum": checksum,
        }
    )

    one_dim_big = np.arange(480_000, dtype=np.float64)

    def einsum_scalar_mul() -> float:
        checksum = 0.0
        for _ in range(100):
            checksum += edge_checksum(np.einsum("i,->i", one_dim_big, 300, optimize=True))
        return checksum

    millis, checksum = median_ms(einsum_scalar_mul, rounds=7)
    cases.append(
        {
            "name": "asv_linalg_einsum_scalar_mul_f64_480000",
            "millis": millis,
            "checksum": checksum,
        }
    )

    def einsum_sum() -> float:
        checksum = 0.0
        for _ in range(100):
            checksum += float(np.einsum("i->", one_dim_big, optimize=True))
        return checksum

    millis, checksum = median_ms(einsum_sum, rounds=7)
    cases.append(
        {"name": "asv_linalg_einsum_sum_f64_480000", "millis": millis, "checksum": checksum}
    )

    two_dim = np.arange(240_000, dtype=np.float64).reshape(400, 600)
    one_dim_small = np.arange(600, dtype=np.float64)

    def einsum_weighted_sum() -> float:
        checksum = 0.0
        for _ in range(100):
            checksum += float(np.einsum("ji,i->", two_dim, one_dim_small, optimize=True))
        return checksum

    millis, checksum = median_ms(einsum_weighted_sum, rounds=7)
    cases.append(
        {
            "name": "asv_linalg_einsum_weighted_sum_f64_400x600",
            "millis": millis,
            "checksum": checksum,
        }
    )

    noncon_dim1 = np.arange(1, 4000, 2, dtype=np.float64)
    millis, checksum = median_ms(
        lambda: edge_checksum(np.einsum("i,j", noncon_dim1, noncon_dim1, optimize=True)),
        rounds=7,
    )
    cases.append(
        {
            "name": "asv_linalg_einsum_noncon_outer_f64_2000",
            "millis": millis,
            "checksum": checksum,
        }
    )

    noncon_dim2 = np.arange(1, 2400, 2, dtype=np.float64).reshape(30, 40)
    noncon_dim3 = np.arange(1, 48_000, 2, dtype=np.float64).reshape(20, 30, 40)
    millis, checksum = median_ms(
        lambda: edge_checksum(np.einsum("..., ...", noncon_dim2, noncon_dim3, optimize=True)),
        rounds=7,
    )
    cases.append(
        {
            "name": "asv_linalg_einsum_noncon_multiply_f64_30x40_20x30x40",
            "millis": millis,
            "checksum": checksum,
        }
    )

    def einsum_noncon_sum_mul() -> float:
        checksum = 0.0
        for _ in range(100):
            checksum += float(np.einsum(",i...->", 300, noncon_dim3, optimize=True))
        return checksum

    millis, checksum = median_ms(einsum_noncon_sum_mul, rounds=7)
    cases.append(
        {
            "name": "asv_linalg_einsum_noncon_sum_mul_f64_scalar_20x30x40",
            "millis": millis,
            "checksum": checksum,
        }
    )

    def einsum_noncon_sum_mul2() -> float:
        checksum = 0.0
        for _ in range(100):
            checksum += float(np.einsum("i...,->", noncon_dim3, 300, optimize=True))
        return checksum

    millis, checksum = median_ms(einsum_noncon_sum_mul2, rounds=7)
    cases.append(
        {
            "name": "asv_linalg_einsum_noncon_sum_mul2_f64_20x30x40_scalar",
            "millis": millis,
            "checksum": checksum,
        }
    )

    def einsum_noncon_scalar_mul() -> float:
        checksum = 0.0
        for _ in range(100):
            checksum += edge_checksum(np.einsum("i,->i", noncon_dim1, 300, optimize=True))
        return checksum

    millis, checksum = median_ms(einsum_noncon_scalar_mul, rounds=7)
    cases.append(
        {
            "name": "asv_linalg_einsum_noncon_scalar_mul_f64_2000",
            "millis": millis,
            "checksum": checksum,
        }
    )

    noncon_dim1_small = np.arange(1, 80, 2, dtype=np.float64)

    def einsum_noncon_weighted_sum() -> float:
        checksum = 0.0
        for _ in range(100):
            checksum += float(np.einsum("ji,i->", noncon_dim2, noncon_dim1_small, optimize=True))
        return checksum

    millis, checksum = median_ms(einsum_noncon_weighted_sum, rounds=7)
    cases.append(
        {
            "name": "asv_linalg_einsum_noncon_weighted_sum_f64_30x40",
            "millis": millis,
            "checksum": checksum,
        }
    )

    def einsum_noncon_sum() -> float:
        checksum = 0.0
        for _ in range(100):
            checksum += float(np.einsum("i->", noncon_dim1, optimize=True))
        return checksum

    millis, checksum = median_ms(einsum_noncon_sum, rounds=7)
    cases.append(
        {
            "name": "asv_linalg_einsum_noncon_sum_f64_2000",
            "millis": millis,
            "checksum": checksum,
        }
    )

    return {
        "engine": "numpy",
        "numpy_version": np.__version__,
        "cases": cases,
    }


def bench_numpy_selected(case_names: list[str]) -> dict:
    requested = set(case_names)
    supported = {
        "asv_linalg_dot_a_b_f64_150x400_400x600",
        "asv_linalg_matmul_a_b_f64_150x400_400x600",
        "asv_linalg_dot_trans_a_atc_f64_150x400_400x150",
        "asv_linalg_dot_trans_atc_a_f64_400x150_150x400",
        "asv_linalg_inner_a_ac_f64_150x400_150x400",
        "asv_linalg_matmul_trans_a_at_f64_150x400_400x150",
        "asv_linalg_matmul_trans_a_atc_f64_150x400_400x150",
        "asv_linalg_matmul_trans_at_a_f64_400x150_150x400",
        "asv_linalg_matmul_trans_atc_a_f64_400x150_150x400",
        "asv_linalg_einsum_scalar_mul_f64_480000",
        "asv_reduce_stats_min_f32_200",
        "asv_reduce_stats_max_f32_200",
        "asv_reduce_stats_mean_f32_200",
        "asv_reduce_stats_std_f32_200",
        "asv_reduce_stats_prod_f32_200",
        "asv_reduce_stats_var_f32_200",
        "asv_reduce_stats_min_i64_200",
        "asv_reduce_stats_max_i64_200",
        "asv_reduce_stats_mean_i64_200",
        "asv_reduce_stats_std_i64_200",
        "asv_reduce_stats_prod_i64_200",
        "asv_reduce_stats_var_i64_200",
        "asv_reduce_stats_min_u64_200",
        "asv_reduce_stats_max_u64_200",
        "asv_reduce_stats_mean_u64_200",
        "asv_reduce_stats_std_u64_200",
        "asv_reduce_stats_prod_u64_200",
        "asv_reduce_stats_var_u64_200",
        "asv_reduce_stats_min_bool_200",
        "asv_reduce_stats_max_bool_200",
        "asv_reduce_stats_mean_bool_200",
        "asv_reduce_stats_std_bool_200",
        "asv_reduce_stats_prod_bool_200",
        "asv_reduce_stats_var_bool_200",
        "asv_reduce_stats_min_c64_200",
        "asv_reduce_stats_max_c64_200",
        "asv_reduce_stats_mean_c64_200",
        "asv_reduce_stats_std_c64_200",
        "asv_reduce_stats_prod_c64_200",
        "asv_reduce_stats_var_c64_200",
        "asv_manipulate_broadcast_arrays_f64_16x32",
        "asv_manipulate_broadcast_arrays_f64_128x256",
        "asv_manipulate_broadcast_arrays_f32_128x256",
        "asv_manipulate_broadcast_arrays_i32_128x256",
        "asv_manipulate_broadcast_arrays_f64_512x1024",
        "asv_manipulate_broadcast_to_f64_16",
        "asv_manipulate_broadcast_to_f64_64",
        "asv_manipulate_broadcast_to_f32_64",
        "asv_manipulate_broadcast_to_i32_64",
        "asv_manipulate_broadcast_to_f64_512",
        "asv_manipulate_concatenate_ax0_f64_32x64_n5",
        "asv_manipulate_concatenate_ax0_f32_32x64_n5",
        "asv_manipulate_concatenate_ax0_i32_32x64_n5",
        "asv_manipulate_concatenate_ax1_f64_32x64_n5",
        "asv_manipulate_concatenate_ax1_f32_32x64_n5",
        "asv_manipulate_concatenate_ax1_i32_32x64_n5",
        "asv_manipulate_stack_ax0_f64_32x64_n5",
        "asv_manipulate_stack_ax0_f32_32x64_n5",
        "asv_manipulate_stack_ax0_i32_32x64_n5",
        "asv_manipulate_stack_ax1_f64_32x64_n5",
        "asv_manipulate_stack_ax1_f32_32x64_n5",
        "asv_manipulate_stack_ax1_i32_32x64_n5",
        "asv_manipulate_flip_all_f64_5x2x3x1",
        "asv_manipulate_flip_one_f64_5x2x3x1_axis1",
        "asv_manipulate_flip_neg_f64_5x2x3x1_axis_neg1",
        "asv_manipulate_moveaxis_f64_5x2x3x1",
        "asv_manipulate_roll_f64_5x2x3x1_shift3",
    }
    unsupported = sorted(requested - supported)
    if unsupported:
        raise ValueError(f"focused NumPy runner does not support cases: {unsupported}")

    cases = []

    def append_case(name: str, fn: Callable[[], float]) -> None:
        if name in requested:
            millis, checksum = median_ms(fn, rounds=7)
            cases.append({"name": name, "millis": millis, "checksum": checksum})

    a = np.arange(60_000.0).reshape(150, 400)
    b = np.arange(240_000.0).reshape(400, 600)
    at = a.T
    atc = a.T.copy()
    ac = a.copy()
    one_dim_big = np.arange(480_000, dtype=np.float64)
    stats_data_f32 = np.ones(200, dtype=np.float32)
    stats_data_i64 = np.ones(200, dtype=np.int64)
    stats_data_u64 = np.ones(200, dtype=np.uint64)
    stats_data_bool = np.ones(200, dtype=np.bool_)
    stats_data_c64 = np.ones(200, dtype=np.complex64)
    stats_data_c64 = stats_data_c64 * stats_data_c64.T * 1j
    concat_arrays = [
        (np.arange(32 * 64, dtype=np.float64) + idx * 32 * 64).reshape(32, 64)
        for idx in range(5)
    ]
    dims_values = np.arange(5 * 2 * 3 * 1, dtype=np.float64).reshape(5, 2, 3, 1)

    def dot_a_b() -> float:
        checksum = 0.0
        for _ in range(1000):
            checksum += edge_checksum(np.dot(a, b))
        return checksum

    append_case("asv_linalg_dot_a_b_f64_150x400_400x600", dot_a_b)

    def matmul_a_b() -> float:
        checksum = 0.0
        for _ in range(1000):
            checksum += edge_checksum(np.matmul(a, b))
        return checksum

    append_case("asv_linalg_matmul_a_b_f64_150x400_400x600", matmul_a_b)

    def dot_trans_a_atc() -> float:
        checksum = 0.0
        for _ in range(1000):
            checksum += edge_checksum(np.dot(a, atc))
        return checksum

    append_case("asv_linalg_dot_trans_a_atc_f64_150x400_400x150", dot_trans_a_atc)

    def dot_trans_atc_a() -> float:
        checksum = 0.0
        for _ in range(1000):
            checksum += edge_checksum(np.dot(atc, a))
        return checksum

    append_case("asv_linalg_dot_trans_atc_a_f64_400x150_150x400", dot_trans_atc_a)

    def inner_a_ac() -> float:
        checksum = 0.0
        for _ in range(1000):
            checksum += edge_checksum(np.inner(a, ac))
        return checksum

    append_case("asv_linalg_inner_a_ac_f64_150x400_150x400", inner_a_ac)

    def matmul_trans_a_at() -> float:
        checksum = 0.0
        for _ in range(1000):
            checksum += edge_checksum(np.matmul(a, at))
        return checksum

    append_case("asv_linalg_matmul_trans_a_at_f64_150x400_400x150", matmul_trans_a_at)

    def matmul_trans_a_atc() -> float:
        checksum = 0.0
        for _ in range(1000):
            checksum += edge_checksum(np.matmul(a, atc))
        return checksum

    append_case("asv_linalg_matmul_trans_a_atc_f64_150x400_400x150", matmul_trans_a_atc)

    def matmul_trans_at_a() -> float:
        checksum = 0.0
        for _ in range(1000):
            checksum += edge_checksum(np.matmul(at, a))
        return checksum

    append_case("asv_linalg_matmul_trans_at_a_f64_400x150_150x400", matmul_trans_at_a)

    def matmul_trans_atc_a() -> float:
        checksum = 0.0
        for _ in range(1000):
            checksum += edge_checksum(np.matmul(atc, a))
        return checksum

    append_case("asv_linalg_matmul_trans_atc_a_f64_400x150_150x400", matmul_trans_atc_a)

    def einsum_scalar_mul() -> float:
        checksum = 0.0
        for _ in range(100):
            checksum += edge_checksum(np.einsum("i,->i", one_dim_big, 300, optimize=True))
        return checksum

    append_case("asv_linalg_einsum_scalar_mul_f64_480000", einsum_scalar_mul)

    def append_selected_stats_f32(op_name: str, op: Callable[[np.ndarray], np.generic]) -> None:
        def stats_op() -> float:
            checksum = 0.0
            for _ in range(100_000):
                checksum += float(op(stats_data_f32))
            return checksum

        append_case(f"asv_reduce_stats_{op_name}_f32_200", stats_op)

    append_selected_stats_f32("min", np.min)
    append_selected_stats_f32("max", np.max)
    append_selected_stats_f32("mean", np.mean)
    append_selected_stats_f32("std", np.std)
    append_selected_stats_f32("prod", np.prod)
    append_selected_stats_f32("var", np.var)

    def append_selected_stats_i64(op_name: str, op: Callable[[np.ndarray], np.generic]) -> None:
        def stats_op() -> float:
            checksum = 0.0
            for _ in range(100_000):
                checksum += float(op(stats_data_i64))
            return checksum

        append_case(f"asv_reduce_stats_{op_name}_i64_200", stats_op)

    append_selected_stats_i64("min", np.min)
    append_selected_stats_i64("max", np.max)
    append_selected_stats_i64("mean", np.mean)
    append_selected_stats_i64("std", np.std)
    append_selected_stats_i64("prod", np.prod)
    append_selected_stats_i64("var", np.var)

    def append_selected_stats_u64(op_name: str, op: Callable[[np.ndarray], np.generic]) -> None:
        def stats_op() -> float:
            checksum = 0.0
            for _ in range(100_000):
                checksum += float(op(stats_data_u64))
            return checksum

        append_case(f"asv_reduce_stats_{op_name}_u64_200", stats_op)

    append_selected_stats_u64("min", np.min)
    append_selected_stats_u64("max", np.max)
    append_selected_stats_u64("mean", np.mean)
    append_selected_stats_u64("std", np.std)
    append_selected_stats_u64("prod", np.prod)
    append_selected_stats_u64("var", np.var)

    def append_selected_stats_bool(op_name: str, op: Callable[[np.ndarray], np.generic]) -> None:
        def stats_op() -> float:
            checksum = 0.0
            for _ in range(100_000):
                checksum += float(op(stats_data_bool))
            return checksum

        append_case(f"asv_reduce_stats_{op_name}_bool_200", stats_op)

    append_selected_stats_bool("min", np.min)
    append_selected_stats_bool("max", np.max)
    append_selected_stats_bool("mean", np.mean)
    append_selected_stats_bool("std", np.std)
    append_selected_stats_bool("prod", np.prod)
    append_selected_stats_bool("var", np.var)

    def append_selected_stats_c64(op_name: str, op: Callable[[np.ndarray], np.generic]) -> None:
        def stats_op() -> float:
            checksum = 0.0
            for _ in range(100_000):
                checksum += complex_scalar_checksum(op(stats_data_c64))
            return checksum

        append_case(f"asv_reduce_stats_{op_name}_c64_200", stats_op)

    append_selected_stats_c64("min", np.min)
    append_selected_stats_c64("max", np.max)
    append_selected_stats_c64("mean", np.mean)

    def append_selected_real_stats_c64(op_name: str, op: Callable[[np.ndarray], np.generic]) -> None:
        def stats_op() -> float:
            checksum = 0.0
            for _ in range(100_000):
                checksum += float(op(stats_data_c64))
            return checksum

        append_case(f"asv_reduce_stats_{op_name}_c64_200", stats_op)

    append_selected_real_stats_c64("std", np.std)
    append_selected_stats_c64("prod", np.prod)
    append_selected_real_stats_c64("var", np.var)

    def append_selected_broadcast_arrays(dtype_name: str, rows: int, cols: int, repetitions: int) -> None:
        dtype = np.dtype(dtype_name)
        source = np.arange(rows * cols, dtype=dtype).reshape(rows, cols)
        scalar_one = np.ones(1, dtype=dtype)

        def broadcast_arrays() -> float:
            checksum = 0.0
            for _ in range(repetitions):
                left, right = np.broadcast_arrays(source, scalar_one)
                checksum += paired_broadcast_checksum(left, right)
            return checksum

        append_case(
            f"asv_manipulate_broadcast_arrays_{dtype_name.replace('float', 'f').replace('int', 'i')}_{rows}x{cols}",
            broadcast_arrays,
        )

    append_selected_broadcast_arrays("float64", 16, 32, 200_000)
    append_selected_broadcast_arrays("float64", 128, 256, 100_000)
    append_selected_broadcast_arrays("float32", 128, 256, 100_000)
    append_selected_broadcast_arrays("int32", 128, 256, 100_000)
    append_selected_broadcast_arrays("float64", 512, 1024, 100_000)

    def append_selected_broadcast_to(dtype_name: str, size: int) -> None:
        dtype = np.dtype(dtype_name)
        source = np.arange(size, dtype=dtype)

        def broadcast_to() -> float:
            checksum = 0.0
            for _ in range(200_000):
                checksum += shape_checksum(np.broadcast_to(source, (size, size)))
            return checksum

        append_case(
            f"asv_manipulate_broadcast_to_{dtype_name.replace('float', 'f').replace('int', 'i')}_{size}",
            broadcast_to,
        )

    append_selected_broadcast_to("float64", 16)
    append_selected_broadcast_to("float64", 64)
    append_selected_broadcast_to("float32", 64)
    append_selected_broadcast_to("int32", 64)
    append_selected_broadcast_to("float64", 512)

    def concatenate_ax0() -> float:
        checksum = 0.0
        for _ in range(2_000):
            checksum += edge_checksum(np.concatenate(concat_arrays, axis=0))
        return checksum

    append_case("asv_manipulate_concatenate_ax0_f64_32x64_n5", concatenate_ax0)

    def concatenate_ax1() -> float:
        checksum = 0.0
        for _ in range(2_000):
            checksum += edge_checksum(np.concatenate(concat_arrays, axis=1))
        return checksum

    append_case("asv_manipulate_concatenate_ax1_f64_32x64_n5", concatenate_ax1)

    def stack_ax0() -> float:
        checksum = 0.0
        for _ in range(2_000):
            checksum += edge_checksum(np.stack(concat_arrays, axis=0))
        return checksum

    append_case("asv_manipulate_stack_ax0_f64_32x64_n5", stack_ax0)

    def stack_ax1() -> float:
        checksum = 0.0
        for _ in range(2_000):
            checksum += edge_checksum(np.stack(concat_arrays, axis=1))
        return checksum

    append_case("asv_manipulate_stack_ax1_f64_32x64_n5", stack_ax1)

    def append_selected_concat_stack_dtype(dtype_name: str) -> None:
        dtype = np.dtype(dtype_name)
        arrays = [
            (np.arange(32 * 64, dtype=dtype) + idx * 32 * 64).reshape(32, 64)
            for idx in range(5)
        ]
        suffix = dtype_name.replace("float", "f").replace("int", "i")

        def concatenate_ax0_dtype() -> float:
            checksum = 0.0
            for _ in range(2_000):
                checksum += edge_checksum(np.concatenate(arrays, axis=0))
            return checksum

        append_case(
            f"asv_manipulate_concatenate_ax0_{suffix}_32x64_n5",
            concatenate_ax0_dtype,
        )

        def concatenate_ax1_dtype() -> float:
            checksum = 0.0
            for _ in range(2_000):
                checksum += edge_checksum(np.concatenate(arrays, axis=1))
            return checksum

        append_case(
            f"asv_manipulate_concatenate_ax1_{suffix}_32x64_n5",
            concatenate_ax1_dtype,
        )

        def stack_ax0_dtype() -> float:
            checksum = 0.0
            for _ in range(2_000):
                checksum += edge_checksum(np.stack(arrays, axis=0))
            return checksum

        append_case(f"asv_manipulate_stack_ax0_{suffix}_32x64_n5", stack_ax0_dtype)

        def stack_ax1_dtype() -> float:
            checksum = 0.0
            for _ in range(2_000):
                checksum += edge_checksum(np.stack(arrays, axis=1))
            return checksum

        append_case(f"asv_manipulate_stack_ax1_{suffix}_32x64_n5", stack_ax1_dtype)

    append_selected_concat_stack_dtype("float32")
    append_selected_concat_stack_dtype("int32")

    def flip_all() -> float:
        checksum = 0.0
        for _ in range(200_000):
            checksum += sample_checksum(np.flip(dims_values, axis=None))
        return checksum

    append_case("asv_manipulate_flip_all_f64_5x2x3x1", flip_all)

    def flip_one() -> float:
        checksum = 0.0
        for _ in range(200_000):
            checksum += sample_checksum(np.flip(dims_values, axis=1))
        return checksum

    append_case("asv_manipulate_flip_one_f64_5x2x3x1_axis1", flip_one)

    def flip_neg() -> float:
        checksum = 0.0
        for _ in range(200_000):
            checksum += sample_checksum(np.flip(dims_values, axis=-1))
        return checksum

    append_case("asv_manipulate_flip_neg_f64_5x2x3x1_axis_neg1", flip_neg)

    def moveaxis() -> float:
        checksum = 0.0
        for _ in range(200_000):
            checksum += sample_checksum(np.moveaxis(dims_values, [0, 1], [-1, -2]))
        return checksum

    append_case("asv_manipulate_moveaxis_f64_5x2x3x1", moveaxis)

    def roll() -> float:
        checksum = 0.0
        for _ in range(100_000):
            checksum += sample_checksum(np.roll(dims_values, 3))
        return checksum

    append_case("asv_manipulate_roll_f64_5x2x3x1_shift3", roll)

    cases_by_name = {case["name"]: case for case in cases}
    return {
        "engine": "numpy",
        "numpy_version": np.__version__,
        "cases": [cases_by_name[name] for name in case_names],
    }


def run_numrust() -> dict:
    env = os.environ.copy()
    env.setdefault("RUSTFLAGS", "-C target-cpu=native")
    proc = subprocess.run(
        ["cargo", "run", "--release", "-p", "numrs-core", "--example", "external_cases_json"],
        cwd=ROOT,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
    for line in reversed(proc.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    raise RuntimeError("NumRust external benchmark did not emit JSON")


def benchmark_passes() -> int:
    value = os.environ.get("NUMRUST_EXTERNAL_BENCH_PASSES", "5")
    try:
        passes = int(value)
    except ValueError as err:
        raise ValueError("NUMRUST_EXTERNAL_BENCH_PASSES must be an integer") from err
    if passes < 1:
        raise ValueError("NUMRUST_EXTERNAL_BENCH_PASSES must be at least 1")
    return passes


def run_benchmark_pass(pass_idx: int) -> dict:
    write_lstsq_fixture()
    if pass_idx % 2 == 0:
        numrust = run_numrust()
        numpy_result = bench_numpy()
    else:
        numpy_result = bench_numpy()
        numrust = run_numrust()
    return {"pass_idx": pass_idx, "numrust": numrust, "numpy": numpy_result}


def aggregate_engine_runs(runs: list[dict]) -> dict:
    if not runs:
        raise ValueError("expected at least one benchmark run")

    cases_by_run = [{case["name"]: case for case in run["cases"]} for run in runs]
    case_names = [case.name for case in RUNNABLE_CASES]
    cases = []
    for name in case_names:
        millis = [run_cases[name]["millis"] for run_cases in cases_by_run]
        checksums = [run_cases[name]["checksum"] for run_cases in cases_by_run]
        cases.append(
            {
                "name": name,
                "millis": statistics.median(millis),
                "checksum": checksums[-1],
                "pass_millis": millis,
            }
        )

    aggregated = {key: value for key, value in runs[0].items() if key != "cases"}
    aggregated["cases"] = cases
    aggregated["benchmark_passes"] = len(runs)
    return aggregated


def aggregate_selected_engine_runs(case_names: list[str], runs: list[dict]) -> dict:
    if not runs:
        raise ValueError("expected at least one benchmark run")

    cases_by_run = [{case["name"]: case for case in run["cases"]} for run in runs]
    cases = []
    for name in case_names:
        millis = [run_cases[name]["millis"] for run_cases in cases_by_run]
        checksums = [run_cases[name]["checksum"] for run_cases in cases_by_run]
        cases.append(
            {
                "name": name,
                "millis": statistics.median(millis),
                "checksum": checksums[-1],
                "pass_millis": millis,
            }
        )

    aggregated = {key: value for key, value in runs[0].items() if key != "cases"}
    aggregated["cases"] = cases
    aggregated["benchmark_passes"] = len(runs)
    return aggregated


def aggregate_pass_payloads(pass_payloads: list[dict], lock: dict) -> dict:
    if not pass_payloads:
        raise ValueError("expected at least one pass payload")
    ordered_passes = sorted(pass_payloads, key=lambda payload: payload["pass_idx"])
    numrust_runs = [payload["numrust"] for payload in ordered_passes]
    numpy_runs = [payload["numpy"] for payload in ordered_passes]
    numrust = aggregate_engine_runs(numrust_runs)
    numpy_result = aggregate_engine_runs(numpy_runs)
    result = compare(numrust, numpy_result, lock)
    result["benchmark_passes_per_engine"] = len(ordered_passes)
    result["benchmark_pass_order"] = "alternating; NumRust first on even passes, NumPy first on odd passes"
    return result


def load_lock() -> dict:
    if not LOCK_IN.exists():
        raise FileNotFoundError(
            f"Missing {LOCK_IN}. Run `uv run benchmarks/external_sources.py --update-lock` first."
        )
    return json.loads(LOCK_IN.read_text(encoding="utf-8"))


def load_array_api_result() -> dict | None:
    if not ARRAY_API_REPORT.exists():
        return None
    report = json.loads(ARRAY_API_REPORT.read_text(encoding="utf-8"))
    summary = report.get("summary", {})
    return {
        "source": "array-api-tests",
        "api_version": report.get("api_version"),
        "commit": report.get("source", {}).get("commit"),
        "status": report.get("status"),
        "returncode": report.get("returncode"),
        "full_suite": report.get("full_suite"),
        "collected": summary.get("collected"),
        "passed": summary.get("passed"),
        "skipped": summary.get("skipped"),
        "total": summary.get("total"),
        "report": str(ARRAY_API_REPORT.relative_to(ROOT)),
    }


def compare(numrust: dict, numpy_result: dict, lock: dict) -> dict:
    rust_cases = {case["name"]: case for case in numrust["cases"]}
    numpy_cases = {case["name"]: case for case in numpy_result["cases"]}
    case_specs = {case.name: case for case in RUNNABLE_CASES}
    rows = []
    failures = []

    for name, spec in case_specs.items():
        rust_case = rust_cases[name]
        numpy_case = numpy_cases[name]
        rust_ms = rust_case["millis"]
        numpy_ms = numpy_case["millis"]
        speedup = numpy_ms / rust_ms
        checksum_abs_diff = abs(float(rust_case["checksum"]) - float(numpy_case["checksum"]))
        checksum_match = math.isclose(
            float(rust_case["checksum"]),
            float(numpy_case["checksum"]),
            rel_tol=1e-9,
            abs_tol=1e-5,
        )
        if not checksum_match:
            failures.append(name)
        rows.append(
            {
                **asdict(spec),
                "numrust_ms": rust_ms,
                "numpy_ms": numpy_ms,
                "speedup_vs_numpy": speedup,
                "winner": "numrust" if speedup > 1.0 else "numpy",
                "near_tie": abs(speedup - 1.0) <= NEAR_TIE_RELATIVE_MARGIN,
                "numrust_checksum": rust_case["checksum"],
                "numpy_checksum": numpy_case["checksum"],
                "numrust_pass_millis": rust_case.get("pass_millis", [rust_ms]),
                "numpy_pass_millis": numpy_case.get("pass_millis", [numpy_ms]),
                "checksum_abs_diff": checksum_abs_diff,
                "checksum_match": checksum_match,
            }
        )

    wins = sum(1 for row in rows if row["winner"] == "numrust")
    near_ties = [row["name"] for row in rows if row["near_tie"]]
    geomean = math.prod(row["speedup_vs_numpy"] for row in rows) ** (1 / len(rows))
    return {
        "source_lock": lock,
        "machine": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
        },
        "numrust": numrust,
        "numpy": numpy_result,
        "comparison": rows,
        "array_api_conformance": load_array_api_result(),
        "unsupported_external_cases": UNSUPPORTED_EXTERNAL_CASES,
        "score": {
            "supported_external_cases": len(rows),
            "unsupported_external_cases": len(UNSUPPORTED_EXTERNAL_CASES),
            "numrust_wins": wins,
            "numpy_wins": len(rows) - wins,
            "geomean_speedup_vs_numpy": geomean,
            "numrust_ranked_higher_on_supported_external_cases": wins > len(rows) / 2,
            "global_numpy_replacement_claim": False,
            "near_tie_relative_margin": NEAR_TIE_RELATIVE_MARGIN,
            "near_tie_cases": near_ties,
            "checksum_failures": failures,
        },
    }


def loss_triage_rows(result: dict) -> list[dict]:
    rows = [row for row in result["comparison"] if row["winner"] == "numpy"]
    rows.sort(key=lambda row: row["speedup_vs_numpy"])
    triage_rows = []
    for priority, row in enumerate(rows, start=1):
        triage_rows.append(
            {
                "priority": priority,
                "name": row["name"],
                "source_id": row["source_id"],
                "source_path": row["source_path"],
                "source_symbol": row["source_symbol"],
                "translation": row["translation"],
                "numrust_ms": row["numrust_ms"],
                "numpy_ms": row["numpy_ms"],
                "speedup_vs_numpy": row["speedup_vs_numpy"],
                "numrust_slowdown_vs_numpy": (row["numrust_ms"] / row["numpy_ms"]) - 1.0,
                "near_tie": row["near_tie"],
                "checksum_match": row["checksum_match"],
                "numrust_pass_millis": row["numrust_pass_millis"],
                "numpy_pass_millis": row["numpy_pass_millis"],
            }
        )
    return triage_rows


def loss_triage_payload(result: dict) -> dict:
    score = result["score"]
    return {
        "source_report": str(JSON_OUT.relative_to(ROOT)),
        "sort": "NumPy-winning rows only, ordered by ascending speedup_vs_numpy",
        "supported_external_cases": score["supported_external_cases"],
        "numrust_wins": score["numrust_wins"],
        "numpy_wins": score["numpy_wins"],
        "geomean_speedup_vs_numpy": score["geomean_speedup_vs_numpy"],
        "near_tie_relative_margin": score["near_tie_relative_margin"],
        "global_numpy_replacement_claim": False,
        "rows": loss_triage_rows(result),
    }


def focused_loss_stability(rows: list[dict]) -> dict:
    numrust_flips = [row["name"] for row in rows if row["winner"] == "numrust"]
    stable_numpy_wins = [row["name"] for row in rows if row["winner"] == "numpy"]
    near_ties = [row["name"] for row in rows if row["near_tie"]]
    return {
        "source_numpy_winners": len(rows),
        "focused_numrust_winner_flips": len(numrust_flips),
        "focused_still_numpy_winners": len(stable_numpy_wins),
        "focused_near_ties": len(near_ties),
        "numrust_flip_cases": numrust_flips,
        "stable_numpy_win_cases": stable_numpy_wins,
        "near_tie_cases": near_ties,
        "authoritative_score_source": str(JSON_OUT.relative_to(ROOT)),
        "interpretation": (
            "Focused reruns are optimization and stability probes for current loss rows; "
            "the full external report remains the authoritative supported-case score."
        ),
    }


def write_loss_triage(result: dict) -> None:
    payload = loss_triage_payload(result)
    LOSS_TRIAGE_JSON_OUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# External NumPy Loss Triage",
        "",
        f"- Source report: `{payload['source_report']}`",
        f"- Supported external cases: {payload['supported_external_cases']}",
        f"- NumRust wins: {payload['numrust_wins']}",
        f"- NumPy wins: {payload['numpy_wins']}",
        f"- Geomean speedup vs NumPy: {payload['geomean_speedup_vs_numpy']:.2f}x",
        "- Global NumPy replacement claim: false",
        "",
        "| Priority | Case | Source | NumRust ms | NumPy ms | Speedup | NumRust Slowdown | Near Tie |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload["rows"]:
        source = f"`{row['source_path']}::{row['source_symbol']}`"
        lines.append(
            f"| {row['priority']} | `{row['name']}` | {source} | "
            f"{row['numrust_ms']:.3f} | {row['numpy_ms']:.3f} | "
            f"{row['speedup_vs_numpy']:.2f}x | {row['numrust_slowdown_vs_numpy']:.1%} | "
            f"{row['near_tie']} |"
        )
    if not payload["rows"]:
        lines.append("|  | No NumPy-winning rows in this report. |  |  |  |  |  |  |")
    lines.append("")
    LOSS_TRIAGE_MD_OUT.write_text("\n".join(lines), encoding="utf-8")


def focused_loss_payload(source_report: dict, numrust: dict, numpy_result: dict) -> dict:
    triage_rows = loss_triage_rows(source_report)
    names = [row["name"] for row in triage_rows]
    rust_cases = {case["name"]: case for case in numrust["cases"]}
    numpy_cases = {case["name"]: case for case in numpy_result["cases"]}
    source_rows = {row["name"]: row for row in source_report["comparison"]}

    rows = []
    failures = []
    for priority, name in enumerate(names, start=1):
        source = source_rows[name]
        rust_case = rust_cases[name]
        numpy_case = numpy_cases[name]
        rust_ms = rust_case["millis"]
        numpy_ms = numpy_case["millis"]
        speedup = numpy_ms / rust_ms
        checksum_abs_diff = abs(float(rust_case["checksum"]) - float(numpy_case["checksum"]))
        checksum_match = math.isclose(
            float(rust_case["checksum"]),
            float(numpy_case["checksum"]),
            rel_tol=1e-9,
            abs_tol=1e-5,
        )
        if not checksum_match:
            failures.append(name)
        rows.append(
            {
                "priority": priority,
                "name": name,
                "source_id": source["source_id"],
                "source_path": source["source_path"],
                "source_symbol": source["source_symbol"],
                "translation": source["translation"],
                "baseline_speedup_vs_numpy": source["speedup_vs_numpy"],
                "numrust_ms": rust_ms,
                "numpy_ms": numpy_ms,
                "speedup_vs_numpy": speedup,
                "winner": "numrust" if speedup > 1.0 else "numpy",
                "near_tie": abs(speedup - 1.0) <= NEAR_TIE_RELATIVE_MARGIN,
                "numrust_checksum": rust_case["checksum"],
                "numpy_checksum": numpy_case["checksum"],
                "numrust_pass_millis": rust_case.get("pass_millis", [rust_ms]),
                "numpy_pass_millis": numpy_case.get("pass_millis", [numpy_ms]),
                "checksum_abs_diff": checksum_abs_diff,
                "checksum_match": checksum_match,
            }
        )

    numrust_wins = sum(row["winner"] == "numrust" for row in rows)
    return {
        "source_report": str(JSON_OUT.relative_to(ROOT)),
        "purpose": "Focused rerun of current NumPy-winning external rows for optimization triage.",
        "focused_cases": len(rows),
        "benchmark_passes_per_engine": numrust.get("benchmark_passes", 1),
        "benchmark_pass_order": "alternating; NumRust first on even passes, NumPy first on odd passes",
        "numrust_wins": numrust_wins,
        "numpy_wins": len(rows) - numrust_wins,
        "near_tie_relative_margin": NEAR_TIE_RELATIVE_MARGIN,
        "checksum_failures": failures,
        "global_numpy_replacement_claim": False,
        "stability": focused_loss_stability(rows),
        "rows": rows,
    }


def write_focused_loss_report(payload: dict) -> None:
    LOSS_FOCUSED_JSON_OUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# External NumPy Focused Loss Rerun",
        "",
        f"- Source report: `{payload['source_report']}`",
        f"- Focused cases: {payload['focused_cases']}",
        f"- Full focused passes per engine: {payload['benchmark_passes_per_engine']}",
        f"- NumRust wins in focused rerun: {payload['numrust_wins']}",
        f"- NumPy wins in focused rerun: {payload['numpy_wins']}",
        f"- Checksum failures: {len(payload['checksum_failures'])}",
        f"- Focused NumRust flips from source NumPy wins: {payload['stability']['focused_numrust_winner_flips']}",
        f"- Source NumPy wins still won by NumPy: {payload['stability']['focused_still_numpy_winners']}",
        f"- Focused near ties within {payload['near_tie_relative_margin']:.0%}: {payload['stability']['focused_near_ties']}",
        f"- Authoritative score source: `{payload['stability']['authoritative_score_source']}`",
        "- Raw per-pass samples: `benchmark-results/external-numpy-loss-focused.json`",
        "- Global NumPy replacement claim: false",
        "",
        "| Priority | Case | NumRust ms | NumPy ms | Speedup | Winner | Baseline Speedup | Checksum |",
        "| ---: | --- | ---: | ---: | ---: | --- | ---: | --- |",
    ]
    for row in payload["rows"]:
        checksum = "ok" if row["checksum_match"] else f"diff {row['checksum_abs_diff']:.3g}"
        lines.append(
            f"| {row['priority']} | `{row['name']}` | {row['numrust_ms']:.3f} | "
            f"{row['numpy_ms']:.3f} | {row['speedup_vs_numpy']:.2f}x | {row['winner']} | "
            f"{row['baseline_speedup_vs_numpy']:.2f}x | {checksum} |"
        )
    lines.append("")
    LOSS_FOCUSED_MD_OUT.write_text("\n".join(lines), encoding="utf-8")


def rerun_focused_losses(passes: int = 1) -> dict:
    if passes < 1:
        raise ValueError("focused loss passes must be at least 1")
    source_report = json.loads(JSON_OUT.read_text(encoding="utf-8"))
    names = [row["name"] for row in loss_triage_rows(source_report)]
    if not names:
        payload = focused_loss_payload(
            source_report,
            {"engine": "numrust", "cases": []},
            {"engine": "numpy", "numpy_version": np.__version__, "cases": []},
        )
        write_focused_loss_report(payload)
        return payload
    write_lstsq_fixture()
    numrust_runs = []
    numpy_runs = []
    for pass_idx in range(passes):
        if pass_idx % 2 == 0:
            numrust_runs.append(run_numrust())
            numpy_runs.append(bench_numpy_selected(names))
        else:
            numpy_runs.append(bench_numpy_selected(names))
            numrust_runs.append(run_numrust())
    numrust = aggregate_selected_engine_runs(names, numrust_runs)
    numpy_result = aggregate_selected_engine_runs(names, numpy_runs)
    payload = focused_loss_payload(source_report, numrust, numpy_result)
    write_focused_loss_report(payload)
    return payload


def write_markdown(result: dict) -> None:
    numpy_source = next(
        source for source in result["source_lock"]["sources"] if source["id"] == "numpy-asv"
    )
    array_api_source = next(
        source for source in result["source_lock"]["sources"] if source["id"] == "array-api-tests"
    )
    lines = [
        "# External NumPy Evidence",
        "",
        "This report uses pinned online sources. It is not a self-authored-only benchmark.",
        "",
        "## Source Lock",
        "",
        f"- NumPy ASV: `{numpy_source['commit']}`",
        f"- Array API tests: `{array_api_source['commit']}`",
        f"- NumPy version measured locally: `{result['numpy']['numpy_version']}`",
        f"- Python: `{result['machine']['python']}`",
        f"- Full benchmark passes per engine: {result['benchmark_passes_per_engine']}",
        "- Pass aggregation: per-case median across alternating engine-order full passes.",
        "",
        "## Array API Conformance",
        "",
    ]
    conformance = result.get("array_api_conformance")
    if conformance is None:
        lines.append("- Full suite result: not run in this benchmark pass.")
    else:
        lines.extend(
            [
                f"- Source: `{conformance['source']}` commit `{conformance['commit']}`",
                f"- API version: `{conformance['api_version']}`",
                f"- Full suite: {conformance['full_suite']}",
                f"- Status: {conformance['status']}",
                f"- Return code: {conformance['returncode']}",
                f"- Summary: {conformance['passed']} passed, {conformance['skipped']} skipped, {conformance['collected']} collected",
                f"- Report: `{conformance['report']}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Supported External Cases",
            "",
            "| Case | Source | Reps | NumRust ms | NumPy ms | Speedup vs NumPy | Winner | Checksum |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in result["comparison"]:
        source = f"`{row['source_path']}::{row['source_symbol']}`"
        checksum = "ok" if row["checksum_match"] else f"diff {row['checksum_abs_diff']:.3g}"
        lines.append(
            f"| `{row['name']}` | {source} | {row['repetitions']} | "
            f"{row['numrust_ms']:.3f} | {row['numpy_ms']:.3f} | "
            f"{row['speedup_vs_numpy']:.2f}x | {row['winner']} | {checksum} |"
        )

    score = result["score"]
    numpy_wins = [row for row in result["comparison"] if row["winner"] == "numpy"]
    lines += [
        "",
        "## Score",
        "",
        f"- Supported external cases: {score['supported_external_cases']}",
        f"- Unsupported external cases tracked: {score['unsupported_external_cases']}",
        f"- NumRust wins: {score['numrust_wins']}",
        f"- NumPy wins: {score['numpy_wins']}",
        f"- Geomean speedup vs NumPy: {score['geomean_speedup_vs_numpy']:.2f}x",
        f"- Near-tie relative margin: {score['near_tie_relative_margin']:.0%}",
        f"- Near-tie cases: {len(score['near_tie_cases'])}",
        f"- Ranked higher on supported external cases: {score['numrust_ranked_higher_on_supported_external_cases']}",
        "- Global NumPy replacement claim: false",
        "",
        "## NumPy-Winning Cases",
        "",
    ]
    if not numpy_wins:
        lines.append("- None in this run.")
    else:
        lines.extend(
            [
                "| Case | NumRust ms | NumPy ms | NumRust pass ms | NumPy pass ms |",
                "| --- | ---: | ---: | --- | --- |",
            ]
        )
        for row in numpy_wins:
            rust_samples = ", ".join(f"{sample:.3f}" for sample in row["numrust_pass_millis"])
            numpy_samples = ", ".join(f"{sample:.3f}" for sample in row["numpy_pass_millis"])
            lines.append(
                f"| `{row['name']}` | {row['numrust_ms']:.3f} | {row['numpy_ms']:.3f} | "
                f"{rust_samples} | {numpy_samples} |"
            )

    lines += [
        "",
        "## Unsupported External Cases",
        "",
        "| Source | Case | Reason |",
        "| --- | --- | --- |",
    ]
    for case in result["unsupported_external_cases"]:
        lines.append(f"| `{case['source']}` | `{case['case']}` | {case['reason']} |")

    lines += [
        "",
        "## Neutrality Controls",
        "",
        "- Benchmark cases come from pinned NumPy ASV files; conformance evidence comes from the pinned Array API suite.",
        "- The supported-case table includes losses and checksum failures.",
        "- Unsupported external cases are counted outside the speed score instead of omitted silently.",
        "- Tiny ASV cases are repeated equally for both engines because ASV normally auto-calibrates repetitions.",
        "- This still does not prove full NumPy replacement status.",
        "",
    ]
    MD_OUT.write_text("\n".join(lines), encoding="utf-8")


def write_report(result: dict) -> None:
    JSON_OUT.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(result)
    write_loss_triage(result)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pass-index", type=int, help="run one alternating benchmark pass")
    parser.add_argument("--pass-out", type=Path, help="write one-pass JSON payload")
    parser.add_argument(
        "--aggregate-passes",
        nargs="+",
        type=Path,
        help="aggregate one-pass JSON payloads into the normal report",
    )
    parser.add_argument(
        "--loss-triage",
        action="store_true",
        help="write loss-triage artifacts from the existing JSON report",
    )
    parser.add_argument(
        "--rerun-losses",
        action="store_true",
        help="rerun current NumPy-winning rows and write focused loss artifacts",
    )
    parser.add_argument(
        "--loss-passes",
        type=int,
        default=1,
        help="number of alternating passes for --rerun-losses",
    )
    args = parser.parse_args()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    lock = load_lock()

    if args.loss_triage:
        if args.pass_index is not None or args.aggregate_passes is not None or args.rerun_losses:
            raise ValueError("--loss-triage cannot be combined with benchmark execution options")
        result = json.loads(JSON_OUT.read_text(encoding="utf-8"))
        write_loss_triage(result)
        payload = loss_triage_payload(result)
        print(json.dumps({"status": "loss-triage-written", "numpy_wins": payload["numpy_wins"]}))
        return 0

    if args.rerun_losses:
        if args.pass_index is not None or args.aggregate_passes is not None:
            raise ValueError("--rerun-losses cannot be combined with benchmark execution options")
        payload = rerun_focused_losses(args.loss_passes)
        print(
            json.dumps(
                {
                    "status": "focused-loss-rerun-written",
                    "focused_cases": payload["focused_cases"],
                    "passes": payload["benchmark_passes_per_engine"],
                    "numrust_wins": payload["numrust_wins"],
                    "numpy_wins": payload["numpy_wins"],
                    "checksum_failures": payload["checksum_failures"],
                },
                sort_keys=True,
            )
        )
        return 1 if payload["checksum_failures"] else 0

    if args.pass_index is not None:
        if args.aggregate_passes is not None:
            raise ValueError("--pass-index and --aggregate-passes are mutually exclusive")
        if args.pass_out is None:
            raise ValueError("--pass-out is required with --pass-index")
        print(json.dumps({"status": "pass-started", "pass": args.pass_index}), flush=True)
        payload = run_benchmark_pass(args.pass_index)
        args.pass_out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps({"status": "pass-written", "pass": args.pass_index, "path": str(args.pass_out)}))
        return 0

    if args.aggregate_passes is not None:
        pass_payloads = [
            json.loads(path.read_text(encoding="utf-8")) for path in args.aggregate_passes
        ]
        result = aggregate_pass_payloads(pass_payloads, lock)
        write_report(result)
        print(json.dumps(result["score"], indent=2, sort_keys=True))
        return 1 if result["score"]["checksum_failures"] else 0

    passes = benchmark_passes()
    pass_payloads = [run_benchmark_pass(pass_idx) for pass_idx in range(passes)]
    result = aggregate_pass_payloads(pass_payloads, lock)
    write_report(result)
    print(json.dumps(result["score"], indent=2, sort_keys=True))
    return 1 if result["score"]["checksum_failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
