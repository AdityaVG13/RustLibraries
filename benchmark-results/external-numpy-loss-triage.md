# External NumPy Loss Triage

- Source report: `benchmark-results/external-numpy-asv-inspired.json`
- Supported external cases: 53
- NumRust wins: 43
- NumPy wins: 10
- Geomean speedup vs NumPy: 8.92x
- Global NumPy replacement claim: false

| Priority | Case | Source | NumRust ms | NumPy ms | Speedup | NumRust Slowdown | Near Tie |
| ---: | --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | `asv_linalg_matmul_trans_a_atc_f64_150x400_400x150` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_matmul_trans_a_atc` | 57.400 | 47.133 | 0.82x | 21.8% | False |
| 2 | `asv_linalg_matmul_trans_at_a_f64_400x150_150x400` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_matmul_trans_at_a` | 107.797 | 91.140 | 0.85x | 18.3% | False |
| 3 | `asv_linalg_dot_trans_a_atc_f64_150x400_400x150` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_dot_trans_a_atc` | 58.976 | 51.520 | 0.87x | 14.5% | False |
| 4 | `asv_linalg_matmul_trans_atc_a_f64_400x150_150x400` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_matmul_trans_atc_a` | 138.744 | 122.039 | 0.88x | 13.7% | False |
| 5 | `asv_linalg_dot_trans_atc_a_f64_400x150_150x400` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_dot_trans_atc_a` | 134.407 | 118.826 | 0.88x | 13.1% | False |
| 6 | `asv_linalg_inner_a_ac_f64_150x400_150x400` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_inner_trans_a_ac` | 59.508 | 54.891 | 0.92x | 8.4% | False |
| 7 | `asv_linalg_matmul_trans_a_at_f64_150x400_400x150` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_matmul_trans_a_at` | 34.652 | 32.056 | 0.93x | 8.1% | False |
| 8 | `asv_linalg_dot_a_b_f64_150x400_400x600` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_dot_a_b` | 208.580 | 198.544 | 0.95x | 5.1% | False |
| 9 | `asv_linalg_einsum_scalar_mul_f64_480000` | `benchmarks/benchmarks/bench_linalg.py::Einsum.time_einsum_mul(dtype=float64)` | 6.253 | 6.020 | 0.96x | 3.9% | False |
| 10 | `asv_linalg_matmul_a_b_f64_150x400_400x600` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_matmul_a_b` | 204.019 | 198.361 | 0.97x | 2.9% | False |
