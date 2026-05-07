# External NumPy Loss Triage

- Source report: `benchmark-results/external-numpy-asv-inspired.json`
- Supported external cases: 80
- NumRust wins: 78
- NumPy wins: 2
- Geomean speedup vs NumPy: 8.79x
- Global NumPy replacement claim: false

| Priority | Case | Source | NumRust ms | NumPy ms | Speedup | NumRust Slowdown | Near Tie |
| ---: | --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | `asv_linalg_matmul_trans_a_at_f64_150x400_400x150` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_matmul_trans_a_at` | 30.569 | 29.160 | 0.95x | 4.8% | False |
| 2 | `asv_linalg_matmul_trans_a_atc_f64_150x400_400x150` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_matmul_trans_a_atc` | 42.125 | 41.240 | 0.98x | 2.1% | False |
