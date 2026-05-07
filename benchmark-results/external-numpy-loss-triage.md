# External NumPy Loss Triage

- Source report: `benchmark-results/external-numpy-asv-inspired.json`
- Supported external cases: 118
- NumRust wins: 117
- NumPy wins: 1
- Geomean speedup vs NumPy: 32.62x
- Global NumPy replacement claim: false

| Priority | Case | Source | NumRust ms | NumPy ms | Speedup | NumRust Slowdown | Near Tie |
| ---: | --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | `asv_linalg_matmul_trans_a_at_f64_150x400_400x150` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_matmul_trans_a_at` | 30.567 | 29.021 | 0.95x | 5.3% | False |
