# External NumPy Loss Triage

- Source report: `benchmark-results/external-numpy-asv-inspired.json`
- Supported external cases: 84
- NumRust wins: 82
- NumPy wins: 2
- Geomean speedup vs NumPy: 8.41x
- Global NumPy replacement claim: false

| Priority | Case | Source | NumRust ms | NumPy ms | Speedup | NumRust Slowdown | Near Tie |
| ---: | --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | `asv_linalg_matmul_trans_a_at_f64_150x400_400x150` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_matmul_trans_a_at` | 30.769 | 30.696 | 1.00x | 0.2% | True |
| 2 | `asv_linalg_inner_a_ac_f64_150x400_150x400` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_inner_trans_a_ac` | 49.295 | 49.239 | 1.00x | 0.1% | True |
