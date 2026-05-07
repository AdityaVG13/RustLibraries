# External NumPy Loss Triage

- Source report: `benchmark-results/external-numpy-asv-inspired.json`
- Supported external cases: 72
- NumRust wins: 71
- NumPy wins: 1
- Geomean speedup vs NumPy: 9.07x
- Global NumPy replacement claim: false

| Priority | Case | Source | NumRust ms | NumPy ms | Speedup | NumRust Slowdown | Near Tie |
| ---: | --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | `asv_linalg_matmul_trans_a_atc_f64_150x400_400x150` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_matmul_trans_a_atc` | 42.526 | 42.493 | 1.00x | 0.1% | True |
