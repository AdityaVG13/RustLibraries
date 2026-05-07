# External NumPy Loss Triage

- Source report: `benchmark-results/external-numpy-asv-inspired.json`
- Supported external cases: 96
- NumRust wins: 95
- NumPy wins: 1
- Geomean speedup vs NumPy: 14.03x
- Global NumPy replacement claim: false

| Priority | Case | Source | NumRust ms | NumPy ms | Speedup | NumRust Slowdown | Near Tie |
| ---: | --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | `asv_linalg_dot_trans_a_atc_f64_150x400_400x150` | `benchmarks/benchmarks/bench_linalg.py::Eindot.time_dot_trans_a_atc` | 42.263 | 42.002 | 0.99x | 0.6% | True |
