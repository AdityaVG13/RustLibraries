# External NumPy Focused Loss Rerun

- Source report: `benchmark-results/external-numpy-asv-inspired.json`
- Focused cases: 10
- Full focused passes per engine: 3
- NumRust wins in focused rerun: 9
- NumPy wins in focused rerun: 1
- Checksum failures: 0
- Focused NumRust flips from source NumPy wins: 9
- Source NumPy wins still won by NumPy: 1
- Focused near ties within 2%: 7
- Authoritative score source: `benchmark-results/external-numpy-asv-inspired.json`
- Raw per-pass samples: `benchmark-results/external-numpy-loss-focused.json`
- Global NumPy replacement claim: false

| Priority | Case | NumRust ms | NumPy ms | Speedup | Winner | Baseline Speedup | Checksum |
| ---: | --- | ---: | ---: | ---: | --- | ---: | --- |
| 1 | `asv_linalg_matmul_trans_a_atc_f64_150x400_400x150` | 41.805 | 42.708 | 1.02x | numrust | 0.82x | ok |
| 2 | `asv_linalg_matmul_trans_at_a_f64_400x150_150x400` | 78.096 | 79.334 | 1.02x | numrust | 0.85x | ok |
| 3 | `asv_linalg_dot_trans_a_atc_f64_150x400_400x150` | 41.687 | 42.001 | 1.01x | numrust | 0.87x | ok |
| 4 | `asv_linalg_matmul_trans_atc_a_f64_400x150_150x400` | 94.986 | 96.394 | 1.01x | numrust | 0.88x | ok |
| 5 | `asv_linalg_dot_trans_atc_a_f64_400x150_150x400` | 95.619 | 100.852 | 1.05x | numrust | 0.88x | ok |
| 6 | `asv_linalg_inner_a_ac_f64_150x400_150x400` | 49.289 | 49.771 | 1.01x | numrust | 0.92x | ok |
| 7 | `asv_linalg_matmul_trans_a_at_f64_150x400_400x150` | 29.577 | 29.458 | 1.00x | numpy | 0.93x | ok |
| 8 | `asv_linalg_dot_a_b_f64_150x400_400x600` | 144.249 | 146.754 | 1.02x | numrust | 0.95x | ok |
| 9 | `asv_linalg_einsum_scalar_mul_f64_480000` | 6.327 | 6.778 | 1.07x | numrust | 0.96x | ok |
| 10 | `asv_linalg_matmul_a_b_f64_150x400_400x600` | 144.250 | 144.644 | 1.00x | numrust | 0.97x | ok |
