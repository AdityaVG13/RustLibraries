# External NumPy Focused Loss Rerun

- Source report: `benchmark-results/external-numpy-asv-inspired.json`
- Focused cases: 2
- Full focused passes per engine: 3
- NumRust wins in focused rerun: 1
- NumPy wins in focused rerun: 1
- Checksum failures: 0
- Focused NumRust flips from source NumPy wins: 1
- Source NumPy wins still won by NumPy: 1
- Focused near ties within 2%: 1
- Authoritative score source: `benchmark-results/external-numpy-asv-inspired.json`
- Raw per-pass samples: `benchmark-results/external-numpy-loss-focused.json`
- Global NumPy replacement claim: false

| Priority | Case | NumRust ms | NumPy ms | Speedup | Winner | Baseline Speedup | Checksum |
| ---: | --- | ---: | ---: | ---: | --- | ---: | --- |
| 1 | `asv_linalg_dot_trans_a_atc_f64_150x400_400x150` | 40.955 | 43.724 | 1.07x | numrust | 0.98x | ok |
| 2 | `asv_linalg_matmul_trans_atc_a_f64_400x150_150x400` | 96.552 | 95.533 | 0.99x | numpy | 0.99x | ok |
