# External NumPy Focused Loss Rerun

- Source report: `benchmark-results/external-numpy-asv-inspired.json`
- Focused cases: 2
- Full focused passes per engine: 3
- NumRust wins in focused rerun: 2
- NumPy wins in focused rerun: 0
- Checksum failures: 0
- Focused NumRust flips from source NumPy wins: 2
- Source NumPy wins still won by NumPy: 0
- Focused near ties within 2%: 1
- Authoritative score source: `benchmark-results/external-numpy-asv-inspired.json`
- Raw per-pass samples: `benchmark-results/external-numpy-loss-focused.json`
- Global NumPy replacement claim: false

| Priority | Case | NumRust ms | NumPy ms | Speedup | Winner | Baseline Speedup | Checksum |
| ---: | --- | ---: | ---: | ---: | --- | ---: | --- |
| 1 | `asv_linalg_matmul_trans_a_at_f64_150x400_400x150` | 30.693 | 30.816 | 1.00x | numrust | 0.97x | ok |
| 2 | `asv_linalg_matmul_trans_atc_a_f64_400x150_150x400` | 94.648 | 96.794 | 1.02x | numrust | 1.00x | ok |
