# NumRust vs NumPy Benchmark

Targeted same-data benchmark for implemented NumRust kernels. This is not a full NumPy replacement claim.

NumPy version: `2.4.4`

| Case | NumRust ms | NumPy ms | Speedup vs NumPy | Checksum diff | Checksum ok | Winner |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `small_add_f64_loop` | 2.806 | 9.918 | 3.53x | 0 | True | numrust |
| `large_add_f64_loop` | 3.522 | 3.884 | 1.10x | 0 | True | numrust |
| `fused_add_sum_f64_loop` | 2.999 | 3.882 | 1.29x | 0 | True | numrust |
| `broadcast_add_f64` | 0.183 | 0.456 | 2.49x | 0 | True | numrust |
| `sum_f64_loop` | 7.101 | 11.129 | 1.57x | 0 | True | numrust |
| `metadata_view_loop` | 31.967 | 40.042 | 1.25x | 0 | True | numrust |
| `take_axis_i64_loop` | 16.526 | 20.588 | 1.25x | 0 | True | numrust |
| `where_select_f64_loop` | 48.615 | 52.859 | 1.09x | 0 | True | numrust |
| `nonzero_bool_loop` | 192.362 | 935.158 | 4.86x | 0 | True | numrust |
| `dot_f64_192` | 16.473 | 17.051 | 1.04x | 0 | True | numrust |

## Score

- NumRust wins: 10
- NumPy wins: 0
- Checksum failures: 0
- Geomean speedup vs NumPy: 1.67x
- Ranked higher on this suite: True
- Global NumPy replacement claim: false

A row with a checksum failure is not counted as a Rust win, regardless of timing.
