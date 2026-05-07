# NumRust vs NumPy Benchmark

Targeted same-data benchmark for implemented NumRust kernels. This is not a full NumPy replacement claim.

NumPy version: `2.4.4`

| Case | NumRust ms | NumPy ms | Speedup vs NumPy | Checksum diff | Checksum ok | Winner |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `small_add_f64_loop` | 2.829 | 9.727 | 3.44x | 0 | True | numrust |
| `large_add_f64_loop` | 3.582 | 3.807 | 1.06x | 0 | True | numrust |
| `fused_add_sum_f64_loop` | 3.035 | 3.816 | 1.26x | 0 | True | numrust |
| `broadcast_add_f64` | 0.192 | 0.422 | 2.20x | 0 | True | numrust |
| `sum_f64_loop` | 7.494 | 10.635 | 1.42x | 0 | True | numrust |
| `metadata_view_loop` | 30.088 | 40.862 | 1.36x | 0 | True | numrust |
| `take_axis_i64_loop` | 16.420 | 20.002 | 1.22x | 0 | True | numrust |
| `where_select_f64_loop` | 85.690 | 51.328 | 0.60x | 0 | True | numpy |
| `nonzero_bool_loop` | 193.816 | 917.348 | 4.73x | 0 | True | numrust |
| `dot_f64_192` | 15.657 | 15.584 | 1.00x | 0 | True | numpy |

## Score

- NumRust wins: 8
- NumPy wins: 2
- Checksum failures: 0
- Geomean speedup vs NumPy: 1.52x
- Ranked higher on this suite: True
- Global NumPy replacement claim: false

A row with a checksum failure is not counted as a Rust win, regardless of timing.
