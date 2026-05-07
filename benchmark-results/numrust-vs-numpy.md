# NumRust vs NumPy Benchmark

NumPy version: `2.4.4`

| Case | NumRust ms | NumPy ms | Speedup vs NumPy | Winner |
| --- | ---: | ---: | ---: | --- |
| `small_add_f64_loop` | 3.599 | 12.661 | 3.52x | numrust |
| `large_add_f64_loop` | 3.190 | 4.153 | 1.30x | numrust |
| `fused_add_sum_f64_loop` | 3.985 | 4.089 | 1.03x | numrust |
| `broadcast_add_f64` | 0.198 | 0.518 | 2.62x | numrust |
| `sum_f64_loop` | 9.167 | 13.675 | 1.49x | numrust |
| `metadata_view_loop` | 38.041 | 47.178 | 1.24x | numrust |
| `take_axis_i64_loop` | 20.989 | 25.758 | 1.23x | numrust |
| `dot_f64_192` | 22.295 | 23.465 | 1.05x | numrust |

## Score

- NumRust wins: 8
- NumPy wins: 0
- Geomean speedup vs NumPy: 1.53x
- Ranked higher on this suite: True
- Global NumPy replacement claim: false

This suite measures targeted core kernels and Python dispatch overhead. It does not prove full NumPy replacement status.
