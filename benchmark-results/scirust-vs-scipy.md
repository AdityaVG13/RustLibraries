# SciRust vs SciPy Benchmark

SciPy version: `1.17.1`

| Case | SciRust ms | SciPy ms | Speedup vs SciPy | Winner | Checksum |
| --- | ---: | ---: | ---: | --- | --- |
| `scipy_integrate_trapezoid_10001` | 15.534 | 20.189 | 1.30x | scirust | ok |
| `scipy_integrate_simpson_10001` | 5.518 | 13.745 | 2.49x | scirust | ok |
| `scipy_asv_cumulative_simpson_1d_1000` | 1.147 | 21.106 | 18.40x | scirust | ok |
| `scipy_asv_cumulative_simpson_multid_100x100x1000` | 13.218 | 61.067 | 4.62x | scirust | ok |
| `scipy_optimize_minimize_scalar_bounded` | 0.994 | 123.261 | 124.06x | scirust | ok |
| `scipy_optimize_bisect` | 1.991 | 128.772 | 64.67x | scirust | ok |
| `scipy_asv_zeros_f2_bisect` | 1.677 | 128.078 | 76.39x | scirust | ok |
| `scipy_optimize_brentq` | 0.796 | 34.184 | 42.96x | scirust | ok |
| `scipy_asv_zeros_f2_brentq` | 0.729 | 34.163 | 46.86x | scirust | ok |

## Score

- SciRust wins: 9
- SciPy wins: 0
- Geomean speedup vs SciPy: 19.11x
- Checksum failures: 0
- Ranked higher on this suite: True
- Global SciPy replacement claim: false

This suite includes translated SciPy ASV root-finding and cumulative Simpson cases, plus same-data integration/optimization cases for the implemented slice. It does not claim full SciPy API parity.
