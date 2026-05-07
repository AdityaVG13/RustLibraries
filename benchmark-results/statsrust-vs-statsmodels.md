# StatsRust vs StatsModels Benchmark

StatsModels version: `0.14.6`

| Case | StatsRust ms | StatsModels ms | Speedup vs StatsModels | Winner | Checksum |
| --- | ---: | ---: | ---: | --- | --- |
| `statsmodels_ols_fit_2000x3` | 1.933 | 10.819 | 5.60x | statsrust | ok |
| `statsmodels_ols_predict_2000x3` | 19.335 | 35.828 | 1.85x | statsrust | ok |
| `statsmodels_wls_fit_2000x3` | 2.265 | 7.570 | 3.34x | statsrust | ok |
| `statsmodels_logit_fit_800x2` | 0.845 | 3.715 | 4.40x | statsrust | ok |

## Score

- StatsRust wins: 4
- StatsModels wins: 0
- Geomean speedup vs StatsModels: 3.51x
- Checksum failures: 0
- Ranked higher on this suite: True
- Global StatsModels replacement claim: false

This is a same-data benchmark for the implemented OLS, WLS, and Logit slice. It does not claim full StatsModels API parity.
