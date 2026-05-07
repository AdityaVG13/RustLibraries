# StatsModels Port

**StatsRust** starts the second library line after NumRust crossed the targeted NumPy benchmark gate.

## Why StatsModels

StatsModels is a major Python statistics library. Rust has useful statistics and machine-learning crates, but no clear full StatsModels-level equivalent with formulas, model families, diagnostics, and inference as the central API.

Primary references:

- https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html
- https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.Logit.html
- https://www.statsmodels.org/stable/regression.html

## Current Slice

- Ordinary least squares fitting.
- Weighted least squares fitting.
- Binary logistic regression fitting.
- Optional intercept column.
- Coefficients.
- Regression prediction.
- Logistic probability and class prediction.
- RSS, TSS, R-squared, adjusted R-squared, and MSE.
- Log-likelihood, null log-likelihood, deviance, McFadden-style pseudo R-squared, and accuracy.
- Singular design detection.
- Invalid weight detection.
- Invalid binary target and threshold detection.

## Benchmark Evidence

Run:

```sh
uv run benchmarks/compare_statsmodels.py
```

Latest same-data benchmark against StatsModels 0.14.6:

- StatsRust wins: 4 of 4.
- StatsModels wins: 0 of 4.
- Geomean speedup vs StatsModels: 3.51x.
- Checksum failures: 0.
- Global StatsModels replacement claim: false.

Recent optimization: OLS prediction now evaluates directly from the raw feature matrix instead of allocating an intercept-expanded design matrix on every prediction call.

## Next Scope

- QR/SVD solver instead of normal equations.
- Standard errors, t-statistics, p-values, and confidence intervals.
- Formula parser.
- More GLM families.
- Robust covariance estimators.
- Time-series models.
