# LearnRust

Pure Rust classical ML primitives inspired by scikit-learn.

This is an implemented slice, not a full scikit-learn replacement. The current API covers:

- Dense row-major `f64` matrices with checked shape construction.
- `StandardScaler` with population variance semantics and zero-variance scale handling.
- `NearestCentroid` classification with deterministic sorted-label behavior.
- `accuracy_score`, `confusion_matrix`, and inferred sorted labels.

## Example

```rust
use learnrust::{DenseMatrix, NearestCentroid, StandardScaler};

let matrix = DenseMatrix::from_vec(
    4,
    2,
    vec![0.0, 0.0, 0.0, 2.0, 10.0, 10.0, 10.0, 12.0],
)?;
let labels = vec![0, 0, 1, 1];

let (_scaler, transformed) = StandardScaler::fit_transform(&matrix)?;
let model = NearestCentroid::fit(&transformed, &labels)?;
let predictions = model.predict(&transformed)?;
```

## Benchmark

Run:

```sh
uv run --with numpy --with scikit-learn benchmarks/compare_sklearn.py
```

Current same-data slice evidence is in `benchmark-results/learnrust-vs-sklearn.md`: 3 LearnRust wins, 0 scikit-learn wins, 6.53x geomean speedup, and 0 checksum failures.

## Not Covered Yet

Pipelines, model selection, estimators beyond nearest centroid, tree models, linear models, clustering beyond this classifier, sparse matrices, feature unions, serialization, probabilistic APIs, sample weights, partial fit, and the broader scikit-learn estimator protocol are still open.
