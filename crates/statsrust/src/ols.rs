use numrs_core::{Array, NumRsError, Result};

#[derive(Debug, Clone, PartialEq)]
pub struct OlsModel {
    coefficients: Array<f64>,
    intercept: bool,
    metrics: RegressionMetrics,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RegressionMetrics {
    pub observations: usize,
    pub features: usize,
    pub rank_parameters: usize,
    pub rss: f64,
    pub tss: f64,
    pub r_squared: f64,
    pub adjusted_r_squared: f64,
    pub mse: f64,
}

pub struct OlsFit;
pub struct WlsFit;

impl OlsFit {
    pub fn fit(x: &Array<f64>, y: &Array<f64>, intercept: bool) -> Result<OlsModel> {
        validate_xy(x, y)?;
        let rows = x.shape()[0];
        let raw_features = x.shape()[1];
        let features = raw_features + usize::from(intercept);

        if rows <= features {
            return Err(NumRsError::InvalidShape(format!(
                "OLS needs observations > parameters, got observations={rows}, parameters={features}"
            )));
        }

        let design = design_matrix(x, intercept);
        let xtx = gram_matrix(&design, rows, features);
        let xty = xt_y(&design, y.as_slice(), rows, features);
        let beta = solve_linear_system(xtx, xty)?;
        let predictions = predict_raw(&design, &beta, rows, features);
        let metrics = metrics(y.as_slice(), &predictions, rows, features);

        Ok(OlsModel {
            coefficients: Array::from_vec(vec![features], beta)?,
            intercept,
            metrics,
        })
    }
}

impl WlsFit {
    pub fn fit(
        x: &Array<f64>,
        y: &Array<f64>,
        weights: &Array<f64>,
        intercept: bool,
    ) -> Result<OlsModel> {
        validate_xy(x, y)?;
        validate_weights(x.shape()[0], weights)?;
        let rows = x.shape()[0];
        let raw_features = x.shape()[1];
        let features = raw_features + usize::from(intercept);

        if rows <= features {
            return Err(NumRsError::InvalidShape(format!(
                "WLS needs observations > parameters, got observations={rows}, parameters={features}"
            )));
        }

        let design = design_matrix(x, intercept);
        let xtx = weighted_gram_matrix(&design, weights.as_slice(), rows, features);
        let xty = weighted_xt_y(&design, y.as_slice(), weights.as_slice(), rows, features);
        let beta = solve_linear_system(xtx, xty)?;
        let predictions = predict_raw(&design, &beta, rows, features);
        let metrics = weighted_metrics(
            y.as_slice(),
            &predictions,
            weights.as_slice(),
            rows,
            features,
        );

        Ok(OlsModel {
            coefficients: Array::from_vec(vec![features], beta)?,
            intercept,
            metrics,
        })
    }
}

impl OlsModel {
    pub fn coefficients(&self) -> &Array<f64> {
        &self.coefficients
    }

    pub fn intercept_enabled(&self) -> bool {
        self.intercept
    }

    pub fn metrics(&self) -> RegressionMetrics {
        self.metrics
    }

    pub fn predict(&self, x: &Array<f64>) -> Result<Array<f64>> {
        if x.ndim() != 2 {
            return Err(NumRsError::InvalidShape(format!(
                "predict expected 2-D x, got shape {:?}",
                x.shape()
            )));
        }
        let expected_features = self.coefficients.len() - usize::from(self.intercept);
        if x.shape()[1] != expected_features {
            return Err(NumRsError::InvalidShape(format!(
                "predict expected {expected_features} features, got {}",
                x.shape()[1]
            )));
        }

        let predictions = predict_from_features(x, self.coefficients.as_slice(), self.intercept);
        let rows = x.shape()[0];
        Array::from_vec(vec![rows], predictions)
    }
}

fn validate_xy(x: &Array<f64>, y: &Array<f64>) -> Result<()> {
    if x.ndim() != 2 {
        return Err(NumRsError::InvalidShape(format!(
            "OLS expected 2-D x, got shape {:?}",
            x.shape()
        )));
    }
    if y.ndim() != 1 {
        return Err(NumRsError::InvalidShape(format!(
            "OLS expected 1-D y, got shape {:?}",
            y.shape()
        )));
    }
    if x.shape()[0] != y.shape()[0] {
        return Err(NumRsError::ShapeDataMismatch {
            shape: x.shape().to_vec(),
            expected: x.shape()[0],
            actual: y.shape()[0],
        });
    }
    Ok(())
}

fn validate_weights(rows: usize, weights: &Array<f64>) -> Result<()> {
    if weights.ndim() != 1 || weights.shape()[0] != rows {
        return Err(NumRsError::InvalidShape(format!(
            "WLS expected weights shape [{rows}], got {:?}",
            weights.shape()
        )));
    }
    if weights
        .as_slice()
        .iter()
        .any(|weight| !weight.is_finite() || *weight < 0.0)
    {
        return Err(NumRsError::InvalidShape(
            "WLS weights must be finite and non-negative".to_string(),
        ));
    }
    if weights.as_slice().iter().all(|weight| *weight == 0.0) {
        return Err(NumRsError::InvalidShape(
            "WLS needs at least one positive weight".to_string(),
        ));
    }
    Ok(())
}

fn design_matrix(x: &Array<f64>, intercept: bool) -> Vec<f64> {
    let rows = x.shape()[0];
    let raw_features = x.shape()[1];
    let features = raw_features + usize::from(intercept);
    let mut out = vec![0.0; rows * features];

    for row in 0..rows {
        let out_base = row * features;
        if intercept {
            out[out_base] = 1.0;
        }
        for col in 0..raw_features {
            out[out_base + col + usize::from(intercept)] = x.as_slice()[row * raw_features + col];
        }
    }

    out
}

fn gram_matrix(design: &[f64], rows: usize, features: usize) -> Vec<f64> {
    let mut xtx = vec![0.0; features * features];
    for row in 0..rows {
        let base = row * features;
        for i in 0..features {
            for j in i..features {
                xtx[i * features + j] += design[base + i] * design[base + j];
            }
        }
    }
    for i in 0..features {
        for j in 0..i {
            xtx[i * features + j] = xtx[j * features + i];
        }
    }
    xtx
}

fn xt_y(design: &[f64], y: &[f64], rows: usize, features: usize) -> Vec<f64> {
    let mut out = vec![0.0; features];
    for (row, target) in y.iter().copied().enumerate().take(rows) {
        let base = row * features;
        for col in 0..features {
            out[col] += design[base + col] * target;
        }
    }
    out
}

fn weighted_gram_matrix(design: &[f64], weights: &[f64], rows: usize, features: usize) -> Vec<f64> {
    let mut xtx = vec![0.0; features * features];
    for (row, weight) in weights.iter().copied().enumerate().take(rows) {
        let base = row * features;
        for i in 0..features {
            for j in i..features {
                xtx[i * features + j] += weight * design[base + i] * design[base + j];
            }
        }
    }
    for i in 0..features {
        for j in 0..i {
            xtx[i * features + j] = xtx[j * features + i];
        }
    }
    xtx
}

fn weighted_xt_y(
    design: &[f64],
    y: &[f64],
    weights: &[f64],
    rows: usize,
    features: usize,
) -> Vec<f64> {
    let mut out = vec![0.0; features];
    for row in 0..rows {
        let base = row * features;
        let weighted_target = weights[row] * y[row];
        for col in 0..features {
            out[col] += design[base + col] * weighted_target;
        }
    }
    out
}

fn predict_raw(design: &[f64], beta: &[f64], rows: usize, features: usize) -> Vec<f64> {
    let mut out = vec![0.0; rows];
    for (row, predicted) in out.iter_mut().enumerate().take(rows) {
        let base = row * features;
        *predicted = beta
            .iter()
            .enumerate()
            .map(|(col, coef)| design[base + col] * coef)
            .sum();
    }
    out
}

fn predict_from_features(x: &Array<f64>, beta: &[f64], intercept: bool) -> Vec<f64> {
    let rows = x.shape()[0];
    let raw_features = x.shape()[1];
    let mut out = vec![0.0; rows];
    let coefficient_offset = usize::from(intercept);

    for (row, predicted) in out.iter_mut().enumerate().take(rows) {
        let row_base = row * raw_features;
        let mut acc = if intercept { beta[0] } else { 0.0 };
        for col in 0..raw_features {
            acc += x.as_slice()[row_base + col] * beta[col + coefficient_offset];
        }
        *predicted = acc;
    }

    out
}

fn metrics(y: &[f64], predictions: &[f64], rows: usize, features: usize) -> RegressionMetrics {
    let mean = y.iter().sum::<f64>() / rows as f64;
    let rss = y
        .iter()
        .zip(predictions.iter())
        .map(|(actual, predicted)| {
            let residual = actual - predicted;
            residual * residual
        })
        .sum::<f64>();
    let tss = y
        .iter()
        .map(|actual| {
            let centered = actual - mean;
            centered * centered
        })
        .sum::<f64>();
    let r_squared = if tss == 0.0 { 1.0 } else { 1.0 - rss / tss };
    let adjusted_r_squared =
        1.0 - (1.0 - r_squared) * ((rows - 1) as f64 / (rows - features) as f64);

    RegressionMetrics {
        observations: rows,
        features: features - 1,
        rank_parameters: features,
        rss,
        tss,
        r_squared,
        adjusted_r_squared,
        mse: rss / (rows - features) as f64,
    }
}

fn weighted_metrics(
    y: &[f64],
    predictions: &[f64],
    weights: &[f64],
    rows: usize,
    features: usize,
) -> RegressionMetrics {
    let weight_sum = weights.iter().sum::<f64>();
    let mean = y
        .iter()
        .zip(weights.iter())
        .map(|(actual, weight)| actual * weight)
        .sum::<f64>()
        / weight_sum;
    let rss = y
        .iter()
        .zip(predictions.iter())
        .zip(weights.iter())
        .map(|((actual, predicted), weight)| {
            let residual = actual - predicted;
            weight * residual * residual
        })
        .sum::<f64>();
    let tss = y
        .iter()
        .zip(weights.iter())
        .map(|(actual, weight)| {
            let centered = actual - mean;
            weight * centered * centered
        })
        .sum::<f64>();
    let r_squared = if tss == 0.0 { 1.0 } else { 1.0 - rss / tss };
    let adjusted_r_squared =
        1.0 - (1.0 - r_squared) * ((rows - 1) as f64 / (rows - features) as f64);

    RegressionMetrics {
        observations: rows,
        features: features - 1,
        rank_parameters: features,
        rss,
        tss,
        r_squared,
        adjusted_r_squared,
        mse: rss / (rows - features) as f64,
    }
}

fn solve_linear_system(mut matrix: Vec<f64>, mut rhs: Vec<f64>) -> Result<Vec<f64>> {
    let n = rhs.len();
    for pivot in 0..n {
        let mut best = pivot;
        let mut best_value = matrix[pivot * n + pivot].abs();
        for row in pivot + 1..n {
            let value = matrix[row * n + pivot].abs();
            if value > best_value {
                best = row;
                best_value = value;
            }
        }

        if best_value <= f64::EPSILON {
            return Err(NumRsError::InvalidShape(
                "OLS normal equation matrix is singular".to_string(),
            ));
        }

        if best != pivot {
            for col in 0..n {
                matrix.swap(pivot * n + col, best * n + col);
            }
            rhs.swap(pivot, best);
        }

        let pivot_value = matrix[pivot * n + pivot];
        for col in pivot..n {
            matrix[pivot * n + col] /= pivot_value;
        }
        rhs[pivot] /= pivot_value;

        for row in 0..n {
            if row == pivot {
                continue;
            }
            let factor = matrix[row * n + pivot];
            if factor == 0.0 {
                continue;
            }
            for col in pivot..n {
                matrix[row * n + col] -= factor * matrix[pivot * n + col];
            }
            rhs[row] -= factor * rhs[pivot];
        }
    }
    Ok(rhs)
}
