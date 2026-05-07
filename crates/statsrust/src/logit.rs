use numrs_core::{Array, NumRsError, Result};

#[derive(Debug, Clone, PartialEq)]
pub struct LogitModel {
    coefficients: Array<f64>,
    intercept: bool,
    metrics: BinaryClassificationMetrics,
    iterations: usize,
    converged: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LogitOptions {
    pub max_iterations: usize,
    pub tolerance: f64,
    pub l2_penalty: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BinaryClassificationMetrics {
    pub observations: usize,
    pub features: usize,
    pub rank_parameters: usize,
    pub log_likelihood: f64,
    pub null_log_likelihood: f64,
    pub deviance: f64,
    pub pseudo_r_squared: f64,
    pub accuracy: f64,
}

pub struct LogitFit;

impl Default for LogitOptions {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            tolerance: 1e-10,
            l2_penalty: 0.0,
        }
    }
}

impl LogitFit {
    pub fn fit(x: &Array<f64>, y: &Array<f64>, intercept: bool) -> Result<LogitModel> {
        Self::fit_with_options(x, y, intercept, LogitOptions::default())
    }

    pub fn fit_with_options(
        x: &Array<f64>,
        y: &Array<f64>,
        intercept: bool,
        options: LogitOptions,
    ) -> Result<LogitModel> {
        validate_options(options)?;
        validate_binary_xy(x, y)?;

        let rows = x.shape()[0];
        let raw_features = x.shape()[1];
        let features = raw_features + usize::from(intercept);

        if rows <= features {
            return Err(NumRsError::InvalidShape(format!(
                "Logit needs observations > parameters, got observations={rows}, parameters={features}"
            )));
        }

        let design = design_matrix(x, intercept);
        let mut beta = vec![0.0; features];
        let mut probabilities = vec![0.5; rows];
        let mut converged = false;
        let mut iterations = 0usize;

        for iteration in 1..=options.max_iterations {
            iterations = iteration;
            predict_probabilities_raw(&design, &beta, rows, features, &mut probabilities);
            let (hessian, gradient) = normal_equation_step(StepInput {
                design: &design,
                y: y.as_slice(),
                probabilities: &probabilities,
                beta: &beta,
                rows,
                features,
                intercept,
                l2_penalty: options.l2_penalty,
            });
            let delta = solve_linear_system(hessian, gradient)?;
            let max_delta = delta
                .iter()
                .fold(0.0_f64, |acc, value| acc.max(value.abs()));

            for (coefficient, step) in beta.iter_mut().zip(delta.iter()) {
                *coefficient += step;
            }

            if max_delta <= options.tolerance {
                converged = true;
                break;
            }
        }

        predict_probabilities_raw(&design, &beta, rows, features, &mut probabilities);
        let metrics = metrics(y.as_slice(), &probabilities, rows, features);

        Ok(LogitModel {
            coefficients: Array::from_vec(vec![features], beta)?,
            intercept,
            metrics,
            iterations,
            converged,
        })
    }
}

impl LogitModel {
    pub fn coefficients(&self) -> &Array<f64> {
        &self.coefficients
    }

    pub fn intercept_enabled(&self) -> bool {
        self.intercept
    }

    pub fn metrics(&self) -> BinaryClassificationMetrics {
        self.metrics
    }

    pub fn iterations(&self) -> usize {
        self.iterations
    }

    pub fn converged(&self) -> bool {
        self.converged
    }

    pub fn predict_proba(&self, x: &Array<f64>) -> Result<Array<f64>> {
        validate_predict_x(x, self.coefficients.len(), self.intercept)?;
        let rows = x.shape()[0];
        let design = design_matrix(x, self.intercept);
        let mut probabilities = vec![0.0; rows];
        predict_probabilities_raw(
            &design,
            self.coefficients.as_slice(),
            rows,
            self.coefficients.len(),
            &mut probabilities,
        );
        Array::from_vec(vec![rows], probabilities)
    }

    pub fn predict_classes(&self, x: &Array<f64>, threshold: f64) -> Result<Array<u8>> {
        if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
            return Err(NumRsError::InvalidShape(format!(
                "predict_classes threshold must be finite and in [0, 1], got {threshold}"
            )));
        }
        let probabilities = self.predict_proba(x)?;
        Array::from_vec(
            probabilities.shape().to_vec(),
            probabilities
                .as_slice()
                .iter()
                .map(|probability| u8::from(*probability >= threshold))
                .collect(),
        )
    }
}

fn validate_options(options: LogitOptions) -> Result<()> {
    if options.max_iterations == 0 {
        return Err(NumRsError::InvalidShape(
            "Logit max_iterations must be positive".to_string(),
        ));
    }
    if !options.tolerance.is_finite() || options.tolerance <= 0.0 {
        return Err(NumRsError::InvalidShape(
            "Logit tolerance must be finite and positive".to_string(),
        ));
    }
    if !options.l2_penalty.is_finite() || options.l2_penalty < 0.0 {
        return Err(NumRsError::InvalidShape(
            "Logit l2_penalty must be finite and non-negative".to_string(),
        ));
    }
    Ok(())
}

fn validate_binary_xy(x: &Array<f64>, y: &Array<f64>) -> Result<()> {
    if x.ndim() != 2 {
        return Err(NumRsError::InvalidShape(format!(
            "Logit expected 2-D x, got shape {:?}",
            x.shape()
        )));
    }
    if y.ndim() != 1 {
        return Err(NumRsError::InvalidShape(format!(
            "Logit expected 1-D y, got shape {:?}",
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
    if y.as_slice()
        .iter()
        .any(|target| !target.is_finite() || (*target != 0.0 && *target != 1.0))
    {
        return Err(NumRsError::InvalidShape(
            "Logit target values must be finite 0/1 labels".to_string(),
        ));
    }
    let positives = y.as_slice().iter().filter(|target| **target == 1.0).count();
    if positives == 0 || positives == y.len() {
        return Err(NumRsError::InvalidShape(
            "Logit needs both positive and negative labels".to_string(),
        ));
    }
    Ok(())
}

fn validate_predict_x(x: &Array<f64>, coefficient_count: usize, intercept: bool) -> Result<()> {
    if x.ndim() != 2 {
        return Err(NumRsError::InvalidShape(format!(
            "predict expected 2-D x, got shape {:?}",
            x.shape()
        )));
    }
    let expected_features = coefficient_count - usize::from(intercept);
    if x.shape()[1] != expected_features {
        return Err(NumRsError::InvalidShape(format!(
            "predict expected {expected_features} features, got {}",
            x.shape()[1]
        )));
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

struct StepInput<'a> {
    design: &'a [f64],
    y: &'a [f64],
    probabilities: &'a [f64],
    beta: &'a [f64],
    rows: usize,
    features: usize,
    intercept: bool,
    l2_penalty: f64,
}

fn normal_equation_step(input: StepInput<'_>) -> (Vec<f64>, Vec<f64>) {
    let StepInput {
        design,
        y,
        probabilities,
        beta,
        rows,
        features,
        intercept,
        l2_penalty,
    } = input;

    let mut hessian = vec![0.0; features * features];
    let mut gradient = vec![0.0; features];

    for row in 0..rows {
        let base = row * features;
        let probability = probabilities[row];
        let weight = (probability * (1.0 - probability)).max(1e-12);
        let residual = y[row] - probability;

        for i in 0..features {
            let xi = design[base + i];
            gradient[i] += xi * residual;
            for j in i..features {
                hessian[i * features + j] += weight * xi * design[base + j];
            }
        }
    }

    for i in 0..features {
        for j in 0..i {
            hessian[i * features + j] = hessian[j * features + i];
        }
    }

    for i in usize::from(intercept)..features {
        hessian[i * features + i] += l2_penalty;
        gradient[i] -= l2_penalty * beta[i];
    }

    (hessian, gradient)
}

fn predict_probabilities_raw(
    design: &[f64],
    beta: &[f64],
    rows: usize,
    features: usize,
    out: &mut [f64],
) {
    for (row, probability) in out.iter_mut().enumerate().take(rows) {
        let base = row * features;
        let mut eta = 0.0;
        for col in 0..features {
            eta += design[base + col] * beta[col];
        }
        *probability = sigmoid(eta);
    }
}

fn sigmoid(value: f64) -> f64 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp_value = value.exp();
        exp_value / (1.0 + exp_value)
    }
}

fn metrics(
    y: &[f64],
    probabilities: &[f64],
    rows: usize,
    features: usize,
) -> BinaryClassificationMetrics {
    let log_likelihood = log_likelihood(y, probabilities);
    let positive_rate = y.iter().sum::<f64>() / rows as f64;
    let null_probability = positive_rate.clamp(1e-15, 1.0 - 1e-15);
    let null_log_likelihood = y
        .iter()
        .map(|target| {
            target * null_probability.ln() + (1.0 - target) * (1.0 - null_probability).ln()
        })
        .sum::<f64>();
    let correct = y
        .iter()
        .zip(probabilities.iter())
        .filter(|(target, probability)| u8::from(**probability >= 0.5) as f64 == **target)
        .count();

    BinaryClassificationMetrics {
        observations: rows,
        features: features - 1,
        rank_parameters: features,
        log_likelihood,
        null_log_likelihood,
        deviance: -2.0 * log_likelihood,
        pseudo_r_squared: if null_log_likelihood == 0.0 {
            1.0
        } else {
            1.0 - log_likelihood / null_log_likelihood
        },
        accuracy: correct as f64 / rows as f64,
    }
}

fn log_likelihood(y: &[f64], probabilities: &[f64]) -> f64 {
    y.iter()
        .zip(probabilities.iter())
        .map(|(target, probability)| {
            let probability = probability.clamp(1e-15, 1.0 - 1e-15);
            target * probability.ln() + (1.0 - target) * (1.0 - probability).ln()
        })
        .sum()
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
                "Logit Hessian matrix is singular".to_string(),
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
