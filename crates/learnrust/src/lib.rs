use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

pub type Result<T> = std::result::Result<T, LearnRustError>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LearnRustError {
    EmptyInput,
    ShapeMismatch { expected: usize, actual: usize },
    InvalidShape(String),
    NotFitted,
}

impl fmt::Display for LearnRustError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LearnRustError::EmptyInput => write!(f, "input must not be empty"),
            LearnRustError::ShapeMismatch { expected, actual } => {
                write!(f, "shape mismatch: expected {expected}, got {actual}")
            }
            LearnRustError::InvalidShape(message) => write!(f, "{message}"),
            LearnRustError::NotFitted => write!(f, "model has not been fitted"),
        }
    }
}

impl Error for LearnRustError {}

#[derive(Debug, Clone, PartialEq)]
pub struct DenseMatrix {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

impl DenseMatrix {
    pub fn from_vec(rows: usize, cols: usize, data: Vec<f64>) -> Result<Self> {
        let expected = rows.checked_mul(cols).ok_or_else(|| {
            LearnRustError::InvalidShape(format!("shape [{rows}, {cols}] overflows"))
        })?;
        if expected != data.len() {
            return Err(LearnRustError::ShapeMismatch {
                expected,
                actual: data.len(),
            });
        }
        Ok(Self { rows, cols, data })
    }

    pub fn from_shape_fn<F>(rows: usize, cols: usize, mut f: F) -> Result<Self>
    where
        F: FnMut(usize, usize) -> f64,
    {
        let len = rows.checked_mul(cols).ok_or_else(|| {
            LearnRustError::InvalidShape(format!("shape [{rows}, {cols}] overflows"))
        })?;
        let mut data = Vec::with_capacity(len);
        for row in 0..rows {
            for col in 0..cols {
                data.push(f(row, col));
            }
        }
        Self::from_vec(rows, cols, data)
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }

    pub fn row(&self, row: usize) -> Result<&[f64]> {
        if row >= self.rows {
            return Err(LearnRustError::InvalidShape(format!(
                "row {row} out of bounds for {} rows",
                self.rows
            )));
        }
        let start = row * self.cols;
        Ok(&self.data[start..start + self.cols])
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct StandardScaler {
    mean: Vec<f64>,
    scale: Vec<f64>,
}

impl StandardScaler {
    pub fn fit(matrix: &DenseMatrix) -> Result<Self> {
        if matrix.rows == 0 || matrix.cols == 0 {
            return Err(LearnRustError::EmptyInput);
        }
        let mut mean = vec![0.0; matrix.cols];
        for row in matrix.data.chunks_exact(matrix.cols) {
            for (col, value) in row.iter().enumerate() {
                mean[col] += *value;
            }
        }
        let inv_rows = 1.0 / matrix.rows as f64;
        for value in &mut mean {
            *value *= inv_rows;
        }

        let mut var = vec![0.0; matrix.cols];
        for row in matrix.data.chunks_exact(matrix.cols) {
            for (col, value) in row.iter().enumerate() {
                let centered = *value - mean[col];
                var[col] += centered * centered;
            }
        }
        let scale = var
            .into_iter()
            .map(|value| {
                let scale = (value * inv_rows).sqrt();
                if scale == 0.0 {
                    1.0
                } else {
                    scale
                }
            })
            .collect();
        Ok(Self { mean, scale })
    }

    pub fn mean(&self) -> &[f64] {
        &self.mean
    }

    pub fn scale(&self) -> &[f64] {
        &self.scale
    }

    pub fn transform(&self, matrix: &DenseMatrix) -> Result<DenseMatrix> {
        if matrix.cols != self.mean.len() {
            return Err(LearnRustError::ShapeMismatch {
                expected: self.mean.len(),
                actual: matrix.cols,
            });
        }
        let mut out = Vec::with_capacity(matrix.data.len());
        for row in matrix.data.chunks_exact(matrix.cols) {
            for (col, value) in row.iter().enumerate() {
                out.push((*value - self.mean[col]) / self.scale[col]);
            }
        }
        DenseMatrix::from_vec(matrix.rows, matrix.cols, out)
    }

    pub fn fit_transform(matrix: &DenseMatrix) -> Result<(Self, DenseMatrix)> {
        let scaler = Self::fit(matrix)?;
        let transformed = scaler.transform(matrix)?;
        Ok((scaler, transformed))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct NearestCentroid {
    labels: Vec<i64>,
    centroids: DenseMatrix,
}

impl NearestCentroid {
    pub fn fit(matrix: &DenseMatrix, labels: &[i64]) -> Result<Self> {
        if matrix.rows == 0 || matrix.cols == 0 {
            return Err(LearnRustError::EmptyInput);
        }
        if matrix.rows != labels.len() {
            return Err(LearnRustError::ShapeMismatch {
                expected: matrix.rows,
                actual: labels.len(),
            });
        }

        let mut by_label = BTreeMap::<i64, (usize, Vec<f64>)>::new();
        for (row_idx, row) in matrix.data.chunks_exact(matrix.cols).enumerate() {
            let (_, sums) = by_label
                .entry(labels[row_idx])
                .or_insert_with(|| (0, vec![0.0; matrix.cols]));
            for (col, value) in row.iter().enumerate() {
                sums[col] += *value;
            }
            by_label.get_mut(&labels[row_idx]).expect("entry exists").0 += 1;
        }

        let mut out_labels = Vec::with_capacity(by_label.len());
        let mut centroids = Vec::with_capacity(by_label.len() * matrix.cols);
        for (label, (count, mut sums)) in by_label {
            if count == 0 {
                return Err(LearnRustError::EmptyInput);
            }
            let inv_count = 1.0 / count as f64;
            for value in &mut sums {
                *value *= inv_count;
            }
            out_labels.push(label);
            centroids.extend(sums);
        }
        let centroids = DenseMatrix::from_vec(out_labels.len(), matrix.cols, centroids)?;
        Ok(Self {
            labels: out_labels,
            centroids,
        })
    }

    pub fn labels(&self) -> &[i64] {
        &self.labels
    }

    pub fn centroids(&self) -> &DenseMatrix {
        &self.centroids
    }

    pub fn predict(&self, matrix: &DenseMatrix) -> Result<Vec<i64>> {
        if self.labels.is_empty() {
            return Err(LearnRustError::NotFitted);
        }
        if matrix.cols != self.centroids.cols {
            return Err(LearnRustError::ShapeMismatch {
                expected: self.centroids.cols,
                actual: matrix.cols,
            });
        }

        let mut predictions = Vec::with_capacity(matrix.rows);
        for row in matrix.data.chunks_exact(matrix.cols) {
            let mut best_label = self.labels[0];
            let mut best_distance = f64::INFINITY;
            for (label_idx, centroid) in self.centroids.data.chunks_exact(matrix.cols).enumerate() {
                let mut distance = 0.0;
                for (value, center) in row.iter().zip(centroid.iter()) {
                    let diff = *value - *center;
                    distance += diff * diff;
                }
                if distance < best_distance {
                    best_distance = distance;
                    best_label = self.labels[label_idx];
                }
            }
            predictions.push(best_label);
        }
        Ok(predictions)
    }

    pub fn fit_predict(matrix: &DenseMatrix, labels: &[i64]) -> Result<Vec<i64>> {
        Self::fit(matrix, labels)?.predict(matrix)
    }
}

pub fn accuracy_score(y_true: &[i64], y_pred: &[i64]) -> Result<f64> {
    if y_true.len() != y_pred.len() {
        return Err(LearnRustError::ShapeMismatch {
            expected: y_true.len(),
            actual: y_pred.len(),
        });
    }
    if y_true.is_empty() {
        return Err(LearnRustError::EmptyInput);
    }
    let correct = y_true
        .iter()
        .zip(y_pred.iter())
        .filter(|(left, right)| left == right)
        .count();
    Ok(correct as f64 / y_true.len() as f64)
}

pub fn confusion_matrix(y_true: &[i64], y_pred: &[i64], labels: &[i64]) -> Result<Vec<usize>> {
    if y_true.len() != y_pred.len() {
        return Err(LearnRustError::ShapeMismatch {
            expected: y_true.len(),
            actual: y_pred.len(),
        });
    }
    if labels.is_empty() {
        return Err(LearnRustError::EmptyInput);
    }
    let mut positions = BTreeMap::new();
    for (idx, label) in labels.iter().copied().enumerate() {
        positions.insert(label, idx);
    }
    let mut out = vec![0usize; labels.len() * labels.len()];
    for (actual, predicted) in y_true.iter().zip(y_pred.iter()) {
        let Some(row) = positions.get(actual).copied() else {
            continue;
        };
        let Some(col) = positions.get(predicted).copied() else {
            continue;
        };
        out[row * labels.len() + col] += 1;
    }
    Ok(out)
}

pub fn inferred_labels(y_true: &[i64], y_pred: &[i64]) -> Vec<i64> {
    let mut labels = BTreeMap::new();
    for label in y_true.iter().chain(y_pred.iter()).copied() {
        labels.insert(label, ());
    }
    labels.into_keys().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_scaler_matches_population_variance() {
        let matrix = DenseMatrix::from_vec(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let (scaler, transformed) = StandardScaler::fit_transform(&matrix).unwrap();
        assert_eq!(scaler.mean(), &[3.0, 4.0]);
        let expected_scale = (8.0_f64 / 3.0).sqrt();
        assert!((scaler.scale()[0] - expected_scale).abs() < 1e-12);
        assert!((scaler.scale()[1] - expected_scale).abs() < 1e-12);
        assert!((transformed.as_slice()[0] + 1.224744871391589).abs() < 1e-12);
        assert!(transformed.as_slice()[2].abs() < 1e-12);
    }

    #[test]
    fn nearest_centroid_predicts_by_squared_distance() {
        let matrix =
            DenseMatrix::from_vec(4, 2, vec![0.0, 0.0, 0.0, 2.0, 10.0, 10.0, 10.0, 12.0]).unwrap();
        let labels = vec![0, 0, 1, 1];
        let model = NearestCentroid::fit(&matrix, &labels).unwrap();
        assert_eq!(model.labels(), &[0, 1]);
        assert_eq!(model.centroids().as_slice(), &[0.0, 1.0, 10.0, 11.0]);

        let test = DenseMatrix::from_vec(3, 2, vec![0.0, 0.5, 8.0, 11.0, 3.0, 1.0]).unwrap();
        assert_eq!(model.predict(&test).unwrap(), vec![0, 1, 0]);
    }

    #[test]
    fn metrics_match_sklearn_style_label_order() {
        let y_true = vec![2, 2, 1, 0, 1];
        let y_pred = vec![2, 1, 1, 0, 0];
        let labels = inferred_labels(&y_true, &y_pred);
        assert_eq!(labels, vec![0, 1, 2]);
        assert_eq!(accuracy_score(&y_true, &y_pred).unwrap(), 0.6);
        assert_eq!(
            confusion_matrix(&y_true, &y_pred, &labels).unwrap(),
            vec![1, 0, 0, 1, 1, 0, 0, 1, 1]
        );
    }
}
