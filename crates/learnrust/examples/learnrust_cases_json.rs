use std::time::{Duration, Instant};

use learnrust::{
    accuracy_score, confusion_matrix, inferred_labels, DenseMatrix, NearestCentroid, Result,
    StandardScaler,
};

fn median_ms(mut samples: Vec<Duration>) -> f64 {
    samples.sort_unstable();
    samples[samples.len() / 2].as_secs_f64() * 1000.0
}

fn bench<F>(mut f: F, rounds: usize) -> (f64, f64)
where
    F: FnMut() -> f64,
{
    let mut checksum = f();
    let mut samples = Vec::with_capacity(rounds);
    for _ in 0..rounds {
        let start = Instant::now();
        checksum = f();
        samples.push(start.elapsed());
    }
    (median_ms(samples), checksum)
}

fn scaler_matrix(rows: usize, cols: usize) -> Result<DenseMatrix> {
    DenseMatrix::from_shape_fn(rows, cols, |row, col| {
        let base = ((row * 31 + col * 17) % 1009) as f64 / 37.0;
        base + col as f64 * 0.25 - (row % 11) as f64 * 0.03
    })
}

fn classification_matrix(
    rows: usize,
    cols: usize,
    classes: usize,
) -> Result<(DenseMatrix, Vec<i64>)> {
    let labels = (0..rows)
        .map(|row| (row % classes) as i64)
        .collect::<Vec<_>>();
    let matrix = DenseMatrix::from_shape_fn(rows, cols, |row, col| {
        let label = (row % classes) as f64;
        let jitter = ((row * 19 + col * 23) % 97) as f64 / 500.0;
        label * 4.0 + col as f64 * 0.2 + jitter
    })?;
    Ok((matrix, labels))
}

fn matrix_checksum(matrix: &DenseMatrix) -> f64 {
    let values = matrix.as_slice();
    if values.is_empty() {
        return 0.0;
    }
    let mid = values.len() / 2;
    values[0] + values[mid] + values[values.len() - 1]
}

fn labels_checksum(values: &[i64]) -> f64 {
    values
        .iter()
        .enumerate()
        .map(|(idx, value)| ((idx % 31) as f64 + 1.0) * *value as f64)
        .sum()
}

fn counts_checksum(values: &[usize]) -> f64 {
    values
        .iter()
        .enumerate()
        .map(|(idx, value)| ((idx % 17) as f64 + 1.0) * *value as f64)
        .sum()
}

fn main() -> Result<()> {
    let scaler_input = scaler_matrix(200_000, 12)?;
    let (millis, checksum) = bench(
        || {
            let (_, transformed) = StandardScaler::fit_transform(&scaler_input).unwrap();
            matrix_checksum(&transformed)
        },
        9,
    );
    let scaler_case = (millis, checksum);

    let (train, train_labels) = classification_matrix(120_000, 8, 6)?;
    let (test, test_labels) = classification_matrix(40_000, 8, 6)?;
    let (millis, checksum) = bench(
        || {
            let model = NearestCentroid::fit(&train, &train_labels).unwrap();
            let predicted = model.predict(&test).unwrap();
            labels_checksum(&predicted) + accuracy_score(&test_labels, &predicted).unwrap()
        },
        9,
    );
    let nearest_centroid_case = (millis, checksum);

    let metric_len = 1_000_000usize;
    let y_true = (0..metric_len)
        .map(|idx| (idx % 8) as i64)
        .collect::<Vec<_>>();
    let y_pred = (0..metric_len)
        .map(|idx| ((idx + usize::from(idx % 19 == 0)) % 8) as i64)
        .collect::<Vec<_>>();
    let labels = inferred_labels(&y_true, &y_pred);
    let (millis, checksum) = bench(
        || {
            let accuracy = accuracy_score(&y_true, &y_pred).unwrap();
            let matrix = confusion_matrix(&y_true, &y_pred, &labels).unwrap();
            accuracy + counts_checksum(&matrix)
        },
        9,
    );
    let metrics_case = (millis, checksum);

    println!(
        "{{\"engine\":\"learnrust\",\"cases\":[\
         {{\"name\":\"standard_scaler_fit_transform_200000x12\",\"millis\":{:.6},\"checksum\":{:.12}}},\
         {{\"name\":\"nearest_centroid_fit_predict_120000x8_40000x8_6\",\"millis\":{:.6},\"checksum\":{:.12}}},\
         {{\"name\":\"accuracy_confusion_1000000_8\",\"millis\":{:.6},\"checksum\":{:.12}}}\
         ]}}",
        scaler_case.0,
        scaler_case.1,
        nearest_centroid_case.0,
        nearest_centroid_case.1,
        metrics_case.0,
        metrics_case.1
    );
    Ok(())
}
