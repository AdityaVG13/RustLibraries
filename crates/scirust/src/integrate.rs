use numrs_core::{NumRsError, Result};

pub fn trapezoid(y: &[f64], x: Option<&[f64]>) -> Result<f64> {
    if y.len() < 2 {
        return Ok(0.0);
    }
    match x {
        Some(x) => {
            validate_xy(y, x)?;
            Ok(y.windows(2)
                .zip(x.windows(2))
                .map(|(y, x)| 0.5 * (y[0] + y[1]) * (x[1] - x[0]))
                .sum())
        }
        None => Ok(y.windows(2).map(|y| 0.5 * (y[0] + y[1])).sum()),
    }
}

pub fn simpson_uniform(y: &[f64], dx: f64) -> Result<f64> {
    validate_dx(dx)?;
    if y.len() < 2 {
        return Ok(0.0);
    }
    if y.len() % 2 == 0 {
        return Err(NumRsError::InvalidShape(
            "simpson_uniform requires an odd number of samples".to_string(),
        ));
    }

    let mut acc = 0.0;
    let mut i = 0usize;
    while i + 2 < y.len() {
        acc += y[i] + 4.0 * y[i + 1] + y[i + 2];
        i += 2;
    }
    Ok(acc * dx / 3.0)
}

pub fn cumulative_simpson_uniform(y: &[f64], dx: f64) -> Result<Vec<f64>> {
    validate_dx(dx)?;
    if y.len() < 2 {
        return Ok(Vec::new());
    }

    let mut out = Vec::with_capacity(y.len() - 1);
    append_cumulative_simpson_uniform(y, dx, &mut out);
    Ok(out)
}

pub fn cumulative_simpson_uniform_axis_last(
    values: &[f64],
    row_len: usize,
    dx: f64,
) -> Result<Vec<f64>> {
    validate_dx(dx)?;
    if row_len == 0 || values.len() % row_len != 0 {
        return Err(NumRsError::InvalidShape(
            "values length must be a multiple of row_len".to_string(),
        ));
    }
    if row_len < 2 {
        return Ok(Vec::new());
    }

    let rows = values.len() / row_len;
    let mut out = Vec::with_capacity(rows * (row_len - 1));
    for row in values.chunks_exact(row_len) {
        append_cumulative_simpson_uniform(row, dx, &mut out);
    }
    Ok(out)
}

fn append_cumulative_simpson_uniform(y: &[f64], dx: f64, out: &mut Vec<f64>) {
    if y.len() == 2 {
        out.push(0.5 * (y[0] + y[1]) * dx);
        return;
    }

    let mut cumulative = 0.0;
    let mut idx = 0usize;
    while idx + 2 < y.len() {
        let (left, right) = simpson_pair_increments(y[idx], y[idx + 1], y[idx + 2], dx);
        cumulative += left;
        out.push(cumulative);
        cumulative += right;
        out.push(cumulative);
        idx += 2;
    }

    if y.len() % 2 == 0 {
        let tail = simpson_pair_increments(y[y.len() - 3], y[y.len() - 2], y[y.len() - 1], dx).1;
        out.push(cumulative + tail);
    }
}

fn simpson_pair_increments(y0: f64, y1: f64, y2: f64, dx: f64) -> (f64, f64) {
    (
        dx * (5.0 * y0 + 8.0 * y1 - y2) / 12.0,
        dx * (-y0 + 8.0 * y1 + 5.0 * y2) / 12.0,
    )
}

fn validate_dx(dx: f64) -> Result<()> {
    if !dx.is_finite() || dx <= 0.0 {
        return Err(NumRsError::InvalidShape(
            "dx must be finite and positive".to_string(),
        ));
    }
    Ok(())
}

fn validate_xy(y: &[f64], x: &[f64]) -> Result<()> {
    if x.len() != y.len() {
        return Err(NumRsError::ShapeDataMismatch {
            shape: vec![y.len()],
            expected: y.len(),
            actual: x.len(),
        });
    }
    if x.windows(2).any(|pair| pair[1] < pair[0]) {
        return Err(NumRsError::InvalidShape(
            "x coordinates must be sorted ascending".to_string(),
        ));
    }
    Ok(())
}
