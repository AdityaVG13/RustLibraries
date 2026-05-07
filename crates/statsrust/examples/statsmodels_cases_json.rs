use std::hint::black_box;
use std::time::{Duration, Instant};

use numrs_core::Array;
use statsrust::{LogitFit, LogitOptions, OlsFit, WlsFit};

struct Case {
    name: &'static str,
    millis: f64,
    checksum: f64,
}

fn median_ms(mut samples: Vec<Duration>) -> f64 {
    samples.sort_unstable();
    samples[samples.len() / 2].as_secs_f64() * 1000.0
}

fn bench<F>(mut f: F, rounds: usize) -> (f64, f64)
where
    F: FnMut() -> f64,
{
    let mut samples = Vec::with_capacity(rounds);
    let mut checksum = black_box(f());
    for _ in 0..rounds {
        let start = Instant::now();
        checksum = black_box(f());
        samples.push(start.elapsed());
    }
    (median_ms(samples), checksum)
}

fn regression_data(rows: usize) -> numrs_core::Result<(Array<f64>, Array<f64>, Array<f64>)> {
    let mut x = Vec::with_capacity(rows * 3);
    let mut y = Vec::with_capacity(rows);
    let mut weights = Vec::with_capacity(rows);

    for row in 0..rows {
        let x0 = (row % 97) as f64 / 48.0 - 1.0;
        let x1 = ((row * 7) % 113) as f64 / 56.0 - 1.0;
        let x2 = ((row * 13) % 89) as f64 / 44.0 - 1.0;
        let noise = (((row * 17) % 23) as f64 - 11.0) * 0.001;
        x.extend_from_slice(&[x0, x1, x2]);
        y.push(1.0 + 2.0 * x0 - 3.0 * x1 + 0.5 * x2 + noise);
        weights.push(0.5 + ((row * 5) % 17) as f64 / 16.0);
    }

    Ok((
        Array::from_vec(vec![rows, 3], x)?,
        Array::from_vec(vec![rows], y)?,
        Array::from_vec(vec![rows], weights)?,
    ))
}

fn logit_data(rows: usize) -> numrs_core::Result<(Array<f64>, Array<f64>)> {
    let mut x = Vec::with_capacity(rows * 2);
    let mut y = Vec::with_capacity(rows);

    for row in 0..rows {
        let x0 = (row % 101) as f64 / 50.0 - 1.0;
        let x1 = ((row * 11) % 97) as f64 / 48.0 - 1.0;
        let eta = -0.35 + 1.2 * x0 - 0.8 * x1;
        let probability = 1.0 / (1.0 + f64::exp(-eta));
        let draw = (((row * 37) % 101) as f64 + 0.5) / 101.0;
        x.extend_from_slice(&[x0, x1]);
        y.push(f64::from(draw < probability));
    }

    Ok((
        Array::from_vec(vec![rows, 2], x)?,
        Array::from_vec(vec![rows], y)?,
    ))
}

fn coefficient_checksum(coefficients: &Array<f64>) -> f64 {
    coefficients
        .as_slice()
        .iter()
        .enumerate()
        .map(|(idx, value)| (idx + 1) as f64 * value)
        .sum()
}

fn edge_checksum(values: &Array<f64>) -> f64 {
    let values = values.as_slice();
    values[0] + values[values.len() / 2] + values[values.len() - 1]
}

fn main() -> numrs_core::Result<()> {
    let mut cases = Vec::new();
    let (x, y, weights) = regression_data(2000)?;

    let (millis, checksum) = bench(
        || {
            let mut checksum = 0.0;
            for _ in 0..50 {
                let model = OlsFit::fit(&x, &y, true).unwrap();
                checksum += coefficient_checksum(model.coefficients());
            }
            checksum
        },
        5,
    );
    cases.push(Case {
        name: "statsmodels_ols_fit_2000x3",
        millis,
        checksum,
    });

    let ols_model = OlsFit::fit(&x, &y, true)?;
    let (millis, checksum) = bench(
        || {
            let mut checksum = 0.0;
            for _ in 0..5_000 {
                checksum += edge_checksum(&ols_model.predict(&x).unwrap());
            }
            checksum
        },
        5,
    );
    cases.push(Case {
        name: "statsmodels_ols_predict_2000x3",
        millis,
        checksum,
    });

    let (millis, checksum) = bench(
        || {
            let mut checksum = 0.0;
            for _ in 0..50 {
                let model = WlsFit::fit(&x, &y, &weights, true).unwrap();
                checksum += coefficient_checksum(model.coefficients());
            }
            checksum
        },
        5,
    );
    cases.push(Case {
        name: "statsmodels_wls_fit_2000x3",
        millis,
        checksum,
    });

    let (logit_x, logit_y) = logit_data(800)?;
    let options = LogitOptions {
        max_iterations: 100,
        tolerance: 1e-10,
        l2_penalty: 0.0,
    };
    let (millis, checksum) = bench(
        || {
            let mut checksum = 0.0;
            for _ in 0..10 {
                let model = LogitFit::fit_with_options(&logit_x, &logit_y, true, options).unwrap();
                checksum += coefficient_checksum(model.coefficients());
            }
            checksum
        },
        5,
    );
    cases.push(Case {
        name: "statsmodels_logit_fit_800x2",
        millis,
        checksum,
    });

    print!("{{\"engine\":\"statsrust\",\"cases\":[");
    for (idx, case) in cases.iter().enumerate() {
        if idx > 0 {
            print!(",");
        }
        print!(
            "{{\"name\":\"{}\",\"millis\":{:.6},\"checksum\":{:.12}}}",
            case.name, case.millis, case.checksum
        );
    }
    println!("]}}");

    Ok(())
}
