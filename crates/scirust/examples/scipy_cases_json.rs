use std::hint::black_box;
use std::time::{Duration, Instant};

use scirust::{integrate, optimize, root};

const ASV_CUMULATIVE_1D_REPETITIONS: usize = 1_000;

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

fn integration_data(samples: usize) -> (Vec<f64>, Vec<f64>, f64) {
    let lower = 0.0;
    let upper = std::f64::consts::PI;
    let dx = (upper - lower) / (samples - 1) as f64;
    let mut x = Vec::with_capacity(samples);
    let mut y = Vec::with_capacity(samples);
    for idx in 0..samples {
        let value = lower + idx as f64 * dx;
        x.push(value);
        y.push(value.sin() + 0.05 * (3.0 * value).cos());
    }
    (x, y, dx)
}

fn cumulative_simpson_asv_data() -> (Vec<f64>, Vec<f64>, f64) {
    let lower = 0.0;
    let upper = 5.0;
    let samples = 1000usize;
    let dx = (upper - lower) / (samples - 1) as f64;
    let mut y = Vec::with_capacity(samples);
    for idx in 0..samples {
        let x = lower + idx as f64 * dx;
        y.push((2.0 * std::f64::consts::PI * x).sin());
    }

    let mut y2 = Vec::with_capacity(100 * 100 * samples);
    for _ in 0..10_000 {
        y2.extend_from_slice(&y);
    }

    (y, y2, dx)
}

fn objective(x: f64) -> f64 {
    let centered = x - 1.2345;
    centered * centered + 2.0
}

fn root_function(x: f64) -> f64 {
    x * x * x - x - 2.0
}

fn scipy_asv_f2(x: f64) -> f64 {
    x * x - 1.0
}

fn main() -> numrs_core::Result<()> {
    let mut cases = Vec::new();
    let (x, y, dx) = integration_data(10_001);

    let (millis, checksum) = bench(
        || {
            let mut checksum = 0.0;
            for _ in 0..2_000 {
                checksum += integrate::trapezoid(&y, Some(&x)).unwrap();
            }
            checksum
        },
        5,
    );
    cases.push(Case {
        name: "scipy_integrate_trapezoid_10001",
        millis,
        checksum,
    });

    let (millis, checksum) = bench(
        || {
            let mut checksum = 0.0;
            for _ in 0..2_000 {
                checksum += integrate::simpson_uniform(&y, dx).unwrap();
            }
            checksum
        },
        5,
    );
    cases.push(Case {
        name: "scipy_integrate_simpson_10001",
        millis,
        checksum,
    });

    let (asv_y, asv_y2, asv_dx) = cumulative_simpson_asv_data();
    let (millis, checksum) = bench(
        || {
            let mut checksum = 0.0;
            for _ in 0..ASV_CUMULATIVE_1D_REPETITIONS {
                let out = integrate::cumulative_simpson_uniform(&asv_y, asv_dx).unwrap();
                checksum += out.iter().sum::<f64>();
            }
            checksum
        },
        5,
    );
    cases.push(Case {
        name: "scipy_asv_cumulative_simpson_1d_1000",
        millis,
        checksum,
    });

    let (millis, checksum) = bench(
        || {
            let out =
                integrate::cumulative_simpson_uniform_axis_last(&asv_y2, 1000, asv_dx).unwrap();
            out.iter().sum()
        },
        5,
    );
    cases.push(Case {
        name: "scipy_asv_cumulative_simpson_multid_100x100x1000",
        millis,
        checksum,
    });

    let (millis, checksum) = bench(
        || {
            let mut checksum = 0.0;
            for _ in 0..10_000 {
                let result =
                    optimize::minimize_scalar_bounded(objective, -5.0, 5.0, 1e-10, 100).unwrap();
                checksum += result.x + result.fun;
            }
            checksum
        },
        5,
    );
    cases.push(Case {
        name: "scipy_optimize_minimize_scalar_bounded",
        millis,
        checksum,
    });

    let (millis, checksum) = bench(
        || {
            let mut checksum = 0.0;
            for _ in 0..10_000 {
                let result = root::bisect(root_function, 1.0, 2.0, 1e-12, 100).unwrap();
                checksum += result.root + result.fun.abs();
            }
            checksum
        },
        5,
    );
    cases.push(Case {
        name: "scipy_optimize_bisect",
        millis,
        checksum,
    });

    let (millis, checksum) = bench(
        || {
            let mut checksum = 0.0;
            for _ in 0..10_000 {
                let result = root::bisect(scipy_asv_f2, 0.5, 3.0_f64.sqrt(), 1e-12, 100).unwrap();
                checksum += result.root + result.fun.abs();
            }
            checksum
        },
        5,
    );
    cases.push(Case {
        name: "scipy_asv_zeros_f2_bisect",
        millis,
        checksum,
    });

    let (millis, checksum) = bench(
        || {
            let mut checksum = 0.0;
            for _ in 0..10_000 {
                let result = root::brentq(root_function, 1.0, 2.0, 1e-12, 1e-12, 100).unwrap();
                checksum += result.root + result.fun.abs();
            }
            checksum
        },
        5,
    );
    cases.push(Case {
        name: "scipy_optimize_brentq",
        millis,
        checksum,
    });

    let (millis, checksum) = bench(
        || {
            let mut checksum = 0.0;
            for _ in 0..10_000 {
                let result =
                    root::brentq(scipy_asv_f2, 0.5, 3.0_f64.sqrt(), 1e-12, 1e-12, 100).unwrap();
                checksum += result.root + result.fun.abs();
            }
            checksum
        },
        5,
    );
    cases.push(Case {
        name: "scipy_asv_zeros_f2_brentq",
        millis,
        checksum,
    });

    print!("{{\"engine\":\"scirust\",\"cases\":[");
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
