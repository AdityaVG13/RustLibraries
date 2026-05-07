use scirust::integrate::{cumulative_simpson_uniform, cumulative_simpson_uniform_axis_last};
use scirust::integrate::{simpson_uniform, trapezoid};
use scirust::optimize::minimize_scalar_bounded;
use scirust::root::{bisect, brentq};

#[test]
fn minimizes_scalar_bounded_quadratic() {
    let result =
        minimize_scalar_bounded(|x| (x - 2.5).powi(2) + 1.0, -10.0, 10.0, 1e-10, 200).unwrap();

    assert!(result.converged);
    assert!((result.x - 2.5).abs() < 1e-6);
    assert!((result.fun - 1.0).abs() < 1e-9);
}

#[test]
fn integrates_with_trapezoid() {
    let x = [0.0, 0.5, 1.0];
    let y = [0.0, 0.25, 1.0];
    let result = trapezoid(&y, Some(&x)).unwrap();

    assert!((result - 0.375).abs() < 1e-12);
}

#[test]
fn integrates_with_uniform_simpson() {
    let y = [0.0, 0.25, 1.0];
    let result = simpson_uniform(&y, 0.5).unwrap();

    assert!((result - (1.0 / 3.0)).abs() < 1e-12);

    let cubic = [0.0_f64, 1.0, 8.0, 27.0, 64.0];
    let result = simpson_uniform(&cubic, 1.0).unwrap();
    assert!((result - 64.0).abs() < 1e-12);
}

#[test]
fn integrates_with_cumulative_uniform_simpson() {
    assert_eq!(
        cumulative_simpson_uniform(&[0.0, 1.0], 1.0).unwrap(),
        vec![0.5]
    );
    assert_close_vec(
        &cumulative_simpson_uniform(&[0.0, 1.0, 4.0], 1.0).unwrap(),
        &[1.0 / 3.0, 8.0 / 3.0],
    );
    assert_close_vec(
        &cumulative_simpson_uniform(&[0.0, 1.0, 4.0, 9.0], 1.0).unwrap(),
        &[1.0 / 3.0, 8.0 / 3.0, 9.0],
    );
    assert_close_vec(
        &cumulative_simpson_uniform(&[0.0, 1.0, 8.0, 27.0, 64.0], 1.0).unwrap(),
        &[0.0, 4.0, 20.0, 64.0],
    );

    let batched =
        cumulative_simpson_uniform_axis_last(&[0.0, 1.0, 4.0, 0.0, 1.0, 8.0], 3, 1.0).unwrap();
    assert_close_vec(&batched, &[1.0 / 3.0, 8.0 / 3.0, 0.0, 4.0]);
}

fn assert_close_vec(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert!((actual - expected).abs() < 1e-12, "{actual} != {expected}");
    }
}

#[test]
fn finds_roots_with_bisection() {
    let result = bisect(|x| x.powi(3) - 2.0, 0.0, 2.0, 1e-12, 200).unwrap();

    assert!(result.converged);
    assert!((result.root - 2.0_f64.cbrt()).abs() < 1e-10);
    assert!(result.fun.abs() < 1e-9);
}

#[test]
fn finds_roots_with_brentq() {
    let result = brentq(|x| x.cos() - x, 0.0, 1.0, 1e-12, 1e-12, 100).unwrap();

    assert!(result.converged);
    assert!(result.iterations < 10);
    assert!((result.root - 0.739_085_133_215_160_7).abs() < 1e-12);
    assert!(result.fun.abs() < 1e-12);
}

#[test]
fn rejects_unbracketed_roots() {
    let err = brentq(|x| x * x + 1.0, -1.0, 1.0, 1e-12, 0.0, 100).unwrap_err();
    assert!(matches!(err, numrs_core::NumRsError::InvalidShape(_)));
}
