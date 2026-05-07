use numrs_core::{Array, NumRsError};
use statsrust::{LogitFit, LogitOptions};

#[test]
fn fits_logistic_regression_with_intercept() {
    let x = Array::from_vec(
        vec![12, 1],
        vec![
            -4.0, -3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0,
        ],
    )
    .unwrap();
    let y = Array::from_vec(
        vec![12],
        vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0],
    )
    .unwrap();

    let model = LogitFit::fit(&x, &y, true).unwrap();
    assert!(model.converged());
    assert!(model.iterations() < 20);
    assert_eq!(model.coefficients().shape(), &[2]);
    assert!(model.coefficients().as_slice()[1] > 0.0);
    assert!(model.metrics().accuracy >= 0.75);
    assert!(model.metrics().pseudo_r_squared > 0.25);

    let probes = Array::from_vec(vec![2, 1], vec![-5.0, 6.0]).unwrap();
    let probabilities = model.predict_proba(&probes).unwrap();
    assert!(probabilities.as_slice()[0] < 0.05);
    assert!(probabilities.as_slice()[1] > 0.95);

    let classes = model.predict_classes(&probes, 0.5).unwrap();
    assert_eq!(classes.as_slice(), &[0, 1]);
}

#[test]
fn supports_regularized_logit_options() {
    let x = Array::from_vec(vec![6, 1], vec![-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]).unwrap();
    let y = Array::from_vec(vec![6], vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();

    let model = LogitFit::fit_with_options(
        &x,
        &y,
        true,
        LogitOptions {
            max_iterations: 100,
            tolerance: 1e-9,
            l2_penalty: 1.0,
        },
    )
    .unwrap();

    assert!(model.converged());
    assert!(model.coefficients().as_slice()[1] > 0.0);
    assert!(model.metrics().accuracy > 0.99);
}

#[test]
fn rejects_non_binary_targets() {
    let x = Array::from_vec(vec![3, 1], vec![1.0, 2.0, 3.0]).unwrap();
    let y = Array::from_vec(vec![3], vec![0.0, 0.5, 1.0]).unwrap();

    let err = LogitFit::fit(&x, &y, true).unwrap_err();
    assert!(matches!(err, NumRsError::InvalidShape(_)));
}

#[test]
fn rejects_invalid_prediction_thresholds() {
    let x = Array::from_vec(vec![5, 1], vec![-2.0, -1.0, 0.0, 1.0, 2.0]).unwrap();
    let y = Array::from_vec(vec![5], vec![0.0, 0.0, 0.0, 1.0, 1.0]).unwrap();
    let model = LogitFit::fit_with_options(
        &x,
        &y,
        true,
        LogitOptions {
            max_iterations: 100,
            tolerance: 1e-9,
            l2_penalty: 0.5,
        },
    )
    .unwrap();

    let err = model.predict_classes(&x, 1.5).unwrap_err();
    assert!(matches!(err, NumRsError::InvalidShape(_)));
}
