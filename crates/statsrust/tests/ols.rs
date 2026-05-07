use numrs_core::{Array, NumRsError};
use statsrust::{OlsFit, WlsFit};

#[test]
fn fits_ordinary_least_squares_with_intercept() {
    let x = Array::from_vec(
        vec![5, 2],
        vec![0.0, 0.0, 1.0, 2.0, 2.0, 1.0, 3.0, 5.0, 4.0, 4.0],
    )
    .unwrap();
    let y = Array::from_vec(vec![5], vec![1.0, 9.0, 8.0, 22.0, 21.0]).unwrap();

    let model = OlsFit::fit(&x, &y, true).unwrap();
    let coefficients = model.coefficients().as_slice();

    assert!((coefficients[0] - 1.0).abs() < 1e-9);
    assert!((coefficients[1] - 2.0).abs() < 1e-9);
    assert!((coefficients[2] - 3.0).abs() < 1e-9);
    assert!(model.metrics().r_squared > 0.999999);

    let predicted = model.predict(&x).unwrap();
    for (actual, predicted) in y.as_slice().iter().zip(predicted.as_slice()) {
        assert!((actual - predicted).abs() < 1e-9);
    }
}

#[test]
fn rejects_singular_designs() {
    let x = Array::from_vec(vec![4, 2], vec![1.0, 2.0, 2.0, 4.0, 3.0, 6.0, 4.0, 8.0]).unwrap();
    let y = Array::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();

    let err = OlsFit::fit(&x, &y, true).unwrap_err();
    assert!(matches!(err, NumRsError::InvalidShape(_)));
}

#[test]
fn fits_weighted_least_squares_with_intercept() {
    let x = Array::from_vec(
        vec![5, 2],
        vec![0.0, 0.0, 1.0, 2.0, 2.0, 1.0, 3.0, 5.0, 4.0, 4.0],
    )
    .unwrap();
    let y = Array::from_vec(vec![5], vec![1.0, 9.0, 8.0, 22.0, 21.0]).unwrap();
    let weights = Array::from_vec(vec![5], vec![1.0, 2.0, 1.0, 4.0, 3.0]).unwrap();

    let model = WlsFit::fit(&x, &y, &weights, true).unwrap();
    let coefficients = model.coefficients().as_slice();

    assert!((coefficients[0] - 1.0).abs() < 1e-9);
    assert!((coefficients[1] - 2.0).abs() < 1e-9);
    assert!((coefficients[2] - 3.0).abs() < 1e-9);
    assert!(model.metrics().r_squared > 0.999999);
}

#[test]
fn rejects_invalid_wls_weights() {
    let x = Array::from_vec(vec![3, 1], vec![1.0, 2.0, 3.0]).unwrap();
    let y = Array::from_vec(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
    let weights = Array::from_vec(vec![3], vec![1.0, -1.0, 1.0]).unwrap();

    let err = WlsFit::fit(&x, &y, &weights, true).unwrap_err();
    assert!(matches!(err, NumRsError::InvalidShape(_)));
}
