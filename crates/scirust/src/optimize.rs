use numrs_core::{NumRsError, Result};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScalarMinResult {
    pub x: f64,
    pub fun: f64,
    pub iterations: usize,
    pub converged: bool,
}

pub fn minimize_scalar_bounded<F>(
    mut f: F,
    lower: f64,
    upper: f64,
    tolerance: f64,
    max_iterations: usize,
) -> Result<ScalarMinResult>
where
    F: FnMut(f64) -> f64,
{
    if !lower.is_finite() || !upper.is_finite() || !tolerance.is_finite() {
        return Err(NumRsError::InvalidShape(
            "bounds and tolerance must be finite".to_string(),
        ));
    }
    if lower >= upper {
        return Err(NumRsError::InvalidShape(format!(
            "lower bound {lower} must be less than upper bound {upper}"
        )));
    }
    if tolerance <= 0.0 {
        return Err(NumRsError::InvalidShape(
            "tolerance must be positive".to_string(),
        ));
    }

    let inv_phi = (5.0_f64.sqrt() - 1.0) / 2.0;
    let inv_phi2 = (3.0 - 5.0_f64.sqrt()) / 2.0;
    let mut a = lower;
    let b = upper;
    let mut h = b - a;
    if h <= tolerance {
        let x = (a + b) * 0.5;
        return Ok(ScalarMinResult {
            x,
            fun: f(x),
            iterations: 0,
            converged: true,
        });
    }

    let mut c = a + inv_phi2 * h;
    let mut d = a + inv_phi * h;
    let mut yc = f(c);
    let mut yd = f(d);

    for iteration in 1..=max_iterations {
        if yc < yd {
            d = c;
            yd = yc;
            h *= inv_phi;
            c = a + inv_phi2 * h;
            yc = f(c);
        } else {
            a = c;
            c = d;
            yc = yd;
            h *= inv_phi;
            d = a + inv_phi * h;
            yd = f(d);
        }

        if h <= tolerance {
            let x = if yc < yd { c } else { d };
            return Ok(ScalarMinResult {
                x,
                fun: f(x),
                iterations: iteration,
                converged: true,
            });
        }
    }

    let x = if yc < yd { c } else { d };
    Ok(ScalarMinResult {
        x,
        fun: f(x),
        iterations: max_iterations,
        converged: false,
    })
}
