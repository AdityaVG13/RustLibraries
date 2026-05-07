use numrs_core::{NumRsError, Result};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RootResult {
    pub root: f64,
    pub fun: f64,
    pub iterations: usize,
    pub converged: bool,
}

pub fn bisect<F>(
    mut f: F,
    lower: f64,
    upper: f64,
    tolerance: f64,
    max_iterations: usize,
) -> Result<RootResult>
where
    F: FnMut(f64) -> f64,
{
    validate_bounds(lower, upper)?;
    validate_tolerance(tolerance, max_iterations)?;

    let mut a = lower;
    let mut b = upper;
    let mut fa = checked_eval(&mut f, a)?;
    let fb = checked_eval(&mut f, b)?;
    validate_bracket(fa, fb)?;
    if fa == 0.0 {
        return Ok(converged(a, fa, 0));
    }
    if fb == 0.0 {
        return Ok(converged(b, fb, 0));
    }

    for iteration in 1..=max_iterations {
        let mid = 0.5 * (a + b);
        let fmid = checked_eval(&mut f, mid)?;
        if fmid == 0.0 || 0.5 * (b - a).abs() <= tolerance {
            return Ok(converged(mid, fmid, iteration));
        }
        if same_sign(fa, fmid) {
            a = mid;
            fa = fmid;
        } else {
            b = mid;
        }
    }

    let mid = 0.5 * (a + b);
    Ok(RootResult {
        root: mid,
        fun: checked_eval(&mut f, mid)?,
        iterations: max_iterations,
        converged: false,
    })
}

pub fn brentq<F>(
    mut f: F,
    lower: f64,
    upper: f64,
    xtolerance: f64,
    rtolerance: f64,
    max_iterations: usize,
) -> Result<RootResult>
where
    F: FnMut(f64) -> f64,
{
    validate_bounds(lower, upper)?;
    validate_tolerance(xtolerance, max_iterations)?;
    if !rtolerance.is_finite() || rtolerance < 0.0 {
        return Err(NumRsError::InvalidShape(
            "rtolerance must be finite and non-negative".to_string(),
        ));
    }

    let mut a = lower;
    let mut b = upper;
    let mut fa = checked_eval(&mut f, a)?;
    let mut fb = checked_eval(&mut f, b)?;
    validate_bracket(fa, fb)?;
    if fa == 0.0 {
        return Ok(converged(a, fa, 0));
    }
    if fb == 0.0 {
        return Ok(converged(b, fb, 0));
    }

    let mut c = a;
    let mut fc = fa;
    let mut d = b - a;
    let mut e = d;

    for iteration in 1..=max_iterations {
        if same_sign(fb, fc) {
            c = a;
            fc = fa;
            d = b - a;
            e = d;
        }
        if fc.abs() < fb.abs() {
            a = b;
            b = c;
            c = a;
            fa = fb;
            fb = fc;
            fc = fa;
        }

        let tolerance = 2.0 * f64::EPSILON * b.abs() + 0.5 * xtolerance + rtolerance * b.abs();
        let midpoint = 0.5 * (c - b);
        if midpoint.abs() <= tolerance || fb == 0.0 {
            return Ok(converged(b, fb, iteration));
        }

        if e.abs() >= tolerance && fa.abs() > fb.abs() {
            let s = fb / fa;
            let (mut p, mut q) = if a == c {
                (2.0 * midpoint * s, 1.0 - s)
            } else {
                let q = fa / fc;
                let r = fb / fc;
                (
                    s * (2.0 * midpoint * q * (q - r) - (b - a) * (r - 1.0)),
                    (q - 1.0) * (r - 1.0) * (s - 1.0),
                )
            };

            if p > 0.0 {
                q = -q;
            }
            p = p.abs();

            let interpolation_bound = (3.0 * midpoint * q - (tolerance * q).abs()).abs();
            let previous_bound = (e * q).abs();
            if 2.0 * p < interpolation_bound.min(previous_bound) {
                e = d;
                d = p / q;
            } else {
                d = midpoint;
                e = d;
            }
        } else {
            d = midpoint;
            e = d;
        }

        a = b;
        fa = fb;
        b += if d.abs() > tolerance {
            d
        } else {
            tolerance.copysign(midpoint)
        };
        fb = checked_eval(&mut f, b)?;
    }

    Ok(RootResult {
        root: b,
        fun: fb,
        iterations: max_iterations,
        converged: false,
    })
}

fn validate_bounds(lower: f64, upper: f64) -> Result<()> {
    if !lower.is_finite() || !upper.is_finite() {
        return Err(NumRsError::InvalidShape(
            "root bounds must be finite".to_string(),
        ));
    }
    if lower >= upper {
        return Err(NumRsError::InvalidShape(format!(
            "lower bound {lower} must be less than upper bound {upper}"
        )));
    }
    Ok(())
}

fn validate_tolerance(tolerance: f64, max_iterations: usize) -> Result<()> {
    if !tolerance.is_finite() || tolerance <= 0.0 {
        return Err(NumRsError::InvalidShape(
            "tolerance must be finite and positive".to_string(),
        ));
    }
    if max_iterations == 0 {
        return Err(NumRsError::InvalidShape(
            "max_iterations must be positive".to_string(),
        ));
    }
    Ok(())
}

fn validate_bracket(left: f64, right: f64) -> Result<()> {
    if same_sign(left, right) && left != 0.0 && right != 0.0 {
        return Err(NumRsError::InvalidShape(
            "root bracket endpoints must have opposite signs".to_string(),
        ));
    }
    Ok(())
}

fn checked_eval<F>(f: &mut F, x: f64) -> Result<f64>
where
    F: FnMut(f64) -> f64,
{
    let value = f(x);
    if !value.is_finite() {
        return Err(NumRsError::InvalidShape(format!(
            "root function returned non-finite value {value} at x={x}"
        )));
    }
    Ok(value)
}

fn same_sign(left: f64, right: f64) -> bool {
    left.is_sign_positive() == right.is_sign_positive()
}

fn converged(root: f64, fun: f64, iterations: usize) -> RootResult {
    RootResult {
        root,
        fun,
        iterations,
        converged: true,
    }
}
