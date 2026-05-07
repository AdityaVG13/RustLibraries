use crate::error::{NumRsError, Result};

pub fn size_of_shape(shape: &[usize]) -> Result<usize> {
    shape.iter().try_fold(1usize, |acc, dim| {
        acc.checked_mul(*dim).ok_or(NumRsError::ShapeOverflow)
    })
}

pub fn c_strides(shape: &[usize]) -> Vec<isize> {
    let mut strides = vec![0isize; shape.len()];
    let mut next = 1isize;
    for (axis, dim) in shape.iter().enumerate().rev() {
        strides[axis] = next;
        next = next.saturating_mul(*dim as isize);
    }
    strides
}

pub fn broadcast_shape(left: &[usize], right: &[usize]) -> Result<Vec<usize>> {
    let ndim = left.len().max(right.len());
    let mut out = vec![1usize; ndim];

    for i in 0..ndim {
        let l = dim_from_right(left, i).unwrap_or(1);
        let r = dim_from_right(right, i).unwrap_or(1);
        out[ndim - 1 - i] = match (l, r) {
            (a, b) if a == b => a,
            (1, b) => b,
            (a, 1) => a,
            _ => {
                return Err(NumRsError::BroadcastMismatch {
                    left: left.to_vec(),
                    right: right.to_vec(),
                });
            }
        };
    }

    Ok(out)
}

pub(crate) fn dim_from_right(shape: &[usize], index_from_right: usize) -> Option<usize> {
    shape
        .len()
        .checked_sub(index_from_right + 1)
        .map(|idx| shape[idx])
}

pub fn resolve_shape(spec: &[isize], len: usize) -> Result<Vec<usize>> {
    let mut inferred_axis = None;
    let mut known = 1usize;
    let mut out = Vec::with_capacity(spec.len());

    for (axis, dim) in spec.iter().copied().enumerate() {
        if dim == -1 {
            if inferred_axis.replace(axis).is_some() {
                return Err(NumRsError::InvalidShape(
                    "only one inferred dimension is allowed".to_string(),
                ));
            }
            out.push(usize::MAX);
        } else if dim < 0 {
            return Err(NumRsError::InvalidShape(format!(
                "negative dimension {dim} at axis {axis}"
            )));
        } else {
            let dim = dim as usize;
            known = known.checked_mul(dim).ok_or(NumRsError::ShapeOverflow)?;
            out.push(dim);
        }
    }

    if let Some(axis) = inferred_axis {
        if known == 0 {
            return Err(NumRsError::InvalidShape(
                "cannot infer dimension when known product is zero".to_string(),
            ));
        }
        if len % known != 0 {
            return Err(NumRsError::InvalidShape(format!(
                "cannot reshape {len} elements with known product {known}"
            )));
        }
        out[axis] = len / known;
    }

    let resolved_len = size_of_shape(&out)?;
    if resolved_len != len {
        return Err(NumRsError::InvalidShape(format!(
            "shape {out:?} has {resolved_len} elements, expected {len}"
        )));
    }

    Ok(out)
}

pub(crate) fn normalize_axis(axis: isize, ndim: usize) -> Result<usize> {
    let normalized = if axis < 0 { ndim as isize + axis } else { axis };

    if normalized < 0 || normalized >= ndim as isize {
        return Err(NumRsError::AxisOutOfBounds { axis, ndim });
    }

    Ok(normalized as usize)
}

pub(crate) fn normalize_insert_axis(axis: isize, ndim: usize) -> Result<usize> {
    let upper = ndim + 1;
    let normalized = if axis < 0 {
        upper as isize + axis
    } else {
        axis
    };

    if normalized < 0 || normalized > ndim as isize {
        return Err(NumRsError::AxisOutOfBounds { axis, ndim: upper });
    }

    Ok(normalized as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn broadcasts_like_numpy() {
        assert_eq!(broadcast_shape(&[2, 1, 3], &[4, 3]).unwrap(), vec![2, 4, 3]);
        assert!(broadcast_shape(&[2, 3], &[4, 3]).is_err());
    }

    #[test]
    fn resolves_inferred_shape() {
        assert_eq!(resolve_shape(&[2, -1, 3], 24).unwrap(), vec![2, 4, 3]);
        assert!(resolve_shape(&[-1, -1], 4).is_err());
    }
}
