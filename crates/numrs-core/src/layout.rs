use crate::error::{NumRsError, Result};
use crate::shape::{broadcast_shape, c_strides, size_of_shape};

pub(crate) fn offset_for_index(
    shape: &[usize],
    strides: &[isize],
    base_offset: isize,
    index: &[usize],
) -> Result<usize> {
    if index.len() != shape.len() {
        return Err(NumRsError::IndexRankMismatch {
            expected: shape.len(),
            actual: index.len(),
        });
    }

    let mut offset = base_offset;
    for (axis, ((idx, len), stride)) in index
        .iter()
        .zip(shape.iter())
        .zip(strides.iter())
        .enumerate()
    {
        if *idx >= *len {
            return Err(NumRsError::IndexOutOfBounds {
                axis,
                index: *idx as isize,
                len: *len,
            });
        }
        offset += *idx as isize * *stride;
    }

    Ok(offset as usize)
}

pub(crate) fn is_c_contiguous(shape: &[usize], strides: &[isize]) -> bool {
    shape.contains(&0) || strides == c_strides(shape)
}

pub(crate) fn broadcast_strides(
    source_shape: &[usize],
    source_strides: &[isize],
    target_shape: &[usize],
) -> Result<Vec<isize>> {
    let checked = broadcast_shape(source_shape, target_shape)?;
    if checked != target_shape {
        return Err(NumRsError::BroadcastMismatch {
            left: source_shape.to_vec(),
            right: target_shape.to_vec(),
        });
    }

    let target_ndim = target_shape.len();
    let source_ndim = source_shape.len();
    let mut out = vec![0isize; target_ndim];

    for target_axis in 0..target_ndim {
        let source_axis = target_axis as isize - (target_ndim as isize - source_ndim as isize);
        if source_axis < 0 {
            out[target_axis] = 0;
            continue;
        }

        let source_axis = source_axis as usize;
        let source_dim = source_shape[source_axis];
        let target_dim = target_shape[target_axis];
        out[target_axis] = if source_dim == target_dim {
            source_strides[source_axis]
        } else if source_dim == 1 {
            0
        } else {
            return Err(NumRsError::BroadcastMismatch {
                left: source_shape.to_vec(),
                right: target_shape.to_vec(),
            });
        };
    }

    Ok(out)
}

#[derive(Debug, Clone)]
pub(crate) struct OffsetIter {
    shape: Vec<usize>,
    strides: Vec<isize>,
    base_offset: isize,
    next_linear: usize,
    len: usize,
}

impl OffsetIter {
    pub(crate) fn new(shape: &[usize], strides: &[isize], base_offset: isize) -> Result<Self> {
        let len = size_of_shape(shape)?;
        Ok(Self {
            shape: shape.to_vec(),
            strides: strides.to_vec(),
            base_offset,
            next_linear: 0,
            len,
        })
    }
}

impl Iterator for OffsetIter {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next_linear >= self.len {
            return None;
        }

        let mut remainder = self.next_linear;
        self.next_linear += 1;

        if self.shape.is_empty() {
            return Some(self.base_offset as usize);
        }

        let mut offset = self.base_offset;
        for axis in (0..self.shape.len()).rev() {
            let dim = self.shape[axis];
            let coord = remainder.checked_rem(dim).unwrap_or(0);
            remainder = remainder.checked_div(dim).unwrap_or(0);
            offset += coord as isize * self.strides[axis];
        }

        Some(offset as usize)
    }
}

pub(crate) fn permute_unique_axes(axes: &[usize], ndim: usize) -> Result<()> {
    if axes.len() != ndim {
        return Err(NumRsError::IndexRankMismatch {
            expected: ndim,
            actual: axes.len(),
        });
    }

    let mut seen = vec![false; ndim];
    for axis in axes.iter().copied() {
        if axis >= ndim {
            return Err(NumRsError::AxisOutOfBounds {
                axis: axis as isize,
                ndim,
            });
        }
        if seen[axis] {
            return Err(NumRsError::DuplicateAxis(axis));
        }
        seen[axis] = true;
    }

    Ok(())
}
