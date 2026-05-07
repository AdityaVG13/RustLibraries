use crate::error::{NumRsError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Slice {
    All,
    Index(isize),
    Range {
        start: Option<isize>,
        end: Option<isize>,
        step: isize,
    },
}

impl Slice {
    pub fn range(start: Option<isize>, end: Option<isize>, step: isize) -> Self {
        Self::Range { start, end, step }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NormalizedSlice {
    Index(isize),
    Range {
        start: isize,
        len: usize,
        step: isize,
    },
}

pub(crate) fn normalize_slice(
    slice: Slice,
    axis: usize,
    dim_len: usize,
) -> Result<NormalizedSlice> {
    match slice {
        Slice::All => Ok(NormalizedSlice::Range {
            start: 0,
            len: dim_len,
            step: 1,
        }),
        Slice::Index(index) => {
            let normalized = normalize_index(index, axis, dim_len)?;
            Ok(NormalizedSlice::Index(normalized))
        }
        Slice::Range { start, end, step } => normalize_range(start, end, step, dim_len),
    }
}

fn normalize_index(index: isize, axis: usize, dim_len: usize) -> Result<isize> {
    let normalized = if index < 0 {
        dim_len as isize + index
    } else {
        index
    };
    if normalized < 0 || normalized >= dim_len as isize {
        return Err(NumRsError::IndexOutOfBounds {
            axis,
            index,
            len: dim_len,
        });
    }
    Ok(normalized)
}

fn normalize_range(
    start: Option<isize>,
    end: Option<isize>,
    step: isize,
    dim_len: usize,
) -> Result<NormalizedSlice> {
    if step == 0 {
        return Err(NumRsError::InvalidSlice(
            "slice step cannot be zero".to_string(),
        ));
    }

    let dim_len = dim_len as isize;
    let (start, end, len) = if step > 0 {
        let start = normalize_positive_bound(start.unwrap_or(0), dim_len);
        let end = normalize_positive_bound(end.unwrap_or(dim_len), dim_len);
        let len = if start >= end {
            0
        } else {
            ((end - start - 1) / step + 1) as usize
        };
        (start, end, len)
    } else {
        let start = normalize_negative_start(start, dim_len);
        let end = normalize_negative_end(end, dim_len);
        let step_abs = -step;
        let len = if end >= start {
            0
        } else {
            ((start - end - 1) / step_abs + 1) as usize
        };
        (start, end, len)
    };

    let _ = end;
    Ok(NormalizedSlice::Range { start, len, step })
}

fn normalize_positive_bound(value: isize, dim_len: isize) -> isize {
    let adjusted = if value < 0 { value + dim_len } else { value };
    adjusted.clamp(0, dim_len)
}

fn normalize_negative_start(value: Option<isize>, dim_len: isize) -> isize {
    match value {
        None => dim_len - 1,
        Some(raw) => {
            let adjusted = if raw < 0 { raw + dim_len } else { raw };
            adjusted.clamp(-1, dim_len - 1)
        }
    }
}

fn normalize_negative_end(value: Option<isize>, dim_len: isize) -> isize {
    match value {
        None => -1,
        Some(raw) => {
            let adjusted = if raw < 0 { raw + dim_len } else { raw };
            adjusted.clamp(-1, dim_len - 1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_forward_slice() {
        assert_eq!(
            normalize_range(Some(1), Some(5), 2, 6).unwrap(),
            NormalizedSlice::Range {
                start: 1,
                len: 2,
                step: 2
            }
        );
    }

    #[test]
    fn normalizes_reverse_slice() {
        assert_eq!(
            normalize_range(None, None, -1, 5).unwrap(),
            NormalizedSlice::Range {
                start: 4,
                len: 5,
                step: -1
            }
        );
    }
}
