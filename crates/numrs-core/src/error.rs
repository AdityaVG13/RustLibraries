use std::fmt;

pub type Result<T> = std::result::Result<T, NumRsError>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NumRsError {
    ShapeDataMismatch {
        shape: Vec<usize>,
        expected: usize,
        actual: usize,
    },
    ShapeOverflow,
    InvalidShape(String),
    BroadcastMismatch {
        left: Vec<usize>,
        right: Vec<usize>,
    },
    IndexRankMismatch {
        expected: usize,
        actual: usize,
    },
    IndexOutOfBounds {
        axis: usize,
        index: isize,
        len: usize,
    },
    InvalidSlice(String),
    AxisOutOfBounds {
        axis: isize,
        ndim: usize,
    },
    DuplicateAxis(usize),
    NonContiguousReshape {
        shape: Vec<usize>,
        strides: Vec<isize>,
    },
    DotShapeMismatch {
        left: Vec<usize>,
        right: Vec<usize>,
    },
    EmptyReduction,
    CannotSqueezeAxis {
        axis: usize,
        len: usize,
    },
    BooleanMaskShapeMismatch {
        array: Vec<usize>,
        mask: Vec<usize>,
    },
}

impl fmt::Display for NumRsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShapeDataMismatch {
                shape,
                expected,
                actual,
            } => write!(
                f,
                "shape {shape:?} expects {expected} elements, got {actual}"
            ),
            Self::ShapeOverflow => write!(f, "shape size overflow"),
            Self::InvalidShape(message) => write!(f, "invalid shape: {message}"),
            Self::BroadcastMismatch { left, right } => {
                write!(f, "cannot broadcast shapes {left:?} and {right:?}")
            }
            Self::IndexRankMismatch { expected, actual } => {
                write!(f, "expected {expected} indices, got {actual}")
            }
            Self::IndexOutOfBounds { axis, index, len } => {
                write!(f, "index {index} out of bounds for axis {axis} with len {len}")
            }
            Self::InvalidSlice(message) => write!(f, "invalid slice: {message}"),
            Self::AxisOutOfBounds { axis, ndim } => {
                write!(f, "axis {axis} out of bounds for ndim {ndim}")
            }
            Self::DuplicateAxis(axis) => write!(f, "duplicate axis {axis}"),
            Self::NonContiguousReshape { shape, strides } => write!(
                f,
                "cannot zero-copy reshape non-contiguous layout shape={shape:?}, strides={strides:?}"
            ),
            Self::DotShapeMismatch { left, right } => {
                write!(f, "2-D dot shape mismatch: {left:?} x {right:?}")
            }
            Self::EmptyReduction => write!(f, "reduction over an empty array"),
            Self::CannotSqueezeAxis { axis, len } => {
                write!(f, "cannot squeeze axis {axis} with len {len}")
            }
            Self::BooleanMaskShapeMismatch { array, mask } => {
                write!(
                    f,
                    "boolean mask shape {mask:?} does not match array shape {array:?}"
                )
            }
        }
    }
}

impl std::error::Error for NumRsError {}
