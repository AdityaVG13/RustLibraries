//! `numrs-core` is the flagship Rust-first NumPy-core port for this repo.
//!
//! The crate intentionally starts with a strict foundation slice:
//! typed n-dimensional arrays, shape/stride layout metadata, zero-copy views,
//! NumPy-style slicing, reshape, transpose, broadcasting, elementwise kernels,
//! reductions, 2-D dot products, and NumPy-style matrix multiplication.

mod array;
mod blas;
mod dtype;
mod error;
mod indexing;
mod layout;
mod ops;
mod shape;

pub use num_complex::{Complex32, Complex64};

pub use array::{Array, ArrayView};
pub use dtype::{promote_dtype, CastElement, DType, DTypeKind};
pub use error::{NumRsError, Result};
pub use indexing::Slice;
pub use ops::{
    ArgReduceKernel, DotKernel, MatmulKernel, ProductKernel, SumKernel, TensordotKernel,
};
pub use shape::{broadcast_shape, c_strides, resolve_shape, size_of_shape};

pub type F64Array = Array<f64>;
pub type F32Array = Array<f32>;
pub type C128Array = Array<Complex64>;
pub type C64Array = Array<Complex32>;
pub type I64Array = Array<i64>;
pub type I32Array = Array<i32>;
pub type U64Array = Array<u64>;
pub type U32Array = Array<u32>;
pub type BoolArray = Array<bool>;
