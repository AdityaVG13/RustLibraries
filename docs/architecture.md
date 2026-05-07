# Architecture

## Goal

Build a pure Rust NumPy-core foundation that can grow toward parity while using Rust to exceed Python/NumPy ergonomics where the language gives stronger guarantees.

## Current v0 Envelope

Implemented target:

- Typed arrays for `f64`, `f32`, signed ints, unsigned ints, and bool.
- Dtype metadata, item sizes, primitive `astype`, and a deterministic promotion table.
- Owned row-major arrays.
- Uniform-value metadata for arrays constructed by `full` and `zeros`; mutating writes invalidate the metadata.
- Borrowed array views.
- Shape, stride, and base-offset metadata.
- Checked indexing.
- NumPy-style integer and range slicing, including negative indices and negative step.
- Zero-copy reshape for contiguous layouts.
- Zero-copy transpose and arbitrary axis permutation.
- Metadata broadcasting with zero strides.
- Fancy indexing foundations: flat `take`, axis `take_axis`, exact-shape boolean masks, flat `put`, `putmask`, and indexed `add_at`/`maximum_at`.
- Elementwise add, sub, mul, div, equality, bool and/or/not.
- In-place broadcast add, sub, mul, and div for owned arrays.
- Fused `f64` add+sum for temporary-free reductions.
- Specialized `f64` outer-broadcast add for column-vector plus row-vector kernels.
- Reductions: `sum_all`, `sum_axis`, `mean_all`, `mean_axis`, `min_all`, `max_all`, `prod_all`, `var_all`, `std_all`, `argmax`, and `argmin`.
- NumPy-style `matmul` for vector dot, matrix-vector, vector-matrix, matrix-matrix, and broadcasted batched matrix multiplication.
- Explicit-axis `tensordot_axes` for tensor contractions.
- Selected einsum-style contractions: `outer_product`, `mul_scalar`, and `weighted_axis1_sum`.
- Small pure-Rust linalg: `norm_l2`, `det`, and vector `solve` for `f64`.
- Accelerate-backed contiguous 2-D dot for `f64` and `f32` on macOS, with generic fallback for other numeric dtypes and platforms.
- Accelerate-backed contiguous `f64` outer product and matrix-vector weighted sum on macOS, with generic fallback elsewhere.
- Packed BLAS-backed `f64` tensor contractions for `tensordot_axes`.
- Accelerate/vDSP-backed contiguous sums for `f64` and `f32` on macOS.
- Uniform-array fast paths for sum/product/stat reductions, arg reductions, and all-true/all-false `putmask`.
- Metadata `ravel`, `expand_dims`, and `squeeze`.

Out of scope for v0:

- Full Python Array API bindings.
- Full NumPy dtype system, object/string/datetime dtypes, and exact NumPy promotion semantics.
- Mutable views.
- Generalized NumPy advanced indexing with mixed advanced/basic index arrays.
- SIMD kernels.
- NumPy ABI compatibility.

## Novel Direction

The architecture separates "semantic layout" from "kernel execution":

1. `Array<T>` owns contiguous data.
2. `ArrayView<'a, T>` is a layout descriptor over borrowed data.
3. Shape transforms produce views, not copies.
4. Broadcast transforms set stride `0` on expanded axes.
5. Kernels dispatch on layout:
   - exact contiguous equal-shape fast path,
   - uniform metadata fast path,
   - broadcasted stride path,
   - future vectorized/tiled path.

This gives NumPy-like semantics without inheriting Python's mutable aliasing hazards.

## Optimization Plan

Near term:

- Add mutable owned kernels and in-place operations with strict borrow rules.
- Add copy-on-write arrays for cheap branching transforms.
- Add small-shape stack storage to avoid heap allocation for common ranks.
- Add axis-specialized reductions for contiguous innermost axes.

Mid term:

- Add portable SIMD for contiguous `f64` and `i64`.
- Add matrix multiply tiling and optional BLAS feature.
- Add Rayon feature for large elementwise kernels.
- Expand dtype promotion toward the Array API and NumPy edge cases.

Long term:

- Add PyO3 bindings after Rust semantics are stable.
- Add property tests against Python/NumPy as an oracle.
- Add pandas/scikit-learn foundations on top of this core.

## Correctness Contracts

- Shape product is checked for overflow.
- Data length must match shape product.
- Index rank must match view rank.
- Broadcast mismatch is a typed error.
- Zero-copy reshape requires C-contiguous layout.
- Reductions over empty arrays are explicit errors where NumPy would often warn or return `NaN`.
