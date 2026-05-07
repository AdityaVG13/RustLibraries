use numrs_core::{promote_dtype, Array, DTypeKind, NumRsError, Slice};

#[test]
fn constructs_typed_arrays_and_reports_dtype() {
    let floats = Array::from_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let floats32 = Array::from_vec(vec![2], vec![1.0_f32, 2.0]).unwrap();
    let ints = Array::from_vec(vec![3], vec![1_i64, 2, 3]).unwrap();
    let ints32 = Array::from_vec(vec![2], vec![1_i32, 2]).unwrap();
    let uints = Array::from_vec(vec![2], vec![1_u64, 2]).unwrap();
    let bools = Array::from_vec(vec![2], vec![true, false]).unwrap();
    let zeros = Array::<i64>::zeros(vec![2, 2]).unwrap();
    let full = Array::full(vec![2, 2], 7_i64).unwrap();
    let generated = Array::from_shape_fn(vec![2, 3], |idx| (idx[0] * 10 + idx[1]) as i64).unwrap();
    let scalar = Array::scalar(42_i64).unwrap();

    assert_eq!(floats.dtype(), DTypeKind::F64);
    assert_eq!(floats32.dtype(), DTypeKind::F32);
    assert_eq!(ints.dtype(), DTypeKind::I64);
    assert_eq!(ints32.dtype(), DTypeKind::I32);
    assert_eq!(uints.dtype(), DTypeKind::U64);
    assert_eq!(bools.dtype(), DTypeKind::Bool);
    assert_eq!(floats.shape(), &[2, 2]);
    assert_eq!(floats.strides(), &[2, 1]);
    assert_eq!(zeros.as_slice(), &[0, 0, 0, 0]);
    assert_eq!(full.as_slice(), &[7, 7, 7, 7]);
    assert_eq!(
        full,
        Array::from_vec(vec![2, 2], vec![7_i64, 7, 7, 7]).unwrap()
    );
    assert_eq!(generated.as_slice(), &[0, 1, 2, 10, 11, 12]);
    assert_eq!(scalar.shape(), &[]);
    assert_eq!(scalar.as_slice(), &[42]);
    assert_eq!(DTypeKind::F64.itemsize(), 8);
    assert!(DTypeKind::F32.is_float());
    assert_eq!(
        promote_dtype(DTypeKind::I32, DTypeKind::F64),
        DTypeKind::F64
    );
    assert_eq!(promote_dtype(DTypeKind::Bool, DTypeKind::U8), DTypeKind::U8);
    assert_eq!(
        promote_dtype(DTypeKind::I16, DTypeKind::U16),
        DTypeKind::I32
    );
    assert_eq!(promote_dtype(DTypeKind::I32, DTypeKind::U8), DTypeKind::I32);
    assert_eq!(promote_dtype(DTypeKind::I64, DTypeKind::U8), DTypeKind::I64);
}

#[test]
fn casts_arrays_between_supported_dtypes() {
    let ints = Array::from_vec(vec![3], vec![1_i64, 2, 3]).unwrap();
    let floats = ints.astype::<f64>().unwrap();
    assert_eq!(floats.dtype(), DTypeKind::F64);
    assert_eq!(floats.as_slice(), &[1.0, 2.0, 3.0]);

    let bools = Array::from_vec(vec![2], vec![true, false]).unwrap();
    let ints = bools.astype::<i32>().unwrap();
    assert_eq!(ints.as_slice(), &[1, 0]);

    let values = Array::from_vec(vec![3], vec![0_i64, -2, 3]).unwrap();
    let truth = values.astype::<bool>().unwrap();
    assert_eq!(truth.as_slice(), &[false, true, true]);

    let matrix = Array::from_vec(vec![2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
    let transposed = matrix.transpose();
    let cast = transposed.astype::<f64>().unwrap();
    assert_eq!(cast.shape(), &[3, 2]);
    assert_eq!(cast.as_slice(), &[0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
}

#[test]
fn integer_sum_wraps_instead_of_panicking_on_overflow() {
    let ints = Array::from_vec(vec![2], vec![i8::MAX, 1]).unwrap();
    assert_eq!(ints.sum_all().unwrap(), i8::MIN);

    let uints = Array::from_vec(vec![2], vec![u8::MAX, 1]).unwrap();
    assert_eq!(uints.sum_all().unwrap(), 0);
}

#[test]
fn supports_fancy_indexing_take_and_boolean_masks() {
    let a = Array::from_vec(vec![3, 4], (0_i64..12).collect()).unwrap();

    let flat = a.take(&[0, -1, 5]).unwrap();
    assert_eq!(flat.shape(), &[3]);
    assert_eq!(flat.as_slice(), &[0, 11, 5]);

    let rows = a.take_axis(&[2, 0], 0).unwrap();
    assert_eq!(rows.shape(), &[2, 4]);
    assert_eq!(rows.as_slice(), &[8, 9, 10, 11, 0, 1, 2, 3]);

    let cols = a.take_axis(&[-1, 1], 1).unwrap();
    assert_eq!(cols.shape(), &[3, 2]);
    assert_eq!(cols.as_slice(), &[3, 1, 7, 5, 11, 9]);

    let identity = a.take_axis(&[0, 1, 2], 0).unwrap();
    assert_eq!(identity.shape(), &[3, 4]);
    assert_eq!(identity.as_slice(), a.as_slice());

    let mask = Array::from_vec(
        vec![3, 4],
        vec![
            true, false, false, true, false, true, false, false, true, false, true, false,
        ],
    )
    .unwrap();
    let filtered = a.boolean_mask(&mask).unwrap();
    assert_eq!(filtered.shape(), &[5]);
    assert_eq!(filtered.as_slice(), &[0, 3, 5, 8, 10]);

    let mut writable = Array::from_vec(vec![6], vec![0_i64; 6]).unwrap();
    writable.put(&[0, -1, 2], &[10, 20, 30]).unwrap();
    assert_eq!(writable.as_slice(), &[10, 0, 30, 0, 0, 20]);

    let mask = Array::from_vec(vec![6], vec![false, true, true, false, true, false]).unwrap();
    writable.putmask(&mask, &[7]).unwrap();
    assert_eq!(writable.as_slice(), &[10, 7, 7, 0, 7, 20]);

    writable
        .putmask(&Array::full(vec![6], false).unwrap(), &[])
        .unwrap();
    assert_eq!(writable.as_slice(), &[10, 7, 7, 0, 7, 20]);
    writable
        .putmask(&Array::full(vec![6], true).unwrap(), &[3])
        .unwrap();
    assert_eq!(writable.as_slice(), &[3, 3, 3, 3, 3, 3]);
    assert_eq!(writable.argmax().unwrap(), 0);

    let mut accum = Array::from_vec(vec![3], vec![0.0_f64, 0.0, 0.0]).unwrap();
    accum.add_at(&[0, 2, 0, -1], &[1.0, 2.0, 3.0, 4.0]).unwrap();
    assert_eq!(accum.as_slice(), &[4.0, 0.0, 6.0]);
    accum.maximum_at(&[0, 1, 2], &[2.0, 9.0, 1.0]).unwrap();
    assert_eq!(accum.as_slice(), &[4.0, 9.0, 6.0]);
}

#[test]
fn supports_nonzero_and_where_selection() {
    let mask = Array::from_vec(
        vec![3, 4],
        vec![
            true, false, false, true, false, true, false, false, true, false, true, false,
        ],
    )
    .unwrap();

    let nonzero = mask.nonzero().unwrap();
    assert_eq!(nonzero.len(), 2);
    assert_eq!(nonzero[0].as_slice(), &[0, 0, 1, 2, 2]);
    assert_eq!(nonzero[1].as_slice(), &[0, 3, 1, 0, 2]);

    let values = Array::from_vec(vec![3, 4], (0_i64..12).collect()).unwrap();
    let fallback = Array::full(vec![1, 4], -1_i64).unwrap();
    let selected = mask.where_select(&values, &fallback).unwrap();
    assert_eq!(selected.shape(), &[3, 4]);
    assert_eq!(
        selected.as_slice(),
        &[0, -1, -1, 3, -1, 5, -1, -1, 8, -1, 10, -1]
    );

    let transposed_mask = mask.transpose();
    let transposed_values = values.transpose();
    let selected = transposed_mask
        .where_select(&transposed_values, &Array::scalar(-2_i64).unwrap().view())
        .unwrap();
    assert_eq!(selected.shape(), &[4, 3]);
    assert_eq!(
        selected.as_slice(),
        &[0, -2, 8, -2, 5, -2, -2, -2, 10, 3, -2, -2]
    );
}

#[test]
fn views_slice_without_copying_and_keep_numpy_negative_index_rules() {
    let a = Array::from_vec(vec![3, 4], (0_i64..12).collect()).unwrap();

    let col = a.slice(&[Slice::All, Slice::Index(-1)]).unwrap();
    assert_eq!(col.shape(), &[3]);
    assert_eq!(col.strides(), &[4]);
    assert_eq!(col.to_vec().unwrap(), vec![3, 7, 11]);

    let reversed = a
        .slice(&[Slice::Index(1), Slice::range(None, None, -1)])
        .unwrap();
    assert_eq!(reversed.shape(), &[4]);
    assert_eq!(reversed.strides(), &[-1]);
    assert_eq!(reversed.to_vec().unwrap(), vec![7, 6, 5, 4]);
}

#[test]
fn reshape_and_transpose_are_metadata_operations() {
    let a = Array::from_vec(vec![2, 3], (0_i64..6).collect()).unwrap();

    let reshaped = a.reshape(&[3, -1]).unwrap();
    assert_eq!(reshaped.shape(), &[3, 2]);
    assert_eq!(reshaped.to_vec().unwrap(), vec![0, 1, 2, 3, 4, 5]);

    let transposed = a.transpose();
    assert_eq!(transposed.shape(), &[3, 2]);
    assert_eq!(transposed.strides(), &[1, 3]);
    assert_eq!(transposed.to_vec().unwrap(), vec![0, 3, 1, 4, 2, 5]);

    let expanded = a.expand_dims(1).unwrap();
    assert_eq!(expanded.shape(), &[2, 1, 3]);
    assert_eq!(expanded.strides(), &[3, 0, 1]);
    assert_eq!(expanded.squeeze(None).unwrap().shape(), &[2, 3]);
    assert_eq!(expanded.squeeze(Some(1)).unwrap().shape(), &[2, 3]);

    let raveled = a.ravel().unwrap();
    assert_eq!(raveled.shape(), &[6]);
    assert_eq!(raveled.to_vec().unwrap(), vec![0, 1, 2, 3, 4, 5]);
}

#[test]
fn rejects_zero_copy_reshape_for_non_contiguous_views() {
    let a = Array::from_vec(vec![2, 3], (0_i64..6).collect()).unwrap();
    let transposed = a.transpose();
    let err = transposed.reshape(&[6]).unwrap_err();
    assert!(matches!(err, NumRsError::NonContiguousReshape { .. }));
}

#[test]
fn broadcasts_elementwise_ops_without_materializing_inputs() {
    let mut a = Array::from_vec(vec![3, 1], vec![1_i64, 2, 3]).unwrap();
    let b = Array::from_vec(vec![1, 4], vec![10_i64, 20, 30, 40]).unwrap();

    let out = a.add(&b).unwrap();
    assert_eq!(out.shape(), &[3, 4]);
    assert_eq!(
        out.as_slice(),
        &[11, 21, 31, 41, 12, 22, 32, 42, 13, 23, 33, 43]
    );

    let mask = out.eq_elem(&Array::scalar(22_i64).unwrap()).unwrap();
    assert_eq!(mask.shape(), &[3, 4]);
    assert_eq!(
        mask.as_slice(),
        &[false, false, false, false, false, true, false, false, false, false, false, false]
    );

    a.add_assign(&Array::scalar(10_i64).unwrap()).unwrap();
    assert_eq!(a.as_slice(), &[11, 12, 13]);

    let matrix = Array::from_vec(vec![2, 3], vec![10_i64, 20, 30, 40, 50, 60]).unwrap();
    let row = Array::from_vec(vec![3], vec![1_i64, 2, 3]).unwrap();
    let shifted = matrix.sub(&row).unwrap();
    assert_eq!(shifted.shape(), &[2, 3]);
    assert_eq!(shifted.as_slice(), &[9, 18, 27, 39, 48, 57]);

    let f64_matrix =
        Array::from_vec(vec![2, 3], vec![10.0_f64, 20.0, 30.0, 40.0, 50.0, 60.0]).unwrap();
    let f64_row = Array::from_vec(vec![3], vec![1.0_f64, 2.0, 3.0]).unwrap();
    let shifted = f64_matrix.sub(&f64_row).unwrap();
    assert_eq!(shifted.shape(), &[2, 3]);
    assert_eq!(shifted.as_slice(), &[9.0, 18.0, 27.0, 39.0, 48.0, 57.0]);

    let col = Array::from_vec(vec![3, 1], vec![1.0_f64, 2.0, 3.0]).unwrap();
    let row = Array::from_vec(vec![1, 4], vec![10.0_f64, 20.0, 30.0, 40.0]).unwrap();
    let fast = col.add_outer2d_f64(&row).unwrap();
    assert_eq!(fast.shape(), &[3, 4]);
    assert_eq!(
        fast.as_slice(),
        &[11.0, 21.0, 31.0, 41.0, 12.0, 22.0, 32.0, 42.0, 13.0, 23.0, 33.0, 43.0]
    );

    let tile = Array::from_vec(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let batch = Array::from_vec(
        vec![2, 2, 3],
        vec![
            10.0_f64, 20.0, 30.0, 40.0, 50.0, 60.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0,
        ],
    )
    .unwrap();
    let tiled_batch = tile.mul(&batch).unwrap();
    assert_eq!(tiled_batch.shape(), &[2, 2, 3]);
    assert_eq!(
        tiled_batch.as_slice(),
        &[10.0, 40.0, 90.0, 160.0, 250.0, 360.0, 1.0, 2.0, 3.0, 8.0, 10.0, 12.0]
    );
    assert_eq!(batch.mul(&tile).unwrap(), tiled_batch);
}

#[test]
fn supports_bool_broadcast_kernels() {
    let a = Array::from_vec(vec![2, 1], vec![true, false]).unwrap();
    let b = Array::from_vec(vec![1, 3], vec![true, false, true]).unwrap();

    let anded = a.logical_and(&b).unwrap();
    assert_eq!(anded.shape(), &[2, 3]);
    assert_eq!(anded.as_slice(), &[true, false, true, false, false, false]);

    let not = anded.logical_not().unwrap();
    assert_eq!(not.as_slice(), &[false, true, false, true, true, true]);
}

#[test]
fn reduces_all_and_by_axis() {
    let a = Array::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

    assert_eq!(a.sum_all().unwrap(), 21.0);
    assert_eq!(a.mean_all().unwrap(), 3.5);
    assert_eq!(a.min_all().unwrap(), 1.0);
    assert_eq!(a.max_all().unwrap(), 6.0);
    assert_eq!(a.prod_all().unwrap(), 720.0);
    assert!((a.var_all().unwrap() - (35.0 / 12.0)).abs() < 1e-12);
    assert!((a.std_all().unwrap() - (35.0_f64 / 12.0).sqrt()).abs() < 1e-12);

    let by_col = a.sum_axis(0).unwrap();
    assert_eq!(by_col.shape(), &[3]);
    assert_eq!(by_col.as_slice(), &[5.0, 7.0, 9.0]);

    let by_row = a.sum_axis(1).unwrap();
    assert_eq!(by_row.shape(), &[2]);
    assert_eq!(by_row.as_slice(), &[6.0, 15.0]);

    let mean_by_row = a.mean_axis(-1).unwrap();
    assert_eq!(mean_by_row.shape(), &[2]);
    assert_eq!(mean_by_row.as_slice(), &[2.0, 5.0]);

    let values = Array::from_vec(vec![2, 3], vec![1_i64, 9, 3, 4, -5, 8]).unwrap();
    assert_eq!(values.argmax().unwrap(), 1);
    assert_eq!(values.argmin().unwrap(), 4);

    let duplicates = Array::from_vec(vec![7], vec![4_i64, 9, 9, 1, -3, -3, 8]).unwrap();
    assert_eq!(duplicates.argmax().unwrap(), 1);
    assert_eq!(duplicates.argmin().unwrap(), 4);

    let equal = Array::from_vec(vec![4], vec![7_i64, 7, 7, 7]).unwrap();
    assert_eq!(equal.argmax().unwrap(), 0);
    assert_eq!(equal.argmin().unwrap(), 0);

    let uniform = Array::full(vec![200_000], 7_i64).unwrap();
    assert_eq!(uniform.argmax().unwrap(), 0);
    assert_eq!(uniform.argmin().unwrap(), 0);

    let reversed = duplicates.slice(&[Slice::range(None, None, -1)]).unwrap();
    assert_eq!(reversed.argmax().unwrap(), 4);
    assert_eq!(reversed.argmin().unwrap(), 1);
    assert!(matches!(
        Array::<i64>::from_vec(vec![0], vec![])
            .unwrap()
            .argmax()
            .unwrap_err(),
        NumRsError::EmptyReduction
    ));
}

#[test]
fn performs_2d_dot_on_contiguous_and_strided_views() {
    let a = Array::from_vec(vec![2, 3], vec![1_i64, 2, 3, 4, 5, 6]).unwrap();
    let b = Array::from_vec(vec![3, 2], vec![7_i64, 8, 9, 10, 11, 12]).unwrap();
    let out = a.dot2d(&b).unwrap();

    assert_eq!(out.shape(), &[2, 2]);
    assert_eq!(out.as_slice(), &[58, 64, 139, 154]);

    let c = Array::from_vec(vec![2, 3], vec![1_i64, 4, 2, 5, 3, 6]).unwrap();
    let c_t = c.transpose();
    let out = a.view().dot2d(&c_t).unwrap();
    assert_eq!(out.shape(), &[2, 2]);
    assert_eq!(out.as_slice(), &[15, 29, 36, 71]);

    let inner = a.inner2d(&c).unwrap();
    assert_eq!(inner.shape(), &[2, 2]);
    assert_eq!(inner.as_slice(), &[15, 29, 36, 71]);
    let inner_view = a.view().inner2d(&c_t.transpose()).unwrap();
    assert_eq!(inner_view, inner);

    let f = Array::from_vec(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let g = Array::from_vec(vec![2, 2], vec![5.0_f64, 6.0, 7.0, 8.0]).unwrap();
    let out = f.dot2d(&g).unwrap();
    assert_eq!(out.as_slice(), &[19.0, 22.0, 43.0, 50.0]);
    assert_eq!(f.inner2d(&g).unwrap().as_slice(), &[17.0, 23.0, 39.0, 53.0]);
    assert!((f.norm_l2().unwrap() - 30.0_f64.sqrt()).abs() < 1e-12);
    assert!((f.det().unwrap() + 2.0).abs() < 1e-12);

    let solve_a = Array::from_vec(vec![2, 2], vec![3.0_f64, 1.0, 1.0, 2.0]).unwrap();
    let solve_b = Array::from_vec(vec![2], vec![9.0_f64, 8.0]).unwrap();
    let solved = solve_a.solve(&solve_b).unwrap();
    assert_eq!(solved.as_slice(), &[2.0, 3.0]);

    let x = Array::from_vec(vec![2], vec![2.0_f64, 3.0]).unwrap();
    let y = Array::from_vec(vec![3], vec![5.0_f64, 7.0, 11.0]).unwrap();
    let outer = x.outer_product(&y).unwrap();
    assert_eq!(outer.shape(), &[2, 3]);
    assert_eq!(outer.as_slice(), &[10.0, 14.0, 22.0, 15.0, 21.0, 33.0]);
    assert_eq!(y.mul_scalar(2.0).unwrap().as_slice(), &[10.0, 14.0, 22.0]);

    let matrix = Array::from_vec(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    assert_eq!(matrix.weighted_axis1_sum(&y).unwrap(), 173.0);
    assert_eq!(matrix.bilinear_form(&x, &y).unwrap(), 467.0);

    let f32_a = Array::from_vec(vec![1, 2], vec![2.0_f32, 3.0]).unwrap();
    let f32_b = Array::from_vec(vec![2, 1], vec![4.0_f32, 5.0]).unwrap();
    let out = f32_a.dot2d(&f32_b).unwrap();
    assert_eq!(out.as_slice(), &[23.0]);
    let f32_inner = f32_a
        .inner2d(&Array::from_vec(vec![1, 2], vec![6.0_f32, 7.0]).unwrap())
        .unwrap();
    assert_eq!(f32_inner.shape(), &[1, 1]);
    assert_eq!(f32_inner.as_slice(), &[33.0]);
}

#[test]
fn performs_float_dot_on_transposed_blas_views() {
    let left = Array::from_vec(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let right_source = Array::from_vec(
        vec![4, 3],
        vec![
            7.0_f64, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
        ],
    )
    .unwrap();
    let right_t = right_source.transpose();
    let out = left.view().dot2d(&right_t).unwrap();
    assert_eq!(out.shape(), &[2, 4]);
    assert_eq!(
        out.as_slice(),
        &[50.0, 68.0, 86.0, 104.0, 122.0, 167.0, 212.0, 257.0]
    );

    let left_source = Array::from_vec(vec![3, 2], vec![1.0_f64, 4.0, 2.0, 5.0, 3.0, 6.0]).unwrap();
    let left_t = left_source.transpose();
    let right = Array::from_vec(vec![3, 2], vec![7.0_f64, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
    let out = left_t.dot2d(&right.view()).unwrap();
    assert_eq!(out.shape(), &[2, 2]);
    assert_eq!(out.as_slice(), &[58.0, 64.0, 139.0, 154.0]);

    let gram_right = left.transpose();
    let out = left.view().matmul(&gram_right).unwrap();
    assert_eq!(out.shape(), &[2, 2]);
    assert_eq!(out.as_slice(), &[14.0, 32.0, 32.0, 77.0]);

    let gram_left = left.transpose();
    let out = gram_left.matmul(&left.view()).unwrap();
    assert_eq!(out.shape(), &[3, 3]);
    assert_eq!(
        out.as_slice(),
        &[17.0, 22.0, 27.0, 22.0, 29.0, 36.0, 27.0, 36.0, 45.0]
    );

    let left32 = Array::from_vec(vec![2, 3], vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let right32_source = Array::from_vec(
        vec![4, 3],
        vec![
            7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
        ],
    )
    .unwrap();
    let right32_t = right32_source.transpose();
    let out = left32.view().dot2d(&right32_t).unwrap();
    assert_eq!(out.shape(), &[2, 4]);
    assert_eq!(
        out.as_slice(),
        &[50.0, 68.0, 86.0, 104.0, 122.0, 167.0, 212.0, 257.0]
    );

    let left32_source =
        Array::from_vec(vec![3, 2], vec![1.0_f32, 4.0, 2.0, 5.0, 3.0, 6.0]).unwrap();
    let left32_t = left32_source.transpose();
    let right32 = Array::from_vec(vec![3, 2], vec![7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
    let out = left32_t.dot2d(&right32.view()).unwrap();
    assert_eq!(out.shape(), &[2, 2]);
    assert_eq!(out.as_slice(), &[58.0, 64.0, 139.0, 154.0]);
}

#[test]
fn performs_numpy_style_matmul() {
    let left = Array::from_vec(vec![3], vec![1_i64, 2, 3]).unwrap();
    let right = Array::from_vec(vec![3], vec![4_i64, 5, 6]).unwrap();
    let out = left.matmul(&right).unwrap();
    assert_eq!(out.shape(), &[]);
    assert_eq!(out.as_slice(), &[32]);

    let matrix = Array::from_vec(vec![2, 3], vec![1_i64, 2, 3, 4, 5, 6]).unwrap();
    let vector = Array::from_vec(vec![3], vec![7_i64, 8, 9]).unwrap();
    let out = matrix.matmul(&vector).unwrap();
    assert_eq!(out.shape(), &[2]);
    assert_eq!(out.as_slice(), &[50, 122]);

    let vector = Array::from_vec(vec![2], vec![7_i64, 8]).unwrap();
    let matrix = Array::from_vec(vec![2, 3], vec![1_i64, 2, 3, 4, 5, 6]).unwrap();
    let out = vector.matmul(&matrix).unwrap();
    assert_eq!(out.shape(), &[3]);
    assert_eq!(out.as_slice(), &[39, 54, 69]);

    let batched_left =
        Array::from_vec(vec![2, 2, 3], vec![1_i64, 2, 3, 4, 5, 6, 2, 0, 1, 3, 1, 4]).unwrap();
    let batched_right = Array::from_vec(
        vec![2, 3, 2],
        vec![7_i64, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6],
    )
    .unwrap();
    let out = batched_left.matmul(&batched_right).unwrap();
    assert_eq!(out.shape(), &[2, 2, 2]);
    assert_eq!(out.as_slice(), &[58, 64, 139, 154, 7, 10, 26, 34]);

    let broadcasted_right = Array::from_vec(vec![1, 3, 2], vec![7_i64, 8, 9, 10, 11, 12]).unwrap();
    let out = batched_left.matmul(&broadcasted_right).unwrap();
    assert_eq!(out.shape(), &[2, 2, 2]);
    assert_eq!(out.as_slice(), &[58, 64, 139, 154, 25, 28, 74, 82]);

    let err = Array::from_vec(vec![], vec![1_i64])
        .unwrap()
        .matmul(&right)
        .unwrap_err();
    assert!(matches!(err, NumRsError::InvalidShape(_)));
}

#[test]
fn performs_numpy_style_tensordot() {
    let left = Array::from_vec(vec![2, 3], (0_i64..6).collect()).unwrap();
    let right = Array::from_vec(vec![3, 4], (0_i64..12).collect()).unwrap();
    let out = left.tensordot_axes(&right, &[1], &[0]).unwrap();
    assert_eq!(out.shape(), &[2, 4]);
    assert_eq!(out.as_slice(), &[20, 23, 26, 29, 56, 68, 80, 92]);

    let left = Array::from_vec(vec![2, 3, 4], (0_i64..24).collect()).unwrap();
    let right = Array::from_vec(vec![3, 2, 5], (0_i64..30).collect()).unwrap();
    let out = left.tensordot_axes(&right, &[1, 0], &[0, 1]).unwrap();
    assert_eq!(out.shape(), &[4, 5]);
    assert_eq!(
        out.as_slice(),
        &[
            1000, 1060, 1120, 1180, 1240, 1075, 1141, 1207, 1273, 1339, 1150, 1222, 1294, 1366,
            1438, 1225, 1303, 1381, 1459, 1537
        ]
    );

    let err = left.tensordot_axes(&right, &[1, 1], &[0, 1]).unwrap_err();
    assert!(matches!(err, NumRsError::DuplicateAxis(1)));

    let err = left.tensordot_axes(&right, &[2], &[2]).unwrap_err();
    assert!(matches!(err, NumRsError::DotShapeMismatch { .. }));
}

#[test]
fn returns_clear_shape_errors() {
    let bad = Array::from_vec(vec![2, 2], vec![1_i64, 2, 3]).unwrap_err();
    assert!(matches!(
        bad,
        NumRsError::ShapeDataMismatch {
            expected: 4,
            actual: 3,
            ..
        }
    ));

    let a = Array::from_vec(vec![2, 3], vec![1_i64, 2, 3, 4, 5, 6]).unwrap();
    let b = Array::from_vec(vec![4], vec![1_i64, 2, 3, 4]).unwrap();
    assert!(matches!(
        a.add(&b).unwrap_err(),
        NumRsError::BroadcastMismatch { .. }
    ));
}
