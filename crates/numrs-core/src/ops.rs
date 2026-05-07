use std::ops::{Add, Div, Mul, Sub};

use num_complex::{Complex32, Complex64};

use crate::array::{axis_without, Array, ArrayView};
use crate::error::{NumRsError, Result};
use crate::layout::{broadcast_strides, OffsetIter};
use crate::shape::{broadcast_shape, normalize_axis, size_of_shape};

impl<T> Array<T>
where
    T: Copy,
{
    pub fn map<U, F>(&self, f: F) -> Result<Array<U>>
    where
        F: Fn(T) -> U,
    {
        self.view().map(f)
    }

    pub fn elementwise_binary<F>(&self, rhs: &Array<T>, f: F) -> Result<Array<T>>
    where
        F: Fn(T, T) -> T,
    {
        self.view().elementwise_binary(&rhs.view(), f)
    }

    pub fn eq_elem<U>(&self, rhs: &Array<U>) -> Result<Array<bool>>
    where
        U: Copy,
        T: PartialEq<U>,
    {
        self.view().eq_elem(&rhs.view())
    }

    pub fn elementwise_assign<F>(&mut self, rhs: &Array<T>, f: F) -> Result<()>
    where
        F: Fn(T, T) -> T,
    {
        let rhs = rhs.view();
        let shape = self.shape().to_vec();

        if rhs.shape() == shape {
            if let Some(rhs_slice) = rhs.contiguous_slice() {
                for (left, right) in self
                    .raw_data_mut()
                    .iter_mut()
                    .zip(rhs_slice.iter().copied())
                {
                    *left = f(*left, right);
                }
                return Ok(());
            }
        }

        let rhs_strides = broadcast_strides(rhs.shape(), rhs.strides(), &shape)?;
        let rhs_offsets = OffsetIter::new(&shape, &rhs_strides, rhs.offset())?;
        let rhs_data = rhs.raw_data();

        for (left, rhs_offset) in self.raw_data_mut().iter_mut().zip(rhs_offsets) {
            *left = f(*left, rhs_data[rhs_offset]);
        }

        Ok(())
    }

    pub fn elementwise_binary_sum<F>(&self, rhs: &Array<T>, f: F) -> Result<T>
    where
        F: Fn(T, T) -> T,
        T: Default + Add<Output = T>,
    {
        self.view().elementwise_binary_sum(&rhs.view(), f)
    }
}

impl<T> Array<T>
where
    T: Copy + Add<Output = T>,
{
    pub fn add(&self, rhs: &Array<T>) -> Result<Array<T>> {
        self.view().add(&rhs.view())
    }

    pub fn add_assign(&mut self, rhs: &Array<T>) -> Result<()> {
        self.elementwise_assign(rhs, |l, r| l + r)
    }

    pub fn add_sum(&self, rhs: &Array<T>) -> Result<T>
    where
        T: Default,
    {
        self.elementwise_binary_sum(rhs, |l, r| l + r)
    }
}

impl<T> Array<T>
where
    T: SubKernel,
{
    pub fn sub(&self, rhs: &Array<T>) -> Result<Array<T>> {
        self.view().sub(&rhs.view())
    }

    pub fn sub_assign(&mut self, rhs: &Array<T>) -> Result<()> {
        self.elementwise_assign(rhs, |l, r| l - r)
    }
}

impl<T> Array<T>
where
    T: MulKernel,
{
    pub fn mul(&self, rhs: &Array<T>) -> Result<Array<T>> {
        self.view().mul(&rhs.view())
    }
}

impl<T> Array<T>
where
    T: Copy + Mul<Output = T>,
{
    pub fn mul_assign(&mut self, rhs: &Array<T>) -> Result<()> {
        self.elementwise_assign(rhs, |l, r| l * r)
    }
}

impl<T> Array<T>
where
    T: Copy + Div<Output = T>,
{
    pub fn div(&self, rhs: &Array<T>) -> Result<Array<T>> {
        self.view().div(&rhs.view())
    }

    pub fn div_assign(&mut self, rhs: &Array<T>) -> Result<()> {
        self.elementwise_assign(rhs, |l, r| l / r)
    }
}

impl<T> Array<T>
where
    T: SumKernel,
{
    pub fn sum_all(&self) -> Result<T> {
        if let Some(value) = self.uniform_value() {
            return Ok(T::sum_uniform(*value, self.len()));
        }
        self.view().sum_all()
    }

    pub fn sum_axis(&self, axis: isize) -> Result<Array<T>> {
        self.view().sum_axis(axis)
    }
}

impl<T> Array<T>
where
    T: ProductKernel,
{
    pub fn prod_all(&self) -> Result<T> {
        if let Some(value) = self.uniform_value() {
            return Ok(T::prod_uniform(*value, self.len()));
        }
        self.view().prod_all()
    }
}

impl<T> Array<T>
where
    T: ExtremeKernel,
{
    pub fn min_all(&self) -> Result<T> {
        if let Some(value) = self.uniform_value() {
            if !self.is_empty() {
                return Ok(*value);
            }
        }
        self.view().min_all()
    }

    pub fn max_all(&self) -> Result<T> {
        if let Some(value) = self.uniform_value() {
            if !self.is_empty() {
                return Ok(*value);
            }
        }
        self.view().max_all()
    }
}

impl<T> Array<T>
where
    T: ArgReduceKernel,
{
    pub fn argmax(&self) -> Result<usize> {
        if !self.is_empty() && self.has_uniform_storage() {
            return Ok(0);
        }
        self.view().argmax()
    }

    pub fn argmin(&self) -> Result<usize> {
        if !self.is_empty() && self.has_uniform_storage() {
            return Ok(0);
        }
        self.view().argmin()
    }
}

impl<T> Array<T>
where
    T: DotKernel,
{
    pub fn dot2d(&self, rhs: &Array<T>) -> Result<Array<T>> {
        T::dot2d_arrays(self, rhs)
    }
}

impl<T> Array<T>
where
    T: Inner2dKernel,
{
    pub fn inner2d(&self, rhs: &Array<T>) -> Result<Array<T>> {
        T::inner2d_arrays(self, rhs)
    }
}

impl<T> Array<T>
where
    T: MatmulKernel,
{
    pub fn matmul(&self, rhs: &Array<T>) -> Result<Array<T>> {
        T::matmul_arrays(self, rhs)
    }
}

impl<T> Array<T>
where
    T: TensordotKernel,
{
    pub fn tensordot_axes(
        &self,
        rhs: &Array<T>,
        left_axes: &[isize],
        right_axes: &[isize],
    ) -> Result<Array<T>> {
        self.view()
            .tensordot_axes(&rhs.view(), left_axes, right_axes)
    }
}

impl Array<f64> {
    pub fn mean_all(&self) -> Result<f64> {
        if let Some(value) = self.uniform_value() {
            if !self.is_empty() {
                return Ok(*value);
            }
        }
        self.view().mean_all()
    }

    pub fn mean_axis(&self, axis: isize) -> Result<Array<f64>> {
        self.view().mean_axis(axis)
    }

    pub fn var_all(&self) -> Result<f64> {
        if self.has_uniform_storage() && !self.is_empty() {
            return Ok(0.0);
        }
        self.view().var_all()
    }

    pub fn std_all(&self) -> Result<f64> {
        if self.has_uniform_storage() && !self.is_empty() {
            return Ok(0.0);
        }
        self.view().std_all()
    }

    pub fn norm_l2(&self) -> Result<f64> {
        self.view().norm_l2()
    }

    pub fn det(&self) -> Result<f64> {
        self.view().det()
    }

    pub fn solve(&self, rhs: &Array<f64>) -> Result<Array<f64>> {
        self.view().solve(&rhs.view())
    }

    pub fn outer_product(&self, rhs: &Array<f64>) -> Result<Array<f64>> {
        self.view().outer_product(&rhs.view())
    }

    pub fn mul_scalar(&self, value: f64) -> Result<Array<f64>> {
        self.view().mul_scalar(value)
    }

    pub fn weighted_axis1_sum(&self, weights: &Array<f64>) -> Result<f64> {
        self.view().weighted_axis1_sum(&weights.view())
    }

    pub fn bilinear_form(
        &self,
        left_weights: &Array<f64>,
        right_weights: &Array<f64>,
    ) -> Result<f64> {
        self.view()
            .bilinear_form(&left_weights.view(), &right_weights.view())
    }

    pub fn add_sum_f64(&self, rhs: &Array<f64>) -> Result<f64> {
        self.view().add_sum_f64(&rhs.view())
    }

    pub fn add_outer2d_f64(&self, rhs: &Array<f64>) -> Result<Array<f64>> {
        self.view().add_outer2d_f64(&rhs.view())
    }
}

impl Array<f32> {
    pub fn mean_all(&self) -> Result<f64> {
        if let Some(value) = self.uniform_value() {
            if !self.is_empty() {
                return Ok(*value as f64);
            }
        }
        self.view().mean_all()
    }

    pub fn var_all(&self) -> Result<f64> {
        if self.has_uniform_storage() && !self.is_empty() {
            return Ok(0.0);
        }
        self.view().var_all()
    }

    pub fn std_all(&self) -> Result<f64> {
        if self.has_uniform_storage() && !self.is_empty() {
            return Ok(0.0);
        }
        self.view().std_all()
    }
}

impl Array<i64> {
    pub fn mean_all(&self) -> Result<f64> {
        if let Some(value) = self.uniform_value() {
            if !self.is_empty() {
                return Ok(*value as f64);
            }
        }
        self.view().mean_all()
    }

    pub fn var_all(&self) -> Result<f64> {
        if self.has_uniform_storage() && !self.is_empty() {
            return Ok(0.0);
        }
        self.view().var_all()
    }

    pub fn std_all(&self) -> Result<f64> {
        if self.has_uniform_storage() && !self.is_empty() {
            return Ok(0.0);
        }
        self.view().std_all()
    }
}

impl Array<u64> {
    pub fn mean_all(&self) -> Result<f64> {
        if let Some(value) = self.uniform_value() {
            if !self.is_empty() {
                return Ok(*value as f64);
            }
        }
        self.view().mean_all()
    }

    pub fn var_all(&self) -> Result<f64> {
        if self.has_uniform_storage() && !self.is_empty() {
            return Ok(0.0);
        }
        self.view().var_all()
    }

    pub fn std_all(&self) -> Result<f64> {
        if self.has_uniform_storage() && !self.is_empty() {
            return Ok(0.0);
        }
        self.view().std_all()
    }
}

impl Array<bool> {
    pub fn mean_all(&self) -> Result<f64> {
        if let Some(value) = self.uniform_value() {
            if !self.is_empty() {
                return Ok(if *value { 1.0 } else { 0.0 });
            }
        }
        self.view().mean_all()
    }

    pub fn var_all(&self) -> Result<f64> {
        if self.has_uniform_storage() && !self.is_empty() {
            return Ok(0.0);
        }
        self.view().var_all()
    }

    pub fn std_all(&self) -> Result<f64> {
        if self.has_uniform_storage() && !self.is_empty() {
            return Ok(0.0);
        }
        self.view().std_all()
    }

    pub fn prod_all(&self) -> Result<bool> {
        if let Some(value) = self.uniform_value() {
            return Ok(*value || self.is_empty());
        }
        self.view().prod_all()
    }

    pub fn logical_not(&self) -> Result<Array<bool>> {
        self.view().logical_not()
    }

    pub fn logical_and(&self, rhs: &Array<bool>) -> Result<Array<bool>> {
        self.view().logical_and(&rhs.view())
    }

    pub fn logical_or(&self, rhs: &Array<bool>) -> Result<Array<bool>> {
        self.view().logical_or(&rhs.view())
    }
}

impl Array<Complex32> {
    pub fn mean_all(&self) -> Result<Complex32> {
        if let Some(value) = self.uniform_value() {
            if !self.is_empty() {
                return Ok(*value);
            }
        }
        self.view().mean_all()
    }

    pub fn var_all(&self) -> Result<f64> {
        if self.uniform_value().is_some() && !self.is_empty() {
            return Ok(0.0);
        }
        self.view().var_all()
    }

    pub fn std_all(&self) -> Result<f64> {
        if self.uniform_value().is_some() && !self.is_empty() {
            return Ok(0.0);
        }
        self.view().std_all()
    }
}

impl<'a, T> ArrayView<'a, T>
where
    T: Copy,
{
    pub fn map<U, F>(&self, f: F) -> Result<Array<U>>
    where
        F: Fn(T) -> U,
    {
        let out = if let Some(slice) = self.contiguous_slice() {
            slice.iter().copied().map(f).collect()
        } else {
            let mut out = Vec::with_capacity(self.len());
            for offset in self.offset_iter()? {
                out.push(f(self.data_at(offset)));
            }
            out
        };
        Ok(Array::from_vec_trusted(self.shape().to_vec(), out))
    }

    pub fn elementwise_binary<F>(&self, rhs: &ArrayView<'_, T>, f: F) -> Result<Array<T>>
    where
        F: Fn(T, T) -> T,
    {
        let out_shape = broadcast_shape(self.shape(), rhs.shape())?;
        let len = size_of_shape(&out_shape)?;

        if self.shape() == rhs.shape() {
            if let (Some(left), Some(right)) = (self.contiguous_slice(), rhs.contiguous_slice()) {
                let out = left
                    .iter()
                    .copied()
                    .zip(right.iter().copied())
                    .map(|(l, r)| f(l, r))
                    .collect();
                return Array::from_vec(out_shape, out);
            }
        }

        if let Some(out) = self.elementwise_binary_outer_2d(rhs, &f)? {
            return Array::from_vec(out_shape, out);
        }

        let left_strides = broadcast_strides(self.shape(), self.strides(), &out_shape)?;
        let right_strides = broadcast_strides(rhs.shape(), rhs.strides(), &out_shape)?;
        let left_offsets = OffsetIter::new(&out_shape, &left_strides, self.offset())?;
        let right_offsets = OffsetIter::new(&out_shape, &right_strides, rhs.offset())?;

        let mut out = Vec::with_capacity(len);
        for (left_offset, right_offset) in left_offsets.zip(right_offsets) {
            out.push(f(self.data_at(left_offset), rhs.data_at(right_offset)));
        }
        Array::from_vec(out_shape, out)
    }

    fn elementwise_binary_outer_2d<F>(
        &self,
        rhs: &ArrayView<'_, T>,
        f: &F,
    ) -> Result<Option<Vec<T>>>
    where
        F: Fn(T, T) -> T,
    {
        let (Some(left), Some(right)) = (self.contiguous_slice(), rhs.contiguous_slice()) else {
            return Ok(None);
        };

        match (self.shape(), rhs.shape()) {
            ([rows, cols], [rhs_cols]) if cols == rhs_cols => {
                let mut out = Vec::with_capacity(rows * cols);
                for row in left.chunks_exact(*cols).take(*rows) {
                    for (left_value, right_value) in row.iter().copied().zip(right.iter().copied())
                    {
                        out.push(f(left_value, right_value));
                    }
                }
                Ok(Some(out))
            }
            ([lhs_cols], [rows, cols]) if lhs_cols == cols => {
                let mut out = Vec::with_capacity(rows * cols);
                for row in right.chunks_exact(*cols).take(*rows) {
                    for (left_value, right_value) in left.iter().copied().zip(row.iter().copied()) {
                        out.push(f(left_value, right_value));
                    }
                }
                Ok(Some(out))
            }
            ([rows, 1], [1, cols]) => {
                let mut out = Vec::with_capacity(rows * cols);
                for left_value in left.iter().take(*rows).copied() {
                    for right_value in right.iter().take(*cols).copied() {
                        out.push(f(left_value, right_value));
                    }
                }
                Ok(Some(out))
            }
            ([1, cols], [rows, 1]) => {
                let mut out = Vec::with_capacity(rows * cols);
                for right_value in right.iter().take(*rows).copied() {
                    for left_value in left.iter().take(*cols).copied() {
                        out.push(f(left_value, right_value));
                    }
                }
                Ok(Some(out))
            }
            _ => Ok(None),
        }
    }

    pub fn eq_elem<U>(&self, rhs: &ArrayView<'_, U>) -> Result<Array<bool>>
    where
        U: Copy,
        T: PartialEq<U>,
    {
        let out_shape = broadcast_shape(self.shape(), rhs.shape())?;
        let len = size_of_shape(&out_shape)?;
        let left_strides = broadcast_strides(self.shape(), self.strides(), &out_shape)?;
        let right_strides = broadcast_strides(rhs.shape(), rhs.strides(), &out_shape)?;
        let left_offsets = OffsetIter::new(&out_shape, &left_strides, self.offset())?;
        let right_offsets = OffsetIter::new(&out_shape, &right_strides, rhs.offset())?;

        let mut out = Vec::with_capacity(len);
        for (left_offset, right_offset) in left_offsets.zip(right_offsets) {
            out.push(self.data_at(left_offset) == rhs.data_at(right_offset));
        }
        Array::from_vec(out_shape, out)
    }

    pub fn elementwise_binary_sum<F>(&self, rhs: &ArrayView<'_, T>, f: F) -> Result<T>
    where
        F: Fn(T, T) -> T,
        T: Default + Add<Output = T>,
    {
        let out_shape = broadcast_shape(self.shape(), rhs.shape())?;

        if self.shape() == rhs.shape() {
            if let (Some(left), Some(right)) = (self.contiguous_slice(), rhs.contiguous_slice()) {
                return Ok(left
                    .iter()
                    .copied()
                    .zip(right.iter().copied())
                    .fold(T::default(), |acc, (l, r)| acc + f(l, r)));
            }
        }

        let left_strides = broadcast_strides(self.shape(), self.strides(), &out_shape)?;
        let right_strides = broadcast_strides(rhs.shape(), rhs.strides(), &out_shape)?;
        let left_offsets = OffsetIter::new(&out_shape, &left_strides, self.offset())?;
        let right_offsets = OffsetIter::new(&out_shape, &right_strides, rhs.offset())?;

        let mut acc = T::default();
        for (left_offset, right_offset) in left_offsets.zip(right_offsets) {
            acc = acc + f(self.data_at(left_offset), rhs.data_at(right_offset));
        }
        Ok(acc)
    }

    fn data_at(&self, offset: usize) -> T {
        self.data()[offset]
    }

    fn data(&self) -> &'a [T] {
        self.raw_data()
    }
}

impl<'a, T> ArrayView<'a, T>
where
    T: Copy + Add<Output = T>,
{
    pub fn add(&self, rhs: &ArrayView<'_, T>) -> Result<Array<T>> {
        self.elementwise_binary(rhs, |l, r| l + r)
    }
}

impl<'a, T> ArrayView<'a, T>
where
    T: SubKernel,
{
    pub fn sub(&self, rhs: &ArrayView<'_, T>) -> Result<Array<T>> {
        <T as SubKernel>::sub(self, rhs)
    }
}

pub trait SubKernel: Copy + Sub<Output = Self> {
    fn sub(left: &ArrayView<'_, Self>, right: &ArrayView<'_, Self>) -> Result<Array<Self>> {
        left.elementwise_binary(right, |l, r| l - r)
    }
}

macro_rules! impl_sub_kernel {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl SubKernel for $ty {}
        )+
    };
}

impl_sub_kernel!(f32, Complex64, Complex32);

macro_rules! impl_wrapping_sub_kernel {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl SubKernel for $ty {
                fn sub(left: &ArrayView<'_, Self>, right: &ArrayView<'_, Self>) -> Result<Array<Self>> {
                    left.elementwise_binary(right, |l, r| <$ty>::wrapping_sub(l, r))
                }
            }
        )+
    };
}

impl_wrapping_sub_kernel!(i64, i32, i16, i8, u64, u32, u16, u8);

impl SubKernel for f64 {
    fn sub(left: &ArrayView<'_, Self>, right: &ArrayView<'_, Self>) -> Result<Array<Self>> {
        if let Some((shape, out)) = sub_row_broadcast_f64(left, right)? {
            return Array::from_vec(shape, out);
        }
        left.elementwise_binary(right, |l, r| l - r)
    }
}

fn sub_row_broadcast_f64(
    left: &ArrayView<'_, f64>,
    right: &ArrayView<'_, f64>,
) -> Result<Option<(Vec<usize>, Vec<f64>)>> {
    let out_shape = broadcast_shape(left.shape(), right.shape())?;
    let (Some(left_data), Some(right_data)) = (left.contiguous_slice(), right.contiguous_slice())
    else {
        return Ok(None);
    };

    match (left.shape(), right.shape()) {
        ([rows, cols], [rhs_cols]) if cols == rhs_cols => {
            let mut out = vec![0.0; rows * cols];
            for (out_row, row) in out
                .chunks_exact_mut(*cols)
                .zip(left_data.chunks_exact(*cols).take(*rows))
            {
                for ((slot, value), rhs_value) in
                    out_row.iter_mut().zip(row.iter()).zip(right_data.iter())
                {
                    *slot = value - rhs_value;
                }
            }
            Ok(Some((out_shape, out)))
        }
        ([lhs_cols], [rows, cols]) if lhs_cols == cols => {
            let mut out = vec![0.0; rows * cols];
            for (out_row, row) in out
                .chunks_exact_mut(*cols)
                .zip(right_data.chunks_exact(*cols).take(*rows))
            {
                for ((slot, lhs_value), value) in
                    out_row.iter_mut().zip(left_data.iter()).zip(row.iter())
                {
                    *slot = lhs_value - value;
                }
            }
            Ok(Some((out_shape, out)))
        }
        _ => Ok(None),
    }
}

impl<'a, T> ArrayView<'a, T>
where
    T: MulKernel,
{
    pub fn mul(&self, rhs: &ArrayView<'_, T>) -> Result<Array<T>> {
        <T as MulKernel>::mul(self, rhs)
    }
}

pub trait MulKernel: Copy + Mul<Output = Self> {
    fn mul(left: &ArrayView<'_, Self>, right: &ArrayView<'_, Self>) -> Result<Array<Self>> {
        left.elementwise_binary(right, |l, r| l * r)
    }
}

macro_rules! impl_mul_kernel {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl MulKernel for $ty {}
        )+
    };
}

impl_mul_kernel!(f32, Complex64, Complex32);

impl MulKernel for f64 {
    fn mul(left: &ArrayView<'_, Self>, right: &ArrayView<'_, Self>) -> Result<Array<Self>> {
        if let Some((shape, out)) = mul_trailing_broadcast_f64(left, right)? {
            return Array::from_vec(shape, out);
        }
        left.elementwise_binary(right, |l, r| l * r)
    }
}

fn mul_trailing_broadcast_f64(
    left: &ArrayView<'_, f64>,
    right: &ArrayView<'_, f64>,
) -> Result<Option<(Vec<usize>, Vec<f64>)>> {
    let out_shape = broadcast_shape(left.shape(), right.shape())?;
    let out_len = size_of_shape(&out_shape)?;
    let (Some(left_data), Some(right_data)) = (left.contiguous_slice(), right.contiguous_slice())
    else {
        return Ok(None);
    };

    match (left.shape(), right.shape()) {
        ([rows, cols], [_batch, rhs_rows, rhs_cols]) if rows == rhs_rows && cols == rhs_cols => {
            let plane_len = rows.checked_mul(*cols).ok_or(NumRsError::ShapeOverflow)?;
            debug_assert_eq!(plane_len, left_data.len());
            debug_assert_eq!(out_len, right_data.len());
            let out = crate::blas::mul_repeated_tile_f64(left_data, right_data);
            Ok(Some((out_shape, out)))
        }
        ([_batch, rows, cols], [rhs_rows, rhs_cols]) if rows == rhs_rows && cols == rhs_cols => {
            let plane_len = rows.checked_mul(*cols).ok_or(NumRsError::ShapeOverflow)?;
            debug_assert_eq!(out_len, left_data.len());
            debug_assert_eq!(plane_len, right_data.len());
            let out = crate::blas::mul_repeated_tile_f64(right_data, left_data);
            Ok(Some((out_shape, out)))
        }
        _ => Ok(None),
    }
}

macro_rules! impl_wrapping_mul_kernel {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl MulKernel for $ty {
                fn mul(left: &ArrayView<'_, Self>, right: &ArrayView<'_, Self>) -> Result<Array<Self>> {
                    left.elementwise_binary(right, |l, r| <$ty>::wrapping_mul(l, r))
                }
            }
        )+
    };
}

impl_wrapping_mul_kernel!(i64, i32, i16, i8, u64, u32, u16, u8);

impl<'a, T> ArrayView<'a, T>
where
    T: Copy + Div<Output = T>,
{
    pub fn div(&self, rhs: &ArrayView<'_, T>) -> Result<Array<T>> {
        self.elementwise_binary(rhs, |l, r| l / r)
    }
}

impl<'a, T> ArrayView<'a, T>
where
    T: SumKernel,
{
    pub fn sum_all(&self) -> Result<T> {
        T::sum_all(self)
    }

    pub fn sum_axis(&self, axis: isize) -> Result<Array<T>> {
        let (axis, out_shape) = axis_without(self.shape(), axis)?;
        let out_len = size_of_shape(&out_shape)?;
        let mut out = Vec::with_capacity(out_len);

        if out_shape.is_empty() {
            return Array::from_vec(out_shape, vec![self.sum_all()?]);
        }

        for out_linear in 0..out_len {
            let mut remainder = out_linear;
            let mut base_offset = self.offset();
            for out_axis in (0..out_shape.len()).rev() {
                let coord = remainder % out_shape[out_axis];
                remainder /= out_shape[out_axis];
                let source_axis = if out_axis < axis {
                    out_axis
                } else {
                    out_axis + 1
                };
                base_offset += coord as isize * self.strides()[source_axis];
            }

            let mut acc = T::default();
            for k in 0..self.shape()[axis] {
                let offset = base_offset + k as isize * self.strides()[axis];
                acc = acc + self.data_at(offset as usize);
            }
            out.push(acc);
        }

        Array::from_vec(out_shape, out)
    }
}

impl<'a, T> ArrayView<'a, T>
where
    T: ProductKernel,
{
    pub fn prod_all(&self) -> Result<T> {
        T::prod_all(self)
    }
}

impl<'a, T> ArrayView<'a, T>
where
    T: ExtremeKernel,
{
    pub fn min_all(&self) -> Result<T> {
        T::min_all(self)
    }

    pub fn max_all(&self) -> Result<T> {
        T::max_all(self)
    }
}

impl<'a, T> ArrayView<'a, T>
where
    T: ArgReduceKernel,
{
    pub fn argmax(&self) -> Result<usize> {
        if self.is_empty() {
            return Err(NumRsError::EmptyReduction);
        }

        if let Some(slice) = self.contiguous_slice() {
            return T::argmax_contiguous(slice).ok_or(NumRsError::EmptyReduction);
        }

        generic_argmax_view(self)
    }

    pub fn argmin(&self) -> Result<usize> {
        if self.is_empty() {
            return Err(NumRsError::EmptyReduction);
        }

        if let Some(slice) = self.contiguous_slice() {
            return T::argmin_contiguous(slice).ok_or(NumRsError::EmptyReduction);
        }

        generic_argmin_view(self)
    }
}

pub trait ExtremeKernel: Copy {
    fn min_all(view: &ArrayView<'_, Self>) -> Result<Self>;

    fn max_all(view: &ArrayView<'_, Self>) -> Result<Self>;
}

macro_rules! impl_partial_ord_extreme_kernel {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl ExtremeKernel for $ty {
                fn min_all(view: &ArrayView<'_, Self>) -> Result<Self> {
                    generic_extreme_by(view, |candidate, best| candidate < best)
                }

                fn max_all(view: &ArrayView<'_, Self>) -> Result<Self> {
                    generic_extreme_by(view, |candidate, best| candidate > best)
                }
            }
        )+
    };
}

impl_partial_ord_extreme_kernel!(f64, f32, i64, i32, i16, i8, u64, u32, u16, u8, bool);

impl ExtremeKernel for Complex32 {
    fn min_all(view: &ArrayView<'_, Self>) -> Result<Self> {
        complex32_extreme_all(view, complex32_less)
    }

    fn max_all(view: &ArrayView<'_, Self>) -> Result<Self> {
        complex32_extreme_all(view, complex32_greater)
    }
}

fn generic_extreme_by<T, F>(view: &ArrayView<'_, T>, better: F) -> Result<T>
where
    T: Copy,
    F: Fn(T, T) -> bool,
{
    if view.is_empty() {
        return Err(NumRsError::EmptyReduction);
    }

    if let Some(slice) = view.contiguous_slice() {
        let (first, rest) = slice.split_first().expect("empty checked above");
        let mut best_value = *first;
        for value in rest.iter().copied() {
            if better(value, best_value) {
                best_value = value;
            }
        }
        return Ok(best_value);
    }

    let mut offsets = view.offset_iter()?;
    let first_offset = offsets.next().expect("empty checked above");
    let mut best_value = view.data_at(first_offset);

    for offset in offsets {
        let value = view.data_at(offset);
        if better(value, best_value) {
            best_value = value;
        }
    }

    Ok(best_value)
}

pub trait ArgReduceKernel: Copy + PartialOrd {
    fn argmax_contiguous(values: &[Self]) -> Option<usize> {
        generic_argmax_slice(values)
    }

    fn argmin_contiguous(values: &[Self]) -> Option<usize> {
        generic_argmin_slice(values)
    }
}

macro_rules! impl_generic_arg_reduce_kernel {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl ArgReduceKernel for $ty {}
        )+
    };
}

impl_generic_arg_reduce_kernel!(f64, f32, i32, i16, i8, u64, u32, u16, u8, bool);
impl ArgReduceKernel for i64 {}

fn generic_argmax_slice<T>(values: &[T]) -> Option<usize>
where
    T: Copy + PartialOrd,
{
    let (first, rest) = values.split_first()?;
    let mut best_value = *first;
    let mut best_index = 0usize;
    for (idx, value) in rest.iter().copied().enumerate() {
        if value > best_value {
            best_value = value;
            best_index = idx + 1;
        }
    }
    Some(best_index)
}

fn generic_argmin_slice<T>(values: &[T]) -> Option<usize>
where
    T: Copy + PartialOrd,
{
    let (first, rest) = values.split_first()?;
    let mut best_value = *first;
    let mut best_index = 0usize;
    for (idx, value) in rest.iter().copied().enumerate() {
        if value < best_value {
            best_value = value;
            best_index = idx + 1;
        }
    }
    Some(best_index)
}

fn generic_argmax_view<T>(view: &ArrayView<'_, T>) -> Result<usize>
where
    T: Copy + PartialOrd,
{
    let mut offsets = view.offset_iter()?;
    let first_offset = offsets.next().expect("empty checked by caller");
    let mut best_value = view.data_at(first_offset);
    let mut best_index = 0usize;

    for (linear, offset) in offsets.enumerate() {
        let value = view.data_at(offset);
        if value > best_value {
            best_value = value;
            best_index = linear + 1;
        }
    }

    Ok(best_index)
}

fn generic_argmin_view<T>(view: &ArrayView<'_, T>) -> Result<usize>
where
    T: Copy + PartialOrd,
{
    let mut offsets = view.offset_iter()?;
    let first_offset = offsets.next().expect("empty checked by caller");
    let mut best_value = view.data_at(first_offset);
    let mut best_index = 0usize;

    for (linear, offset) in offsets.enumerate() {
        let value = view.data_at(offset);
        if value < best_value {
            best_value = value;
            best_index = linear + 1;
        }
    }

    Ok(best_index)
}

pub trait SumKernel: Copy + Default + Add<Output = Self> {
    fn sum_all(view: &ArrayView<'_, Self>) -> Result<Self>;

    fn sum_uniform(value: Self, len: usize) -> Self {
        let mut acc = Self::default();
        for _ in 0..len {
            acc = acc + value;
        }
        acc
    }
}

macro_rules! impl_generic_sum_kernel {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl SumKernel for $ty {
                fn sum_all(view: &ArrayView<'_, Self>) -> Result<Self> {
                    generic_sum_all(view)
                }
            }
        )+
    };
}

impl_generic_sum_kernel!(Complex64, Complex32);

macro_rules! impl_wrapping_sum_kernel {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl SumKernel for $ty {
                fn sum_all(view: &ArrayView<'_, Self>) -> Result<Self> {
                    let mut acc = Self::default();
                    for offset in view.offset_iter()? {
                        acc = acc.wrapping_add(view.data_at(offset));
                    }
                    Ok(acc)
                }

                fn sum_uniform(value: Self, len: usize) -> Self {
                    let mut acc = Self::default();
                    for _ in 0..len {
                        acc = acc.wrapping_add(value);
                    }
                    acc
                }
            }
        )+
    };
}

impl_wrapping_sum_kernel!(i64, i32, i16, i8, u64, u32, u16, u8);

impl SumKernel for f64 {
    fn sum_all(view: &ArrayView<'_, Self>) -> Result<Self> {
        if let Some(slice) = view.contiguous_slice() {
            return Ok(crate::blas::sum_f64(slice));
        }
        generic_sum_all(view)
    }
}

impl SumKernel for f32 {
    fn sum_all(view: &ArrayView<'_, Self>) -> Result<Self> {
        if let Some(slice) = view.contiguous_slice() {
            return Ok(crate::blas::sum_f32(slice));
        }
        generic_sum_all(view)
    }
}

fn generic_sum_all<T>(view: &ArrayView<'_, T>) -> Result<T>
where
    T: Copy + Default + Add<Output = T>,
{
    let mut acc = T::default();
    for offset in view.offset_iter()? {
        acc = acc + view.data_at(offset);
    }
    Ok(acc)
}

pub trait ProductKernel: Copy + Mul<Output = Self> {
    fn one() -> Self;

    fn prod_uniform(value: Self, len: usize) -> Self {
        let mut acc = Self::one();
        for _ in 0..len {
            acc = acc * value;
        }
        acc
    }

    fn prod_all(view: &ArrayView<'_, Self>) -> Result<Self> {
        let mut acc = Self::one();
        if let Some(slice) = view.contiguous_slice() {
            for value in slice.iter().copied() {
                acc = acc * value;
            }
            return Ok(acc);
        }
        for offset in view.offset_iter()? {
            acc = acc * view.data_at(offset);
        }
        Ok(acc)
    }
}

macro_rules! impl_product_kernel {
    ($($ty:ty => $one:expr),+ $(,)?) => {
        $(
            impl ProductKernel for $ty {
                fn one() -> Self {
                    $one
                }
            }
        )+
    };
}

impl_product_kernel!(
    f64 => 1.0,
    f32 => 1.0,
    i64 => 1,
    i32 => 1,
    i16 => 1,
    i8 => 1,
    u64 => 1,
    u32 => 1,
    u16 => 1,
    u8 => 1,
);

fn complex32_powu(mut base: Complex32, mut exp: usize) -> Complex32 {
    let mut acc = Complex32::new(1.0, 0.0);
    while exp > 0 {
        if exp & 1 == 1 {
            acc *= base;
        }
        exp >>= 1;
        if exp > 0 {
            base *= base;
        }
    }
    acc
}

fn complex64_powu(mut base: Complex64, mut exp: usize) -> Complex64 {
    let mut acc = Complex64::new(1.0, 0.0);
    while exp > 0 {
        if exp & 1 == 1 {
            acc *= base;
        }
        exp >>= 1;
        if exp > 0 {
            base *= base;
        }
    }
    acc
}

impl ProductKernel for Complex32 {
    fn one() -> Self {
        Complex32::new(1.0, 0.0)
    }

    fn prod_uniform(value: Self, len: usize) -> Self {
        complex32_powu(value, len)
    }
}

impl ProductKernel for Complex64 {
    fn one() -> Self {
        Complex64::new(1.0, 0.0)
    }

    fn prod_uniform(value: Self, len: usize) -> Self {
        complex64_powu(value, len)
    }
}

impl<'a, T> ArrayView<'a, T>
where
    T: DotKernel,
{
    #[inline(always)]
    pub fn dot2d(&self, rhs: &ArrayView<'_, T>) -> Result<Array<T>> {
        T::dot2d(self, rhs)
    }
}

impl<'a, T> ArrayView<'a, T>
where
    T: Inner2dKernel,
{
    #[inline(always)]
    pub fn inner2d(&self, rhs: &ArrayView<'_, T>) -> Result<Array<T>> {
        T::inner2d(self, rhs)
    }
}

impl<'a, T> ArrayView<'a, T>
where
    T: MatmulKernel,
{
    #[inline(always)]
    pub fn matmul(&self, rhs: &ArrayView<'_, T>) -> Result<Array<T>> {
        T::matmul(self, rhs)
    }
}

impl<'a, T> ArrayView<'a, T>
where
    T: TensordotKernel,
{
    pub fn tensordot_axes(
        &self,
        rhs: &ArrayView<'_, T>,
        left_axes: &[isize],
        right_axes: &[isize],
    ) -> Result<Array<T>> {
        T::tensordot_axes(self, rhs, left_axes, right_axes)
    }
}

pub trait DotKernel: Copy + Default + Add<Output = Self> + Mul<Output = Self> {
    fn dot2d(left: &ArrayView<'_, Self>, right: &ArrayView<'_, Self>) -> Result<Array<Self>>;

    fn dot2d_arrays(left: &Array<Self>, right: &Array<Self>) -> Result<Array<Self>> {
        Self::dot2d(&left.view(), &right.view())
    }
}

macro_rules! impl_generic_dot_kernel {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl DotKernel for $ty {
                fn dot2d(
                    left: &ArrayView<'_, Self>,
                    right: &ArrayView<'_, Self>,
                ) -> Result<Array<Self>> {
                    generic_dot2d(left, right)
                }
            }
        )+
    };
}

impl_generic_dot_kernel!(Complex64, Complex32, i64, i32, i16, i8, u64, u32, u16, u8);

impl DotKernel for f64 {
    #[inline(always)]
    fn dot2d(left: &ArrayView<'_, Self>, right: &ArrayView<'_, Self>) -> Result<Array<Self>> {
        check_dot2d(left, right)?;
        dot2d_f64_unchecked(left, right)
    }

    fn dot2d_arrays(left: &Array<Self>, right: &Array<Self>) -> Result<Array<Self>> {
        check_dot2d_arrays(left, right)?;
        let rows = left.shape()[0];
        let inner = left.shape()[1];
        let cols = right.shape()[1];
        let out =
            crate::blas::dgemm_row_major(rows, inner, cols, left.as_slice(), right.as_slice());
        Ok(Array::from_vec_trusted(vec![rows, cols], out))
    }
}

#[inline(always)]
fn dot2d_f64_unchecked(
    left: &ArrayView<'_, f64>,
    right: &ArrayView<'_, f64>,
) -> Result<Array<f64>> {
    if let (Some(left_slice), Some(right_slice)) =
        (left.contiguous_slice(), right.contiguous_slice())
    {
        let rows = left.shape()[0];
        let inner = left.shape()[1];
        let cols = right.shape()[1];
        let out = crate::blas::dgemm_row_major(rows, inner, cols, left_slice, right_slice);
        return Ok(Array::from_vec_trusted(vec![rows, cols], out));
    }
    if let (Some(left_slice), Some(right_transposed)) =
        (left.contiguous_slice(), transposed_source_slice(right))
    {
        let rows = left.shape()[0];
        let inner = left.shape()[1];
        let cols = right.shape()[1];
        let out = crate::blas::dgemm_row_major_trans_b_f64(
            rows,
            inner,
            cols,
            left_slice,
            right_transposed,
        );
        return Ok(Array::from_vec_trusted(vec![rows, cols], out));
    }
    if let (Some(left_transposed), Some(right_slice)) =
        (transposed_source_slice(left), right.contiguous_slice())
    {
        let rows = left.shape()[0];
        let inner = left.shape()[1];
        let cols = right.shape()[1];
        let out = crate::blas::dgemm_row_major_trans_a_f64(
            rows,
            inner,
            cols,
            left_transposed,
            right_slice,
        );
        return Ok(Array::from_vec_trusted(vec![rows, cols], out));
    }
    generic_dot2d_unchecked(left, right)
}

impl DotKernel for f32 {
    #[inline(always)]
    fn dot2d(left: &ArrayView<'_, Self>, right: &ArrayView<'_, Self>) -> Result<Array<Self>> {
        check_dot2d(left, right)?;
        dot2d_f32_unchecked(left, right)
    }

    fn dot2d_arrays(left: &Array<Self>, right: &Array<Self>) -> Result<Array<Self>> {
        check_dot2d_arrays(left, right)?;
        let rows = left.shape()[0];
        let inner = left.shape()[1];
        let cols = right.shape()[1];
        let out =
            crate::blas::sgemm_row_major(rows, inner, cols, left.as_slice(), right.as_slice());
        Ok(Array::from_vec_trusted(vec![rows, cols], out))
    }
}

#[inline(always)]
fn dot2d_f32_unchecked(
    left: &ArrayView<'_, f32>,
    right: &ArrayView<'_, f32>,
) -> Result<Array<f32>> {
    if let (Some(left_slice), Some(right_slice)) =
        (left.contiguous_slice(), right.contiguous_slice())
    {
        let rows = left.shape()[0];
        let inner = left.shape()[1];
        let cols = right.shape()[1];
        let out = crate::blas::sgemm_row_major(rows, inner, cols, left_slice, right_slice);
        return Ok(Array::from_vec_trusted(vec![rows, cols], out));
    }
    if let (Some(left_slice), Some(right_transposed)) =
        (left.contiguous_slice(), transposed_source_slice(right))
    {
        let rows = left.shape()[0];
        let inner = left.shape()[1];
        let cols = right.shape()[1];
        let out =
            crate::blas::sgemm_row_major_trans_b(rows, inner, cols, left_slice, right_transposed);
        return Ok(Array::from_vec_trusted(vec![rows, cols], out));
    }
    if let (Some(left_transposed), Some(right_slice)) =
        (transposed_source_slice(left), right.contiguous_slice())
    {
        let rows = left.shape()[0];
        let inner = left.shape()[1];
        let cols = right.shape()[1];
        let out =
            crate::blas::sgemm_row_major_trans_a(rows, inner, cols, left_transposed, right_slice);
        return Ok(Array::from_vec_trusted(vec![rows, cols], out));
    }
    generic_dot2d_unchecked(left, right)
}

fn check_dot2d<T>(left: &ArrayView<'_, T>, right: &ArrayView<'_, T>) -> Result<()> {
    if left.ndim() != 2 || right.ndim() != 2 || left.shape()[1] != right.shape()[0] {
        return Err(NumRsError::DotShapeMismatch {
            left: left.shape().to_vec(),
            right: right.shape().to_vec(),
        });
    }
    Ok(())
}

fn check_dot2d_arrays<T>(left: &Array<T>, right: &Array<T>) -> Result<()> {
    if left.ndim() != 2 || right.ndim() != 2 || left.shape()[1] != right.shape()[0] {
        return Err(NumRsError::DotShapeMismatch {
            left: left.shape().to_vec(),
            right: right.shape().to_vec(),
        });
    }
    Ok(())
}

fn transposed_source_slice<'a, T>(view: &ArrayView<'a, T>) -> Option<&'a [T]> {
    if view.ndim() != 2 {
        return None;
    }
    let source_cols = view.shape()[0];
    let source_rows = view.shape()[1];
    if view.strides()[0] != 1 || view.strides()[1] != source_cols as isize || view.offset() < 0 {
        return None;
    }
    let start = view.offset() as usize;
    let len = source_rows.checked_mul(source_cols)?;
    view.raw_data().get(start..start.checked_add(len)?)
}

fn generic_dot2d<T>(left: &ArrayView<'_, T>, right: &ArrayView<'_, T>) -> Result<Array<T>>
where
    T: Copy + Default + Add<Output = T> + Mul<Output = T>,
{
    check_dot2d(left, right)?;
    generic_dot2d_unchecked(left, right)
}

fn generic_dot2d_unchecked<T>(left: &ArrayView<'_, T>, right: &ArrayView<'_, T>) -> Result<Array<T>>
where
    T: Copy + Default + Add<Output = T> + Mul<Output = T>,
{
    let rows = left.shape()[0];
    let inner = left.shape()[1];
    let cols = right.shape()[1];
    let mut out = vec![T::default(); rows * cols];

    for row in 0..rows {
        for col in 0..cols {
            let mut acc = T::default();
            for k in 0..inner {
                let left_offset = left.offset()
                    + row as isize * left.strides()[0]
                    + k as isize * left.strides()[1];
                let right_offset = right.offset()
                    + k as isize * right.strides()[0]
                    + col as isize * right.strides()[1];
                acc =
                    acc + left.data_at(left_offset as usize) * right.data_at(right_offset as usize);
            }
            out[row * cols + col] = acc;
        }
    }

    Ok(Array::from_vec_trusted(vec![rows, cols], out))
}

pub trait Inner2dKernel: Copy + Default + Add<Output = Self> + Mul<Output = Self> {
    fn inner2d(left: &ArrayView<'_, Self>, right: &ArrayView<'_, Self>) -> Result<Array<Self>>;

    fn inner2d_arrays(left: &Array<Self>, right: &Array<Self>) -> Result<Array<Self>> {
        Self::inner2d(&left.view(), &right.view())
    }
}

macro_rules! impl_generic_inner2d_kernel {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl Inner2dKernel for $ty {
                fn inner2d(
                    left: &ArrayView<'_, Self>,
                    right: &ArrayView<'_, Self>,
                ) -> Result<Array<Self>> {
                    generic_inner2d(left, right)
                }
            }
        )+
    };
}

impl_generic_inner2d_kernel!(Complex64, Complex32, i64, i32, i16, i8, u64, u32, u16, u8);

impl Inner2dKernel for f64 {
    #[inline(always)]
    fn inner2d(left: &ArrayView<'_, Self>, right: &ArrayView<'_, Self>) -> Result<Array<Self>> {
        check_inner2d(left, right)?;
        if let (Some(left_slice), Some(right_slice)) =
            (left.contiguous_slice(), right.contiguous_slice())
        {
            let rows = left.shape()[0];
            let inner = left.shape()[1];
            let cols = right.shape()[0];
            let out = crate::blas::dgemm_row_major_trans_b_f64(
                rows,
                inner,
                cols,
                left_slice,
                right_slice,
            );
            return Ok(Array::from_vec_trusted(vec![rows, cols], out));
        }
        generic_inner2d_unchecked(left, right)
    }
}

impl Inner2dKernel for f32 {
    #[inline(always)]
    fn inner2d(left: &ArrayView<'_, Self>, right: &ArrayView<'_, Self>) -> Result<Array<Self>> {
        check_inner2d(left, right)?;
        if let (Some(left_slice), Some(right_slice)) =
            (left.contiguous_slice(), right.contiguous_slice())
        {
            let rows = left.shape()[0];
            let inner = left.shape()[1];
            let cols = right.shape()[0];
            let out =
                crate::blas::sgemm_row_major_trans_b(rows, inner, cols, left_slice, right_slice);
            return Ok(Array::from_vec_trusted(vec![rows, cols], out));
        }
        generic_inner2d_unchecked(left, right)
    }
}

fn check_inner2d<T>(left: &ArrayView<'_, T>, right: &ArrayView<'_, T>) -> Result<()> {
    if left.ndim() != 2 || right.ndim() != 2 || left.shape()[1] != right.shape()[1] {
        return Err(NumRsError::DotShapeMismatch {
            left: left.shape().to_vec(),
            right: right.shape().to_vec(),
        });
    }
    Ok(())
}

fn generic_inner2d<T>(left: &ArrayView<'_, T>, right: &ArrayView<'_, T>) -> Result<Array<T>>
where
    T: Copy + Default + Add<Output = T> + Mul<Output = T>,
{
    check_inner2d(left, right)?;
    generic_inner2d_unchecked(left, right)
}

fn generic_inner2d_unchecked<T>(
    left: &ArrayView<'_, T>,
    right: &ArrayView<'_, T>,
) -> Result<Array<T>>
where
    T: Copy + Default + Add<Output = T> + Mul<Output = T>,
{
    let rows = left.shape()[0];
    let cols = right.shape()[0];
    let inner = left.shape()[1];
    let mut out = vec![T::default(); rows * cols];

    for row in 0..rows {
        for col in 0..cols {
            let mut acc = T::default();
            for k in 0..inner {
                let left_offset = left.offset()
                    + row as isize * left.strides()[0]
                    + k as isize * left.strides()[1];
                let right_offset = right.offset()
                    + col as isize * right.strides()[0]
                    + k as isize * right.strides()[1];
                acc =
                    acc + left.data_at(left_offset as usize) * right.data_at(right_offset as usize);
            }
            out[row * cols + col] = acc;
        }
    }

    Ok(Array::from_vec_trusted(vec![rows, cols], out))
}

pub trait MatmulKernel: Copy + Default + Add<Output = Self> + Mul<Output = Self> {
    fn matmul(left: &ArrayView<'_, Self>, right: &ArrayView<'_, Self>) -> Result<Array<Self>>;

    fn matmul_arrays(left: &Array<Self>, right: &Array<Self>) -> Result<Array<Self>> {
        Self::matmul(&left.view(), &right.view())
    }
}

macro_rules! impl_matmul_kernel {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl MatmulKernel for $ty {
                fn matmul(
                    left: &ArrayView<'_, Self>,
                    right: &ArrayView<'_, Self>,
                ) -> Result<Array<Self>> {
                    generic_matmul(left, right)
                }
            }
        )+
    };
}

impl_matmul_kernel!(Complex64, Complex32, i64, i32, i16, i8, u64, u32, u16, u8);

impl MatmulKernel for f64 {
    #[inline(always)]
    fn matmul(left: &ArrayView<'_, Self>, right: &ArrayView<'_, Self>) -> Result<Array<Self>> {
        if left.ndim() == 2 && right.ndim() == 2 {
            if left.shape()[1] != right.shape()[0] {
                return Err(NumRsError::DotShapeMismatch {
                    left: left.shape().to_vec(),
                    right: right.shape().to_vec(),
                });
            }
            return dot2d_f64_unchecked(left, right);
        }
        generic_matmul(left, right)
    }

    fn matmul_arrays(left: &Array<Self>, right: &Array<Self>) -> Result<Array<Self>> {
        match (left.ndim(), right.ndim()) {
            (1, 1) => {
                if left.shape()[0] != right.shape()[0] {
                    return Err(NumRsError::DotShapeMismatch {
                        left: left.shape().to_vec(),
                        right: right.shape().to_vec(),
                    });
                }
                return Array::scalar(crate::blas::dot_f64(left.as_slice(), right.as_slice()));
            }
            (2, 1) => {
                if left.shape()[1] != right.shape()[0] {
                    return Err(NumRsError::DotShapeMismatch {
                        left: left.shape().to_vec(),
                        right: right.shape().to_vec(),
                    });
                }
                let rows = left.shape()[0];
                let cols = left.shape()[1];
                let out =
                    crate::blas::dgemv_row_major(rows, cols, left.as_slice(), right.as_slice());
                return Ok(Array::from_vec_trusted(vec![rows], out));
            }
            (1, 2) => {
                if left.shape()[0] != right.shape()[0] {
                    return Err(NumRsError::DotShapeMismatch {
                        left: left.shape().to_vec(),
                        right: right.shape().to_vec(),
                    });
                }
                let rows = right.shape()[0];
                let cols = right.shape()[1];
                let out = crate::blas::dgemv_row_major_trans(
                    rows,
                    cols,
                    right.as_slice(),
                    left.as_slice(),
                );
                return Ok(Array::from_vec_trusted(vec![cols], out));
            }
            (2, 2) => {
                if left.shape()[1] != right.shape()[0] {
                    return Err(NumRsError::DotShapeMismatch {
                        left: left.shape().to_vec(),
                        right: right.shape().to_vec(),
                    });
                }
                let rows = left.shape()[0];
                let inner = left.shape()[1];
                let cols = right.shape()[1];
                let out = crate::blas::dgemm_row_major(
                    rows,
                    inner,
                    cols,
                    left.as_slice(),
                    right.as_slice(),
                );
                return Ok(Array::from_vec_trusted(vec![rows, cols], out));
            }
            _ => {}
        }
        generic_matmul(&left.view(), &right.view())
    }
}

impl MatmulKernel for f32 {
    #[inline(always)]
    fn matmul(left: &ArrayView<'_, Self>, right: &ArrayView<'_, Self>) -> Result<Array<Self>> {
        if left.ndim() == 2 && right.ndim() == 2 {
            if left.shape()[1] != right.shape()[0] {
                return Err(NumRsError::DotShapeMismatch {
                    left: left.shape().to_vec(),
                    right: right.shape().to_vec(),
                });
            }
            return dot2d_f32_unchecked(left, right);
        }
        generic_matmul(left, right)
    }

    fn matmul_arrays(left: &Array<Self>, right: &Array<Self>) -> Result<Array<Self>> {
        match (left.ndim(), right.ndim()) {
            (1, 1) => {
                if left.shape()[0] != right.shape()[0] {
                    return Err(NumRsError::DotShapeMismatch {
                        left: left.shape().to_vec(),
                        right: right.shape().to_vec(),
                    });
                }
                return Array::scalar(crate::blas::dot_f32(left.as_slice(), right.as_slice()));
            }
            (2, 1) => {
                if left.shape()[1] != right.shape()[0] {
                    return Err(NumRsError::DotShapeMismatch {
                        left: left.shape().to_vec(),
                        right: right.shape().to_vec(),
                    });
                }
                let rows = left.shape()[0];
                let cols = left.shape()[1];
                let out =
                    crate::blas::sgemv_row_major(rows, cols, left.as_slice(), right.as_slice());
                return Ok(Array::from_vec_trusted(vec![rows], out));
            }
            (1, 2) => {
                if left.shape()[0] != right.shape()[0] {
                    return Err(NumRsError::DotShapeMismatch {
                        left: left.shape().to_vec(),
                        right: right.shape().to_vec(),
                    });
                }
                let rows = right.shape()[0];
                let cols = right.shape()[1];
                let out = crate::blas::sgemv_row_major_trans(
                    rows,
                    cols,
                    right.as_slice(),
                    left.as_slice(),
                );
                return Ok(Array::from_vec_trusted(vec![cols], out));
            }
            (2, 2) => {
                if left.shape()[1] != right.shape()[0] {
                    return Err(NumRsError::DotShapeMismatch {
                        left: left.shape().to_vec(),
                        right: right.shape().to_vec(),
                    });
                }
                let rows = left.shape()[0];
                let inner = left.shape()[1];
                let cols = right.shape()[1];
                let out = crate::blas::sgemm_row_major(
                    rows,
                    inner,
                    cols,
                    left.as_slice(),
                    right.as_slice(),
                );
                return Ok(Array::from_vec_trusted(vec![rows, cols], out));
            }
            _ => {}
        }
        generic_matmul(&left.view(), &right.view())
    }
}

#[derive(Debug, Clone)]
struct MatmulLayout {
    batch_shape: Vec<usize>,
    out_shape: Vec<usize>,
    left_batch_strides: Vec<isize>,
    right_batch_strides: Vec<isize>,
    left_m: usize,
    inner: usize,
    right_n: usize,
    left_row_stride: isize,
    left_inner_stride: isize,
    right_inner_stride: isize,
    right_col_stride: isize,
    left_vector: bool,
    right_vector: bool,
}

fn generic_matmul<T>(left: &ArrayView<'_, T>, right: &ArrayView<'_, T>) -> Result<Array<T>>
where
    T: Copy + Default + Add<Output = T> + Mul<Output = T>,
{
    let layout = matmul_layout(left, right)?;
    let out_len = size_of_shape(&layout.out_shape)?;
    let mut out = Vec::with_capacity(out_len);
    let left_batch_offsets = OffsetIter::new(
        &layout.batch_shape,
        &layout.left_batch_strides,
        left.offset(),
    )?;
    let right_batch_offsets = OffsetIter::new(
        &layout.batch_shape,
        &layout.right_batch_strides,
        right.offset(),
    )?;

    for (left_batch_offset, right_batch_offset) in left_batch_offsets.zip(right_batch_offsets) {
        if layout.left_vector && layout.right_vector {
            out.push(matmul_cell(
                left,
                right,
                &layout,
                left_batch_offset,
                right_batch_offset,
                0,
                0,
            ));
        } else if layout.left_vector {
            for col in 0..layout.right_n {
                out.push(matmul_cell(
                    left,
                    right,
                    &layout,
                    left_batch_offset,
                    right_batch_offset,
                    0,
                    col,
                ));
            }
        } else if layout.right_vector {
            for row in 0..layout.left_m {
                out.push(matmul_cell(
                    left,
                    right,
                    &layout,
                    left_batch_offset,
                    right_batch_offset,
                    row,
                    0,
                ));
            }
        } else {
            for row in 0..layout.left_m {
                for col in 0..layout.right_n {
                    out.push(matmul_cell(
                        left,
                        right,
                        &layout,
                        left_batch_offset,
                        right_batch_offset,
                        row,
                        col,
                    ));
                }
            }
        }
    }

    Array::from_vec(layout.out_shape, out)
}

fn matmul_cell<T>(
    left: &ArrayView<'_, T>,
    right: &ArrayView<'_, T>,
    layout: &MatmulLayout,
    left_batch_offset: usize,
    right_batch_offset: usize,
    row: usize,
    col: usize,
) -> T
where
    T: Copy + Default + Add<Output = T> + Mul<Output = T>,
{
    let mut acc = T::default();
    for inner in 0..layout.inner {
        let left_offset = left_batch_offset as isize
            + row as isize * layout.left_row_stride
            + inner as isize * layout.left_inner_stride;
        let right_offset = right_batch_offset as isize
            + inner as isize * layout.right_inner_stride
            + col as isize * layout.right_col_stride;
        acc = acc + left.data_at(left_offset as usize) * right.data_at(right_offset as usize);
    }
    acc
}

fn matmul_layout<T>(left: &ArrayView<'_, T>, right: &ArrayView<'_, T>) -> Result<MatmulLayout> {
    if left.ndim() == 0 || right.ndim() == 0 {
        return Err(NumRsError::InvalidShape(format!(
            "matmul does not accept scalar operands, got {:?} and {:?}",
            left.shape(),
            right.shape()
        )));
    }

    let left_vector = left.ndim() == 1;
    let right_vector = right.ndim() == 1;
    let left_batch_ndim = left.ndim().saturating_sub(2);
    let right_batch_ndim = right.ndim().saturating_sub(2);
    let left_batch_shape = &left.shape()[..left_batch_ndim];
    let right_batch_shape = &right.shape()[..right_batch_ndim];
    let batch_shape = broadcast_shape(left_batch_shape, right_batch_shape)?;
    let left_batch_strides = broadcast_strides(
        left_batch_shape,
        &left.strides()[..left_batch_ndim],
        &batch_shape,
    )?;
    let right_batch_strides = broadcast_strides(
        right_batch_shape,
        &right.strides()[..right_batch_ndim],
        &batch_shape,
    )?;

    let (left_m, left_inner, left_row_stride, left_inner_stride) = if left_vector {
        (1, left.shape()[0], 0, left.strides()[0])
    } else {
        let ndim = left.ndim();
        (
            left.shape()[ndim - 2],
            left.shape()[ndim - 1],
            left.strides()[ndim - 2],
            left.strides()[ndim - 1],
        )
    };
    let (right_inner, right_n, right_inner_stride, right_col_stride) = if right_vector {
        (right.shape()[0], 1, right.strides()[0], 0)
    } else {
        let ndim = right.ndim();
        (
            right.shape()[ndim - 2],
            right.shape()[ndim - 1],
            right.strides()[ndim - 2],
            right.strides()[ndim - 1],
        )
    };

    if left_inner != right_inner {
        return Err(NumRsError::DotShapeMismatch {
            left: left.shape().to_vec(),
            right: right.shape().to_vec(),
        });
    }

    let mut out_shape = batch_shape.clone();
    match (left_vector, right_vector) {
        (true, true) => {}
        (true, false) => out_shape.push(right_n),
        (false, true) => out_shape.push(left_m),
        (false, false) => {
            out_shape.push(left_m);
            out_shape.push(right_n);
        }
    }

    Ok(MatmulLayout {
        batch_shape,
        out_shape,
        left_batch_strides,
        right_batch_strides,
        left_m,
        inner: left_inner,
        right_n,
        left_row_stride,
        left_inner_stride,
        right_inner_stride,
        right_col_stride,
        left_vector,
        right_vector,
    })
}

pub trait TensordotKernel: Copy + Default + Add<Output = Self> + Mul<Output = Self> {
    fn tensordot_axes(
        left: &ArrayView<'_, Self>,
        right: &ArrayView<'_, Self>,
        left_axes: &[isize],
        right_axes: &[isize],
    ) -> Result<Array<Self>>;
}

macro_rules! impl_tensordot_kernel {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl TensordotKernel for $ty {
                fn tensordot_axes(
                    left: &ArrayView<'_, Self>,
                    right: &ArrayView<'_, Self>,
                    left_axes: &[isize],
                    right_axes: &[isize],
                ) -> Result<Array<Self>> {
                    generic_tensordot_axes(left, right, left_axes, right_axes)
                }
            }
        )+
    };
}

impl_tensordot_kernel!(f32, i64, i32, i16, i8, u64, u32, u16, u8);

impl TensordotKernel for f64 {
    fn tensordot_axes(
        left: &ArrayView<'_, Self>,
        right: &ArrayView<'_, Self>,
        left_axes: &[isize],
        right_axes: &[isize],
    ) -> Result<Array<Self>> {
        tensordot_axes_f64(left, right, left_axes, right_axes)
    }
}

#[derive(Debug, Clone)]
struct TensordotLayout {
    out_shape: Vec<usize>,
    contract_shape: Vec<usize>,
    left_free_shape: Vec<usize>,
    right_free_shape: Vec<usize>,
    left_free_strides: Vec<isize>,
    right_free_strides: Vec<isize>,
    left_contract_strides: Vec<isize>,
    right_contract_strides: Vec<isize>,
}

fn generic_tensordot_axes<T>(
    left: &ArrayView<'_, T>,
    right: &ArrayView<'_, T>,
    left_axes: &[isize],
    right_axes: &[isize],
) -> Result<Array<T>>
where
    T: Copy + Default + Add<Output = T> + Mul<Output = T>,
{
    let layout = tensordot_layout(left, right, left_axes, right_axes)?;
    let out_len = size_of_shape(&layout.out_shape)?;
    let contract_len = size_of_shape(&layout.contract_shape)?;
    let left_free_len = size_of_shape(&layout.left_free_shape)?;
    let right_free_len = size_of_shape(&layout.right_free_shape)?;
    let mut out = Vec::with_capacity(out_len);

    for out_linear in 0..out_len {
        let left_free_linear = out_linear.checked_div(right_free_len).unwrap_or(0);
        let right_free_linear = out_linear.checked_rem(right_free_len).unwrap_or(0);
        debug_assert!(left_free_linear < left_free_len.max(1));
        let left_base = offset_from_linear(
            left.offset(),
            &layout.left_free_shape,
            &layout.left_free_strides,
            left_free_linear,
        );
        let right_base = offset_from_linear(
            right.offset(),
            &layout.right_free_shape,
            &layout.right_free_strides,
            right_free_linear,
        );
        let mut acc = T::default();
        for contract_linear in 0..contract_len {
            let left_offset = offset_from_linear(
                left_base,
                &layout.contract_shape,
                &layout.left_contract_strides,
                contract_linear,
            );
            let right_offset = offset_from_linear(
                right_base,
                &layout.contract_shape,
                &layout.right_contract_strides,
                contract_linear,
            );
            acc = acc + left.data_at(left_offset as usize) * right.data_at(right_offset as usize);
        }
        out.push(acc);
    }

    Array::from_vec(layout.out_shape, out)
}

fn tensordot_axes_f64(
    left: &ArrayView<'_, f64>,
    right: &ArrayView<'_, f64>,
    left_axes: &[isize],
    right_axes: &[isize],
) -> Result<Array<f64>> {
    let layout = tensordot_layout(left, right, left_axes, right_axes)?;
    let left_free_len = size_of_shape(&layout.left_free_shape)?;
    let right_free_len = size_of_shape(&layout.right_free_shape)?;
    let contract_len = size_of_shape(&layout.contract_shape)?;
    let out_len = size_of_shape(&layout.out_shape)?;

    if out_len == 0 || contract_len == 0 {
        return Array::from_vec(layout.out_shape, vec![0.0; out_len]);
    }

    if layout.left_free_shape.len() == 1
        && layout.contract_shape.len() == 2
        && layout.right_free_shape.len() == 1
    {
        return tensordot_1x2x1_f64(left, right, layout);
    }

    let mut packed_left = vec![0.0; left_free_len * contract_len];
    for left_free_linear in 0..left_free_len {
        let left_base = offset_from_linear(
            left.offset(),
            &layout.left_free_shape,
            &layout.left_free_strides,
            left_free_linear,
        );
        let out_base = left_free_linear * contract_len;
        for contract_linear in 0..contract_len {
            let offset = offset_from_linear(
                left_base,
                &layout.contract_shape,
                &layout.left_contract_strides,
                contract_linear,
            );
            packed_left[out_base + contract_linear] = left.data_at(offset as usize);
        }
    }

    let mut packed_right = vec![0.0; contract_len * right_free_len];
    for contract_linear in 0..contract_len {
        let right_contract_base = offset_from_linear(
            right.offset(),
            &layout.contract_shape,
            &layout.right_contract_strides,
            contract_linear,
        );
        let out_base = contract_linear * right_free_len;
        for right_free_linear in 0..right_free_len {
            let offset = offset_from_linear(
                right_contract_base,
                &layout.right_free_shape,
                &layout.right_free_strides,
                right_free_linear,
            );
            packed_right[out_base + right_free_linear] = right.data_at(offset as usize);
        }
    }

    let out = crate::blas::dgemm_row_major(
        left_free_len,
        contract_len,
        right_free_len,
        &packed_left,
        &packed_right,
    );
    Array::from_vec(layout.out_shape, out)
}

fn tensordot_1x2x1_f64(
    left: &ArrayView<'_, f64>,
    right: &ArrayView<'_, f64>,
    layout: TensordotLayout,
) -> Result<Array<f64>> {
    let left_free_len = layout.left_free_shape[0];
    let contract_outer = layout.contract_shape[0];
    let contract_inner = layout.contract_shape[1];
    let right_free_len = layout.right_free_shape[0];
    let contract_len = contract_outer * contract_inner;

    let mut packed_left_t = vec![0.0; contract_len * left_free_len];
    for outer in 0..contract_outer {
        let outer_base = left.offset() + outer as isize * layout.left_contract_strides[0];
        for inner in 0..contract_inner {
            let contract_linear = outer * contract_inner + inner;
            let contract_base = outer_base + inner as isize * layout.left_contract_strides[1];
            let out_base = contract_linear * left_free_len;
            for left_free in 0..left_free_len {
                let offset = contract_base + left_free as isize * layout.left_free_strides[0];
                packed_left_t[out_base + left_free] = left.data_at(offset as usize);
            }
        }
    }

    let right_direct = right.contiguous_slice().filter(|_| {
        layout.right_free_strides[0] == 1
            && layout.right_contract_strides[1] == right_free_len as isize
            && layout.right_contract_strides[0] == (contract_inner * right_free_len) as isize
    });
    let packed_right;
    let right_matrix = if let Some(slice) = right_direct {
        slice
    } else {
        let mut packed = vec![0.0; contract_len * right_free_len];
        for outer in 0..contract_outer {
            let outer_base = right.offset() + outer as isize * layout.right_contract_strides[0];
            for inner in 0..contract_inner {
                let contract_linear = outer * contract_inner + inner;
                let contract_base = outer_base + inner as isize * layout.right_contract_strides[1];
                let out_base = contract_linear * right_free_len;
                for right_free in 0..right_free_len {
                    let offset = contract_base + right_free as isize * layout.right_free_strides[0];
                    packed[out_base + right_free] = right.data_at(offset as usize);
                }
            }
        }
        packed_right = packed;
        &packed_right
    };

    let out = crate::blas::dgemm_row_major_trans_a_f64(
        left_free_len,
        contract_len,
        right_free_len,
        &packed_left_t,
        right_matrix,
    );
    Array::from_vec(layout.out_shape, out)
}

fn tensordot_layout<T>(
    left: &ArrayView<'_, T>,
    right: &ArrayView<'_, T>,
    left_axes: &[isize],
    right_axes: &[isize],
) -> Result<TensordotLayout> {
    if left_axes.len() != right_axes.len() {
        return Err(NumRsError::InvalidShape(format!(
            "tensordot expected equal axes lengths, got {} and {}",
            left_axes.len(),
            right_axes.len()
        )));
    }

    let left_axes = normalize_unique_axes(left_axes, left.ndim())?;
    let right_axes = normalize_unique_axes(right_axes, right.ndim())?;
    let mut contract_shape = Vec::with_capacity(left_axes.len());
    let mut left_contract_strides = Vec::with_capacity(left_axes.len());
    let mut right_contract_strides = Vec::with_capacity(right_axes.len());

    for (left_axis, right_axis) in left_axes.iter().copied().zip(right_axes.iter().copied()) {
        if left.shape()[left_axis] != right.shape()[right_axis] {
            return Err(NumRsError::DotShapeMismatch {
                left: left.shape().to_vec(),
                right: right.shape().to_vec(),
            });
        }
        contract_shape.push(left.shape()[left_axis]);
        left_contract_strides.push(left.strides()[left_axis]);
        right_contract_strides.push(right.strides()[right_axis]);
    }

    let left_free_axes = complement_axes(left.ndim(), &left_axes);
    let right_free_axes = complement_axes(right.ndim(), &right_axes);
    let left_free_shape = gather_usize(left.shape(), &left_free_axes);
    let right_free_shape = gather_usize(right.shape(), &right_free_axes);
    let left_free_strides = gather_isize(left.strides(), &left_free_axes);
    let right_free_strides = gather_isize(right.strides(), &right_free_axes);
    let mut out_shape = left_free_shape.clone();
    out_shape.extend(right_free_shape.iter().copied());

    Ok(TensordotLayout {
        out_shape,
        contract_shape,
        left_free_shape,
        right_free_shape,
        left_free_strides,
        right_free_strides,
        left_contract_strides,
        right_contract_strides,
    })
}

fn normalize_unique_axes(axes: &[isize], ndim: usize) -> Result<Vec<usize>> {
    let mut normalized = Vec::with_capacity(axes.len());
    let mut seen = vec![false; ndim];
    for axis in axes.iter().copied() {
        let axis = normalize_axis(axis, ndim)?;
        if seen[axis] {
            return Err(NumRsError::DuplicateAxis(axis));
        }
        seen[axis] = true;
        normalized.push(axis);
    }
    Ok(normalized)
}

fn complement_axes(ndim: usize, axes: &[usize]) -> Vec<usize> {
    let mut contracted = vec![false; ndim];
    for axis in axes.iter().copied() {
        contracted[axis] = true;
    }
    (0..ndim).filter(|axis| !contracted[*axis]).collect()
}

fn gather_usize(values: &[usize], axes: &[usize]) -> Vec<usize> {
    axes.iter().map(|axis| values[*axis]).collect()
}

fn gather_isize(values: &[isize], axes: &[usize]) -> Vec<isize> {
    axes.iter().map(|axis| values[*axis]).collect()
}

fn offset_from_linear(
    base_offset: isize,
    shape: &[usize],
    strides: &[isize],
    mut linear: usize,
) -> isize {
    let mut offset = base_offset;
    for axis in (0..shape.len()).rev() {
        let dim = shape[axis];
        let coord = linear % dim;
        linear /= dim;
        offset += coord as isize * strides[axis];
    }
    offset
}

impl<'a> ArrayView<'a, f64> {
    pub fn mean_all(&self) -> Result<f64> {
        let len = self.len();
        if len == 0 {
            return Err(NumRsError::EmptyReduction);
        }
        Ok(self.sum_all()? / len as f64)
    }

    pub fn mean_axis(&self, axis: isize) -> Result<Array<f64>> {
        let axis = normalize_axis(axis, self.ndim())?;
        let axis_len = self.shape()[axis];
        if axis_len == 0 {
            return Err(NumRsError::EmptyReduction);
        }
        self.sum_axis(axis as isize)?
            .map(|value| value / axis_len as f64)
    }

    pub fn var_all(&self) -> Result<f64> {
        let len = self.len();
        if len == 0 {
            return Err(NumRsError::EmptyReduction);
        }
        let mean = self.mean_all()?;
        let mut acc = 0.0;
        for offset in self.offset_iter()? {
            let diff = self.data_at(offset) - mean;
            acc += diff * diff;
        }
        Ok(acc / len as f64)
    }

    pub fn std_all(&self) -> Result<f64> {
        Ok(self.var_all()?.sqrt())
    }

    pub fn norm_l2(&self) -> Result<f64> {
        let mut acc = 0.0;
        for offset in self.offset_iter()? {
            let value = self.data_at(offset);
            acc += value * value;
        }
        Ok(acc.sqrt())
    }

    pub fn det(&self) -> Result<f64> {
        let n = square_matrix_len(self)?;
        if n == 0 {
            return Ok(1.0);
        }
        let mut matrix = self.to_vec()?;
        let mut sign = 1.0;
        let mut det = 1.0;

        for pivot_col in 0..n {
            let mut pivot_row = pivot_col;
            let mut pivot_abs = matrix[pivot_col * n + pivot_col].abs();
            for row in pivot_col + 1..n {
                let candidate_abs = matrix[row * n + pivot_col].abs();
                if candidate_abs > pivot_abs {
                    pivot_abs = candidate_abs;
                    pivot_row = row;
                }
            }

            if pivot_abs <= f64::EPSILON {
                return Ok(0.0);
            }

            if pivot_row != pivot_col {
                swap_rows(&mut matrix, n, pivot_col, pivot_row);
                sign = -sign;
            }

            let pivot = matrix[pivot_col * n + pivot_col];
            det *= pivot;
            for row in pivot_col + 1..n {
                let factor = matrix[row * n + pivot_col] / pivot;
                matrix[row * n + pivot_col] = 0.0;
                for col in pivot_col + 1..n {
                    matrix[row * n + col] -= factor * matrix[pivot_col * n + col];
                }
            }
        }

        Ok(sign * det)
    }

    pub fn solve(&self, rhs: &ArrayView<'_, f64>) -> Result<Array<f64>> {
        let n = square_matrix_len(self)?;
        if rhs.ndim() != 1 || rhs.shape()[0] != n {
            return Err(NumRsError::InvalidShape(format!(
                "solve expected rhs shape [{n}], got {:?}",
                rhs.shape()
            )));
        }

        let mut matrix = self.to_vec()?;
        let mut values = rhs.to_vec()?;

        for pivot_col in 0..n {
            let mut pivot_row = pivot_col;
            let mut pivot_abs = matrix[pivot_col * n + pivot_col].abs();
            for row in pivot_col + 1..n {
                let candidate_abs = matrix[row * n + pivot_col].abs();
                if candidate_abs > pivot_abs {
                    pivot_abs = candidate_abs;
                    pivot_row = row;
                }
            }

            if pivot_abs <= f64::EPSILON {
                return Err(NumRsError::InvalidShape(
                    "cannot solve singular matrix".to_string(),
                ));
            }

            if pivot_row != pivot_col {
                swap_rows(&mut matrix, n, pivot_col, pivot_row);
                values.swap(pivot_col, pivot_row);
            }

            let pivot = matrix[pivot_col * n + pivot_col];
            for row in pivot_col + 1..n {
                let factor = matrix[row * n + pivot_col] / pivot;
                matrix[row * n + pivot_col] = 0.0;
                values[row] -= factor * values[pivot_col];
                for col in pivot_col + 1..n {
                    matrix[row * n + col] -= factor * matrix[pivot_col * n + col];
                }
            }
        }

        let mut solution = vec![0.0; n];
        for row in (0..n).rev() {
            let mut acc = values[row];
            for col in row + 1..n {
                acc -= matrix[row * n + col] * solution[col];
            }
            solution[row] = acc / matrix[row * n + row];
        }

        Array::from_vec(vec![n], solution)
    }

    pub fn outer_product(&self, rhs: &ArrayView<'_, f64>) -> Result<Array<f64>> {
        if self.ndim() != 1 || rhs.ndim() != 1 {
            return Err(NumRsError::InvalidShape(format!(
                "outer_product expected vectors, got {:?} and {:?}",
                self.shape(),
                rhs.shape()
            )));
        }

        let rows = self.shape()[0];
        let cols = rhs.shape()[0];

        if let (Some(left), Some(right)) = (self.contiguous_slice(), rhs.contiguous_slice()) {
            let out = crate::blas::outer_product_f64(rows, cols, left, right);
            return Ok(Array::from_vec_trusted(vec![rows, cols], out));
        }

        let mut out = Vec::with_capacity(rows * cols);
        for left_offset in self.offset_iter()? {
            let left = self.data_at(left_offset);
            for right_offset in rhs.offset_iter()? {
                out.push(left * rhs.data_at(right_offset));
            }
        }
        Ok(Array::from_vec_trusted(vec![rows, cols], out))
    }

    pub fn mul_scalar(&self, value: f64) -> Result<Array<f64>> {
        if let Some(slice) = self.contiguous_slice() {
            return Ok(Array::from_vec_trusted(
                self.shape().to_vec(),
                crate::blas::mul_scalar_f64(slice, value),
            ));
        }
        self.map(|item| item * value)
    }

    pub fn weighted_axis1_sum(&self, weights: &ArrayView<'_, f64>) -> Result<f64> {
        if self.ndim() != 2 || weights.ndim() != 1 || self.shape()[1] != weights.shape()[0] {
            return Err(NumRsError::InvalidShape(format!(
                "weighted_axis1_sum expected [rows, cols] and [cols], got {:?} and {:?}",
                self.shape(),
                weights.shape()
            )));
        }

        let cols = self.shape()[1];
        let mut acc = 0.0;
        if let (Some(matrix), Some(weights)) = (self.contiguous_slice(), weights.contiguous_slice())
        {
            return Ok(crate::blas::weighted_axis1_sum_f64(
                self.shape()[0],
                cols,
                matrix,
                weights,
            ));
        }

        for row in 0..self.shape()[0] {
            for col in 0..cols {
                let left_offset = self.offset()
                    + row as isize * self.strides()[0]
                    + col as isize * self.strides()[1];
                let weight_offset = weights.offset() + col as isize * weights.strides()[0];
                acc += self.data_at(left_offset as usize) * weights.data_at(weight_offset as usize);
            }
        }
        Ok(acc)
    }

    pub fn bilinear_form(
        &self,
        left_weights: &ArrayView<'_, f64>,
        right_weights: &ArrayView<'_, f64>,
    ) -> Result<f64> {
        if self.ndim() != 2
            || left_weights.ndim() != 1
            || right_weights.ndim() != 1
            || self.shape()[0] != left_weights.shape()[0]
            || self.shape()[1] != right_weights.shape()[0]
        {
            return Err(NumRsError::InvalidShape(format!(
                "bilinear_form expected [rows, cols], [rows], and [cols], got {:?}, {:?}, and {:?}",
                self.shape(),
                left_weights.shape(),
                right_weights.shape()
            )));
        }

        if let (Some(matrix), Some(left), Some(right)) = (
            self.contiguous_slice(),
            left_weights.contiguous_slice(),
            right_weights.contiguous_slice(),
        ) {
            let row_weighted =
                crate::blas::dgemv_row_major(self.shape()[0], self.shape()[1], matrix, right);
            return Ok(crate::blas::dot_f64(left, &row_weighted));
        }

        let rows = self.shape()[0];
        let cols = self.shape()[1];
        let mut acc = 0.0;
        for row in 0..rows {
            let left_offset = left_weights.offset() + row as isize * left_weights.strides()[0];
            let left = left_weights.data_at(left_offset as usize);
            for col in 0..cols {
                let matrix_offset = self.offset()
                    + row as isize * self.strides()[0]
                    + col as isize * self.strides()[1];
                let right_offset =
                    right_weights.offset() + col as isize * right_weights.strides()[0];
                acc += left
                    * self.data_at(matrix_offset as usize)
                    * right_weights.data_at(right_offset as usize);
            }
        }
        Ok(acc)
    }

    pub fn add_sum_f64(&self, rhs: &ArrayView<'_, f64>) -> Result<f64> {
        let out_shape = broadcast_shape(self.shape(), rhs.shape())?;

        if self.shape() == rhs.shape() {
            if let (Some(left), Some(right)) = (self.contiguous_slice(), rhs.contiguous_slice()) {
                let left_chunks = left.chunks_exact(4);
                let right_chunks = right.chunks_exact(4);
                let left_remainder = left_chunks.remainder();
                let right_remainder = right_chunks.remainder();
                let mut acc0 = 0.0;
                let mut acc1 = 0.0;
                let mut acc2 = 0.0;
                let mut acc3 = 0.0;
                for (l, r) in left_chunks.zip(right_chunks) {
                    acc0 += l[0] + r[0];
                    acc1 += l[1] + r[1];
                    acc2 += l[2] + r[2];
                    acc3 += l[3] + r[3];
                }
                let mut acc = acc0 + acc1 + acc2 + acc3;
                for (l, r) in left_remainder.iter().zip(right_remainder.iter()) {
                    acc += l + r;
                }
                return Ok(acc);
            }
        }

        let left_strides = broadcast_strides(self.shape(), self.strides(), &out_shape)?;
        let right_strides = broadcast_strides(rhs.shape(), rhs.strides(), &out_shape)?;
        let left_offsets = OffsetIter::new(&out_shape, &left_strides, self.offset())?;
        let right_offsets = OffsetIter::new(&out_shape, &right_strides, rhs.offset())?;

        let mut acc0 = 0.0;
        let mut acc1 = 0.0;
        let mut acc2 = 0.0;
        let mut acc3 = 0.0;
        for (i, (left_offset, right_offset)) in left_offsets.zip(right_offsets).enumerate() {
            match i % 4 {
                0 => acc0 += self.data_at(left_offset) + rhs.data_at(right_offset),
                1 => acc1 += self.data_at(left_offset) + rhs.data_at(right_offset),
                2 => acc2 += self.data_at(left_offset) + rhs.data_at(right_offset),
                _ => acc3 += self.data_at(left_offset) + rhs.data_at(right_offset),
            }
        }
        Ok(acc0 + acc1 + acc2 + acc3)
    }

    pub fn add_outer2d_f64(&self, rhs: &ArrayView<'_, f64>) -> Result<Array<f64>> {
        let out_shape = broadcast_shape(self.shape(), rhs.shape())?;
        let (Some(left), Some(right)) = (self.contiguous_slice(), rhs.contiguous_slice()) else {
            return self.add(rhs);
        };

        match (self.shape(), rhs.shape()) {
            ([rows, 1], [1, cols]) => {
                let mut out = vec![0.0; rows * cols];
                for (row, left_value) in left.iter().take(*rows).copied().enumerate() {
                    let start = row * cols;
                    let end = start + cols;
                    for (slot, right_value) in out[start..end].iter_mut().zip(right.iter()) {
                        *slot = left_value + *right_value;
                    }
                }
                Array::from_vec(out_shape, out)
            }
            ([1, cols], [rows, 1]) => {
                let mut out = vec![0.0; rows * cols];
                for (row, right_value) in right.iter().take(*rows).copied().enumerate() {
                    let start = row * cols;
                    let end = start + cols;
                    for (slot, left_value) in out[start..end].iter_mut().zip(left.iter()) {
                        *slot = *left_value + right_value;
                    }
                }
                Array::from_vec(out_shape, out)
            }
            _ => self.add(rhs),
        }
    }
}

impl<'a> ArrayView<'a, f32> {
    pub fn mean_all(&self) -> Result<f64> {
        let len = self.len();
        if len == 0 {
            return Err(NumRsError::EmptyReduction);
        }
        let mut acc = 0.0;
        for offset in self.offset_iter()? {
            acc += self.data_at(offset) as f64;
        }
        Ok(acc / len as f64)
    }

    pub fn var_all(&self) -> Result<f64> {
        let len = self.len();
        if len == 0 {
            return Err(NumRsError::EmptyReduction);
        }
        let mean = self.mean_all()?;
        let mut acc = 0.0;
        for offset in self.offset_iter()? {
            let diff = self.data_at(offset) as f64 - mean;
            acc += diff * diff;
        }
        Ok(acc / len as f64)
    }

    pub fn std_all(&self) -> Result<f64> {
        Ok(self.var_all()?.sqrt())
    }
}

impl<'a> ArrayView<'a, i64> {
    pub fn mean_all(&self) -> Result<f64> {
        let len = self.len();
        if len == 0 {
            return Err(NumRsError::EmptyReduction);
        }
        let mut acc = 0.0;
        for offset in self.offset_iter()? {
            acc += self.data_at(offset) as f64;
        }
        Ok(acc / len as f64)
    }

    pub fn var_all(&self) -> Result<f64> {
        let len = self.len();
        if len == 0 {
            return Err(NumRsError::EmptyReduction);
        }
        let mean = self.mean_all()?;
        let mut acc = 0.0;
        for offset in self.offset_iter()? {
            let diff = self.data_at(offset) as f64 - mean;
            acc += diff * diff;
        }
        Ok(acc / len as f64)
    }

    pub fn std_all(&self) -> Result<f64> {
        Ok(self.var_all()?.sqrt())
    }
}

impl<'a> ArrayView<'a, u64> {
    pub fn mean_all(&self) -> Result<f64> {
        let len = self.len();
        if len == 0 {
            return Err(NumRsError::EmptyReduction);
        }
        let mut acc = 0.0;
        for offset in self.offset_iter()? {
            acc += self.data_at(offset) as f64;
        }
        Ok(acc / len as f64)
    }

    pub fn var_all(&self) -> Result<f64> {
        let len = self.len();
        if len == 0 {
            return Err(NumRsError::EmptyReduction);
        }
        let mean = self.mean_all()?;
        let mut acc = 0.0;
        for offset in self.offset_iter()? {
            let diff = self.data_at(offset) as f64 - mean;
            acc += diff * diff;
        }
        Ok(acc / len as f64)
    }

    pub fn std_all(&self) -> Result<f64> {
        Ok(self.var_all()?.sqrt())
    }
}

fn complex32_has_nan(value: Complex32) -> bool {
    value.re.is_nan() || value.im.is_nan()
}

fn complex32_less(candidate: Complex32, best: Complex32) -> bool {
    if complex32_has_nan(candidate) {
        return true;
    }
    if complex32_has_nan(best) {
        return false;
    }
    candidate.re < best.re || (candidate.re == best.re && candidate.im < best.im)
}

fn complex32_greater(candidate: Complex32, best: Complex32) -> bool {
    if complex32_has_nan(candidate) {
        return true;
    }
    if complex32_has_nan(best) {
        return false;
    }
    candidate.re > best.re || (candidate.re == best.re && candidate.im > best.im)
}

fn complex32_extreme_all<F>(view: &ArrayView<'_, Complex32>, better: F) -> Result<Complex32>
where
    F: Fn(Complex32, Complex32) -> bool,
{
    if view.is_empty() {
        return Err(NumRsError::EmptyReduction);
    }

    if let Some(slice) = view.contiguous_slice() {
        let (first, rest) = slice.split_first().expect("empty checked above");
        let mut best_value = *first;
        for value in rest.iter().copied() {
            if better(value, best_value) {
                best_value = value;
            }
        }
        return Ok(best_value);
    }

    let mut offsets = view.offset_iter()?;
    let first_offset = offsets.next().expect("empty checked above");
    let mut best_value = view.data_at(first_offset);

    for offset in offsets {
        let value = view.data_at(offset);
        if better(value, best_value) {
            best_value = value;
        }
    }

    Ok(best_value)
}

impl<'a> ArrayView<'a, Complex32> {
    pub fn mean_all(&self) -> Result<Complex32> {
        let len = self.len();
        if len == 0 {
            return Err(NumRsError::EmptyReduction);
        }
        Ok(self.sum_all()? / len as f32)
    }

    pub fn var_all(&self) -> Result<f64> {
        let len = self.len();
        if len == 0 {
            return Err(NumRsError::EmptyReduction);
        }
        let mean = self.mean_all()?;
        let mut acc = 0.0;
        for offset in self.offset_iter()? {
            let diff = self.data_at(offset) - mean;
            acc += diff.norm_sqr() as f64;
        }
        Ok(acc / len as f64)
    }

    pub fn std_all(&self) -> Result<f64> {
        Ok(self.var_all()?.sqrt())
    }
}

fn square_matrix_len<T>(matrix: &ArrayView<'_, T>) -> Result<usize> {
    if matrix.ndim() != 2 || matrix.shape()[0] != matrix.shape()[1] {
        return Err(NumRsError::InvalidShape(format!(
            "expected square matrix, got {:?}",
            matrix.shape()
        )));
    }
    Ok(matrix.shape()[0])
}

fn swap_rows(matrix: &mut [f64], n: usize, left: usize, right: usize) {
    for col in 0..n {
        matrix.swap(left * n + col, right * n + col);
    }
}

impl<'a> ArrayView<'a, bool> {
    pub fn mean_all(&self) -> Result<f64> {
        let len = self.len();
        if len == 0 {
            return Err(NumRsError::EmptyReduction);
        }
        let mut acc = 0usize;
        for offset in self.offset_iter()? {
            acc += usize::from(self.data_at(offset));
        }
        Ok(acc as f64 / len as f64)
    }

    pub fn var_all(&self) -> Result<f64> {
        let len = self.len();
        if len == 0 {
            return Err(NumRsError::EmptyReduction);
        }
        let mean = self.mean_all()?;
        let mut acc = 0.0;
        for offset in self.offset_iter()? {
            let diff = f64::from(self.data_at(offset)) - mean;
            acc += diff * diff;
        }
        Ok(acc / len as f64)
    }

    pub fn std_all(&self) -> Result<f64> {
        Ok(self.var_all()?.sqrt())
    }

    pub fn prod_all(&self) -> Result<bool> {
        if let Some(slice) = self.contiguous_slice() {
            return Ok(slice.iter().copied().all(|value| value));
        }
        for offset in self.offset_iter()? {
            if !self.data_at(offset) {
                return Ok(false);
            }
        }
        Ok(true)
    }

    pub fn logical_not(&self) -> Result<Array<bool>> {
        self.map(|value| !value)
    }

    pub fn logical_and(&self, rhs: &ArrayView<'_, bool>) -> Result<Array<bool>> {
        self.elementwise_binary(rhs, |l, r| l && r)
    }

    pub fn logical_or(&self, rhs: &ArrayView<'_, bool>) -> Result<Array<bool>> {
        self.elementwise_binary(rhs, |l, r| l || r)
    }
}

#[allow(dead_code)]
fn _axis_check(axis: isize, ndim: usize) -> Result<usize> {
    normalize_axis(axis, ndim)
}
