use std::ops::Add;

use crate::dtype::{CastElement, DType, DTypeKind};
use crate::error::{NumRsError, Result};
use crate::indexing::{normalize_slice, NormalizedSlice, Slice};
use crate::layout::{
    broadcast_strides, is_c_contiguous, offset_for_index, permute_unique_axes, OffsetIter,
};
use crate::shape::{
    broadcast_shape, c_strides, normalize_axis, normalize_insert_axis, resolve_shape, size_of_shape,
};

#[derive(Debug, Clone)]
pub struct Array<T> {
    data: Vec<T>,
    shape: Vec<usize>,
    strides: Vec<isize>,
    uniform: Option<T>,
}

impl<T: PartialEq> PartialEq for Array<T> {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data && self.shape == other.shape && self.strides == other.strides
    }
}

#[derive(Debug, Clone)]
pub struct ArrayView<'a, T> {
    data: &'a [T],
    shape: Vec<usize>,
    strides: Vec<isize>,
    offset: isize,
}

impl<T> Array<T> {
    pub fn from_vec(shape: impl Into<Vec<usize>>, data: Vec<T>) -> Result<Self> {
        let shape = shape.into();
        let expected = size_of_shape(&shape)?;
        if expected != data.len() {
            return Err(NumRsError::ShapeDataMismatch {
                shape,
                expected,
                actual: data.len(),
            });
        }
        let strides = c_strides(&shape);
        Ok(Self {
            data,
            shape,
            strides,
            uniform: None,
        })
    }

    pub(crate) fn from_vec_trusted(shape: Vec<usize>, data: Vec<T>) -> Self {
        debug_assert_eq!(size_of_shape(&shape).ok(), Some(data.len()));
        let strides = c_strides(&shape);
        Self {
            data,
            shape,
            strides,
            uniform: None,
        }
    }

    pub fn full(shape: impl Into<Vec<usize>>, value: T) -> Result<Self>
    where
        T: Clone,
    {
        let shape = shape.into();
        let len = size_of_shape(&shape)?;
        let strides = c_strides(&shape);
        Ok(Self {
            data: vec![value.clone(); len],
            shape,
            strides,
            uniform: Some(value),
        })
    }

    pub fn zeros(shape: impl Into<Vec<usize>>) -> Result<Self>
    where
        T: Default + Clone,
    {
        Self::full(shape, T::default())
    }

    pub fn scalar(value: T) -> Result<Self> {
        Self::from_vec(Vec::new(), vec![value])
    }

    pub fn from_shape_fn<F>(shape: impl Into<Vec<usize>>, mut f: F) -> Result<Self>
    where
        F: FnMut(&[usize]) -> T,
    {
        let shape = shape.into();
        let len = size_of_shape(&shape)?;
        let mut data = Vec::with_capacity(len);
        for linear in 0..len {
            let mut remainder = linear;
            let mut index = vec![0usize; shape.len()];
            for axis in (0..shape.len()).rev() {
                let dim = shape[axis];
                index[axis] = remainder.checked_rem(dim).unwrap_or(0);
                remainder = remainder.checked_div(dim).unwrap_or(0);
            }
            data.push(f(&index));
        }
        Self::from_vec(shape, data)
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn strides(&self) -> &[isize] {
        &self.strides
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    pub fn into_vec(self) -> Vec<T> {
        self.data
    }

    pub fn view(&self) -> ArrayView<'_, T> {
        ArrayView {
            data: &self.data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            offset: 0,
        }
    }

    pub fn get(&self, index: &[usize]) -> Result<&T> {
        self.view().get(index)
    }

    pub fn slice(&self, slices: &[Slice]) -> Result<ArrayView<'_, T>> {
        self.view().slice(slices)
    }

    pub fn reshape(&self, shape: &[isize]) -> Result<ArrayView<'_, T>> {
        self.view().reshape(shape)
    }

    pub fn reshape_exact(&self, shape: &[usize]) -> Result<ArrayView<'_, T>> {
        self.view().reshape_exact(shape)
    }

    pub fn transpose(&self) -> ArrayView<'_, T> {
        self.view().transpose()
    }

    pub fn permute_axes(&self, axes: &[usize]) -> Result<ArrayView<'_, T>> {
        self.view().permute_axes(axes)
    }

    pub fn broadcast_to(&self, shape: &[usize]) -> Result<ArrayView<'_, T>> {
        self.view().broadcast_to(shape)
    }

    pub fn ravel(&self) -> Result<ArrayView<'_, T>> {
        self.view().ravel()
    }

    pub fn expand_dims(&self, axis: isize) -> Result<ArrayView<'_, T>> {
        self.view().expand_dims(axis)
    }

    pub fn squeeze(&self, axis: Option<isize>) -> Result<ArrayView<'_, T>> {
        self.view().squeeze(axis)
    }

    pub(crate) fn raw_data_mut(&mut self) -> &mut [T] {
        self.uniform = None;
        &mut self.data
    }

    pub(crate) fn has_uniform_storage(&self) -> bool {
        self.uniform.is_some()
    }

    pub(crate) fn uniform_value(&self) -> Option<&T> {
        self.uniform.as_ref()
    }
}

impl<T: Clone> Array<T> {
    pub fn take(&self, indices: &[isize]) -> Result<Array<T>> {
        self.view().take(indices)
    }

    pub fn take_axis(&self, indices: &[isize], axis: isize) -> Result<Array<T>> {
        self.view().take_axis(indices, axis)
    }

    pub fn boolean_mask(&self, mask: &Array<bool>) -> Result<Array<T>> {
        self.view().boolean_mask(&mask.view())
    }

    pub fn put(&mut self, indices: &[isize], values: &[T]) -> Result<()> {
        if indices.is_empty() {
            return Ok(());
        }
        if values.is_empty() {
            return Err(NumRsError::InvalidShape(
                "put values cannot be empty when indices are non-empty".to_string(),
            ));
        }

        self.uniform = None;
        let len = self.len();
        for (value_index, index) in indices.iter().copied().enumerate() {
            let target = normalize_take_index(index, 0, len)?;
            self.data[target] = values[value_index % values.len()].clone();
        }
        Ok(())
    }

    pub fn putmask(&mut self, mask: &Array<bool>, values: &[T]) -> Result<()> {
        if self.shape() != mask.shape() {
            return Err(NumRsError::BooleanMaskShapeMismatch {
                array: self.shape().to_vec(),
                mask: mask.shape().to_vec(),
            });
        }

        if mask.uniform == Some(false) {
            return Ok(());
        }
        if mask.uniform == Some(true) {
            if values.is_empty() {
                return Err(NumRsError::InvalidShape(
                    "putmask values cannot be empty when mask selects values".to_string(),
                ));
            }
            if values.len() == 1 {
                self.data.fill(values[0].clone());
                self.uniform = Some(values[0].clone());
                return Ok(());
            }
        }

        self.uniform = None;
        let mut value_index = 0usize;
        for (target, selected) in self.data.iter_mut().zip(mask.as_slice().iter().copied()) {
            if selected {
                if values.is_empty() {
                    return Err(NumRsError::InvalidShape(
                        "putmask values cannot be empty when mask selects values".to_string(),
                    ));
                }
                *target = values[value_index % values.len()].clone();
                value_index += 1;
            }
        }
        Ok(())
    }
}

impl<T> Array<T>
where
    T: Copy + Add<Output = T>,
{
    pub fn add_at(&mut self, indices: &[isize], values: &[T]) -> Result<()> {
        if indices.len() != values.len() {
            return Err(NumRsError::InvalidShape(format!(
                "add_at expected {} values, got {}",
                indices.len(),
                values.len()
            )));
        }

        self.uniform = None;
        let len = self.len();
        for (index, value) in indices.iter().copied().zip(values.iter().copied()) {
            let target = normalize_take_index(index, 0, len)?;
            self.data[target] = self.data[target] + value;
        }
        Ok(())
    }
}

impl<T> Array<T>
where
    T: Copy + PartialOrd,
{
    pub fn maximum_at(&mut self, indices: &[isize], values: &[T]) -> Result<()> {
        if indices.len() != values.len() {
            return Err(NumRsError::InvalidShape(format!(
                "maximum_at expected {} values, got {}",
                indices.len(),
                values.len()
            )));
        }

        self.uniform = None;
        let len = self.len();
        for (index, value) in indices.iter().copied().zip(values.iter().copied()) {
            let target = normalize_take_index(index, 0, len)?;
            if value > self.data[target] {
                self.data[target] = value;
            }
        }
        Ok(())
    }
}

impl<T: DType> Array<T> {
    pub fn dtype(&self) -> DTypeKind {
        T::KIND
    }
}

impl Array<bool> {
    pub fn nonzero(&self) -> Result<Vec<Array<i64>>> {
        self.view().nonzero()
    }

    pub fn where_select<T: Clone>(
        &self,
        true_values: &Array<T>,
        false_values: &Array<T>,
    ) -> Result<Array<T>> {
        self.view()
            .where_select(&true_values.view(), &false_values.view())
    }

    pub fn where_select_f64(
        &self,
        true_values: &Array<f64>,
        false_values: &Array<f64>,
    ) -> Result<Array<f64>> {
        self.view()
            .where_select_f64(&true_values.view(), &false_values.view())
    }
}

impl<T> Array<T>
where
    T: Copy,
{
    pub fn astype<U>(&self) -> Result<Array<U>>
    where
        T: CastElement<U>,
        U: DType,
    {
        self.view().astype()
    }
}

impl Array<i64> {
    pub fn arange(stop: i64) -> Self {
        let data = (0..stop).collect::<Vec<_>>();
        Self::from_vec(vec![stop.max(0) as usize], data).expect("arange shape is valid")
    }
}

impl<'a, T> ArrayView<'a, T> {
    pub(crate) fn from_parts(
        data: &'a [T],
        shape: Vec<usize>,
        strides: Vec<isize>,
        offset: isize,
    ) -> ArrayView<'a, T> {
        ArrayView {
            data,
            shape,
            strides,
            offset,
        }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn strides(&self) -> &[isize] {
        &self.strides
    }

    pub fn offset(&self) -> isize {
        self.offset
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn len(&self) -> usize {
        size_of_shape(&self.shape).expect("view shape was previously validated")
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn is_c_contiguous(&self) -> bool {
        is_c_contiguous(&self.shape, &self.strides)
    }

    pub fn get(&self, index: &[usize]) -> Result<&'a T> {
        let offset = offset_for_index(&self.shape, &self.strides, self.offset, index)?;
        Ok(&self.data[offset])
    }

    pub fn slice(&self, slices: &[Slice]) -> Result<ArrayView<'a, T>> {
        if slices.len() > self.ndim() {
            return Err(NumRsError::IndexRankMismatch {
                expected: self.ndim(),
                actual: slices.len(),
            });
        }

        let mut out_shape = Vec::with_capacity(self.ndim());
        let mut out_strides = Vec::with_capacity(self.ndim());
        let mut out_offset = self.offset;

        for axis in 0..self.ndim() {
            let slice = slices.get(axis).copied().unwrap_or(Slice::All);
            match normalize_slice(slice, axis, self.shape[axis])? {
                NormalizedSlice::Index(index) => {
                    out_offset += index * self.strides[axis];
                }
                NormalizedSlice::Range { start, len, step } => {
                    out_offset += start * self.strides[axis];
                    out_shape.push(len);
                    out_strides.push(self.strides[axis] * step);
                }
            }
        }

        Ok(ArrayView::from_parts(
            self.data,
            out_shape,
            out_strides,
            out_offset,
        ))
    }

    pub fn reshape(&self, shape: &[isize]) -> Result<ArrayView<'a, T>> {
        let resolved = resolve_shape(shape, self.len())?;
        self.reshape_exact(&resolved)
    }

    pub fn reshape_exact(&self, shape: &[usize]) -> Result<ArrayView<'a, T>> {
        let len = size_of_shape(shape)?;
        if len != self.len() {
            return Err(NumRsError::InvalidShape(format!(
                "shape {shape:?} has {len} elements, expected {}",
                self.len()
            )));
        }
        if !self.is_c_contiguous() {
            return Err(NumRsError::NonContiguousReshape {
                shape: self.shape.to_vec(),
                strides: self.strides.to_vec(),
            });
        }

        Ok(ArrayView::from_parts(
            self.data,
            shape.to_vec(),
            c_strides(shape),
            self.offset,
        ))
    }

    pub fn transpose(&self) -> ArrayView<'a, T> {
        let axes = (0..self.ndim()).rev().collect::<Vec<_>>();
        self.permute_axes(&axes).expect("reverse axes are unique")
    }

    pub fn permute_axes(&self, axes: &[usize]) -> Result<ArrayView<'a, T>> {
        permute_unique_axes(axes, self.ndim())?;
        let shape = axes.iter().map(|axis| self.shape[*axis]).collect();
        let strides = axes.iter().map(|axis| self.strides[*axis]).collect();
        Ok(ArrayView::from_parts(
            self.data,
            shape,
            strides,
            self.offset,
        ))
    }

    pub fn broadcast_to(&self, shape: &[usize]) -> Result<ArrayView<'a, T>> {
        let strides = broadcast_strides(&self.shape, &self.strides, shape)?;
        Ok(ArrayView::from_parts(
            self.data,
            shape.to_vec(),
            strides,
            self.offset,
        ))
    }

    pub fn ravel(&self) -> Result<ArrayView<'a, T>> {
        self.reshape_exact(&[self.len()])
    }

    pub fn expand_dims(&self, axis: isize) -> Result<ArrayView<'a, T>> {
        let axis = normalize_insert_axis(axis, self.ndim())?;
        let mut shape = self.shape.clone();
        let mut strides = self.strides.clone();
        shape.insert(axis, 1);
        strides.insert(axis, 0);
        Ok(ArrayView::from_parts(
            self.data,
            shape,
            strides,
            self.offset,
        ))
    }

    pub fn squeeze(&self, axis: Option<isize>) -> Result<ArrayView<'a, T>> {
        let mut shape = Vec::with_capacity(self.ndim());
        let mut strides = Vec::with_capacity(self.ndim());

        if let Some(axis) = axis {
            let axis = normalize_axis(axis, self.ndim())?;
            if self.shape[axis] != 1 {
                return Err(NumRsError::CannotSqueezeAxis {
                    axis,
                    len: self.shape[axis],
                });
            }
            for i in 0..self.ndim() {
                if i != axis {
                    shape.push(self.shape[i]);
                    strides.push(self.strides[i]);
                }
            }
        } else {
            for i in 0..self.ndim() {
                if self.shape[i] != 1 {
                    shape.push(self.shape[i]);
                    strides.push(self.strides[i]);
                }
            }
        }

        Ok(ArrayView::from_parts(
            self.data,
            shape,
            strides,
            self.offset,
        ))
    }

    pub(crate) fn offset_iter(&self) -> Result<OffsetIter> {
        OffsetIter::new(&self.shape, &self.strides, self.offset)
    }

    pub(crate) fn contiguous_slice(&self) -> Option<&'a [T]> {
        if !self.is_c_contiguous() {
            return None;
        }
        let start = self.offset as usize;
        let end = start.checked_add(self.len())?;
        self.data.get(start..end)
    }

    pub(crate) fn raw_data(&self) -> &'a [T] {
        self.data
    }
}

impl<'a, T: Clone> ArrayView<'a, T> {
    pub fn to_vec(&self) -> Result<Vec<T>> {
        if let Some(slice) = self.contiguous_slice() {
            return Ok(slice.to_vec());
        }

        let mut out = Vec::with_capacity(self.len());
        for offset in self.offset_iter()? {
            out.push(self.data[offset].clone());
        }
        Ok(out)
    }

    pub fn to_owned_array(&self) -> Result<Array<T>> {
        Array::from_vec(self.shape.to_vec(), self.to_vec()?)
    }

    pub fn take(&self, indices: &[isize]) -> Result<Array<T>> {
        let len = self.len();
        let offsets = self.offset_iter()?.collect::<Vec<_>>();
        let mut out = Vec::with_capacity(indices.len());

        for index in indices.iter().copied() {
            let index = normalize_take_index(index, 0, len)?;
            out.push(self.data[offsets[index]].clone());
        }

        Array::from_vec(vec![indices.len()], out)
    }

    pub fn take_axis(&self, indices: &[isize], axis: isize) -> Result<Array<T>> {
        let axis = normalize_axis(axis, self.ndim())?;

        if self.ndim() == 2 && self.is_c_contiguous() {
            let rows = self.shape[0];
            let cols = self.shape[1];
            let data = self.contiguous_slice().expect("contiguous checked above");
            if axis == 0 && is_positive_identity_take(indices, rows) {
                return Array::from_vec(vec![rows, cols], data.to_vec());
            }
            if axis == 1 && is_positive_identity_take(indices, cols) {
                return Array::from_vec(vec![rows, cols], data.to_vec());
            }
        }

        let normalized = indices
            .iter()
            .copied()
            .map(|index| normalize_take_index(index, axis, self.shape[axis]))
            .collect::<Result<Vec<_>>>()?;

        if self.ndim() == 2 && self.is_c_contiguous() {
            let rows = self.shape[0];
            let cols = self.shape[1];
            let data = self.contiguous_slice().expect("contiguous checked above");
            if axis == 1 {
                if normalized.len() == cols
                    && normalized
                        .iter()
                        .copied()
                        .enumerate()
                        .all(|(idx, source)| idx == source)
                {
                    return Array::from_vec(vec![rows, cols], data.to_vec());
                }
                let mut out = Vec::with_capacity(rows * normalized.len());
                for row in 0..rows {
                    let base = row * cols;
                    for col in normalized.iter().copied() {
                        out.push(data[base + col].clone());
                    }
                }
                return Array::from_vec(vec![rows, normalized.len()], out);
            }
            if axis == 0 {
                if normalized.len() == rows
                    && normalized
                        .iter()
                        .copied()
                        .enumerate()
                        .all(|(idx, source)| idx == source)
                {
                    return Array::from_vec(vec![rows, cols], data.to_vec());
                }
                let mut out = Vec::with_capacity(normalized.len() * cols);
                for row in normalized.iter().copied() {
                    let base = row * cols;
                    out.extend_from_slice(&data[base..base + cols]);
                }
                return Array::from_vec(vec![normalized.len(), cols], out);
            }
        }

        let mut out_shape = self.shape.clone();
        out_shape[axis] = indices.len();
        let out_len = size_of_shape(&out_shape)?;
        let mut out = Vec::with_capacity(out_len);

        for linear in 0..out_len {
            let mut remainder = linear;
            let mut source_index = vec![0usize; self.ndim()];
            for current_axis in (0..out_shape.len()).rev() {
                let dim = out_shape[current_axis];
                let coord = remainder.checked_rem(dim).unwrap_or(0);
                remainder = remainder.checked_div(dim).unwrap_or(0);
                source_index[current_axis] = if current_axis == axis {
                    normalized[coord]
                } else {
                    coord
                };
            }
            let offset = offset_for_index(&self.shape, &self.strides, self.offset, &source_index)?;
            out.push(self.data[offset].clone());
        }

        Array::from_vec(out_shape, out)
    }

    pub fn boolean_mask(&self, mask: &ArrayView<'_, bool>) -> Result<Array<T>> {
        if self.shape() != mask.shape() {
            return Err(NumRsError::BooleanMaskShapeMismatch {
                array: self.shape().to_vec(),
                mask: mask.shape().to_vec(),
            });
        }

        let value_offsets = self.offset_iter()?;
        let mask_offsets = mask.offset_iter()?;
        let mut out = Vec::new();

        for (value_offset, mask_offset) in value_offsets.zip(mask_offsets) {
            if mask.raw_data()[mask_offset] {
                out.push(self.data[value_offset].clone());
            }
        }

        Array::from_vec(vec![out.len()], out)
    }
}

impl<'a, T: DType> ArrayView<'a, T> {
    pub fn dtype(&self) -> DTypeKind {
        T::KIND
    }
}

impl<'a> ArrayView<'a, bool> {
    pub fn nonzero(&self) -> Result<Vec<Array<i64>>> {
        if let Some(slice) = self.contiguous_slice() {
            if self.ndim() == 1 {
                let mut coordinates = Vec::new();
                for (index, selected) in slice.iter().copied().enumerate() {
                    if selected {
                        coordinates.push(index as i64);
                    }
                }
                return Ok(vec![Array::from_vec(vec![coordinates.len()], coordinates)?]);
            }

            if self.ndim() == 2 {
                let rows = self.shape[0];
                let cols = self.shape[1];
                let mut row_coordinates = Vec::new();
                let mut col_coordinates = Vec::new();
                for row in 0..rows {
                    let row_start = row * cols;
                    for col in 0..cols {
                        if slice[row_start + col] {
                            row_coordinates.push(row as i64);
                            col_coordinates.push(col as i64);
                        }
                    }
                }
                return Ok(vec![
                    Array::from_vec(vec![row_coordinates.len()], row_coordinates)?,
                    Array::from_vec(vec![col_coordinates.len()], col_coordinates)?,
                ]);
            }
        }

        let mut coordinates = vec![Vec::<i64>::new(); self.ndim()];
        for (linear, offset) in self.offset_iter()?.enumerate() {
            if !self.data[offset] {
                continue;
            }

            let mut remainder = linear;
            for axis in (0..self.ndim()).rev() {
                let dim = self.shape[axis];
                let coordinate = remainder.checked_rem(dim).unwrap_or(0);
                remainder = remainder.checked_div(dim).unwrap_or(0);
                coordinates[axis].push(coordinate as i64);
            }
        }

        coordinates
            .into_iter()
            .map(|axis_coordinates| Array::from_vec(vec![axis_coordinates.len()], axis_coordinates))
            .collect()
    }

    pub fn where_select<T: Clone>(
        &self,
        true_values: &ArrayView<'_, T>,
        false_values: &ArrayView<'_, T>,
    ) -> Result<Array<T>> {
        let output_shape = broadcast_shape(self.shape(), true_values.shape())?;
        let output_shape = broadcast_shape(&output_shape, false_values.shape())?;
        let output_len = size_of_shape(&output_shape)?;

        if self.shape() == output_shape && true_values.shape() == output_shape {
            if let (Some(condition), Some(true_slice), Some(false_slice)) = (
                self.contiguous_slice(),
                true_values.contiguous_slice(),
                false_values.contiguous_slice(),
            ) {
                if false_values.shape() == output_shape {
                    let mut out = Vec::with_capacity(output_len);
                    for ((selected, true_value), false_value) in condition
                        .iter()
                        .copied()
                        .zip(true_slice.iter())
                        .zip(false_slice.iter())
                    {
                        out.push(if selected {
                            true_value.clone()
                        } else {
                            false_value.clone()
                        });
                    }
                    return Array::from_vec(output_shape, out);
                }

                if false_values.shape().is_empty() {
                    let fallback = &false_slice[0];
                    let mut out = Vec::with_capacity(output_len);
                    for (selected, true_value) in condition.iter().copied().zip(true_slice.iter()) {
                        out.push(if selected {
                            true_value.clone()
                        } else {
                            fallback.clone()
                        });
                    }
                    return Array::from_vec(output_shape, out);
                }

                if output_shape.len() == 2 && false_values.shape() == [1, output_shape[1]] {
                    let rows = output_shape[0];
                    let cols = output_shape[1];
                    let mut out = Vec::with_capacity(output_len);
                    for row in 0..rows {
                        let row_start = row * cols;
                        for (col, false_value) in false_slice.iter().enumerate().take(cols) {
                            let index = row_start + col;
                            out.push(if condition[index] {
                                true_slice[index].clone()
                            } else {
                                false_value.clone()
                            });
                        }
                    }
                    return Array::from_vec(output_shape, out);
                }

                if output_shape.len() == 2 && false_values.shape() == [output_shape[0], 1] {
                    let rows = output_shape[0];
                    let cols = output_shape[1];
                    let mut out = Vec::with_capacity(output_len);
                    for (row, false_value) in false_slice.iter().enumerate().take(rows) {
                        let row_start = row * cols;
                        for col in 0..cols {
                            let index = row_start + col;
                            out.push(if condition[index] {
                                true_slice[index].clone()
                            } else {
                                false_value.clone()
                            });
                        }
                    }
                    return Array::from_vec(output_shape, out);
                }
            }
        }

        let condition_strides = broadcast_strides(self.shape(), self.strides(), &output_shape)?;
        let true_strides =
            broadcast_strides(true_values.shape(), true_values.strides(), &output_shape)?;
        let false_strides =
            broadcast_strides(false_values.shape(), false_values.strides(), &output_shape)?;

        let condition_offsets = OffsetIter::new(&output_shape, &condition_strides, self.offset())?;
        let true_offsets = OffsetIter::new(&output_shape, &true_strides, true_values.offset())?;
        let false_offsets = OffsetIter::new(&output_shape, &false_strides, false_values.offset())?;
        let mut out = Vec::with_capacity(output_len);

        for ((condition_offset, true_offset), false_offset) in
            condition_offsets.zip(true_offsets).zip(false_offsets)
        {
            if self.data[condition_offset] {
                out.push(true_values.raw_data()[true_offset].clone());
            } else {
                out.push(false_values.raw_data()[false_offset].clone());
            }
        }

        Array::from_vec(output_shape, out)
    }

    pub fn where_select_f64(
        &self,
        true_values: &ArrayView<'_, f64>,
        false_values: &ArrayView<'_, f64>,
    ) -> Result<Array<f64>> {
        let output_shape = broadcast_shape(self.shape(), true_values.shape())?;
        let output_shape = broadcast_shape(&output_shape, false_values.shape())?;

        if self.shape() == output_shape && true_values.shape() == output_shape {
            if let (Some(condition), Some(true_slice), Some(false_slice)) = (
                self.contiguous_slice(),
                true_values.contiguous_slice(),
                false_values.contiguous_slice(),
            ) {
                if false_values.shape() == output_shape {
                    return array_from_f64_selector(&output_shape, |index| {
                        if condition[index] {
                            true_slice[index]
                        } else {
                            false_slice[index]
                        }
                    });
                }

                if false_values.shape().is_empty() {
                    let fallback = false_slice[0];
                    return array_from_f64_selector(&output_shape, |index| {
                        if condition[index] {
                            true_slice[index]
                        } else {
                            fallback
                        }
                    });
                }

                if output_shape.len() == 2 && false_values.shape() == [1, output_shape[1]] {
                    let rows = output_shape[0];
                    let cols = output_shape[1];
                    let len = size_of_shape(&output_shape)?;
                    let mut out: Vec<f64> = Vec::with_capacity(len);
                    let out_ptr = out.as_mut_ptr();
                    for row in 0..rows {
                        let row_start = row * cols;
                        for (col, false_value) in false_slice.iter().copied().enumerate().take(cols)
                        {
                            let index = row_start + col;
                            let value = if condition[index] {
                                true_slice[index]
                            } else {
                                false_value
                            };
                            unsafe {
                                out_ptr.add(index).write(value);
                            }
                        }
                    }
                    unsafe {
                        out.set_len(len);
                    }
                    return Array::from_vec(output_shape, out);
                }

                if output_shape.len() == 2 && false_values.shape() == [output_shape[0], 1] {
                    let cols = output_shape[1];
                    let len = size_of_shape(&output_shape)?;
                    let mut out: Vec<f64> = Vec::with_capacity(len);
                    let out_ptr = out.as_mut_ptr();
                    for (row, false_value) in false_slice.iter().copied().enumerate() {
                        let row_start = row * cols;
                        for col in 0..cols {
                            let index = row_start + col;
                            let value = if condition[index] {
                                true_slice[index]
                            } else {
                                false_value
                            };
                            unsafe {
                                out_ptr.add(index).write(value);
                            }
                        }
                    }
                    unsafe {
                        out.set_len(len);
                    }
                    return Array::from_vec(output_shape, out);
                }
            }
        }

        self.where_select(true_values, false_values)
    }
}

impl<'a, T> ArrayView<'a, T>
where
    T: Copy,
{
    pub fn astype<U>(&self) -> Result<Array<U>>
    where
        T: CastElement<U>,
        U: DType,
    {
        let out = if let Some(slice) = self.contiguous_slice() {
            slice.iter().copied().map(CastElement::cast).collect()
        } else {
            let mut out = Vec::with_capacity(self.len());
            for offset in self.offset_iter()? {
                out.push(self.data[offset].cast());
            }
            out
        };
        Array::from_vec(self.shape.to_vec(), out)
    }
}

pub(crate) fn axis_without(shape: &[usize], axis: isize) -> Result<(usize, Vec<usize>)> {
    let axis = normalize_axis(axis, shape.len())?;
    let mut out = Vec::with_capacity(shape.len().saturating_sub(1));
    for (i, dim) in shape.iter().copied().enumerate() {
        if i != axis {
            out.push(dim);
        }
    }
    Ok((axis, out))
}

fn normalize_take_index(index: isize, axis: usize, len: usize) -> Result<usize> {
    let normalized = if index < 0 {
        len as isize + index
    } else {
        index
    };
    if normalized < 0 || normalized >= len as isize {
        return Err(NumRsError::IndexOutOfBounds { axis, index, len });
    }
    Ok(normalized as usize)
}

fn is_positive_identity_take(indices: &[isize], len: usize) -> bool {
    indices.len() == len
        && indices
            .iter()
            .copied()
            .enumerate()
            .all(|(idx, source)| source >= 0 && source as usize == idx)
}

fn array_from_f64_selector<F>(shape: &[usize], mut selector: F) -> Result<Array<f64>>
where
    F: FnMut(usize) -> f64,
{
    let len = size_of_shape(shape)?;
    let mut out: Vec<f64> = Vec::with_capacity(len);
    let out_ptr = out.as_mut_ptr();
    for index in 0..len {
        // Every element is written exactly once before the vector length is set.
        unsafe {
            out_ptr.add(index).write(selector(index));
        }
    }
    unsafe {
        out.set_len(len);
    }
    Array::from_vec(shape.to_vec(), out)
}
