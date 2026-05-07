use numrs_core::{promote_dtype, Array, Complex32, Complex64, DTypeKind};
use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{
    PyAny, PyComplex, PyComplexMethods, PyList, PyModule, PySlice, PyTuple, PyTupleMethods,
};
use rustfft::FftPlanner;

#[derive(Clone)]
enum Storage {
    F64(Array<f64>),
    F32(Array<f32>),
    C128(Array<Complex64>),
    C64(Array<Complex32>),
    I64(Array<i64>),
    I32(Array<i32>),
    I16(Array<i16>),
    I8(Array<i8>),
    U64(Array<u64>),
    U32(Array<u32>),
    U16(Array<u16>),
    U8(Array<u8>),
    Bool(Array<bool>),
}

enum RawIndexItem {
    Index(isize),
    Slice {
        start: Option<isize>,
        stop: Option<isize>,
        step: Option<isize>,
    },
    NewAxis,
    Ellipsis,
}

enum NormalizedIndexItem {
    Index(usize),
    Range(Vec<usize>),
    NewAxis,
}

struct IndexPlan {
    out_shape: Vec<usize>,
    source_offsets: Vec<usize>,
}

macro_rules! with_array {
    ($storage:expr, $array:ident => $body:block) => {{
        match $storage {
            Storage::F64($array) => $body,
            Storage::F32($array) => $body,
            Storage::C128($array) => $body,
            Storage::C64($array) => $body,
            Storage::I64($array) => $body,
            Storage::I32($array) => $body,
            Storage::I16($array) => $body,
            Storage::I8($array) => $body,
            Storage::U64($array) => $body,
            Storage::U32($array) => $body,
            Storage::U16($array) => $body,
            Storage::U8($array) => $body,
            Storage::Bool($array) => $body,
        }
    }};
}

#[pyclass(name = "Array", module = "numrust._numrust", from_py_object)]
#[derive(Clone)]
struct PyArray {
    storage: Storage,
}

#[pymethods]
impl PyArray {
    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        Ok(PyTuple::new(py, self.shape_vec())?.unbind().into())
    }

    #[getter]
    fn ndim(&self) -> usize {
        self.shape_vec().len()
    }

    #[getter]
    fn size(&self) -> usize {
        with_array!(&self.storage, array => { array.len() })
    }

    #[getter]
    fn device(&self) -> &'static str {
        "cpu"
    }

    #[getter]
    fn dtype<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        Ok(PyModule::import(py, "numrust")?
            .getattr(dtype_name(self.dtype_kind()))?
            .unbind())
    }

    fn astype(&self, dtype: &str) -> PyResult<Self> {
        match dtype {
            "float64" => Ok(Self::from(to_f64_array(self)?)),
            "float32" => Ok(Self::from(to_f32_array(self)?)),
            "complex128" => Ok(Self::from(to_c128_array(self)?)),
            "complex64" => Ok(Self::from(to_c64_array(self)?)),
            "int64" => Ok(Self::from(to_i64_array(self)?)),
            "int32" => Ok(Self::from(to_i32_array(self)?)),
            "int16" => Ok(Self::from(to_i16_array(self)?)),
            "int8" => Ok(Self::from(to_i8_array(self)?)),
            "uint64" => Ok(Self::from(to_u64_array(self)?)),
            "uint32" => Ok(Self::from(to_u32_array(self)?)),
            "uint16" => Ok(Self::from(to_u16_array(self)?)),
            "uint8" => Ok(Self::from(to_u8_array(self)?)),
            "bool" => Ok(Self::from(to_bool_array(self)?)),
            _ => Err(dtype_error(dtype)),
        }
    }

    #[pyo3(signature = (api_version = None))]
    fn __array_namespace__<'py>(
        &self,
        py: Python<'py>,
        api_version: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        if let Some(version) = api_version {
            if version != "2021.12" && version != "2023.12" {
                return Err(PyValueError::new_err(format!(
                    "unsupported Array API version {version:?}"
                )));
            }
        }
        Ok(PyModule::import(py, "numrust")?.unbind().into())
    }

    fn __dlpack_device__(&self) -> (i32, i32) {
        (1, 0)
    }

    #[pyo3(signature = (*, stream = None, max_version = None, dl_device = None, copy = None))]
    fn __dlpack__(
        &self,
        stream: Option<&Bound<'_, PyAny>>,
        max_version: Option<(u8, u8)>,
        dl_device: Option<(i32, i32)>,
        copy: Option<bool>,
    ) -> PyResult<()> {
        let _ = (stream, max_version, copy);
        if let Some(device) = dl_device {
            if device != (1, 0) {
                return Err(PyValueError::new_err(format!(
                    "unsupported DLPack device {device:?}; only CPU device 0 is available"
                )));
            }
        }
        Ok(())
    }

    fn tolist<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        match &self.storage {
            Storage::F64(array) => Ok(PyList::new(py, array.as_slice().iter().copied())?
                .unbind()
                .into()),
            Storage::F32(array) => Ok(PyList::new(py, array.as_slice().iter().copied())?
                .unbind()
                .into()),
            Storage::C128(array) => complex_list(
                py,
                array.as_slice().iter().map(|value| (value.re, value.im)),
            ),
            Storage::C64(array) => complex_list(
                py,
                array
                    .as_slice()
                    .iter()
                    .map(|value| (value.re as f64, value.im as f64)),
            ),
            Storage::I64(array) => Ok(PyList::new(py, array.as_slice().iter().copied())?
                .unbind()
                .into()),
            Storage::I32(array) => Ok(PyList::new(py, array.as_slice().iter().copied())?
                .unbind()
                .into()),
            Storage::I16(array) => Ok(PyList::new(py, array.as_slice().iter().copied())?
                .unbind()
                .into()),
            Storage::I8(array) => Ok(PyList::new(py, array.as_slice().iter().copied())?
                .unbind()
                .into()),
            Storage::U64(array) => Ok(PyList::new(py, array.as_slice().iter().copied())?
                .unbind()
                .into()),
            Storage::U32(array) => Ok(PyList::new(py, array.as_slice().iter().copied())?
                .unbind()
                .into()),
            Storage::U16(array) => Ok(PyList::new(py, array.as_slice().iter().copied())?
                .unbind()
                .into()),
            Storage::U8(array) => Ok(PyList::new(py, array.as_slice().iter().copied())?
                .unbind()
                .into()),
            Storage::Bool(array) => Ok(PyList::new(py, array.as_slice().iter().copied())?
                .unbind()
                .into()),
        }
    }

    fn reshape(&self, shape: Vec<isize>) -> PyResult<Self> {
        with_array!(&self.storage, array => {
            Ok(Self::from(
                array
                    .reshape(&shape)
                    .map_err(py_value_error)?
                    .to_owned_array()
                    .map_err(py_value_error)?,
            ))
        })
    }

    fn transpose(&self) -> PyResult<Self> {
        with_array!(&self.storage, array => {
            Ok(Self::from(
                array.transpose().to_owned_array().map_err(py_value_error)?,
            ))
        })
    }

    fn sum(&self) -> PyResult<Self> {
        match &self.storage {
            Storage::F64(array) => scalar_array(array.sum_all().map_err(py_value_error)?),
            Storage::F32(array) => scalar_array(array.sum_all().map_err(py_value_error)?),
            Storage::C128(array) => scalar_array(array.sum_all().map_err(py_value_error)?),
            Storage::C64(array) => scalar_array(array.sum_all().map_err(py_value_error)?),
            Storage::I64(array) => scalar_array(array.sum_all().map_err(py_value_error)?),
            Storage::I32(array) => scalar_array(array.sum_all().map_err(py_value_error)?),
            Storage::I16(array) => scalar_array(array.sum_all().map_err(py_value_error)?),
            Storage::I8(array) => scalar_array(array.sum_all().map_err(py_value_error)?),
            Storage::U64(array) => scalar_array(array.sum_all().map_err(py_value_error)?),
            Storage::U32(array) => scalar_array(array.sum_all().map_err(py_value_error)?),
            Storage::U16(array) => scalar_array(array.sum_all().map_err(py_value_error)?),
            Storage::U8(array) => scalar_array(array.sum_all().map_err(py_value_error)?),
            Storage::Bool(_) => Err(PyTypeError::new_err(
                "sum is not implemented for bool arrays",
            )),
        }
    }

    fn mean(&self) -> PyResult<Self> {
        Ok(Self::from(
            Array::scalar(to_f64_array(self)?.mean_all().map_err(py_value_error)?)
                .map_err(py_value_error)?,
        ))
    }

    fn matmul(&self, rhs: &PyArray) -> PyResult<Self> {
        match (&self.storage, &rhs.storage) {
            (Storage::F64(left), Storage::F64(right)) => {
                Ok(Self::from(left.matmul(right).map_err(py_value_error)?))
            }
            (Storage::F32(left), Storage::F32(right)) => {
                Ok(Self::from(left.matmul(right).map_err(py_value_error)?))
            }
            (Storage::C128(left), Storage::C128(right)) => {
                Ok(Self::from(left.matmul(right).map_err(py_value_error)?))
            }
            (Storage::C64(left), Storage::C64(right)) => {
                Ok(Self::from(left.matmul(right).map_err(py_value_error)?))
            }
            (Storage::I64(left), Storage::I64(right)) => {
                Ok(Self::from(left.matmul(right).map_err(py_value_error)?))
            }
            (Storage::I32(left), Storage::I32(right)) => {
                Ok(Self::from(left.matmul(right).map_err(py_value_error)?))
            }
            (Storage::I16(left), Storage::I16(right)) => {
                Ok(Self::from(left.matmul(right).map_err(py_value_error)?))
            }
            (Storage::I8(left), Storage::I8(right)) => {
                Ok(Self::from(left.matmul(right).map_err(py_value_error)?))
            }
            (Storage::U64(left), Storage::U64(right)) => {
                Ok(Self::from(left.matmul(right).map_err(py_value_error)?))
            }
            (Storage::U32(left), Storage::U32(right)) => {
                Ok(Self::from(left.matmul(right).map_err(py_value_error)?))
            }
            (Storage::U16(left), Storage::U16(right)) => {
                Ok(Self::from(left.matmul(right).map_err(py_value_error)?))
            }
            (Storage::U8(left), Storage::U8(right)) => {
                Ok(Self::from(left.matmul(right).map_err(py_value_error)?))
            }
            _ => Err(PyTypeError::new_err(
                "matmul requires arrays with matching numeric dtypes",
            )),
        }
    }

    fn take_axis(&self, indices: &PyArray, axis: isize) -> PyResult<Self> {
        let indices = to_i64_array(indices)?
            .as_slice()
            .iter()
            .copied()
            .map(|value| value as isize)
            .collect::<Vec<_>>();
        with_array!(&self.storage, array => {
            Ok(Self::from(array.take_axis(&indices, axis).map_err(py_value_error)?))
        })
    }

    fn det(&self) -> PyResult<Self> {
        match &self.storage {
            Storage::F64(array) => Ok(Self::from(stacked_det_f64(array, |value| value)?)),
            Storage::F32(array) => Ok(Self::from(stacked_det_f32(array)?)),
            Storage::C128(array) => Ok(Self::from(stacked_det_c128(array)?)),
            Storage::C64(array) => Ok(Self::from(stacked_det_c64(array)?)),
            _ => Err(PyTypeError::new_err(
                "det is currently implemented for floating and complex arrays",
            )),
        }
    }

    fn diagonal(&self, offset: isize) -> PyResult<Self> {
        with_array!(&self.storage, array => {
            Ok(Self::from(diagonal_array(array, offset)?))
        })
    }

    fn matrix_transpose(&self) -> PyResult<Self> {
        with_array!(&self.storage, array => {
            Ok(Self::from(matrix_transpose_array(array)?))
        })
    }

    #[pyo3(signature = (n = None, axis = -1, inverse = false, norm = "backward"))]
    fn fft_axis(&self, n: Option<usize>, axis: isize, inverse: bool, norm: &str) -> PyResult<Self> {
        match &self.storage {
            Storage::C128(array) => Ok(Self::from(fft_axis_c128(array, n, axis, inverse, norm)?)),
            Storage::C64(array) => Ok(Self::from(fft_axis_c64(array, n, axis, inverse, norm)?)),
            _ => Err(PyTypeError::new_err(
                "fft_axis requires a complex64 or complex128 array",
            )),
        }
    }

    #[pyo3(signature = (n = None, axis = -1, norm = "backward"))]
    fn rfft_axis(&self, n: Option<usize>, axis: isize, norm: &str) -> PyResult<Self> {
        match &self.storage {
            Storage::F64(array) => Ok(Self::from(rfft_axis_f64(array, n, axis, norm)?)),
            Storage::F32(array) => Ok(Self::from(rfft_axis_f32(array, n, axis, norm)?)),
            _ => Err(PyTypeError::new_err(
                "rfft_axis requires a float32 or float64 array",
            )),
        }
    }

    #[pyo3(signature = (n = None, axis = -1, norm = "backward"))]
    fn irfft_axis(&self, n: Option<usize>, axis: isize, norm: &str) -> PyResult<Self> {
        match &self.storage {
            Storage::C128(array) => Ok(Self::from(irfft_axis_c128(array, n, axis, norm)?)),
            Storage::C64(array) => Ok(Self::from(irfft_axis_c64(array, n, axis, norm)?)),
            _ => Err(PyTypeError::new_err(
                "irfft_axis requires a complex64 or complex128 array",
            )),
        }
    }

    #[pyo3(signature = (axes = None, inverse = false))]
    fn fft_shift(&self, axes: Option<Vec<isize>>, inverse: bool) -> PyResult<Self> {
        match &self.storage {
            Storage::F64(array) => Ok(Self::from(shift_array(array, axes.as_deref(), inverse)?)),
            Storage::F32(array) => Ok(Self::from(shift_array(array, axes.as_deref(), inverse)?)),
            Storage::C128(array) => Ok(Self::from(shift_array(array, axes.as_deref(), inverse)?)),
            Storage::C64(array) => Ok(Self::from(shift_array(array, axes.as_deref(), inverse)?)),
            Storage::I64(array) => Ok(Self::from(shift_array(array, axes.as_deref(), inverse)?)),
            Storage::I32(array) => Ok(Self::from(shift_array(array, axes.as_deref(), inverse)?)),
            Storage::I16(array) => Ok(Self::from(shift_array(array, axes.as_deref(), inverse)?)),
            Storage::I8(array) => Ok(Self::from(shift_array(array, axes.as_deref(), inverse)?)),
            Storage::U64(array) => Ok(Self::from(shift_array(array, axes.as_deref(), inverse)?)),
            Storage::U32(array) => Ok(Self::from(shift_array(array, axes.as_deref(), inverse)?)),
            Storage::U16(array) => Ok(Self::from(shift_array(array, axes.as_deref(), inverse)?)),
            Storage::U8(array) => Ok(Self::from(shift_array(array, axes.as_deref(), inverse)?)),
            Storage::Bool(array) => Ok(Self::from(shift_array(array, axes.as_deref(), inverse)?)),
        }
    }

    #[getter]
    #[allow(non_snake_case)]
    fn T(&self) -> PyResult<Self> {
        self.transpose()
    }

    #[getter]
    #[allow(non_snake_case)]
    fn mT(&self) -> PyResult<Self> {
        self.transpose()
    }

    #[pyo3(signature = (device, /, *, stream = None))]
    fn to_device(&self, device: &str, stream: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let _ = stream;
        if device == "cpu" {
            Ok(self.clone())
        } else {
            Err(PyValueError::new_err(format!(
                "unsupported device {device:?}; only 'cpu' is available"
            )))
        }
    }

    fn __matmul__(&self, rhs: &PyArray) -> PyResult<Self> {
        self.matmul(rhs)
    }

    fn __abs__(&self) -> PyResult<Self> {
        match &self.storage {
            Storage::F64(array) => Ok(Self::from(cast_array(array, |value| value.abs())?)),
            Storage::F32(array) => Ok(Self::from(cast_array(array, |value| value.abs())?)),
            Storage::C128(array) => Ok(Self::from(cast_array(array, |value| value.norm())?)),
            Storage::C64(array) => Ok(Self::from(cast_array(array, |value| value.norm())?)),
            Storage::I64(array) => Ok(Self::from(cast_array(array, |value| value.abs())?)),
            Storage::I32(array) => Ok(Self::from(cast_array(array, |value| value.abs())?)),
            Storage::I16(array) => Ok(Self::from(cast_array(array, |value| value.abs())?)),
            Storage::I8(array) => Ok(Self::from(cast_array(array, |value| value.abs())?)),
            Storage::U64(_) | Storage::U32(_) | Storage::U16(_) | Storage::U8(_) => {
                Ok(self.clone())
            }
            Storage::Bool(_) => Err(PyTypeError::new_err("abs is not defined for bool arrays")),
        }
    }

    fn __and__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        let rhs = self.coerce_rhs(rhs)?;
        binary_bitwise_and(self, &rhs)
    }

    fn __iand__(&mut self, rhs: &Bound<'_, PyAny>) -> PyResult<()> {
        let result = self.__and__(rhs)?;
        self.storage = result.storage;
        Ok(())
    }

    fn __floordiv__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        let rhs = self.coerce_rhs(rhs)?;
        binary_floor_divide(self, &rhs)
    }

    fn __ifloordiv__(&mut self, rhs: &Bound<'_, PyAny>) -> PyResult<()> {
        let result = self.__floordiv__(rhs)?;
        self.storage = result.storage;
        Ok(())
    }

    fn __invert__(&self) -> PyResult<Self> {
        unary_bitwise_invert(self)
    }

    fn __lshift__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        let rhs = self.coerce_rhs(rhs)?;
        binary_bitwise_left_shift(self, &rhs)
    }

    fn __ilshift__(&mut self, rhs: &Bound<'_, PyAny>) -> PyResult<()> {
        let result = self.__lshift__(rhs)?;
        self.storage = result.storage;
        Ok(())
    }

    fn __mod__(&self, _rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        array_api_not_implemented("__mod__")
    }

    fn __neg__(&self) -> PyResult<Self> {
        match &self.storage {
            Storage::F64(array) => Ok(Self::from(cast_array(array, |value| -value)?)),
            Storage::F32(array) => Ok(Self::from(cast_array(array, |value| -value)?)),
            Storage::C128(array) => Ok(Self::from(cast_array(array, |value| -value)?)),
            Storage::C64(array) => Ok(Self::from(cast_array(array, |value| -value)?)),
            Storage::I64(array) => Ok(Self::from(cast_array(array, |value| value.wrapping_neg())?)),
            Storage::I32(array) => Ok(Self::from(cast_array(array, |value| value.wrapping_neg())?)),
            Storage::I16(array) => Ok(Self::from(cast_array(array, |value| value.wrapping_neg())?)),
            Storage::I8(array) => Ok(Self::from(cast_array(array, |value| value.wrapping_neg())?)),
            Storage::U64(array) => Ok(Self::from(cast_array(array, |value| value.wrapping_neg())?)),
            Storage::U32(array) => Ok(Self::from(cast_array(array, |value| value.wrapping_neg())?)),
            Storage::U16(array) => Ok(Self::from(cast_array(array, |value| value.wrapping_neg())?)),
            Storage::U8(array) => Ok(Self::from(cast_array(array, |value| value.wrapping_neg())?)),
            Storage::Bool(_) => Err(PyTypeError::new_err(
                "negative is not defined for bool arrays",
            )),
        }
    }

    fn __or__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        let rhs = self.coerce_rhs(rhs)?;
        binary_bitwise_or(self, &rhs)
    }

    fn __ior__(&mut self, rhs: &Bound<'_, PyAny>) -> PyResult<()> {
        let result = self.__or__(rhs)?;
        self.storage = result.storage;
        Ok(())
    }

    fn __pos__(&self) -> Self {
        self.clone()
    }

    fn __pow__(&self, rhs: &Bound<'_, PyAny>, modulo: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        if modulo.is_some() {
            return Err(PyTypeError::new_err(
                "__pow__ modulo argument is unsupported",
            ));
        }
        let rhs = self.coerce_rhs(rhs)?;
        binary_power(self, &rhs)
    }

    fn __ipow__(
        &mut self,
        rhs: &Bound<'_, PyAny>,
        modulo: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let result = self.__pow__(rhs, modulo)?;
        self.storage = result.storage;
        Ok(())
    }

    fn __rshift__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        let rhs = self.coerce_rhs(rhs)?;
        binary_bitwise_right_shift(self, &rhs)
    }

    fn __irshift__(&mut self, rhs: &Bound<'_, PyAny>) -> PyResult<()> {
        let result = self.__rshift__(rhs)?;
        self.storage = result.storage;
        Ok(())
    }

    fn __xor__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        let rhs = self.coerce_rhs(rhs)?;
        binary_bitwise_xor(self, &rhs)
    }

    fn __ixor__(&mut self, rhs: &Bound<'_, PyAny>) -> PyResult<()> {
        let result = self.__xor__(rhs)?;
        self.storage = result.storage;
        Ok(())
    }

    fn __add__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        let rhs = self.coerce_rhs(rhs)?;
        binary_add(self, &rhs)
    }

    fn __iadd__(&mut self, rhs: &Bound<'_, PyAny>) -> PyResult<()> {
        let result = self.__add__(rhs)?;
        self.storage = result.storage;
        Ok(())
    }

    fn __sub__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        let rhs = self.coerce_rhs(rhs)?;
        binary_sub(self, &rhs)
    }

    fn __isub__(&mut self, rhs: &Bound<'_, PyAny>) -> PyResult<()> {
        let result = self.__sub__(rhs)?;
        self.storage = result.storage;
        Ok(())
    }

    fn __mul__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        let rhs = self.coerce_rhs(rhs)?;
        binary_mul(self, &rhs)
    }

    fn __imul__(&mut self, rhs: &Bound<'_, PyAny>) -> PyResult<()> {
        let result = self.__mul__(rhs)?;
        self.storage = result.storage;
        Ok(())
    }

    fn __truediv__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        let rhs = self.coerce_rhs(rhs)?;
        binary_divide(self, &rhs)
    }

    fn __itruediv__(&mut self, rhs: &Bound<'_, PyAny>) -> PyResult<()> {
        let result = self.__truediv__(rhs)?;
        self.storage = result.storage;
        Ok(())
    }

    fn __eq__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        let rhs = self.coerce_rhs(rhs)?;
        compare_equal(self, &rhs)
    }

    fn __ne__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        let rhs = self.coerce_rhs(rhs)?;
        let equals = compare_equal(self, &rhs)?;
        match equals.storage {
            Storage::Bool(array) => Ok(Self::from(
                array.map(|value| !value).map_err(py_value_error)?,
            )),
            _ => unreachable!("compare_equal returns bool storage"),
        }
    }

    fn __lt__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        let rhs = self.coerce_rhs(rhs)?;
        compare_ordered(
            self,
            &rhs,
            |left, right| left < right,
            |left, right| left < right,
        )
    }

    fn __le__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        let rhs = self.coerce_rhs(rhs)?;
        compare_ordered(
            self,
            &rhs,
            |left, right| left <= right,
            |left, right| left <= right,
        )
    }

    fn __gt__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        let rhs = self.coerce_rhs(rhs)?;
        compare_ordered(
            self,
            &rhs,
            |left, right| left > right,
            |left, right| left > right,
        )
    }

    fn __ge__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        let rhs = self.coerce_rhs(rhs)?;
        compare_ordered(
            self,
            &rhs,
            |left, right| left >= right,
            |left, right| left >= right,
        )
    }

    fn __getitem__(&self, key: &Bound<'_, PyAny>) -> PyResult<Self> {
        with_array!(&self.storage, array => {
            Ok(Self::from(getitem_array(array, key, self.bool_mask_from_key(key)?)?))
        })
    }

    fn __setitem__(&mut self, key: &Bound<'_, PyAny>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let bool_mask = self.bool_mask_from_key(key)?;
        self.storage = match &self.storage {
            Storage::F64(array) => Storage::F64(setitem_array(
                array,
                key,
                bool_mask.as_ref(),
                assignment_values_f64(value)?,
            )?),
            Storage::F32(array) => Storage::F32(setitem_array(
                array,
                key,
                bool_mask.as_ref(),
                assignment_values_f32(value)?,
            )?),
            Storage::C128(array) => Storage::C128(setitem_array(
                array,
                key,
                bool_mask.as_ref(),
                assignment_values_c128(value)?,
            )?),
            Storage::C64(array) => Storage::C64(setitem_array(
                array,
                key,
                bool_mask.as_ref(),
                assignment_values_c64(value)?,
            )?),
            Storage::I64(array) => Storage::I64(setitem_array(
                array,
                key,
                bool_mask.as_ref(),
                assignment_values_i64(value)?,
            )?),
            Storage::I32(array) => Storage::I32(setitem_array(
                array,
                key,
                bool_mask.as_ref(),
                assignment_values_i32(value)?,
            )?),
            Storage::I16(array) => Storage::I16(setitem_array(
                array,
                key,
                bool_mask.as_ref(),
                assignment_values_i16(value)?,
            )?),
            Storage::I8(array) => Storage::I8(setitem_array(
                array,
                key,
                bool_mask.as_ref(),
                assignment_values_i8(value)?,
            )?),
            Storage::U64(array) => Storage::U64(setitem_array(
                array,
                key,
                bool_mask.as_ref(),
                assignment_values_u64(value)?,
            )?),
            Storage::U32(array) => Storage::U32(setitem_array(
                array,
                key,
                bool_mask.as_ref(),
                assignment_values_u32(value)?,
            )?),
            Storage::U16(array) => Storage::U16(setitem_array(
                array,
                key,
                bool_mask.as_ref(),
                assignment_values_u16(value)?,
            )?),
            Storage::U8(array) => Storage::U8(setitem_array(
                array,
                key,
                bool_mask.as_ref(),
                assignment_values_u8(value)?,
            )?),
            Storage::Bool(array) => Storage::Bool(setitem_array(
                array,
                key,
                bool_mask.as_ref(),
                assignment_values_bool(value)?,
            )?),
        };
        Ok(())
    }

    fn __bool__(&self) -> PyResult<bool> {
        self.ensure_scalar("__bool__")?;
        match &self.storage {
            Storage::Bool(array) => Ok(array.as_slice()[0]),
            _ => Err(PyTypeError::new_err(
                "__bool__ is only defined for bool scalar arrays",
            )),
        }
    }

    fn __int__<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        self.scalar_int(py, "__int__")
    }

    fn __index__<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        self.scalar_int(py, "__index__")
    }

    fn __float__(&self) -> PyResult<f64> {
        self.ensure_scalar("__float__")?;
        match &self.storage {
            Storage::F64(array) => Ok(array.as_slice()[0]),
            Storage::F32(array) => Ok(array.as_slice()[0] as f64),
            Storage::I64(array) => Ok(array.as_slice()[0] as f64),
            Storage::I32(array) => Ok(array.as_slice()[0] as f64),
            Storage::I16(array) => Ok(array.as_slice()[0] as f64),
            Storage::I8(array) => Ok(array.as_slice()[0] as f64),
            Storage::U64(array) => Ok(array.as_slice()[0] as f64),
            Storage::U32(array) => Ok(array.as_slice()[0] as f64),
            Storage::U16(array) => Ok(array.as_slice()[0] as f64),
            Storage::U8(array) => Ok(array.as_slice()[0] as f64),
            Storage::Bool(array) => Ok(if array.as_slice()[0] { 1.0 } else { 0.0 }),
            _ => Err(PyTypeError::new_err(
                "__float__ is only defined for real numeric scalar arrays",
            )),
        }
    }

    fn __complex__<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        self.ensure_scalar("__complex__")?;
        let (real, imag) = match &self.storage {
            Storage::F64(array) => (array.as_slice()[0], 0.0),
            Storage::F32(array) => (array.as_slice()[0] as f64, 0.0),
            Storage::C128(array) => {
                let value = array.as_slice()[0];
                (value.re, value.im)
            }
            Storage::C64(array) => {
                let value = array.as_slice()[0];
                (value.re as f64, value.im as f64)
            }
            Storage::I64(array) => (array.as_slice()[0] as f64, 0.0),
            Storage::I32(array) => (array.as_slice()[0] as f64, 0.0),
            Storage::I16(array) => (array.as_slice()[0] as f64, 0.0),
            Storage::I8(array) => (array.as_slice()[0] as f64, 0.0),
            Storage::U64(array) => (array.as_slice()[0] as f64, 0.0),
            Storage::U32(array) => (array.as_slice()[0] as f64, 0.0),
            Storage::U16(array) => (array.as_slice()[0] as f64, 0.0),
            Storage::U8(array) => (array.as_slice()[0] as f64, 0.0),
            Storage::Bool(array) => (if array.as_slice()[0] { 1.0 } else { 0.0 }, 0.0),
        };
        Ok(PyComplex::from_doubles(py, real, imag).unbind().into())
    }

    fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        if self.ndim() != 1 {
            return Err(PyTypeError::new_err(
                "__iter__ currently supports one-dimensional arrays",
            ));
        }
        let list = self.tolist(py)?;
        Ok(list.bind(py).try_iter()?.unbind().into())
    }

    fn __repr__(&self) -> String {
        format!(
            "numrust.Array(shape={:?}, dtype={})",
            self.shape_vec(),
            dtype_name(self.dtype_kind())
        )
    }
}

impl PyArray {
    fn shape_vec(&self) -> Vec<usize> {
        with_array!(&self.storage, array => { array.shape().to_vec() })
    }

    fn dtype_kind(&self) -> DTypeKind {
        with_array!(&self.storage, array => { array.dtype() })
    }

    fn ensure_scalar(&self, method: &str) -> PyResult<()> {
        if self.shape_vec().is_empty() {
            Ok(())
        } else {
            Err(PyTypeError::new_err(format!(
                "{method} is only defined for 0-dimensional arrays"
            )))
        }
    }

    fn scalar_int<'py>(&self, py: Python<'py>, method: &str) -> PyResult<Py<PyAny>> {
        self.ensure_scalar(method)?;
        match &self.storage {
            Storage::I64(array) => Ok(array.as_slice()[0].into_pyobject(py)?.into_any().unbind()),
            Storage::I32(array) => Ok(array.as_slice()[0].into_pyobject(py)?.into_any().unbind()),
            Storage::I16(array) => Ok(array.as_slice()[0].into_pyobject(py)?.into_any().unbind()),
            Storage::I8(array) => Ok(array.as_slice()[0].into_pyobject(py)?.into_any().unbind()),
            Storage::U64(array) => Ok(array.as_slice()[0].into_pyobject(py)?.into_any().unbind()),
            Storage::U32(array) => Ok(array.as_slice()[0].into_pyobject(py)?.into_any().unbind()),
            Storage::U16(array) => Ok(array.as_slice()[0].into_pyobject(py)?.into_any().unbind()),
            Storage::U8(array) => Ok(array.as_slice()[0].into_pyobject(py)?.into_any().unbind()),
            _ => Err(PyTypeError::new_err(format!(
                "{method} is only defined for integer scalar arrays"
            ))),
        }
    }

    fn bool_mask_from_key(&self, key: &Bound<'_, PyAny>) -> PyResult<Option<Array<bool>>> {
        let Ok(mask) = key.extract::<PyRef<'_, PyArray>>() else {
            return Ok(None);
        };
        match &mask.storage {
            Storage::Bool(array) => Ok(Some(array.clone())),
            _ => Ok(None),
        }
    }

    fn coerce_rhs(&self, rhs: &Bound<'_, PyAny>) -> PyResult<PyArray> {
        if let Ok(array) = rhs.extract::<PyRef<'_, PyArray>>() {
            return Ok(array.clone());
        }
        match self.dtype_kind() {
            DTypeKind::F64 => scalar_array(value_as_f64(rhs)?),
            DTypeKind::F32 => scalar_array(value_as_f32(rhs)?),
            DTypeKind::Complex128 => scalar_array(value_as_c128(rhs)?),
            DTypeKind::Complex64 => scalar_array(value_as_c64(rhs)?),
            DTypeKind::I64 => scalar_array(value_as_i64(rhs)?),
            DTypeKind::I32 => scalar_array(value_as_i32(rhs)?),
            DTypeKind::I16 => scalar_array(value_as_i16(rhs)?),
            DTypeKind::I8 => scalar_array(value_as_i8(rhs)?),
            DTypeKind::U64 => scalar_array(value_as_u64(rhs)?),
            DTypeKind::U32 => scalar_array(value_as_u32(rhs)?),
            DTypeKind::U16 => scalar_array(value_as_u16(rhs)?),
            DTypeKind::U8 => scalar_array(value_as_u8(rhs)?),
            DTypeKind::Bool => scalar_array(value_as_bool(rhs)?),
        }
    }
}

macro_rules! impl_from_array {
    ($($variant:ident, $ty:ty);+ $(;)?) => {
        $(
            impl From<Array<$ty>> for PyArray {
                fn from(array: Array<$ty>) -> Self {
                    Self {
                        storage: Storage::$variant(array),
                    }
                }
            }
        )+
    };
}

impl_from_array!(
    F64, f64;
    F32, f32;
    C128, Complex64;
    C64, Complex32;
    I64, i64;
    I32, i32;
    I16, i16;
    I8, i8;
    U64, u64;
    U32, u32;
    U16, u16;
    U8, u8;
    Bool, bool;
);

macro_rules! from_vec_fn {
    ($name:ident, $ty:ty) => {
        #[pyfunction]
        fn $name(data: Vec<$ty>, shape: Vec<usize>) -> PyResult<PyArray> {
            Ok(PyArray::from(
                Array::from_vec(shape, data).map_err(py_value_error)?,
            ))
        }
    };
}

from_vec_fn!(from_f64, f64);
from_vec_fn!(from_f32, f32);
from_vec_fn!(from_i64, i64);
from_vec_fn!(from_i32, i32);
from_vec_fn!(from_i16, i16);
from_vec_fn!(from_i8, i8);
from_vec_fn!(from_u64, u64);
from_vec_fn!(from_u32, u32);
from_vec_fn!(from_u16, u16);
from_vec_fn!(from_u8, u8);
from_vec_fn!(from_bool, bool);

#[pyfunction]
fn from_c128(data: Vec<(f64, f64)>, shape: Vec<usize>) -> PyResult<PyArray> {
    Ok(PyArray::from(
        Array::from_vec(
            shape,
            data.into_iter()
                .map(|(real, imag)| Complex64::new(real, imag))
                .collect(),
        )
        .map_err(py_value_error)?,
    ))
}

#[pyfunction]
fn from_c64(data: Vec<(f64, f64)>, shape: Vec<usize>) -> PyResult<PyArray> {
    Ok(PyArray::from(
        Array::from_vec(
            shape,
            data.into_iter()
                .map(|(real, imag)| Complex32::new(real as f32, imag as f32))
                .collect(),
        )
        .map_err(py_value_error)?,
    ))
}

#[pyfunction]
fn zeros(shape: Vec<usize>, dtype: &str) -> PyResult<PyArray> {
    match dtype {
        "float64" => zeros_array::<f64>(shape),
        "float32" => zeros_array::<f32>(shape),
        "complex128" => zeros_array::<Complex64>(shape),
        "complex64" => zeros_array::<Complex32>(shape),
        "int64" => zeros_array::<i64>(shape),
        "int32" => zeros_array::<i32>(shape),
        "int16" => zeros_array::<i16>(shape),
        "int8" => zeros_array::<i8>(shape),
        "uint64" => zeros_array::<u64>(shape),
        "uint32" => zeros_array::<u32>(shape),
        "uint16" => zeros_array::<u16>(shape),
        "uint8" => zeros_array::<u8>(shape),
        "bool" => zeros_array::<bool>(shape),
        _ => Err(dtype_error(dtype)),
    }
}

#[pyfunction]
fn ones(shape: Vec<usize>, dtype: &str) -> PyResult<PyArray> {
    full(shape, dtype, 1.0)
}

#[pyfunction]
fn full(shape: Vec<usize>, dtype: &str, value: f64) -> PyResult<PyArray> {
    match dtype {
        "float64" => full_array(shape, value),
        "float32" => full_array(shape, value as f32),
        "complex128" => full_array(shape, Complex64::new(value, 0.0)),
        "complex64" => full_array(shape, Complex32::new(value as f32, 0.0)),
        "int64" => full_array(shape, value as i64),
        "int32" => full_array(shape, value as i32),
        "int16" => full_array(shape, value as i16),
        "int8" => full_array(shape, value as i8),
        "uint64" => full_array(shape, value as u64),
        "uint32" => full_array(shape, value as u32),
        "uint16" => full_array(shape, value as u16),
        "uint8" => full_array(shape, value as u8),
        "bool" => full_array(shape, value != 0.0),
        _ => Err(dtype_error(dtype)),
    }
}

#[pyfunction]
fn arange(stop: i64, dtype: &str) -> PyResult<PyArray> {
    let len = stop.max(0) as usize;
    match dtype {
        "float64" => arange_array::<f64, _>(stop, len, |value| value as f64),
        "float32" => arange_array::<f32, _>(stop, len, |value| value as f32),
        "complex128" => {
            arange_array::<Complex64, _>(stop, len, |value| Complex64::new(value as f64, 0.0))
        }
        "complex64" => {
            arange_array::<Complex32, _>(stop, len, |value| Complex32::new(value as f32, 0.0))
        }
        "int64" => arange_array::<i64, _>(stop, len, |value| value),
        "int32" => arange_array::<i32, _>(stop, len, |value| value as i32),
        "int16" => arange_array::<i16, _>(stop, len, |value| value as i16),
        "int8" => arange_array::<i8, _>(stop, len, |value| value as i8),
        "uint64" => arange_array::<u64, _>(stop, len, |value| value as u64),
        "uint32" => arange_array::<u32, _>(stop, len, |value| value as u32),
        "uint16" => arange_array::<u16, _>(stop, len, |value| value as u16),
        "uint8" => arange_array::<u8, _>(stop, len, |value| value as u8),
        _ => Err(dtype_error(dtype)),
    }
}

#[pyfunction]
fn fftfreq(n: usize, d: f64, dtype: &str) -> PyResult<PyArray> {
    let scale = if d == 0.0 {
        f64::INFINITY
    } else {
        1.0 / (n as f64 * d)
    };
    let split = n.div_ceil(2);
    let values = (0..n)
        .map(|idx| {
            if idx < split {
                idx as f64 * scale
            } else {
                -((n - idx) as f64) * scale
            }
        })
        .collect::<Vec<_>>();
    freq_array(values, dtype)
}

#[pyfunction]
fn rfftfreq(n: usize, d: f64, dtype: &str) -> PyResult<PyArray> {
    let scale = if d == 0.0 {
        f64::INFINITY
    } else {
        1.0 / (n as f64 * d)
    };
    let values = (0..(n / 2 + 1))
        .map(|idx| idx as f64 * scale)
        .collect::<Vec<_>>();
    freq_array(values, dtype)
}

#[pymodule]
fn _numrust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyArray>()?;
    m.add_function(wrap_pyfunction!(from_f64, m)?)?;
    m.add_function(wrap_pyfunction!(from_f32, m)?)?;
    m.add_function(wrap_pyfunction!(from_c128, m)?)?;
    m.add_function(wrap_pyfunction!(from_c64, m)?)?;
    m.add_function(wrap_pyfunction!(from_i64, m)?)?;
    m.add_function(wrap_pyfunction!(from_i32, m)?)?;
    m.add_function(wrap_pyfunction!(from_i16, m)?)?;
    m.add_function(wrap_pyfunction!(from_i8, m)?)?;
    m.add_function(wrap_pyfunction!(from_u64, m)?)?;
    m.add_function(wrap_pyfunction!(from_u32, m)?)?;
    m.add_function(wrap_pyfunction!(from_u16, m)?)?;
    m.add_function(wrap_pyfunction!(from_u8, m)?)?;
    m.add_function(wrap_pyfunction!(from_bool, m)?)?;
    m.add_function(wrap_pyfunction!(zeros, m)?)?;
    m.add_function(wrap_pyfunction!(ones, m)?)?;
    m.add_function(wrap_pyfunction!(full, m)?)?;
    m.add_function(wrap_pyfunction!(arange, m)?)?;
    m.add_function(wrap_pyfunction!(fftfreq, m)?)?;
    m.add_function(wrap_pyfunction!(rfftfreq, m)?)?;
    Ok(())
}

fn zeros_array<T>(shape: Vec<usize>) -> PyResult<PyArray>
where
    T: Default + Clone + numrs_core::DType,
    PyArray: From<Array<T>>,
{
    Ok(PyArray::from(
        Array::<T>::zeros(shape).map_err(py_value_error)?,
    ))
}

fn full_array<T>(shape: Vec<usize>, value: T) -> PyResult<PyArray>
where
    T: Clone,
    PyArray: From<Array<T>>,
{
    Ok(PyArray::from(
        Array::full(shape, value).map_err(py_value_error)?,
    ))
}

fn freq_array(values: Vec<f64>, dtype: &str) -> PyResult<PyArray> {
    let shape = vec![values.len()];
    match dtype {
        "float64" => Ok(PyArray::from(
            Array::from_vec(shape, values).map_err(py_value_error)?,
        )),
        "float32" => Ok(PyArray::from(
            Array::from_vec(
                shape,
                values.into_iter().map(|value| value as f32).collect(),
            )
            .map_err(py_value_error)?,
        )),
        _ => Err(PyTypeError::new_err(
            "frequency helpers require float32 or float64 dtype",
        )),
    }
}

fn scalar_array<T>(value: T) -> PyResult<PyArray>
where
    PyArray: From<Array<T>>,
{
    Ok(PyArray::from(Array::scalar(value).map_err(py_value_error)?))
}

fn arange_array<T, F>(stop: i64, len: usize, cast: F) -> PyResult<PyArray>
where
    F: Fn(i64) -> T,
    PyArray: From<Array<T>>,
{
    let data = (0..stop).map(cast).collect::<Vec<_>>();
    Ok(PyArray::from(
        Array::from_vec(vec![len], data).map_err(py_value_error)?,
    ))
}

fn binary_add(left: &PyArray, right: &PyArray) -> PyResult<PyArray> {
    match promoted_numeric_kind(left, right)? {
        DTypeKind::F64 => Ok(PyArray::from(
            to_f64_array(left)?
                .add(&to_f64_array(right)?)
                .map_err(py_value_error)?,
        )),
        DTypeKind::F32 => Ok(PyArray::from(
            to_f32_array(left)?
                .add(&to_f32_array(right)?)
                .map_err(py_value_error)?,
        )),
        DTypeKind::Complex128 => Ok(PyArray::from(
            to_c128_array(left)?
                .add(&to_c128_array(right)?)
                .map_err(py_value_error)?,
        )),
        DTypeKind::Complex64 => Ok(PyArray::from(
            to_c64_array(left)?
                .add(&to_c64_array(right)?)
                .map_err(py_value_error)?,
        )),
        DTypeKind::I64 => Ok(PyArray::from(
            to_i64_array(left)?
                .add(&to_i64_array(right)?)
                .map_err(py_value_error)?,
        )),
        DTypeKind::I32 => Ok(PyArray::from(
            to_i32_array(left)?
                .add(&to_i32_array(right)?)
                .map_err(py_value_error)?,
        )),
        DTypeKind::I16 => Ok(PyArray::from(
            to_i16_array(left)?
                .add(&to_i16_array(right)?)
                .map_err(py_value_error)?,
        )),
        DTypeKind::I8 => Ok(PyArray::from(
            to_i8_array(left)?
                .add(&to_i8_array(right)?)
                .map_err(py_value_error)?,
        )),
        DTypeKind::U64 => Ok(PyArray::from(
            to_u64_array(left)?
                .add(&to_u64_array(right)?)
                .map_err(py_value_error)?,
        )),
        DTypeKind::U32 => Ok(PyArray::from(
            to_u32_array(left)?
                .add(&to_u32_array(right)?)
                .map_err(py_value_error)?,
        )),
        DTypeKind::U16 => Ok(PyArray::from(
            to_u16_array(left)?
                .add(&to_u16_array(right)?)
                .map_err(py_value_error)?,
        )),
        DTypeKind::U8 => Ok(PyArray::from(
            to_u8_array(left)?
                .add(&to_u8_array(right)?)
                .map_err(py_value_error)?,
        )),
        DTypeKind::Bool => Err(PyTypeError::new_err(
            "arithmetic is not defined for bool arrays",
        )),
    }
}

fn binary_sub(left: &PyArray, right: &PyArray) -> PyResult<PyArray> {
    match promoted_numeric_kind(left, right)? {
        DTypeKind::F64 => Ok(PyArray::from(
            to_f64_array(left)?
                .sub(&to_f64_array(right)?)
                .map_err(py_value_error)?,
        )),
        DTypeKind::F32 => Ok(PyArray::from(
            to_f32_array(left)?
                .sub(&to_f32_array(right)?)
                .map_err(py_value_error)?,
        )),
        DTypeKind::Complex128 => Ok(PyArray::from(
            to_c128_array(left)?
                .sub(&to_c128_array(right)?)
                .map_err(py_value_error)?,
        )),
        DTypeKind::Complex64 => Ok(PyArray::from(
            to_c64_array(left)?
                .sub(&to_c64_array(right)?)
                .map_err(py_value_error)?,
        )),
        DTypeKind::I64 => Ok(PyArray::from(
            to_i64_array(left)?
                .sub(&to_i64_array(right)?)
                .map_err(py_value_error)?,
        )),
        DTypeKind::I32 => Ok(PyArray::from(
            to_i32_array(left)?
                .sub(&to_i32_array(right)?)
                .map_err(py_value_error)?,
        )),
        DTypeKind::I16 => Ok(PyArray::from(
            to_i16_array(left)?
                .sub(&to_i16_array(right)?)
                .map_err(py_value_error)?,
        )),
        DTypeKind::I8 => Ok(PyArray::from(
            to_i8_array(left)?
                .sub(&to_i8_array(right)?)
                .map_err(py_value_error)?,
        )),
        DTypeKind::U64 => Ok(PyArray::from(
            to_u64_array(left)?
                .sub(&to_u64_array(right)?)
                .map_err(py_value_error)?,
        )),
        DTypeKind::U32 => Ok(PyArray::from(
            to_u32_array(left)?
                .sub(&to_u32_array(right)?)
                .map_err(py_value_error)?,
        )),
        DTypeKind::U16 => Ok(PyArray::from(
            to_u16_array(left)?
                .sub(&to_u16_array(right)?)
                .map_err(py_value_error)?,
        )),
        DTypeKind::U8 => Ok(PyArray::from(
            to_u8_array(left)?
                .sub(&to_u8_array(right)?)
                .map_err(py_value_error)?,
        )),
        DTypeKind::Bool => Err(PyTypeError::new_err(
            "arithmetic is not defined for bool arrays",
        )),
    }
}

fn binary_mul(left: &PyArray, right: &PyArray) -> PyResult<PyArray> {
    match promoted_numeric_kind(left, right)? {
        DTypeKind::F64 => Ok(PyArray::from(
            to_f64_array(left)?
                .mul(&to_f64_array(right)?)
                .map_err(py_value_error)?,
        )),
        DTypeKind::F32 => Ok(PyArray::from(
            to_f32_array(left)?
                .mul(&to_f32_array(right)?)
                .map_err(py_value_error)?,
        )),
        DTypeKind::Complex128 => Ok(PyArray::from(
            to_c128_array(left)?
                .mul(&to_c128_array(right)?)
                .map_err(py_value_error)?,
        )),
        DTypeKind::Complex64 => Ok(PyArray::from(
            to_c64_array(left)?
                .mul(&to_c64_array(right)?)
                .map_err(py_value_error)?,
        )),
        DTypeKind::I64 => Ok(PyArray::from(
            to_i64_array(left)?
                .mul(&to_i64_array(right)?)
                .map_err(py_value_error)?,
        )),
        DTypeKind::I32 => Ok(PyArray::from(
            to_i32_array(left)?
                .mul(&to_i32_array(right)?)
                .map_err(py_value_error)?,
        )),
        DTypeKind::I16 => Ok(PyArray::from(
            to_i16_array(left)?
                .mul(&to_i16_array(right)?)
                .map_err(py_value_error)?,
        )),
        DTypeKind::I8 => Ok(PyArray::from(
            to_i8_array(left)?
                .mul(&to_i8_array(right)?)
                .map_err(py_value_error)?,
        )),
        DTypeKind::U64 => Ok(PyArray::from(
            to_u64_array(left)?
                .mul(&to_u64_array(right)?)
                .map_err(py_value_error)?,
        )),
        DTypeKind::U32 => Ok(PyArray::from(
            to_u32_array(left)?
                .mul(&to_u32_array(right)?)
                .map_err(py_value_error)?,
        )),
        DTypeKind::U16 => Ok(PyArray::from(
            to_u16_array(left)?
                .mul(&to_u16_array(right)?)
                .map_err(py_value_error)?,
        )),
        DTypeKind::U8 => Ok(PyArray::from(
            to_u8_array(left)?
                .mul(&to_u8_array(right)?)
                .map_err(py_value_error)?,
        )),
        DTypeKind::Bool => Err(PyTypeError::new_err(
            "arithmetic is not defined for bool arrays",
        )),
    }
}

fn binary_divide(left: &PyArray, right: &PyArray) -> PyResult<PyArray> {
    match promoted_numeric_kind(left, right)? {
        DTypeKind::F64 => Ok(PyArray::from(
            to_f64_array(left)?
                .div(&to_f64_array(right)?)
                .map_err(py_value_error)?,
        )),
        DTypeKind::F32 => Ok(PyArray::from(
            to_f32_array(left)?
                .div(&to_f32_array(right)?)
                .map_err(py_value_error)?,
        )),
        DTypeKind::Complex128 => Ok(PyArray::from(
            to_c128_array(left)?
                .div(&to_c128_array(right)?)
                .map_err(py_value_error)?,
        )),
        DTypeKind::Complex64 => Ok(PyArray::from({
            let left_array = to_c64_array(left)?;
            let right_array = to_c64_array(right)?;
            binary_slices(
                left_array.shape(),
                left_array.as_slice(),
                right_array.shape(),
                right_array.as_slice(),
                |left, right| {
                    let quotient = Complex64::new(left.re as f64, left.im as f64)
                        / Complex64::new(right.re as f64, right.im as f64);
                    Complex32::new(quotient.re as f32, quotient.im as f32)
                },
            )?
        })),
        DTypeKind::I64
        | DTypeKind::I32
        | DTypeKind::I16
        | DTypeKind::I8
        | DTypeKind::U64
        | DTypeKind::U32
        | DTypeKind::U16
        | DTypeKind::U8
        | DTypeKind::Bool => Err(PyTypeError::new_err(
            "divide is only defined for floating-point arrays",
        )),
    }
}

macro_rules! floor_div_float_array {
    ($left:expr, $right:expr, $to_array:ident) => {{
        let left_array = $to_array($left)?;
        let right_array = $to_array($right)?;
        Ok(PyArray::from(binary_slices(
            left_array.shape(),
            left_array.as_slice(),
            right_array.shape(),
            right_array.as_slice(),
            |left, right| (left / right).floor(),
        )?))
    }};
}

macro_rules! floor_div_signed_array {
    ($left:expr, $right:expr, $to_array:ident, $ty:ty) => {{
        let left_array = $to_array($left)?;
        let right_array = $to_array($right)?;
        Ok(PyArray::from(binary_slices(
            left_array.shape(),
            left_array.as_slice(),
            right_array.shape(),
            right_array.as_slice(),
            |left: $ty, right: $ty| {
                if right == 0 {
                    0
                } else {
                    let quotient = left.wrapping_div(right);
                    let remainder = left.wrapping_rem(right);
                    if remainder != 0 && ((remainder > 0) != (right > 0)) {
                        quotient.wrapping_sub(1)
                    } else {
                        quotient
                    }
                }
            },
        )?))
    }};
}

macro_rules! floor_div_unsigned_array {
    ($left:expr, $right:expr, $to_array:ident, $ty:ty) => {{
        let left_array = $to_array($left)?;
        let right_array = $to_array($right)?;
        Ok(PyArray::from(binary_slices(
            left_array.shape(),
            left_array.as_slice(),
            right_array.shape(),
            right_array.as_slice(),
            |left: $ty, right: $ty| if right == 0 { 0 } else { left / right },
        )?))
    }};
}

fn binary_floor_divide(left: &PyArray, right: &PyArray) -> PyResult<PyArray> {
    match promoted_numeric_kind(left, right)? {
        DTypeKind::F64 => floor_div_float_array!(left, right, to_f64_array),
        DTypeKind::F32 => floor_div_float_array!(left, right, to_f32_array),
        DTypeKind::I64 => floor_div_signed_array!(left, right, to_i64_array, i64),
        DTypeKind::I32 => floor_div_signed_array!(left, right, to_i32_array, i32),
        DTypeKind::I16 => floor_div_signed_array!(left, right, to_i16_array, i16),
        DTypeKind::I8 => floor_div_signed_array!(left, right, to_i8_array, i8),
        DTypeKind::U64 => floor_div_unsigned_array!(left, right, to_u64_array, u64),
        DTypeKind::U32 => floor_div_unsigned_array!(left, right, to_u32_array, u32),
        DTypeKind::U16 => floor_div_unsigned_array!(left, right, to_u16_array, u16),
        DTypeKind::U8 => floor_div_unsigned_array!(left, right, to_u8_array, u8),
        DTypeKind::Complex128 | DTypeKind::Complex64 | DTypeKind::Bool => Err(
            PyTypeError::new_err("floor_divide is only defined for real numeric arrays"),
        ),
    }
}

macro_rules! power_float_array {
    ($left:expr, $right:expr, $to_array:ident) => {{
        let left_array = $to_array($left)?;
        let right_array = $to_array($right)?;
        Ok(PyArray::from(binary_slices(
            left_array.shape(),
            left_array.as_slice(),
            right_array.shape(),
            right_array.as_slice(),
            |left, right| left.powf(right),
        )?))
    }};
}

macro_rules! power_signed_array {
    ($left:expr, $right:expr, $to_array:ident, $ty:ty) => {{
        let left_array = $to_array($left)?;
        let right_array = $to_array($right)?;
        Ok(PyArray::from(binary_slices(
            left_array.shape(),
            left_array.as_slice(),
            right_array.shape(),
            right_array.as_slice(),
            |left: $ty, right: $ty| {
                let exponent = if right < 0 { 0 } else { right as u32 };
                left.wrapping_pow(exponent)
            },
        )?))
    }};
}

macro_rules! power_unsigned_array {
    ($left:expr, $right:expr, $to_array:ident, $ty:ty) => {{
        let left_array = $to_array($left)?;
        let right_array = $to_array($right)?;
        Ok(PyArray::from(binary_slices(
            left_array.shape(),
            left_array.as_slice(),
            right_array.shape(),
            right_array.as_slice(),
            |left: $ty, right: $ty| left.wrapping_pow(right.min(u32::MAX as $ty) as u32),
        )?))
    }};
}

fn binary_power(left: &PyArray, right: &PyArray) -> PyResult<PyArray> {
    match promoted_numeric_kind(left, right)? {
        DTypeKind::F64 => power_float_array!(left, right, to_f64_array),
        DTypeKind::F32 => power_float_array!(left, right, to_f32_array),
        DTypeKind::Complex128 => {
            let left_array = to_c128_array(left)?;
            let right_array = to_c128_array(right)?;
            Ok(PyArray::from(binary_slices(
                left_array.shape(),
                left_array.as_slice(),
                right_array.shape(),
                right_array.as_slice(),
                |left, right| left.powc(right),
            )?))
        }
        DTypeKind::Complex64 => {
            let left_array = to_c64_array(left)?;
            let right_array = to_c64_array(right)?;
            Ok(PyArray::from(binary_slices(
                left_array.shape(),
                left_array.as_slice(),
                right_array.shape(),
                right_array.as_slice(),
                |left, right| left.powc(right),
            )?))
        }
        DTypeKind::I64 => power_signed_array!(left, right, to_i64_array, i64),
        DTypeKind::I32 => power_signed_array!(left, right, to_i32_array, i32),
        DTypeKind::I16 => power_signed_array!(left, right, to_i16_array, i16),
        DTypeKind::I8 => power_signed_array!(left, right, to_i8_array, i8),
        DTypeKind::U64 => power_unsigned_array!(left, right, to_u64_array, u64),
        DTypeKind::U32 => power_unsigned_array!(left, right, to_u32_array, u32),
        DTypeKind::U16 => power_unsigned_array!(left, right, to_u16_array, u16),
        DTypeKind::U8 => power_unsigned_array!(left, right, to_u8_array, u8),
        DTypeKind::Bool => Err(PyTypeError::new_err(
            "pow is only defined for numeric arrays",
        )),
    }
}

macro_rules! bitwise_binary_array {
    ($left:expr, $right:expr, $to_array:ident, $op:tt) => {{
        let left_array = $to_array($left)?;
        let right_array = $to_array($right)?;
        Ok(PyArray::from(binary_slices(
            left_array.shape(),
            left_array.as_slice(),
            right_array.shape(),
            right_array.as_slice(),
            |left, right| left $op right,
        )?))
    }};
}

macro_rules! shift_left_signed_array {
    ($left:expr, $right:expr, $to_array:ident, $ty:ty, $bits:expr) => {{
        let left_array = $to_array($left)?;
        let right_array = $to_array($right)?;
        Ok(PyArray::from(binary_slices(
            left_array.shape(),
            left_array.as_slice(),
            right_array.shape(),
            right_array.as_slice(),
            |left: $ty, right: $ty| {
                if right < 0 || right as u32 >= $bits {
                    0 as $ty
                } else {
                    left.wrapping_shl(right as u32)
                }
            },
        )?))
    }};
}

macro_rules! shift_left_unsigned_array {
    ($left:expr, $right:expr, $to_array:ident, $ty:ty, $bits:expr) => {{
        let left_array = $to_array($left)?;
        let right_array = $to_array($right)?;
        Ok(PyArray::from(binary_slices(
            left_array.shape(),
            left_array.as_slice(),
            right_array.shape(),
            right_array.as_slice(),
            |left: $ty, right: $ty| {
                if right as u128 >= $bits {
                    0 as $ty
                } else {
                    left.wrapping_shl(right as u32)
                }
            },
        )?))
    }};
}

macro_rules! shift_right_signed_array {
    ($left:expr, $right:expr, $to_array:ident, $ty:ty, $bits:expr) => {{
        let left_array = $to_array($left)?;
        let right_array = $to_array($right)?;
        Ok(PyArray::from(binary_slices(
            left_array.shape(),
            left_array.as_slice(),
            right_array.shape(),
            right_array.as_slice(),
            |left: $ty, right: $ty| {
                if right < 0 {
                    0 as $ty
                } else if right as u32 >= $bits {
                    if left < 0 {
                        -1 as $ty
                    } else {
                        0 as $ty
                    }
                } else {
                    left >> (right as u32)
                }
            },
        )?))
    }};
}

macro_rules! shift_right_unsigned_array {
    ($left:expr, $right:expr, $to_array:ident, $ty:ty, $bits:expr) => {{
        let left_array = $to_array($left)?;
        let right_array = $to_array($right)?;
        Ok(PyArray::from(binary_slices(
            left_array.shape(),
            left_array.as_slice(),
            right_array.shape(),
            right_array.as_slice(),
            |left: $ty, right: $ty| {
                if right as u128 >= $bits {
                    0 as $ty
                } else {
                    left >> (right as u32)
                }
            },
        )?))
    }};
}

fn unary_bitwise_invert(array: &PyArray) -> PyResult<PyArray> {
    match &array.storage {
        Storage::Bool(array) => Ok(PyArray::from(cast_array(array, |value| !value)?)),
        Storage::I64(array) => Ok(PyArray::from(cast_array(array, |value| !value)?)),
        Storage::I32(array) => Ok(PyArray::from(cast_array(array, |value| !value)?)),
        Storage::I16(array) => Ok(PyArray::from(cast_array(array, |value| !value)?)),
        Storage::I8(array) => Ok(PyArray::from(cast_array(array, |value| !value)?)),
        Storage::U64(array) => Ok(PyArray::from(cast_array(array, |value| !value)?)),
        Storage::U32(array) => Ok(PyArray::from(cast_array(array, |value| !value)?)),
        Storage::U16(array) => Ok(PyArray::from(cast_array(array, |value| !value)?)),
        Storage::U8(array) => Ok(PyArray::from(cast_array(array, |value| !value)?)),
        Storage::F64(_) | Storage::F32(_) | Storage::C128(_) | Storage::C64(_) => Err(
            PyTypeError::new_err("bitwise operations are only defined for bool and integer arrays"),
        ),
    }
}

fn binary_bitwise_and(left: &PyArray, right: &PyArray) -> PyResult<PyArray> {
    match promote_dtype(left.dtype_kind(), right.dtype_kind()) {
        DTypeKind::Bool => bitwise_binary_array!(left, right, to_bool_array, &),
        DTypeKind::I64 => bitwise_binary_array!(left, right, to_i64_array, &),
        DTypeKind::I32 => bitwise_binary_array!(left, right, to_i32_array, &),
        DTypeKind::I16 => bitwise_binary_array!(left, right, to_i16_array, &),
        DTypeKind::I8 => bitwise_binary_array!(left, right, to_i8_array, &),
        DTypeKind::U64 => bitwise_binary_array!(left, right, to_u64_array, &),
        DTypeKind::U32 => bitwise_binary_array!(left, right, to_u32_array, &),
        DTypeKind::U16 => bitwise_binary_array!(left, right, to_u16_array, &),
        DTypeKind::U8 => bitwise_binary_array!(left, right, to_u8_array, &),
        DTypeKind::F64 | DTypeKind::F32 | DTypeKind::Complex128 | DTypeKind::Complex64 => Err(
            PyTypeError::new_err("bitwise operations are only defined for bool and integer arrays"),
        ),
    }
}

fn binary_bitwise_or(left: &PyArray, right: &PyArray) -> PyResult<PyArray> {
    match promote_dtype(left.dtype_kind(), right.dtype_kind()) {
        DTypeKind::Bool => bitwise_binary_array!(left, right, to_bool_array, |),
        DTypeKind::I64 => bitwise_binary_array!(left, right, to_i64_array, |),
        DTypeKind::I32 => bitwise_binary_array!(left, right, to_i32_array, |),
        DTypeKind::I16 => bitwise_binary_array!(left, right, to_i16_array, |),
        DTypeKind::I8 => bitwise_binary_array!(left, right, to_i8_array, |),
        DTypeKind::U64 => bitwise_binary_array!(left, right, to_u64_array, |),
        DTypeKind::U32 => bitwise_binary_array!(left, right, to_u32_array, |),
        DTypeKind::U16 => bitwise_binary_array!(left, right, to_u16_array, |),
        DTypeKind::U8 => bitwise_binary_array!(left, right, to_u8_array, |),
        DTypeKind::F64 | DTypeKind::F32 | DTypeKind::Complex128 | DTypeKind::Complex64 => Err(
            PyTypeError::new_err("bitwise operations are only defined for bool and integer arrays"),
        ),
    }
}

fn binary_bitwise_xor(left: &PyArray, right: &PyArray) -> PyResult<PyArray> {
    match promote_dtype(left.dtype_kind(), right.dtype_kind()) {
        DTypeKind::Bool => bitwise_binary_array!(left, right, to_bool_array, ^),
        DTypeKind::I64 => bitwise_binary_array!(left, right, to_i64_array, ^),
        DTypeKind::I32 => bitwise_binary_array!(left, right, to_i32_array, ^),
        DTypeKind::I16 => bitwise_binary_array!(left, right, to_i16_array, ^),
        DTypeKind::I8 => bitwise_binary_array!(left, right, to_i8_array, ^),
        DTypeKind::U64 => bitwise_binary_array!(left, right, to_u64_array, ^),
        DTypeKind::U32 => bitwise_binary_array!(left, right, to_u32_array, ^),
        DTypeKind::U16 => bitwise_binary_array!(left, right, to_u16_array, ^),
        DTypeKind::U8 => bitwise_binary_array!(left, right, to_u8_array, ^),
        DTypeKind::F64 | DTypeKind::F32 | DTypeKind::Complex128 | DTypeKind::Complex64 => Err(
            PyTypeError::new_err("bitwise operations are only defined for bool and integer arrays"),
        ),
    }
}

fn binary_bitwise_left_shift(left: &PyArray, right: &PyArray) -> PyResult<PyArray> {
    if left.dtype_kind() == DTypeKind::Bool || right.dtype_kind() == DTypeKind::Bool {
        return Err(PyTypeError::new_err(
            "shift operations are only defined for integer arrays",
        ));
    }
    match promote_dtype(left.dtype_kind(), right.dtype_kind()) {
        DTypeKind::I64 => shift_left_signed_array!(left, right, to_i64_array, i64, 64),
        DTypeKind::I32 => shift_left_signed_array!(left, right, to_i32_array, i32, 32),
        DTypeKind::I16 => shift_left_signed_array!(left, right, to_i16_array, i16, 16),
        DTypeKind::I8 => shift_left_signed_array!(left, right, to_i8_array, i8, 8),
        DTypeKind::U64 => shift_left_unsigned_array!(left, right, to_u64_array, u64, 64),
        DTypeKind::U32 => shift_left_unsigned_array!(left, right, to_u32_array, u32, 32),
        DTypeKind::U16 => shift_left_unsigned_array!(left, right, to_u16_array, u16, 16),
        DTypeKind::U8 => shift_left_unsigned_array!(left, right, to_u8_array, u8, 8),
        DTypeKind::Bool
        | DTypeKind::F64
        | DTypeKind::F32
        | DTypeKind::Complex128
        | DTypeKind::Complex64 => Err(PyTypeError::new_err(
            "shift operations are only defined for integer arrays",
        )),
    }
}

fn binary_bitwise_right_shift(left: &PyArray, right: &PyArray) -> PyResult<PyArray> {
    if left.dtype_kind() == DTypeKind::Bool || right.dtype_kind() == DTypeKind::Bool {
        return Err(PyTypeError::new_err(
            "shift operations are only defined for integer arrays",
        ));
    }
    match promote_dtype(left.dtype_kind(), right.dtype_kind()) {
        DTypeKind::I64 => shift_right_signed_array!(left, right, to_i64_array, i64, 64),
        DTypeKind::I32 => shift_right_signed_array!(left, right, to_i32_array, i32, 32),
        DTypeKind::I16 => shift_right_signed_array!(left, right, to_i16_array, i16, 16),
        DTypeKind::I8 => shift_right_signed_array!(left, right, to_i8_array, i8, 8),
        DTypeKind::U64 => shift_right_unsigned_array!(left, right, to_u64_array, u64, 64),
        DTypeKind::U32 => shift_right_unsigned_array!(left, right, to_u32_array, u32, 32),
        DTypeKind::U16 => shift_right_unsigned_array!(left, right, to_u16_array, u16, 16),
        DTypeKind::U8 => shift_right_unsigned_array!(left, right, to_u8_array, u8, 8),
        DTypeKind::Bool
        | DTypeKind::F64
        | DTypeKind::F32
        | DTypeKind::Complex128
        | DTypeKind::Complex64 => Err(PyTypeError::new_err(
            "shift operations are only defined for integer arrays",
        )),
    }
}

fn promoted_numeric_kind(left: &PyArray, right: &PyArray) -> PyResult<DTypeKind> {
    let promoted = promote_dtype(left.dtype_kind(), right.dtype_kind());
    if promoted == DTypeKind::Bool {
        return Err(PyTypeError::new_err(
            "arithmetic is not defined for bool arrays",
        ));
    }
    Ok(promoted)
}

fn compare_equal(left: &PyArray, right: &PyArray) -> PyResult<PyArray> {
    match (&left.storage, &right.storage) {
        (Storage::Bool(left), Storage::Bool(right)) => Ok(PyArray::from(compare_slices(
            left.shape(),
            left.as_slice(),
            right.shape(),
            right.as_slice(),
            |l, r| l == r,
        )?)),
        _ if is_complex_kind(left.dtype_kind()) || is_complex_kind(right.dtype_kind()) => {
            let left = to_c128_array(left)?;
            let right = to_c128_array(right)?;
            Ok(PyArray::from(compare_slices(
                left.shape(),
                left.as_slice(),
                right.shape(),
                right.as_slice(),
                |l, r| l == r,
            )?))
        }
        _ if left.dtype_kind().is_float() || right.dtype_kind().is_float() => {
            let left = to_f64_array(left)?;
            let right = to_f64_array(right)?;
            Ok(PyArray::from(compare_slices(
                left.shape(),
                left.as_slice(),
                right.shape(),
                right.as_slice(),
                |l, r| l == r,
            )?))
        }
        _ => {
            let left = to_i64_array(left)?;
            let right = to_i64_array(right)?;
            Ok(PyArray::from(compare_slices(
                left.shape(),
                left.as_slice(),
                right.shape(),
                right.as_slice(),
                |l, r| l == r,
            )?))
        }
    }
}

fn compare_ordered<FF64, FI64>(
    left: &PyArray,
    right: &PyArray,
    f64_op: FF64,
    i64_op: FI64,
) -> PyResult<PyArray>
where
    FF64: Fn(&f64, &f64) -> bool,
    FI64: Fn(&i64, &i64) -> bool,
{
    if is_complex_kind(left.dtype_kind()) || is_complex_kind(right.dtype_kind()) {
        return Err(PyTypeError::new_err(
            "ordered comparisons are not defined for complex arrays",
        ));
    }
    if left.dtype_kind().is_float() || right.dtype_kind().is_float() {
        let left = to_f64_array(left)?;
        let right = to_f64_array(right)?;
        return Ok(PyArray::from(compare_slices(
            left.shape(),
            left.as_slice(),
            right.shape(),
            right.as_slice(),
            f64_op,
        )?));
    }
    let left = to_i64_array(left)?;
    let right = to_i64_array(right)?;
    Ok(PyArray::from(compare_slices(
        left.shape(),
        left.as_slice(),
        right.shape(),
        right.as_slice(),
        i64_op,
    )?))
}

fn compare_slices<T, F>(
    left_shape: &[usize],
    left: &[T],
    right_shape: &[usize],
    right: &[T],
    op: F,
) -> PyResult<Array<bool>>
where
    F: Fn(&T, &T) -> bool,
{
    let out_shape = broadcast_shape_for_compare(left_shape, right_shape)?;
    let out_len = shape_size(&out_shape);
    let mut out = Vec::with_capacity(out_len);
    for linear in 0..out_len {
        let out_index = unravel_index(linear, &out_shape);
        let left_offset = broadcast_offset(left_shape, &out_index)?;
        let right_offset = broadcast_offset(right_shape, &out_index)?;
        out.push(op(&left[left_offset], &right[right_offset]));
    }
    Array::from_vec(out_shape, out).map_err(py_value_error)
}

fn binary_slices<T, F>(
    left_shape: &[usize],
    left: &[T],
    right_shape: &[usize],
    right: &[T],
    op: F,
) -> PyResult<Array<T>>
where
    T: Copy,
    F: Fn(T, T) -> T,
{
    let out_shape = broadcast_shape_for_compare(left_shape, right_shape)?;
    let out_len = shape_size(&out_shape);
    let mut out = Vec::with_capacity(out_len);
    for linear in 0..out_len {
        let out_index = unravel_index(linear, &out_shape);
        let left_offset = broadcast_offset(left_shape, &out_index)?;
        let right_offset = broadcast_offset(right_shape, &out_index)?;
        out.push(op(left[left_offset], right[right_offset]));
    }
    Array::from_vec(out_shape, out).map_err(py_value_error)
}

fn getitem_array<T>(
    array: &Array<T>,
    key: &Bound<'_, PyAny>,
    bool_mask: Option<Array<bool>>,
) -> PyResult<Array<T>>
where
    T: Clone,
{
    let plan = if let Some(mask) = bool_mask {
        bool_mask_index_plan(array.shape(), &mask)?
    } else {
        index_plan(array.shape(), key)?
    };
    Array::from_vec(
        plan.out_shape,
        plan.source_offsets
            .into_iter()
            .map(|offset| array.as_slice()[offset].clone())
            .collect(),
    )
    .map_err(py_value_error)
}

fn setitem_array<T>(
    array: &Array<T>,
    key: &Bound<'_, PyAny>,
    bool_mask: Option<&Array<bool>>,
    values: Vec<T>,
) -> PyResult<Array<T>>
where
    T: Clone,
{
    let plan = if let Some(mask) = bool_mask {
        bool_mask_index_plan(array.shape(), mask)?
    } else {
        index_plan(array.shape(), key)?
    };
    if !plan.source_offsets.is_empty()
        && values.len() != 1
        && values.len() != plan.source_offsets.len()
    {
        return Err(PyValueError::new_err(format!(
            "assignment value has {} elements, expected 1 or {}",
            values.len(),
            plan.source_offsets.len()
        )));
    }
    let mut data = array.as_slice().to_vec();
    for (value_index, offset) in plan.source_offsets.into_iter().enumerate() {
        data[offset] = values[value_index % values.len()].clone();
    }
    Array::from_vec(array.shape().to_vec(), data).map_err(py_value_error)
}

fn index_plan(shape: &[usize], key: &Bound<'_, PyAny>) -> PyResult<IndexPlan> {
    let raw = raw_index_items(key)?;
    let items = normalize_index_items(shape, raw)?;
    let out_shape = output_shape(&items);
    let out_len = shape_size(&out_shape);
    let mut source_offsets = Vec::with_capacity(out_len);

    for linear in 0..out_len {
        let out_index = unravel_index(linear, &out_shape);
        let mut out_axis = 0usize;
        let mut source_index = Vec::with_capacity(shape.len());
        for item in &items {
            match item {
                NormalizedIndexItem::Index(index) => source_index.push(*index),
                NormalizedIndexItem::Range(indices) => {
                    source_index.push(indices[out_index[out_axis]]);
                    out_axis += 1;
                }
                NormalizedIndexItem::NewAxis => {
                    out_axis += 1;
                }
            }
        }
        source_offsets.push(row_major_offset(shape, &source_index)?);
    }

    Ok(IndexPlan {
        out_shape,
        source_offsets,
    })
}

fn bool_mask_index_plan(shape: &[usize], mask: &Array<bool>) -> PyResult<IndexPlan> {
    if mask.shape().is_empty() {
        let mut out_shape = Vec::with_capacity(shape.len() + 1);
        out_shape.push(if mask.as_slice()[0] { 1 } else { 0 });
        out_shape.extend_from_slice(shape);
        let source_offsets = if mask.as_slice()[0] {
            (0..shape_size(shape)).collect()
        } else {
            Vec::new()
        };
        return Ok(IndexPlan {
            out_shape,
            source_offsets,
        });
    }

    if mask.ndim() > shape.len()
        || !mask
            .shape()
            .iter()
            .zip(shape.iter())
            .all(|(mask_dim, array_dim)| *mask_dim == *array_dim || *mask_dim == 0)
    {
        return Err(PyIndexError::new_err(format!(
            "boolean mask shape {:?} is not valid for array shape {:?}",
            mask.shape(),
            shape
        )));
    }

    let tail_shape = &shape[mask.ndim()..];
    let tail_len = shape_size(tail_shape);
    let mut source_offsets = Vec::new();
    let mut selected = 0usize;
    if !mask.shape().contains(&0) {
        for (mask_linear, keep) in mask.as_slice().iter().copied().enumerate() {
            if keep {
                selected += 1;
                let base = mask_linear * tail_len;
                source_offsets.extend(base..base + tail_len);
            }
        }
    }
    let mut out_shape = Vec::with_capacity(tail_shape.len() + 1);
    out_shape.push(selected);
    out_shape.extend_from_slice(tail_shape);
    Ok(IndexPlan {
        out_shape,
        source_offsets,
    })
}

fn raw_index_items(key: &Bound<'_, PyAny>) -> PyResult<Vec<RawIndexItem>> {
    if let Ok(tuple) = key.cast::<PyTuple>() {
        return tuple.iter().map(|item| raw_index_item(&item)).collect();
    }
    Ok(vec![raw_index_item(key)?])
}

fn raw_index_item(item: &Bound<'_, PyAny>) -> PyResult<RawIndexItem> {
    if item.is_none() {
        return Ok(RawIndexItem::NewAxis);
    }
    if item.is(item.py().Ellipsis()) {
        return Ok(RawIndexItem::Ellipsis);
    }
    if let Ok(slice) = item.cast::<PySlice>() {
        return Ok(RawIndexItem::Slice {
            start: slice.getattr("start")?.extract()?,
            stop: slice.getattr("stop")?.extract()?,
            step: slice.getattr("step")?.extract()?,
        });
    }
    Ok(RawIndexItem::Index(item.extract()?))
}

fn normalize_index_items(
    shape: &[usize],
    raw: Vec<RawIndexItem>,
) -> PyResult<Vec<NormalizedIndexItem>> {
    let mut out = Vec::new();
    let mut axis = 0usize;
    let mut seen_ellipsis = false;
    let consumed = raw
        .iter()
        .filter(|item| !matches!(item, RawIndexItem::NewAxis | RawIndexItem::Ellipsis))
        .count();
    if consumed > shape.len() {
        return Err(PyIndexError::new_err(format!(
            "too many indices for array: array is {}-dimensional, but {consumed} were indexed",
            shape.len()
        )));
    }

    for item in raw {
        match item {
            RawIndexItem::NewAxis => out.push(NormalizedIndexItem::NewAxis),
            RawIndexItem::Ellipsis => {
                if seen_ellipsis {
                    return Err(PyIndexError::new_err("an index can only have one ellipsis"));
                }
                seen_ellipsis = true;
                let fill = shape.len() - consumed;
                for _ in 0..fill {
                    out.push(NormalizedIndexItem::Range((0..shape[axis]).collect()));
                    axis += 1;
                }
            }
            RawIndexItem::Index(index) => {
                out.push(NormalizedIndexItem::Index(normalize_int_index(
                    index,
                    axis,
                    shape[axis],
                )?));
                axis += 1;
            }
            RawIndexItem::Slice { start, stop, step } => {
                out.push(NormalizedIndexItem::Range(slice_indices(
                    start,
                    stop,
                    step,
                    shape[axis],
                )?));
                axis += 1;
            }
        }
    }
    while axis < shape.len() {
        out.push(NormalizedIndexItem::Range((0..shape[axis]).collect()));
        axis += 1;
    }
    Ok(out)
}

fn normalize_int_index(index: isize, axis: usize, len: usize) -> PyResult<usize> {
    let normalized = if index < 0 {
        len as isize + index
    } else {
        index
    };
    if normalized < 0 || normalized >= len as isize {
        return Err(PyIndexError::new_err(format!(
            "index {index} is out of bounds for axis {axis} with size {len}"
        )));
    }
    Ok(normalized as usize)
}

fn slice_indices(
    start: Option<isize>,
    stop: Option<isize>,
    step: Option<isize>,
    len: usize,
) -> PyResult<Vec<usize>> {
    let step = step.unwrap_or(1);
    if step == 0 {
        return Err(PyValueError::new_err("slice step cannot be zero"));
    }
    let len = len as isize;
    let mut out = Vec::new();
    if step > 0 {
        let start = normalize_positive_bound(start.unwrap_or(0), len);
        let stop = normalize_positive_bound(stop.unwrap_or(len), len);
        let mut current = start;
        while current < stop {
            out.push(current as usize);
            current += step;
        }
    } else {
        let start = normalize_negative_start(start, len);
        let stop = normalize_negative_stop(stop, len);
        let mut current = start;
        while current > stop {
            out.push(current as usize);
            current += step;
        }
    }
    Ok(out)
}

fn normalize_positive_bound(value: isize, len: isize) -> isize {
    let adjusted = if value < 0 { value + len } else { value };
    adjusted.clamp(0, len)
}

fn normalize_negative_start(value: Option<isize>, len: isize) -> isize {
    match value {
        None => len - 1,
        Some(raw) => {
            let adjusted = if raw < 0 { raw + len } else { raw };
            adjusted.clamp(-1, len - 1)
        }
    }
}

fn normalize_negative_stop(value: Option<isize>, len: isize) -> isize {
    match value {
        None => -1,
        Some(raw) => {
            let adjusted = if raw < 0 { raw + len } else { raw };
            adjusted.clamp(-1, len - 1)
        }
    }
}

fn output_shape(items: &[NormalizedIndexItem]) -> Vec<usize> {
    items
        .iter()
        .filter_map(|item| match item {
            NormalizedIndexItem::Index(_) => None,
            NormalizedIndexItem::Range(indices) => Some(indices.len()),
            NormalizedIndexItem::NewAxis => Some(1),
        })
        .collect()
}

fn shape_size(shape: &[usize]) -> usize {
    shape.iter().product()
}

fn unravel_index(mut linear: usize, shape: &[usize]) -> Vec<usize> {
    let mut index = vec![0; shape.len()];
    for axis in (0..shape.len()).rev() {
        let dim = shape[axis];
        index[axis] = linear.checked_rem(dim).unwrap_or(0);
        linear = linear.checked_div(dim).unwrap_or(0);
    }
    index
}

fn row_major_offset(shape: &[usize], index: &[usize]) -> PyResult<usize> {
    if shape.len() != index.len() {
        return Err(PyIndexError::new_err(format!(
            "expected {} indices, got {}",
            shape.len(),
            index.len()
        )));
    }
    let mut offset = 0usize;
    for (axis, (&dim, &coord)) in shape.iter().zip(index.iter()).enumerate() {
        if coord >= dim {
            return Err(PyIndexError::new_err(format!(
                "index {coord} is out of bounds for axis {axis} with size {dim}"
            )));
        }
        offset = offset * dim + coord;
    }
    Ok(offset)
}

fn broadcast_shape_for_compare(left: &[usize], right: &[usize]) -> PyResult<Vec<usize>> {
    let ndim = left.len().max(right.len());
    let mut out = Vec::with_capacity(ndim);
    for axis_from_end in 0..ndim {
        let left_dim = left
            .len()
            .checked_sub(axis_from_end + 1)
            .map(|axis| left[axis])
            .unwrap_or(1);
        let right_dim = right
            .len()
            .checked_sub(axis_from_end + 1)
            .map(|axis| right[axis])
            .unwrap_or(1);
        let dim = if left_dim == 0 || right_dim == 0 {
            if matches!(left_dim, 0 | 1) && matches!(right_dim, 0 | 1) {
                0
            } else {
                return Err(PyValueError::new_err(format!(
                    "cannot broadcast comparison shapes {left:?} and {right:?}"
                )));
            }
        } else if left_dim == 1 {
            right_dim
        } else if right_dim == 1 || left_dim == right_dim {
            left_dim
        } else {
            return Err(PyValueError::new_err(format!(
                "cannot broadcast comparison shapes {left:?} and {right:?}"
            )));
        };
        out.push(dim);
    }
    out.reverse();
    Ok(out)
}

fn broadcast_offset(shape: &[usize], out_index: &[usize]) -> PyResult<usize> {
    if shape.is_empty() {
        return Ok(0);
    }
    let lead = out_index.len().saturating_sub(shape.len());
    let mut source_index = Vec::with_capacity(shape.len());
    for (axis, &dim) in shape.iter().enumerate() {
        source_index.push(if dim == 1 { 0 } else { out_index[lead + axis] });
    }
    row_major_offset(shape, &source_index)
}

fn stacked_det_f64<T>(array: &Array<T>, cast_out: impl Fn(f64) -> T) -> PyResult<Array<T>>
where
    T: Copy + IntoF64,
{
    let shape = array.shape();
    if shape.len() < 2 {
        return Err(PyValueError::new_err(
            "det requires an array with at least two dimensions",
        ));
    }
    let rows = shape[shape.len() - 2];
    let cols = shape[shape.len() - 1];
    if rows != cols {
        return Err(PyValueError::new_err(format!(
            "det requires square matrices, got trailing shape ({rows}, {cols})"
        )));
    }
    let stack_shape = &shape[..shape.len() - 2];
    let matrix_len = rows * cols;
    let stack_len = shape_size(stack_shape);
    let mut out = Vec::with_capacity(stack_len);
    for stack in 0..stack_len {
        let start = stack * matrix_len;
        let matrix = array.as_slice()[start..start + matrix_len]
            .iter()
            .copied()
            .map(|value| value_as_f64_number(value))
            .collect::<Vec<_>>();
        let det = Array::from_vec(vec![rows, cols], matrix)
            .map_err(py_value_error)?
            .det()
            .map_err(py_value_error)?;
        out.push(cast_out(det));
    }
    Array::from_vec(stack_shape.to_vec(), out).map_err(py_value_error)
}

fn stacked_det_f32(array: &Array<f32>) -> PyResult<Array<f32>> {
    stacked_det_f64(array, |value| value as f32)
}

fn stacked_det_c128(array: &Array<Complex64>) -> PyResult<Array<Complex64>> {
    let shape = array.shape();
    let (stack_shape, rows, cols) = det_shape(shape)?;
    let matrix_len = rows * cols;
    let stack_len = shape_size(stack_shape);
    let mut out = Vec::with_capacity(stack_len);
    for stack in 0..stack_len {
        let start = stack * matrix_len;
        out.push(det_complex64(
            array.as_slice()[start..start + matrix_len].to_vec(),
            rows,
        ));
    }
    Array::from_vec(stack_shape.to_vec(), out).map_err(py_value_error)
}

fn stacked_det_c64(array: &Array<Complex32>) -> PyResult<Array<Complex32>> {
    let shape = array.shape();
    let (stack_shape, rows, cols) = det_shape(shape)?;
    let matrix_len = rows * cols;
    let stack_len = shape_size(stack_shape);
    let mut out = Vec::with_capacity(stack_len);
    for stack in 0..stack_len {
        let start = stack * matrix_len;
        let matrix = array.as_slice()[start..start + matrix_len]
            .iter()
            .map(|value| Complex64::new(value.re as f64, value.im as f64))
            .collect::<Vec<_>>();
        let det = det_complex64(matrix, rows);
        out.push(Complex32::new(det.re as f32, det.im as f32));
    }
    Array::from_vec(stack_shape.to_vec(), out).map_err(py_value_error)
}

fn det_shape(shape: &[usize]) -> PyResult<(&[usize], usize, usize)> {
    if shape.len() < 2 {
        return Err(PyValueError::new_err(
            "det requires an array with at least two dimensions",
        ));
    }
    let rows = shape[shape.len() - 2];
    let cols = shape[shape.len() - 1];
    if rows != cols {
        return Err(PyValueError::new_err(format!(
            "det requires square matrices, got trailing shape ({rows}, {cols})"
        )));
    }
    Ok((&shape[..shape.len() - 2], rows, cols))
}

fn det_complex64(mut matrix: Vec<Complex64>, n: usize) -> Complex64 {
    if n == 0 {
        return Complex64::new(1.0, 0.0);
    }
    let mut det = Complex64::new(1.0, 0.0);
    for pivot in 0..n {
        let mut pivot_row = pivot;
        let mut pivot_norm = matrix[pivot * n + pivot].norm_sqr();
        for row in pivot + 1..n {
            let norm = matrix[row * n + pivot].norm_sqr();
            if norm > pivot_norm {
                pivot_norm = norm;
                pivot_row = row;
            }
        }
        if pivot_norm == 0.0 {
            return Complex64::new(0.0, 0.0);
        }
        if pivot_row != pivot {
            for col in 0..n {
                matrix.swap(pivot * n + col, pivot_row * n + col);
            }
            det = -det;
        }
        let pivot_value = matrix[pivot * n + pivot];
        det *= pivot_value;
        for row in pivot + 1..n {
            let factor = matrix[row * n + pivot] / pivot_value;
            for col in pivot + 1..n {
                let pivot_col = matrix[pivot * n + col];
                matrix[row * n + col] -= factor * pivot_col;
            }
        }
    }
    det
}

fn value_as_f64_number<T>(value: T) -> f64
where
    T: IntoF64,
{
    value.into_f64()
}

trait IntoF64 {
    fn into_f64(self) -> f64;
}

impl IntoF64 for f64 {
    fn into_f64(self) -> f64 {
        self
    }
}

impl IntoF64 for f32 {
    fn into_f64(self) -> f64 {
        self as f64
    }
}

fn diagonal_array<T>(array: &Array<T>, offset: isize) -> PyResult<Array<T>>
where
    T: Clone,
{
    let shape = array.shape();
    if shape.len() < 2 {
        return Err(PyValueError::new_err(
            "diagonal requires an array with at least two dimensions",
        ));
    }
    let rows = shape[shape.len() - 2] as isize;
    let cols = shape[shape.len() - 1] as isize;
    let diag_len = if offset < 0 {
        rows.min(cols).min((rows + offset).max(0))
    } else if offset == 0 {
        rows.min(cols)
    } else {
        rows.min(cols).min((cols - offset).max(0))
    } as usize;
    let mut out_shape = shape[..shape.len() - 2].to_vec();
    out_shape.push(diag_len);
    let out_len = shape_size(&out_shape);
    let mut out = Vec::with_capacity(out_len);
    for linear in 0..out_len {
        let out_index = unravel_index(linear, &out_shape);
        let diag = *out_index.last().unwrap_or(&0);
        let mut source_index = out_index[..out_index.len().saturating_sub(1)].to_vec();
        if offset >= 0 {
            source_index.push(diag);
            source_index.push(diag + offset as usize);
        } else {
            source_index.push(diag + (-offset) as usize);
            source_index.push(diag);
        }
        let offset = row_major_offset(shape, &source_index)?;
        out.push(array.as_slice()[offset].clone());
    }
    Array::from_vec(out_shape, out).map_err(py_value_error)
}

fn matrix_transpose_array<T>(array: &Array<T>) -> PyResult<Array<T>>
where
    T: Clone,
{
    let shape = array.shape();
    if shape.len() < 2 {
        return Err(PyValueError::new_err(
            "matrix_transpose requires an array with at least two dimensions",
        ));
    }
    let mut out_shape = shape.to_vec();
    let last = out_shape.len() - 1;
    out_shape.swap(last - 1, last);
    let out_len = shape_size(&out_shape);
    let mut out = Vec::with_capacity(out_len);
    for linear in 0..out_len {
        let mut source_index = unravel_index(linear, &out_shape);
        source_index.swap(last - 1, last);
        let offset = row_major_offset(shape, &source_index)?;
        out.push(array.as_slice()[offset].clone());
    }
    Array::from_vec(out_shape, out).map_err(py_value_error)
}

fn fft_axis_c128(
    array: &Array<Complex64>,
    n: Option<usize>,
    axis: isize,
    inverse: bool,
    norm: &str,
) -> PyResult<Array<Complex64>> {
    let shape = array.shape();
    let axis = normalize_axis_for_shape(axis, shape.len())?;
    let fft_len = n.unwrap_or(shape[axis]);
    let mut out_shape = shape.to_vec();
    out_shape[axis] = fft_len;
    if fft_len == 0 {
        return Array::from_vec(out_shape, Vec::new()).map_err(py_value_error);
    }

    let scale = fft_scale_f64(fft_len, inverse, norm)?;
    let lane_shape = shape_without_axis(&out_shape, axis);
    let mut out = vec![Complex64::new(0.0, 0.0); shape_size(&out_shape)];
    let mut planner = FftPlanner::<f64>::new();
    let fft = if inverse {
        planner.plan_fft_inverse(fft_len)
    } else {
        planner.plan_fft_forward(fft_len)
    };

    for lane_linear in 0..shape_size(&lane_shape) {
        let lane_index = unravel_index(lane_linear, &lane_shape);
        let mut buffer = vec![Complex64::new(0.0, 0.0); fft_len];
        for (sample, slot) in buffer.iter_mut().enumerate().take(fft_len.min(shape[axis])) {
            let source_index = insert_axis_index(&lane_index, axis, sample);
            *slot = array.as_slice()[row_major_offset(shape, &source_index)?];
        }
        fft.process(&mut buffer);
        if scale != 1.0 {
            for value in &mut buffer {
                *value *= scale;
            }
        }
        for (sample, value) in buffer.into_iter().enumerate() {
            let out_index = insert_axis_index(&lane_index, axis, sample);
            let out_offset = row_major_offset(&out_shape, &out_index)?;
            out[out_offset] = value;
        }
    }

    Array::from_vec(out_shape, out).map_err(py_value_error)
}

fn fft_axis_c64(
    array: &Array<Complex32>,
    n: Option<usize>,
    axis: isize,
    inverse: bool,
    norm: &str,
) -> PyResult<Array<Complex32>> {
    let shape = array.shape();
    let axis = normalize_axis_for_shape(axis, shape.len())?;
    let fft_len = n.unwrap_or(shape[axis]);
    let mut out_shape = shape.to_vec();
    out_shape[axis] = fft_len;
    if fft_len == 0 {
        return Array::from_vec(out_shape, Vec::new()).map_err(py_value_error);
    }

    let scale = fft_scale_f32(fft_len, inverse, norm)?;
    let lane_shape = shape_without_axis(&out_shape, axis);
    let mut out = vec![Complex32::new(0.0, 0.0); shape_size(&out_shape)];
    let mut planner = FftPlanner::<f32>::new();
    let fft = if inverse {
        planner.plan_fft_inverse(fft_len)
    } else {
        planner.plan_fft_forward(fft_len)
    };

    for lane_linear in 0..shape_size(&lane_shape) {
        let lane_index = unravel_index(lane_linear, &lane_shape);
        let mut buffer = vec![Complex32::new(0.0, 0.0); fft_len];
        for (sample, slot) in buffer.iter_mut().enumerate().take(fft_len.min(shape[axis])) {
            let source_index = insert_axis_index(&lane_index, axis, sample);
            *slot = array.as_slice()[row_major_offset(shape, &source_index)?];
        }
        fft.process(&mut buffer);
        if scale != 1.0 {
            for value in &mut buffer {
                *value *= scale;
            }
        }
        for (sample, value) in buffer.into_iter().enumerate() {
            let out_index = insert_axis_index(&lane_index, axis, sample);
            let out_offset = row_major_offset(&out_shape, &out_index)?;
            out[out_offset] = value;
        }
    }

    Array::from_vec(out_shape, out).map_err(py_value_error)
}

fn rfft_axis_f64(
    array: &Array<f64>,
    n: Option<usize>,
    axis: isize,
    norm: &str,
) -> PyResult<Array<Complex64>> {
    let complex = cast_array(array, |value| Complex64::new(value, 0.0))?;
    let full = fft_axis_c128(&complex, n, axis, false, norm)?;
    let axis = normalize_axis_for_shape(axis, full.shape().len())?;
    trim_axis_c128(&full, axis, full.shape()[axis] / 2 + 1)
}

fn rfft_axis_f32(
    array: &Array<f32>,
    n: Option<usize>,
    axis: isize,
    norm: &str,
) -> PyResult<Array<Complex32>> {
    let complex = cast_array(array, |value| Complex32::new(value, 0.0))?;
    let full = fft_axis_c64(&complex, n, axis, false, norm)?;
    let axis = normalize_axis_for_shape(axis, full.shape().len())?;
    trim_axis_c64(&full, axis, full.shape()[axis] / 2 + 1)
}

fn irfft_axis_c128(
    array: &Array<Complex64>,
    n: Option<usize>,
    axis: isize,
    norm: &str,
) -> PyResult<Array<f64>> {
    let shape = array.shape();
    let axis = normalize_axis_for_shape(axis, shape.len())?;
    let fft_len = n.unwrap_or_else(|| shape[axis].saturating_sub(1) * 2);
    let full = hermitian_spectrum_c128(array, axis, fft_len)?;
    let transformed = fft_axis_c128(&full, Some(fft_len), axis as isize, true, norm)?;
    cast_array(&transformed, |value| value.re)
}

fn irfft_axis_c64(
    array: &Array<Complex32>,
    n: Option<usize>,
    axis: isize,
    norm: &str,
) -> PyResult<Array<f32>> {
    let shape = array.shape();
    let axis = normalize_axis_for_shape(axis, shape.len())?;
    let fft_len = n.unwrap_or_else(|| shape[axis].saturating_sub(1) * 2);
    let full = hermitian_spectrum_c64(array, axis, fft_len)?;
    let transformed = fft_axis_c64(&full, Some(fft_len), axis as isize, true, norm)?;
    cast_array(&transformed, |value| value.re)
}

fn trim_axis_c128(array: &Array<Complex64>, axis: usize, len: usize) -> PyResult<Array<Complex64>> {
    trim_axis(array, axis, len)
}

fn trim_axis_c64(array: &Array<Complex32>, axis: usize, len: usize) -> PyResult<Array<Complex32>> {
    trim_axis(array, axis, len)
}

fn trim_axis<T>(array: &Array<T>, axis: usize, len: usize) -> PyResult<Array<T>>
where
    T: Clone,
{
    let shape = array.shape();
    let mut out_shape = shape.to_vec();
    out_shape[axis] = len.min(shape[axis]);
    let mut out = Vec::with_capacity(shape_size(&out_shape));
    for linear in 0..shape_size(&out_shape) {
        let index = unravel_index(linear, &out_shape);
        out.push(array.as_slice()[row_major_offset(shape, &index)?].clone());
    }
    Array::from_vec(out_shape, out).map_err(py_value_error)
}

fn hermitian_spectrum_c128(
    array: &Array<Complex64>,
    axis: usize,
    fft_len: usize,
) -> PyResult<Array<Complex64>> {
    let shape = array.shape();
    let mut out_shape = shape.to_vec();
    out_shape[axis] = fft_len;
    if fft_len == 0 {
        return Array::from_vec(out_shape, Vec::new()).map_err(py_value_error);
    }
    let lane_shape = shape_without_axis(&out_shape, axis);
    let mut out = vec![Complex64::new(0.0, 0.0); shape_size(&out_shape)];
    for lane_linear in 0..shape_size(&lane_shape) {
        let lane_index = unravel_index(lane_linear, &lane_shape);
        for sample in 0..fft_len {
            let value = if sample < shape[axis] {
                let source_index = insert_axis_index(&lane_index, axis, sample);
                array.as_slice()[row_major_offset(shape, &source_index)?]
            } else {
                let mirror = fft_len - sample;
                if mirror < shape[axis] {
                    let source_index = insert_axis_index(&lane_index, axis, mirror);
                    array.as_slice()[row_major_offset(shape, &source_index)?].conj()
                } else {
                    Complex64::new(0.0, 0.0)
                }
            };
            let out_index = insert_axis_index(&lane_index, axis, sample);
            let out_offset = row_major_offset(&out_shape, &out_index)?;
            out[out_offset] = value;
        }
    }
    Array::from_vec(out_shape, out).map_err(py_value_error)
}

fn hermitian_spectrum_c64(
    array: &Array<Complex32>,
    axis: usize,
    fft_len: usize,
) -> PyResult<Array<Complex32>> {
    let shape = array.shape();
    let mut out_shape = shape.to_vec();
    out_shape[axis] = fft_len;
    if fft_len == 0 {
        return Array::from_vec(out_shape, Vec::new()).map_err(py_value_error);
    }
    let lane_shape = shape_without_axis(&out_shape, axis);
    let mut out = vec![Complex32::new(0.0, 0.0); shape_size(&out_shape)];
    for lane_linear in 0..shape_size(&lane_shape) {
        let lane_index = unravel_index(lane_linear, &lane_shape);
        for sample in 0..fft_len {
            let value = if sample < shape[axis] {
                let source_index = insert_axis_index(&lane_index, axis, sample);
                array.as_slice()[row_major_offset(shape, &source_index)?]
            } else {
                let mirror = fft_len - sample;
                if mirror < shape[axis] {
                    let source_index = insert_axis_index(&lane_index, axis, mirror);
                    array.as_slice()[row_major_offset(shape, &source_index)?].conj()
                } else {
                    Complex32::new(0.0, 0.0)
                }
            };
            let out_index = insert_axis_index(&lane_index, axis, sample);
            let out_offset = row_major_offset(&out_shape, &out_index)?;
            out[out_offset] = value;
        }
    }
    Array::from_vec(out_shape, out).map_err(py_value_error)
}

fn shift_array<T>(array: &Array<T>, axes: Option<&[isize]>, inverse: bool) -> PyResult<Array<T>>
where
    T: Clone,
{
    let shape = array.shape();
    let axes = normalize_axes_for_shape(axes, shape.len())?;
    let mut out = Vec::with_capacity(array.len());
    for linear in 0..array.len() {
        let out_index = unravel_index(linear, shape);
        let mut source_index = out_index.clone();
        for &axis in &axes {
            let dim = shape[axis];
            if dim == 0 {
                continue;
            }
            let roll = if inverse { dim.div_ceil(2) } else { dim / 2 };
            source_index[axis] = (out_index[axis] + dim - roll % dim) % dim;
        }
        out.push(array.as_slice()[row_major_offset(shape, &source_index)?].clone());
    }
    Array::from_vec(shape.to_vec(), out).map_err(py_value_error)
}

fn fft_scale_f64(n: usize, inverse: bool, norm: &str) -> PyResult<f64> {
    match norm {
        "backward" => Ok(if inverse { 1.0 / n as f64 } else { 1.0 }),
        "forward" => Ok(if inverse { 1.0 } else { 1.0 / n as f64 }),
        "ortho" => Ok(1.0 / (n as f64).sqrt()),
        _ => Err(PyValueError::new_err(format!(
            "unsupported FFT norm {norm:?}; expected 'backward', 'forward', or 'ortho'"
        ))),
    }
}

fn fft_scale_f32(n: usize, inverse: bool, norm: &str) -> PyResult<f32> {
    Ok(fft_scale_f64(n, inverse, norm)? as f32)
}

fn normalize_axis_for_shape(axis: isize, ndim: usize) -> PyResult<usize> {
    if ndim == 0 {
        return Err(PyValueError::new_err(
            "axis is not valid for a scalar array",
        ));
    }
    let normalized = if axis < 0 { ndim as isize + axis } else { axis };
    if normalized < 0 || normalized >= ndim as isize {
        return Err(PyValueError::new_err(format!(
            "axis {axis} is out of bounds for array of dimension {ndim}"
        )));
    }
    Ok(normalized as usize)
}

fn normalize_axes_for_shape(axes: Option<&[isize]>, ndim: usize) -> PyResult<Vec<usize>> {
    let raw = axes
        .map(|axes| axes.to_vec())
        .unwrap_or_else(|| (0..ndim as isize).collect());
    let mut out = Vec::with_capacity(raw.len());
    for axis in raw {
        let normalized = normalize_axis_for_shape(axis, ndim)?;
        if !out.contains(&normalized) {
            out.push(normalized);
        }
    }
    Ok(out)
}

fn shape_without_axis(shape: &[usize], axis: usize) -> Vec<usize> {
    shape
        .iter()
        .enumerate()
        .filter_map(|(idx, &dim)| (idx != axis).then_some(dim))
        .collect()
}

fn insert_axis_index(index: &[usize], axis: usize, value: usize) -> Vec<usize> {
    let mut out = Vec::with_capacity(index.len() + 1);
    out.extend_from_slice(&index[..axis]);
    out.push(value);
    out.extend_from_slice(&index[axis..]);
    out
}

macro_rules! value_as_numeric {
    ($name:ident, $ty:ty) => {
        fn $name(value: &Bound<'_, PyAny>) -> PyResult<$ty> {
            value.extract::<$ty>()
        }
    };
}

value_as_numeric!(value_as_f64, f64);
value_as_numeric!(value_as_f32, f32);
value_as_numeric!(value_as_i64, i64);
value_as_numeric!(value_as_i32, i32);
value_as_numeric!(value_as_i16, i16);
value_as_numeric!(value_as_i8, i8);
value_as_numeric!(value_as_u64, u64);
value_as_numeric!(value_as_u32, u32);
value_as_numeric!(value_as_u16, u16);
value_as_numeric!(value_as_u8, u8);
value_as_numeric!(value_as_bool, bool);

fn value_as_c128(value: &Bound<'_, PyAny>) -> PyResult<Complex64> {
    if let Ok(complex) = value.cast::<PyComplex>() {
        return Ok(Complex64::new(complex.real(), complex.imag()));
    }
    Ok(Complex64::new(value.extract::<f64>()?, 0.0))
}

fn value_as_c64(value: &Bound<'_, PyAny>) -> PyResult<Complex32> {
    if let Ok(complex) = value.cast::<PyComplex>() {
        return Ok(Complex32::new(complex.real() as f32, complex.imag() as f32));
    }
    Ok(Complex32::new(value.extract::<f32>()?, 0.0))
}

macro_rules! assignment_values_numeric {
    ($name:ident, $array_conv:ident, $ty:ty) => {
        fn $name(value: &Bound<'_, PyAny>) -> PyResult<Vec<$ty>> {
            if let Ok(array) = value.extract::<PyRef<'_, PyArray>>() {
                return Ok($array_conv(&array)?.as_slice().to_vec());
            }
            Ok(vec![value.extract::<$ty>()?])
        }
    };
}

assignment_values_numeric!(assignment_values_f64, to_f64_array, f64);
assignment_values_numeric!(assignment_values_f32, to_f32_array, f32);
assignment_values_numeric!(assignment_values_i64, to_i64_array, i64);
assignment_values_numeric!(assignment_values_i32, to_i32_array, i32);
assignment_values_numeric!(assignment_values_i16, to_i16_array, i16);
assignment_values_numeric!(assignment_values_i8, to_i8_array, i8);
assignment_values_numeric!(assignment_values_u64, to_u64_array, u64);
assignment_values_numeric!(assignment_values_u32, to_u32_array, u32);
assignment_values_numeric!(assignment_values_u16, to_u16_array, u16);
assignment_values_numeric!(assignment_values_u8, to_u8_array, u8);

fn assignment_values_bool(value: &Bound<'_, PyAny>) -> PyResult<Vec<bool>> {
    if let Ok(array) = value.extract::<PyRef<'_, PyArray>>() {
        return Ok(to_bool_array(&array)?.as_slice().to_vec());
    }
    Ok(vec![value.extract::<bool>()?])
}

fn assignment_values_c128(value: &Bound<'_, PyAny>) -> PyResult<Vec<Complex64>> {
    if let Ok(array) = value.extract::<PyRef<'_, PyArray>>() {
        return Ok(to_c128_array(&array)?.as_slice().to_vec());
    }
    let complex = value.cast::<PyComplex>()?;
    Ok(vec![Complex64::new(complex.real(), complex.imag())])
}

fn assignment_values_c64(value: &Bound<'_, PyAny>) -> PyResult<Vec<Complex32>> {
    if let Ok(array) = value.extract::<PyRef<'_, PyArray>>() {
        return Ok(to_c64_array(&array)?.as_slice().to_vec());
    }
    let complex = value.cast::<PyComplex>()?;
    Ok(vec![Complex32::new(
        complex.real() as f32,
        complex.imag() as f32,
    )])
}

fn cast_array<T, U, F>(array: &Array<T>, cast: F) -> PyResult<Array<U>>
where
    T: Copy,
    F: Fn(T) -> U,
{
    Array::from_vec(
        array.shape().to_vec(),
        array.as_slice().iter().copied().map(cast).collect(),
    )
    .map_err(py_value_error)
}

macro_rules! impl_to_numeric_array {
    ($name:ident, $ty:ty) => {
        fn $name(array: &PyArray) -> PyResult<Array<$ty>> {
            match &array.storage {
                Storage::F64(array) => cast_array(array, |value| value as $ty),
                Storage::F32(array) => cast_array(array, |value| value as $ty),
                Storage::C128(_) | Storage::C64(_) => Err(PyTypeError::new_err(
                    "cannot cast complex arrays to real dtypes",
                )),
                Storage::I64(array) => cast_array(array, |value| value as $ty),
                Storage::I32(array) => cast_array(array, |value| value as $ty),
                Storage::I16(array) => cast_array(array, |value| value as $ty),
                Storage::I8(array) => cast_array(array, |value| value as $ty),
                Storage::U64(array) => cast_array(array, |value| value as $ty),
                Storage::U32(array) => cast_array(array, |value| value as $ty),
                Storage::U16(array) => cast_array(array, |value| value as $ty),
                Storage::U8(array) => cast_array(array, |value| value as $ty),
                Storage::Bool(array) => {
                    cast_array(array, |value| if value { 1 as $ty } else { 0 as $ty })
                }
            }
        }
    };
}

impl_to_numeric_array!(to_f64_array, f64);
impl_to_numeric_array!(to_f32_array, f32);
impl_to_numeric_array!(to_i64_array, i64);
impl_to_numeric_array!(to_i32_array, i32);
impl_to_numeric_array!(to_i16_array, i16);
impl_to_numeric_array!(to_i8_array, i8);
impl_to_numeric_array!(to_u64_array, u64);
impl_to_numeric_array!(to_u32_array, u32);
impl_to_numeric_array!(to_u16_array, u16);
impl_to_numeric_array!(to_u8_array, u8);

fn to_c128_array(array: &PyArray) -> PyResult<Array<Complex64>> {
    match &array.storage {
        Storage::F64(array) => cast_array(array, |value| Complex64::new(value, 0.0)),
        Storage::F32(array) => cast_array(array, |value| Complex64::new(value as f64, 0.0)),
        Storage::C128(array) => Ok(array.clone()),
        Storage::C64(array) => cast_array(array, |value| {
            Complex64::new(value.re as f64, value.im as f64)
        }),
        Storage::I64(array) => cast_array(array, |value| Complex64::new(value as f64, 0.0)),
        Storage::I32(array) => cast_array(array, |value| Complex64::new(value as f64, 0.0)),
        Storage::I16(array) => cast_array(array, |value| Complex64::new(value as f64, 0.0)),
        Storage::I8(array) => cast_array(array, |value| Complex64::new(value as f64, 0.0)),
        Storage::U64(array) => cast_array(array, |value| Complex64::new(value as f64, 0.0)),
        Storage::U32(array) => cast_array(array, |value| Complex64::new(value as f64, 0.0)),
        Storage::U16(array) => cast_array(array, |value| Complex64::new(value as f64, 0.0)),
        Storage::U8(array) => cast_array(array, |value| Complex64::new(value as f64, 0.0)),
        Storage::Bool(array) => cast_array(array, |value| {
            Complex64::new(if value { 1.0 } else { 0.0 }, 0.0)
        }),
    }
}

fn to_c64_array(array: &PyArray) -> PyResult<Array<Complex32>> {
    match &array.storage {
        Storage::F64(array) => cast_array(array, |value| Complex32::new(value as f32, 0.0)),
        Storage::F32(array) => cast_array(array, |value| Complex32::new(value, 0.0)),
        Storage::C128(array) => cast_array(array, |value| {
            Complex32::new(value.re as f32, value.im as f32)
        }),
        Storage::C64(array) => Ok(array.clone()),
        Storage::I64(array) => cast_array(array, |value| Complex32::new(value as f32, 0.0)),
        Storage::I32(array) => cast_array(array, |value| Complex32::new(value as f32, 0.0)),
        Storage::I16(array) => cast_array(array, |value| Complex32::new(value as f32, 0.0)),
        Storage::I8(array) => cast_array(array, |value| Complex32::new(value as f32, 0.0)),
        Storage::U64(array) => cast_array(array, |value| Complex32::new(value as f32, 0.0)),
        Storage::U32(array) => cast_array(array, |value| Complex32::new(value as f32, 0.0)),
        Storage::U16(array) => cast_array(array, |value| Complex32::new(value as f32, 0.0)),
        Storage::U8(array) => cast_array(array, |value| Complex32::new(value as f32, 0.0)),
        Storage::Bool(array) => cast_array(array, |value| {
            Complex32::new(if value { 1.0 } else { 0.0 }, 0.0)
        }),
    }
}

fn to_bool_array(array: &PyArray) -> PyResult<Array<bool>> {
    match &array.storage {
        Storage::F64(array) => cast_array(array, |value| value != 0.0),
        Storage::F32(array) => cast_array(array, |value| value != 0.0),
        Storage::C128(array) => cast_array(array, |value| value.re != 0.0 || value.im != 0.0),
        Storage::C64(array) => cast_array(array, |value| value.re != 0.0 || value.im != 0.0),
        Storage::I64(array) => cast_array(array, |value| value != 0),
        Storage::I32(array) => cast_array(array, |value| value != 0),
        Storage::I16(array) => cast_array(array, |value| value != 0),
        Storage::I8(array) => cast_array(array, |value| value != 0),
        Storage::U64(array) => cast_array(array, |value| value != 0),
        Storage::U32(array) => cast_array(array, |value| value != 0),
        Storage::U16(array) => cast_array(array, |value| value != 0),
        Storage::U8(array) => cast_array(array, |value| value != 0),
        Storage::Bool(array) => Ok(array.clone()),
    }
}

fn dtype_name(dtype: DTypeKind) -> &'static str {
    match dtype {
        DTypeKind::F64 => "float64",
        DTypeKind::F32 => "float32",
        DTypeKind::Complex128 => "complex128",
        DTypeKind::Complex64 => "complex64",
        DTypeKind::I64 => "int64",
        DTypeKind::I32 => "int32",
        DTypeKind::I16 => "int16",
        DTypeKind::I8 => "int8",
        DTypeKind::U64 => "uint64",
        DTypeKind::U32 => "uint32",
        DTypeKind::U16 => "uint16",
        DTypeKind::U8 => "uint8",
        DTypeKind::Bool => "bool",
    }
}

fn dtype_error(dtype: &str) -> PyErr {
    PyTypeError::new_err(format!(
        "unsupported dtype {dtype:?}; supported dtypes: bool, int8, int16, int32, int64, uint8, uint16, uint32, uint64, float32, float64, complex64, complex128"
    ))
}

fn is_complex_kind(dtype: DTypeKind) -> bool {
    matches!(dtype, DTypeKind::Complex64 | DTypeKind::Complex128)
}

fn complex_list<'py>(
    py: Python<'py>,
    values: impl Iterator<Item = (f64, f64)>,
) -> PyResult<Py<PyAny>> {
    let out = PyList::empty(py);
    for (real, imag) in values {
        out.append(PyComplex::from_doubles(py, real, imag))?;
    }
    Ok(out.unbind().into())
}

fn py_value_error(error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn array_api_not_implemented<T>(name: &str) -> PyResult<T> {
    Err(PyTypeError::new_err(format!(
        "{name} exists for Array API surface discovery but is not implemented yet"
    )))
}
