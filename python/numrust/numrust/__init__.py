from __future__ import annotations

import builtins as _builtins
import cmath as _cmath
import math as _math
from collections import namedtuple as _namedtuple
from types import SimpleNamespace as _SimpleNamespace
from collections.abc import Sequence
from typing import Any

from . import _numrust

Array = _numrust.Array
__array_api_version__ = "2023.12"
e = _math.e
pi = _math.pi
inf = _math.inf
nan = _math.nan
newaxis = None


class DType:
    def __init__(self, name: str | "DType") -> None:
        self.name = name.name if isinstance(name, DType) else str(name)

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"numrust.{self.name}"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, DType):
            return self.name == other.name
        if isinstance(other, str):
            return self.name == other
        return False

    def __hash__(self) -> int:
        return hash(self.name)


bool = DType("bool")
int8 = DType("int8")
int16 = DType("int16")
int32 = DType("int32")
int64 = DType("int64")
uint8 = DType("uint8")
uint16 = DType("uint16")
uint32 = DType("uint32")
uint64 = DType("uint64")
float32 = DType("float32")
float64 = DType("float64")
complex64 = DType("complex64")
complex128 = DType("complex128")

_PRIMITIVE_DTYPES = {
    bool,
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
    float32,
    float64,
}
_DTYPES = _PRIMITIVE_DTYPES | {complex64, complex128}


def _not_implemented(*args: Any, **kwargs: Any) -> None:
    raise NotImplementedError("this Array API extension function is not implemented yet")


linalg = _SimpleNamespace(
    **{
        name: _not_implemented
        for name in [
            "cholesky",
            "cross",
            "det",
            "diagonal",
            "eigh",
            "eigvalsh",
            "inv",
            "matmul",
            "matrix_norm",
            "matrix_power",
            "matrix_rank",
            "matrix_transpose",
            "outer",
            "pinv",
            "qr",
            "slogdet",
            "solve",
            "svd",
            "svdvals",
            "tensordot",
            "trace",
            "vecdot",
            "vector_norm",
        ]
    }
)
fft = _SimpleNamespace(
    **{
        name: _not_implemented
        for name in [
            "fft",
            "ifft",
            "fftn",
            "ifftn",
            "rfft",
            "irfft",
            "rfftn",
            "irfftn",
            "hfft",
            "ihfft",
            "fftfreq",
            "rfftfreq",
            "fftshift",
            "ifftshift",
        ]
    }
)


def asarray(obj: Any, dtype: str | DType | None = None, *, device: str | None = None, copy: bool | None = None) -> Array:
    _check_device(device)
    if isinstance(obj, Array):
        if dtype is not None:
            dtype_name = _dtype_name(dtype)
            if dtype_name == obj.dtype and copy is not True:
                return obj
            return obj.astype(dtype_name)
        return obj.astype(_dtype_name(obj.dtype)) if copy is True else obj

    shape, flat = _flatten(obj)
    dtype = _dtype_name(dtype) if dtype is not None else _infer_dtype(flat)
    if dtype == float64:
        return _numrust.from_f64([float(x) for x in flat], shape)
    if dtype == float32:
        return _numrust.from_f32([float(x) for x in flat], shape)
    if dtype == complex128:
        return _numrust.from_c128([_complex_pair(x) for x in flat], shape)
    if dtype == complex64:
        return _numrust.from_c64([_complex_pair(x) for x in flat], shape)
    if dtype == int64:
        return _numrust.from_i64([_wrap_signed_int(x, 64) for x in flat], shape)
    if dtype == int32:
        return _numrust.from_i32([_wrap_signed_int(x, 32) for x in flat], shape)
    if dtype == int16:
        return _numrust.from_i16([_wrap_signed_int(x, 16) for x in flat], shape)
    if dtype == int8:
        return _numrust.from_i8([_wrap_signed_int(x, 8) for x in flat], shape)
    if dtype == uint64:
        return _numrust.from_u64([_wrap_unsigned_int(x, 64) for x in flat], shape)
    if dtype == uint32:
        return _numrust.from_u32([_wrap_unsigned_int(x, 32) for x in flat], shape)
    if dtype == uint16:
        return _numrust.from_u16([_wrap_unsigned_int(x, 16) for x in flat], shape)
    if dtype == uint8:
        return _numrust.from_u8([_wrap_unsigned_int(x, 8) for x in flat], shape)
    if dtype == bool:
        return _numrust.from_bool([_builtins.bool(x) for x in flat], shape)
    raise TypeError(f"unsupported dtype {dtype!r}")


def zeros(shape: int | Sequence[int], dtype: str | DType = float64, *, device: str | None = None) -> Array:
    _check_device(device)
    return _numrust.zeros(_normalize_shape(shape), _dtype_name(dtype, default=float64))


def ones(shape: int | Sequence[int], dtype: str | DType = float64, *, device: str | None = None) -> Array:
    _check_device(device)
    return _numrust.ones(_normalize_shape(shape), _dtype_name(dtype, default=float64))


def empty(shape: int | Sequence[int], dtype: str | DType = float64, *, device: str | None = None) -> Array:
    return zeros(shape, dtype=dtype, device=device)


def full(
    shape: int | Sequence[int],
    fill_value: float,
    dtype: str | DType | None = None,
    *,
    device: str | None = None,
) -> Array:
    _check_device(device)
    normalized = _normalize_shape(shape)
    dtype_name = _dtype_name(dtype, default=_infer_scalar_dtype(fill_value))
    if dtype_name == complex128:
        return _numrust.from_c128([_complex_pair(fill_value)] * _size_of_shape(normalized), normalized)
    if dtype_name == complex64:
        return _numrust.from_c64([_complex_pair(fill_value)] * _size_of_shape(normalized), normalized)
    return _numrust.full(normalized, dtype_name, float(fill_value))


def arange(
    start: int | float,
    /,
    stop: int | float | None = None,
    step: int | float = 1,
    *,
    dtype: str | DType | None = None,
    device: str | None = None,
) -> Array:
    _check_device(device)
    if stop is None:
        start, stop = 0, start
    if step == 0:
        raise ValueError("arange step cannot be zero")
    dtype_name = _dtype_name(
        dtype,
        default=float64 if _builtins.any(isinstance(v, float) for v in (start, stop, step)) else int64,
    )
    values = []
    current = start
    if step > 0:
        while current < stop:
            values.append(current)
            current += step
    else:
        while current > stop:
            values.append(current)
            current += step
    return asarray(values, dtype=dtype_name)


def astype(
    x: Array,
    dtype: str | DType,
    /,
    *,
    copy: bool = True,
    device: str | None = None,
) -> Array:
    _check_device(device)
    if copy is False and x.dtype == _dtype_name(dtype):
        return x
    return x.astype(_dtype_name(dtype))


def reshape(x: Array, /, shape: Sequence[int], *, copy: bool | None = None) -> Array:
    del copy
    return x.reshape([int(dim) for dim in shape])


def permute_dims(x: Array, axes: Sequence[int]) -> Array:
    axes = tuple(_normalize_axis(axis, x.ndim) for axis in axes)
    if len(axes) != x.ndim or len(set(axes)) != x.ndim:
        raise ValueError("permute_dims axes must be a permutation of all axes")
    out_shape = tuple(x.shape[axis] for axis in axes)
    inverse = [0] * x.ndim
    for out_axis, source_axis in enumerate(axes):
        inverse[source_axis] = out_axis
    values = []
    for source_idx in _ndindex(x.shape):
        out_idx = tuple(source_idx[axis] for axis in axes)
        values.append((out_idx, _scalar_at(x, source_idx)))
    ordered = [None] * _size_of_shape(out_shape)
    for out_idx, value in values:
        ordered[_row_major_offset(out_shape, out_idx)] = value
    return asarray(ordered, dtype=DType(x.dtype)).reshape(out_shape)


def expand_dims(x: Array, axis: int | Sequence[int]) -> Array:
    axes = (axis,) if isinstance(axis, int) else tuple(axis)
    out_ndim = x.ndim + len(axes)
    normalized = []
    for raw_axis in axes:
        raw_axis = int(raw_axis)
        axis_value = raw_axis + out_ndim if raw_axis < 0 else raw_axis
        if axis_value < 0 or axis_value >= out_ndim:
            raise IndexError(f"axis {raw_axis} is out of bounds for expanded dimension {out_ndim}")
        if axis_value in normalized:
            raise ValueError("expand_dims axes must be unique")
        normalized.append(axis_value)
    out_shape = list(x.shape)
    for axis_value in sorted(normalized):
        out_shape.insert(axis_value, 1)
    return asarray(x, dtype=DType(x.dtype)).reshape(tuple(out_shape))


def squeeze(x: Array, axis: int | Sequence[int] | None = None) -> Array:
    if axis is None:
        axes = tuple(idx for idx, side in enumerate(x.shape) if side == 1)
    else:
        axes = (_normalize_axis(axis, x.ndim),) if isinstance(axis, int) else tuple(_normalize_axis(item, x.ndim) for item in axis)
    if _builtins.any(x.shape[item] != 1 for item in axes):
        raise ValueError("cannot squeeze an axis whose length is not one")
    out_shape = tuple(side for idx, side in enumerate(x.shape) if idx not in set(axes))
    return asarray(x, dtype=DType(x.dtype)).reshape(out_shape)


def moveaxis(x: Array, source: int | Sequence[int], destination: int | Sequence[int]) -> Array:
    source_axes = (_normalize_axis(source, x.ndim),) if isinstance(source, int) else tuple(_normalize_axis(axis, x.ndim) for axis in source)
    destination_axes = (
        (_normalize_axis(destination, x.ndim),)
        if isinstance(destination, int)
        else tuple(_normalize_axis(axis, x.ndim) for axis in destination)
    )
    if len(source_axes) != len(destination_axes):
        raise ValueError("source and destination must have the same number of axes")
    if len(set(source_axes)) != len(source_axes) or len(set(destination_axes)) != len(destination_axes):
        raise ValueError("source and destination axes must be unique")
    axes = [axis for axis in range(x.ndim) if axis not in source_axes]
    for dest, src in sorted(zip(destination_axes, source_axes)):
        axes.insert(dest, src)
    return permute_dims(x, axes)


def flip(x: Array, axis: int | Sequence[int] | None = None) -> Array:
    axes = set(_axes_or_all(axis, x.ndim))
    values = []
    for out_idx in _ndindex(x.shape):
        source_idx = tuple(x.shape[i] - 1 - coord if i in axes else coord for i, coord in enumerate(out_idx))
        values.append(_scalar_at(x, source_idx))
    return asarray(values, dtype=DType(x.dtype)).reshape(x.shape)


def add(x1: Array, x2: Array) -> Array:
    return x1 + x2


def subtract(x1: Array, x2: Array) -> Array:
    return x1 - x2


def multiply(x1: Array, x2: Array) -> Array:
    return x1 * x2


def pow(x1: Array, x2: Any) -> Array:
    return x1 ** x2


def divide(x1: Array, x2: Array) -> Array:
    return x1 / x2


def equal(x1: Array, x2: Array) -> Array:
    return x1 == x2


def not_equal(x1: Array, x2: Array) -> Array:
    return x1 != x2


def less(x1: Array, x2: Array) -> Array:
    return x1 < x2


def less_equal(x1: Array, x2: Array) -> Array:
    return x1 <= x2


def greater(x1: Array, x2: Array) -> Array:
    return x1 > x2


def greater_equal(x1: Array, x2: Array) -> Array:
    return x1 >= x2


def logical_and(x1: Array, x2: Array) -> Array:
    return _logical_binary(x1, x2, lambda left, right: left and right)


def logical_or(x1: Array, x2: Array) -> Array:
    return _logical_binary(x1, x2, lambda left, right: left or right)


def logical_xor(x1: Array, x2: Array) -> Array:
    return _logical_binary(x1, x2, lambda left, right: left != right)


def logical_not(x: Array) -> Array:
    array = asarray(x, dtype=bool)
    return asarray([not _builtins.bool(value) for value in array.tolist()], dtype=bool).reshape(array.shape)


def bitwise_and(x1: Array, x2: Any) -> Array:
    return x1 & x2


def bitwise_left_shift(x1: Array, x2: Any) -> Array:
    return x1 << x2


def bitwise_invert(x: Array) -> Array:
    return ~x


def bitwise_or(x1: Array, x2: Any) -> Array:
    return x1 | x2


def bitwise_right_shift(x1: Array, x2: Any) -> Array:
    return x1 >> x2


def bitwise_xor(x1: Array, x2: Any) -> Array:
    return x1 ^ x2


def ceil(x: Array) -> Array:
    array = asarray(x)
    dtype = DType(array.dtype)
    values = []
    for value in array.tolist():
        if dtype in {float32, float64} and not _math.isfinite(float(value)):
            values.append(value)
        else:
            values.append(_math.ceil(value))
    return asarray(values, dtype=dtype).reshape(array.shape)


def floor(x: Array) -> Array:
    array = asarray(x)
    dtype = DType(array.dtype)
    values = []
    for value in array.tolist():
        if dtype in {float32, float64} and not _math.isfinite(float(value)):
            values.append(value)
        else:
            values.append(_math.floor(value))
    return asarray(values, dtype=dtype).reshape(array.shape)


def clip(x: Array, /, min: Any | None = None, max: Any | None = None) -> Array:
    array = asarray(x)
    dtype = DType(array.dtype)
    min_array = None if min is None else asarray(min, dtype=dtype)
    max_array = None if max is None else asarray(max, dtype=dtype)
    shapes = [array.shape]
    if min_array is not None:
        shapes.append(min_array.shape)
    if max_array is not None:
        shapes.append(max_array.shape)
    out_shape = _broadcast_shape(*shapes)
    x_values = _flat_for_shape(array, out_shape)
    min_values = [None] * len(x_values) if min_array is None else _flat_for_shape(min_array, out_shape)
    max_values = [None] * len(x_values) if max_array is None else _flat_for_shape(max_array, out_shape)
    values = []
    for value, lower, upper in zip(x_values, min_values, max_values):
        if _isnan_scalar(value) or (lower is not None and _isnan_scalar(lower)) or (upper is not None and _isnan_scalar(upper)):
            values.append(_math.nan)
            continue
        clipped = value
        if lower is not None and clipped < lower:
            clipped = lower
        if upper is not None and clipped > upper:
            clipped = upper
        values.append(clipped)
    return asarray(values, dtype=dtype).reshape(out_shape)


def copysign(x1: Array, x2: Array) -> Array:
    left, right = broadcast_arrays(asarray(x1), asarray(x2))
    dtype = _promote_dtype_array_api(DType(left.dtype), DType(right.dtype))
    left = asarray(left, dtype=dtype)
    right = asarray(right, dtype=dtype)
    values = [_math.copysign(float(left_value), float(right_value)) for left_value, right_value in zip(left.tolist(), right.tolist())]
    return asarray(values, dtype=dtype).reshape(left.shape)


def floor_divide(x1: Array, x2: Any) -> Array:
    return x1 // x2


def hypot(x1: Array, x2: Array) -> Array:
    return _real_float_binary(x1, x2, _math.hypot)


def log(x: Array) -> Array:
    def real_log(value: float) -> float:
        if value == 0.0:
            return -_math.inf
        return _math.log(value)

    def complex_log(value: complex) -> complex:
        if value.real == 0.0 and value.imag == 0.0:
            angle = _math.pi if _math.copysign(1.0, value.real) < 0 else _math.copysign(0.0, value.imag)
            return complex(-_math.inf, angle)
        return _cmath.log(value)

    return _floating_unary(asarray(x), real_log, complex_log)


def log1p(x: Array) -> Array:
    def real_log1p(value: float) -> float:
        if value == -1.0:
            return -_math.inf
        return _math.log1p(value)

    def complex_log1p(value: complex) -> complex:
        shifted = complex(1.0 + value.real, value.imag)
        if shifted.real == 0.0 and shifted.imag == 0.0:
            return complex(-_math.inf, _math.copysign(0.0, shifted.imag))
        return _cmath.log(shifted)

    return _floating_unary(asarray(x), real_log1p, complex_log1p)


def log2(x: Array) -> Array:
    def real_log2(value: float) -> float:
        if value == 0.0:
            return -_math.inf
        return _math.log2(value)

    return _floating_unary(asarray(x), real_log2, lambda value: log(asarray(value)).tolist()[0] / _math.log(2))


def log10(x: Array) -> Array:
    def real_log10(value: float) -> float:
        if value == 0.0:
            return -_math.inf
        return _math.log10(value)

    return _floating_unary(asarray(x), real_log10, lambda value: log(asarray(value)).tolist()[0] / _math.log(10))


def logaddexp(x1: Array, x2: Array) -> Array:
    def op(left: float, right: float) -> float:
        if _math.isinf(left) or _math.isinf(right):
            return _builtins.max(left, right)
        pivot = _builtins.max(left, right)
        return pivot + _math.log(_math.exp(left - pivot) + _math.exp(right - pivot))

    return _real_float_binary(x1, x2, op)


def maximum(x1: Array, x2: Array) -> Array:
    return _real_float_binary(x1, x2, _builtins.max)


def minimum(x1: Array, x2: Array) -> Array:
    return _real_float_binary(x1, x2, _builtins.min)


def negative(x: Array) -> Array:
    return -x


def positive(x: Array) -> Array:
    return +x


def round(x: Array) -> Array:
    array = asarray(x)
    dtype = DType(array.dtype)
    values = []
    if dtype in {complex64, complex128}:
        for value in array.tolist():
            value = complex(value)
            values.append(complex(_round_real_scalar(value.real), _round_real_scalar(value.imag)))
    else:
        values = [_round_real_scalar(value) for value in array.tolist()]
    return asarray(values, dtype=dtype).reshape(array.shape)


def signbit(x: Array) -> Array:
    array = asarray(x)
    values = [_math.copysign(1.0, float(value)) < 0 for value in array.tolist()]
    return asarray(values, dtype=bool).reshape(array.shape)


def sign(x: Array) -> Array:
    array = asarray(x)
    dtype = DType(array.dtype)
    values = []
    for value in array.tolist():
        if dtype in {complex64, complex128}:
            value = complex(value)
            values.append(0j if value == 0 else value / _builtins.abs(value))
        else:
            if isinstance(value, float) and _math.isnan(value):
                values.append(value)
                continue
            values.append(0 if value == 0 else (1 if value > 0 else -1))
    return asarray(values, dtype=dtype).reshape(array.shape)


def sin(x: Array) -> Array:
    return _floating_unary(asarray(x), _math.sin, _cmath.sin)


def sinh(x: Array) -> Array:
    def complex_sinh(value: complex) -> complex:
        if value.real == 0.0 and _math.isinf(value.imag):
            return complex(_math.copysign(0.0, value.real), _math.nan)
        if _math.isinf(value.real) and not _math.isfinite(value.imag):
            return complex(_math.copysign(_math.inf, value.real), _math.nan)
        return _cmath.sinh(value)

    return _floating_unary(asarray(x), _math.sinh, complex_sinh)


def square(x: Array) -> Array:
    return x * x


def sqrt(x: Array) -> Array:
    return _floating_unary(asarray(x), _math.sqrt, _cmath.sqrt)


def tan(x: Array) -> Array:
    return _floating_unary(asarray(x), _math.tan, _cmath.tan)


def tanh(x: Array) -> Array:
    def complex_tanh(value: complex) -> complex:
        if value.real == 0.0 and (_math.isinf(value.imag) or _math.isnan(value.imag)):
            return complex(_math.copysign(0.0, value.real), _math.nan)
        if _math.isinf(value.real):
            return complex(_math.copysign(1.0, value.real), 0.0)
        return _cmath.tanh(value)

    return _floating_unary(asarray(x), _math.tanh, complex_tanh)


def trunc(x: Array) -> Array:
    array = asarray(x)
    dtype = DType(array.dtype)
    values = []
    for value in array.tolist():
        if dtype in {float32, float64} and not _math.isfinite(float(value)):
            values.append(value)
        else:
            values.append(_math.trunc(value))
    return asarray(values, dtype=dtype).reshape(array.shape)


def cos(x: Array) -> Array:
    return _floating_unary(asarray(x), _math.cos, _cmath.cos)


def cosh(x: Array) -> Array:
    def complex_cosh(value: complex) -> complex:
        if value.real == 0.0 and _math.isinf(value.imag):
            return complex(_math.nan, _math.copysign(0.0, value.real))
        if _math.isinf(value.real) and not _math.isfinite(value.imag):
            return complex(_math.inf, _math.nan)
        return _cmath.cosh(value)

    return _floating_unary(asarray(x), _math.cosh, complex_cosh)


def exp(x: Array) -> Array:
    def complex_exp(value: complex) -> complex:
        if _math.isinf(value.real) and not _math.isfinite(value.imag):
            if value.real > 0:
                return complex(_math.inf, _math.nan)
            return complex(0.0, 0.0)
        return _cmath.exp(value)

    return _floating_unary(asarray(x), _math.exp, complex_exp)


def expm1(x: Array) -> Array:
    def complex_expm1(value: complex) -> complex:
        real = value.real
        imag = value.imag
        if real == _math.inf:
            if imag == 0.0:
                return complex(_math.inf, _math.copysign(0.0, imag))
            if not _math.isfinite(imag):
                return complex(_math.inf, _math.nan)
        if real == -_math.inf:
            return complex(-1.0, 0.0)
        if _math.isnan(real) and imag == 0.0:
            return complex(_math.nan, _math.copysign(0.0, imag))
        if _math.isfinite(real) and not _math.isfinite(imag):
            return complex(_math.nan, _math.nan)
        return _math.expm1(real) * _math.cos(imag) - 2 * _math.sin(imag / 2) ** 2 + 1j * _math.exp(real) * _math.sin(imag)

    return _floating_unary(asarray(x), _math.expm1, complex_expm1)


def abs(x: Array) -> Array:
    if x.dtype == complex64:
        return asarray([_builtins.abs(complex(value)) for value in x.tolist()], dtype=float32).reshape(x.shape)
    if x.dtype == complex128:
        return asarray([_builtins.abs(complex(value)) for value in x.tolist()], dtype=float64).reshape(x.shape)
    return asarray([_builtins.abs(value) for value in x.tolist()], dtype=DType(x.dtype)).reshape(x.shape)


def acos(x: Array) -> Array:
    return _floating_unary(x, _math.acos, _cmath.acos)


def acosh(x: Array) -> Array:
    def complex_acosh(value: complex) -> complex:
        if value.real == 0 and _math.isnan(value.imag):
            return complex(_math.nan, _math.pi / 2)
        return _cmath.acosh(value)

    return _floating_unary(x, _math.acosh, complex_acosh)


def asin(x: Array) -> Array:
    return _floating_unary(x, _math.asin, _cmath.asin)


def asinh(x: Array) -> Array:
    return _floating_unary(x, _math.asinh, _cmath.asinh)


def atan(x: Array) -> Array:
    return _floating_unary(x, _math.atan, _cmath.atan)


def atan2(x1: Array, x2: Array) -> Array:
    left, right = broadcast_arrays(asarray(x1), asarray(x2))
    dtype = _promote_dtype_array_api(DType(left.dtype), DType(right.dtype))
    left = asarray(left, dtype=dtype)
    right = asarray(right, dtype=dtype)
    values = []
    for left_value, right_value in zip(left.tolist(), right.tolist()):
        try:
            values.append(_math.atan2(float(left_value), float(right_value)))
        except ValueError:
            values.append(_math.nan)
    return asarray(values, dtype=dtype).reshape(left.shape)


def atanh(x: Array) -> Array:
    def real_atanh(value: float) -> float:
        if value == -1.0:
            return -_math.inf
        if value == 1.0:
            return _math.inf
        return _math.atanh(value)

    def complex_atanh(value: complex) -> complex:
        if value.imag == 0.0:
            if value.real == -1.0:
                return complex(-_math.inf, _math.copysign(0.0, value.imag))
            if value.real == 1.0:
                return complex(_math.inf, _math.copysign(0.0, value.imag))
        return _cmath.atanh(value)

    return _floating_unary(x, real_atanh, complex_atanh)


def matmul(x1: Array, x2: Array) -> Array:
    dtype = _promote_dtype_array_api(DType(x1.dtype), DType(x2.dtype))
    return asarray(x1, dtype=dtype).matmul(asarray(x2, dtype=dtype))


def sum(
    x: Array,
    /,
    *,
    axis: int | Sequence[int] | None = None,
    dtype: str | DType | None = None,
    keepdims: bool = False,
) -> Array:
    array = asarray(x)
    target_dtype = _sum_dtype(DType(array.dtype)) if dtype is None else DType(dtype)
    return _stat_reduce(
        asarray(array, dtype=target_dtype),
        axis=axis,
        keepdims=keepdims,
        op=_builtins.sum,
        dtype=target_dtype,
    )


def cumulative_sum(
    x: Array,
    /,
    *,
    axis: int | None = None,
    dtype: str | DType | None = None,
    include_initial: bool = False,
) -> Array:
    array = asarray(x, dtype=_sum_dtype(DType(x.dtype)) if dtype is None else DType(dtype))
    if axis is None:
        values = array.tolist()
        out = []
        total = 0
        if include_initial:
            out.append(total)
        for value in values:
            total += value
            out.append(total)
        return asarray(out, dtype=DType(array.dtype))
    axis = _normalize_axis(axis, array.ndim)
    out_shape = list(array.shape)
    if include_initial:
        out_shape[axis] += 1
    out = [0] * _size_of_shape(out_shape)
    start = 1 if include_initial else 0
    for outer_idx in _ndindex(tuple(side for idx, side in enumerate(array.shape) if idx != axis)):
        total = 0
        if include_initial:
            target_idx = outer_idx[:axis] + (0,) + outer_idx[axis:]
            out[_row_major_offset(out_shape, target_idx)] = total
        for coord in range(array.shape[axis]):
            source_idx = outer_idx[:axis] + (coord,) + outer_idx[axis:]
            total += _scalar_at(array, source_idx)
            target_idx = outer_idx[:axis] + (coord + start,) + outer_idx[axis:]
            out[_row_major_offset(out_shape, target_idx)] = total
    return asarray(out, dtype=DType(array.dtype)).reshape(tuple(out_shape))


def max(x: Array, /, *, axis: int | Sequence[int] | None = None, keepdims: bool = False) -> Array:
    def maximum(values):
        if _builtins.any(_isnan_scalar(value) for value in values):
            return _math.nan
        return _builtins.max(values)

    return _stat_reduce(x, axis=axis, keepdims=keepdims, op=maximum, dtype=DType(x.dtype))


def mean(x: Array, /, *, axis: int | Sequence[int] | None = None, keepdims: bool = False) -> Array:
    dtype = _real_result_dtype(DType(x.dtype))
    def average(values):
        if not values:
            return _math.nan
        return _builtins.sum(values) / len(values)

    return _stat_reduce(x, axis=axis, keepdims=keepdims, op=average, dtype=dtype)


def min(x: Array, /, *, axis: int | Sequence[int] | None = None, keepdims: bool = False) -> Array:
    def minimum(values):
        if _builtins.any(_isnan_scalar(value) for value in values):
            return _math.nan
        return _builtins.min(values)

    return _stat_reduce(x, axis=axis, keepdims=keepdims, op=minimum, dtype=DType(x.dtype))


def prod(
    x: Array,
    /,
    *,
    axis: int | Sequence[int] | None = None,
    dtype: str | DType | None = None,
    keepdims: bool = False,
) -> Array:
    target_dtype = _sum_dtype(DType(x.dtype)) if dtype is None else DType(dtype)
    def product(values):
        total = 1
        for value in values:
            total *= value
        return total
    return _stat_reduce(asarray(x, dtype=target_dtype), axis=axis, keepdims=keepdims, op=product, dtype=target_dtype)


def std(
    x: Array,
    /,
    *,
    axis: int | Sequence[int] | None = None,
    correction: int | float = 0.0,
    keepdims: bool = False,
) -> Array:
    dtype = _real_result_dtype(DType(x.dtype))
    def stdev(values):
        n = len(values)
        if n <= correction:
            return _math.nan
        avg = _builtins.sum(values) / n
        return _math.sqrt(_builtins.sum((value - avg) ** 2 for value in values) / (n - correction))
    return _stat_reduce(asarray(x, dtype=dtype), axis=axis, keepdims=keepdims, op=stdev, dtype=dtype)


def var(
    x: Array,
    /,
    *,
    axis: int | Sequence[int] | None = None,
    correction: int | float = 0.0,
    keepdims: bool = False,
) -> Array:
    dtype = _real_result_dtype(DType(x.dtype))
    def variance(values):
        n = len(values)
        if n <= correction:
            return _math.nan
        avg = _builtins.sum(values) / n
        return _builtins.sum((value - avg) ** 2 for value in values) / (n - correction)
    return _stat_reduce(asarray(x, dtype=dtype), axis=axis, keepdims=keepdims, op=variance, dtype=dtype)


def all(
    x: Array,
    /,
    *,
    axis: int | Sequence[int] | None = None,
    keepdims: bool = False,
) -> Array:
    return _stat_reduce(asarray(x, dtype=bool), axis=axis, keepdims=keepdims, op=_builtins.all, dtype=bool)


def any(
    x: Array,
    /,
    *,
    axis: int | Sequence[int] | None = None,
    keepdims: bool = False,
) -> Array:
    return _stat_reduce(asarray(x, dtype=bool), axis=axis, keepdims=keepdims, op=_builtins.any, dtype=bool)


def argmax(x: Array, /, *, axis: int | None = None, keepdims: bool = False) -> Array:
    return _arg_reduce(x, axis=axis, keepdims=keepdims, choose_max=True)


def argmin(x: Array, /, *, axis: int | None = None, keepdims: bool = False) -> Array:
    return _arg_reduce(x, axis=axis, keepdims=keepdims, choose_max=False)


def nonzero(x: Array) -> tuple[Array, ...]:
    array = asarray(x)
    if array.ndim == 0:
        raise ValueError("nonzero is not defined for zero-dimensional arrays")
    indices = [[] for _ in range(array.ndim)]
    for idx in _ndindex(array.shape):
        value = _scalar_at(array, idx)
        if value != 0:
            for axis, coord in enumerate(idx):
                indices[axis].append(coord)
    return tuple(asarray(axis_indices, dtype=int64) for axis_indices in indices)


def sort(x: Array, /, *, axis: int = -1, descending: bool = False, stable: bool = True) -> Array:
    array = asarray(x)
    del stable
    axis = _normalize_axis(axis, array.ndim)
    out = [None] * _size_of_shape(array.shape)
    outer_shape = tuple(side for idx, side in enumerate(array.shape) if idx != axis)
    for outer_idx in _ndindex(outer_shape):
        line = []
        for coord in range(array.shape[axis]):
            source_idx = outer_idx[:axis] + (coord,) + outer_idx[axis:]
            line.append((coord, _scalar_at(array, source_idx)))
        for target_coord, (_, value) in enumerate(sorted(line, key=lambda item: item[1], reverse=descending)):
            target_idx = outer_idx[:axis] + (target_coord,) + outer_idx[axis:]
            out[_row_major_offset(array.shape, target_idx)] = value
    return asarray(out, dtype=DType(array.dtype)).reshape(array.shape)


def argsort(x: Array, /, *, axis: int = -1, descending: bool = False, stable: bool = True) -> Array:
    array = asarray(x)
    del stable
    axis = _normalize_axis(axis, array.ndim)
    out = [0] * _size_of_shape(array.shape)
    outer_shape = tuple(side for idx, side in enumerate(array.shape) if idx != axis)
    for outer_idx in _ndindex(outer_shape):
        line = []
        for coord in range(array.shape[axis]):
            source_idx = outer_idx[:axis] + (coord,) + outer_idx[axis:]
            line.append((coord, _scalar_at(array, source_idx)))
        for target_coord, (source_coord, _) in enumerate(sorted(line, key=lambda item: item[1], reverse=descending)):
            target_idx = outer_idx[:axis] + (target_coord,) + outer_idx[axis:]
            out[_row_major_offset(array.shape, target_idx)] = source_coord
    return asarray(out, dtype=int64).reshape(array.shape)


def searchsorted(x1: Array, x2: Any, /, *, side: str = "left", sorter: Array | None = None) -> Array:
    import bisect as _bisect

    haystack = asarray(x1).tolist()
    if sorter is not None:
        haystack = [haystack[int(idx)] for idx in asarray(sorter).tolist()]
    values = asarray(x2, dtype=DType(asarray(x1).dtype))
    func = _bisect.bisect_left if side == "left" else _bisect.bisect_right
    out = [func(haystack, value) for value in values.tolist()]
    return asarray(out, dtype=int64).reshape(values.shape)


def isinf(x: Any) -> bool | Array:
    if isinstance(x, Array):
        return asarray([_isinf_scalar(value) for value in x.tolist()], dtype=bool).reshape(x.shape)
    return _isinf_scalar(x)


def isnan(x: Any) -> bool | Array:
    if isinstance(x, Array):
        return asarray([_isnan_scalar(value) for value in x.tolist()], dtype=bool).reshape(x.shape)
    return _isnan_scalar(x)


def isfinite(x: Any) -> bool | Array:
    if isinstance(x, Array):
        return asarray([_isfinite_scalar(value) for value in x.tolist()], dtype=bool).reshape(x.shape)
    return _isfinite_scalar(x)


def where(condition: Array, x1: Any, x2: Any) -> Array:
    cond = condition if isinstance(condition, Array) else asarray(condition, dtype=bool)
    target_dtype = _result_dtype_for_where(x1, x2)
    left = x1 if isinstance(x1, Array) and x1.dtype == target_dtype else asarray(x1, dtype=target_dtype)
    right = x2 if isinstance(x2, Array) and x2.dtype == target_dtype else asarray(x2, dtype=target_dtype)
    out_shape = _broadcast_shape_for_where(cond, left, right)
    cond_values = _flat_for_shape(cond, out_shape)
    left_values = _flat_for_shape(left, out_shape)
    right_values = _flat_for_shape(right, out_shape)
    values = [left if keep else right for keep, left, right in zip(cond_values, left_values, right_values)]
    return asarray(values, dtype=target_dtype).reshape(out_shape)


def unique_all(x: Array):
    values, indices, inverse, counts = _unique_parts(asarray(x))
    return _SimpleNamespace(values=values, indices=indices, inverse_indices=inverse, counts=counts)


def unique_counts(x: Array):
    values, _indices, _inverse, counts = _unique_parts(asarray(x))
    return _SimpleNamespace(values=values, counts=counts)


def unique_inverse(x: Array):
    values, _indices, inverse, _counts = _unique_parts(asarray(x))
    return _SimpleNamespace(values=values, inverse_indices=inverse)


def unique_values(x: Array) -> Array:
    values, _indices, _inverse, _counts = _unique_parts(asarray(x))
    return values


def empty_like(x: Array, dtype: str | DType | None = None, *, device: str | None = None) -> Array:
    return empty(x.shape, dtype=_dtype_name(dtype, default=DType(x.dtype)), device=device)


def zeros_like(x: Array, dtype: str | DType | None = None, *, device: str | None = None) -> Array:
    return zeros(x.shape, dtype=_dtype_name(dtype, default=DType(x.dtype)), device=device)


def ones_like(x: Array, dtype: str | DType | None = None, *, device: str | None = None) -> Array:
    return ones(x.shape, dtype=_dtype_name(dtype, default=DType(x.dtype)), device=device)


def full_like(
    x: Array,
    fill_value: Any,
    dtype: str | DType | None = None,
    *,
    device: str | None = None,
) -> Array:
    return full(x.shape, fill_value, dtype=_dtype_name(dtype, default=DType(x.dtype)), device=device)


def eye(
    n_rows: int,
    n_cols: int | None = None,
    *,
    k: int = 0,
    dtype: str | DType = float64,
    device: str | None = None,
) -> Array:
    _check_device(device)
    cols = n_rows if n_cols is None else int(n_cols)
    data = [[1 if col - row == k else 0 for col in range(cols)] for row in range(int(n_rows))]
    flat = [value for row in data for value in row]
    return asarray(flat, dtype=_dtype_name(dtype, default=float64)).reshape((int(n_rows), cols))


def linspace(
    start: int | float,
    stop: int | float,
    num: int,
    *,
    dtype: str | DType = float64,
    endpoint: bool = True,
    device: str | None = None,
) -> Array:
    _check_device(device)
    num = int(num)
    dtype_name = _dtype_name(dtype, default=float64)
    if num <= 0:
        return asarray([], dtype=dtype_name)
    if num == 1:
        return asarray([start], dtype=dtype_name)
    denom = num - 1 if endpoint else num
    step = (stop - start) / denom
    values = [start + step * i for i in range(num)]
    if endpoint:
        values[-1] = stop
    return asarray(values, dtype=dtype_name)


def broadcast_to(x: Array, shape: Sequence[int]) -> Array:
    normalized = tuple(_normalize_shape(shape))
    return asarray(_broadcast_flat_values(x, normalized), dtype=DType(x.dtype)).reshape(normalized)


def broadcast_arrays(*arrays: Array) -> list[Array]:
    shape = _broadcast_shape(*(array.shape for array in arrays))
    return [broadcast_to(array, shape) for array in arrays]


def meshgrid(*arrays: Array, indexing: str = "xy") -> list[Array]:
    if indexing not in {"xy", "ij"}:
        raise ValueError("meshgrid indexing must be 'xy' or 'ij'")
    shape = [array.shape[0] for array in arrays]
    axes = list(range(len(arrays)))
    if len(arrays) > 1 and indexing == "xy":
        shape[0], shape[1] = shape[1], shape[0]
        axes[0], axes[1] = axes[1], axes[0]
    out = []
    for source_axis, array in enumerate(arrays):
        values = []
        for idx in _ndindex(shape):
            axis = axes.index(source_axis)
            values.append(array.tolist()[idx[axis]])
        out.append(asarray(values, dtype=DType(array.dtype)).reshape(shape))
    return out


def can_cast(_from: str | DType, to: str | DType) -> bool:
    _dtype_name(_from)
    _dtype_name(to)
    return True


def from_dlpack(x: Array, *, device: str | None = None, copy: bool | None = None) -> Array:
    _check_device(device)
    if hasattr(x, "__dlpack__"):
        x.__dlpack__(copy=copy)
    return asarray(x, copy=True if copy else None)


def real(x: Array) -> Array:
    if x.dtype == complex64:
        return asarray([complex(value).real for value in x.tolist()], dtype=float32).reshape(x.shape)
    if x.dtype == complex128:
        return asarray([complex(value).real for value in x.tolist()], dtype=float64).reshape(x.shape)
    return x


def imag(x: Array) -> Array:
    if x.dtype == complex64:
        return asarray([complex(value).imag for value in x.tolist()], dtype=float32).reshape(x.shape)
    if x.dtype == complex128:
        return asarray([complex(value).imag for value in x.tolist()], dtype=float64).reshape(x.shape)
    dtype = DType(x.dtype)
    return zeros(x.shape, dtype=dtype)


def conj(x: Array) -> Array:
    if x.dtype in {complex64, complex128}:
        return asarray([complex(value).conjugate() for value in x.tolist()], dtype=DType(x.dtype)).reshape(x.shape)
    return x


def stack(arrays: Sequence[Array], *, axis: int = 0) -> Array:
    arrays = [array if isinstance(array, Array) else asarray(array) for array in arrays]
    if not arrays:
        raise ValueError("stack requires at least one array")
    source_shape = tuple(arrays[0].shape)
    dtype = _promoted_dtype_for_arrays(arrays)
    arrays = [asarray(array, dtype=dtype) for array in arrays]
    if _builtins.any(tuple(array.shape) != source_shape for array in arrays):
        raise ValueError("all input arrays must have the same shape")
    ndim = len(source_shape) + 1
    axis = _normalize_axis(axis, ndim)
    out_shape = source_shape[:axis] + (len(arrays),) + source_shape[axis:]
    values = []
    for out_idx in _ndindex(out_shape):
        source_array = arrays[out_idx[axis]]
        source_idx = out_idx[:axis] + out_idx[axis + 1 :]
        values.append(_scalar_at(source_array, source_idx))
    return asarray(values, dtype=dtype).reshape(out_shape)


def concat(arrays: Sequence[Array], *, axis: int | None = 0) -> Array:
    arrays = [array if isinstance(array, Array) else asarray(array) for array in arrays]
    if not arrays:
        raise ValueError("concat requires at least one array")
    dtype = _promoted_dtype_for_arrays(arrays)
    arrays = [asarray(array, dtype=dtype) for array in arrays]
    if axis is None:
        values = []
        for array in arrays:
            values.extend(array.tolist())
        return asarray(values, dtype=dtype)
    axis = _normalize_axis(axis, arrays[0].ndim)
    out_shape = list(arrays[0].shape)
    out_shape[axis] = 0
    for array in arrays:
        if array.ndim != len(out_shape):
            raise ValueError("concat arrays must have matching ranks")
        for dim_axis, (left, right) in enumerate(zip(arrays[0].shape, array.shape)):
            if dim_axis != axis and left != right:
                raise ValueError("concat arrays must match outside the concatenation axis")
        out_shape[axis] += array.shape[axis]
    values = []
    starts = []
    current = 0
    for array in arrays:
        starts.append(current)
        current += array.shape[axis]
    for out_idx in _ndindex(out_shape):
        axis_coord = out_idx[axis]
        array_index = 0
        while array_index + 1 < len(arrays) and axis_coord >= starts[array_index + 1]:
            array_index += 1
        source_idx = list(out_idx)
        source_idx[axis] = axis_coord - starts[array_index]
        values.append(_scalar_at(arrays[array_index], tuple(source_idx)))
    return asarray(values, dtype=dtype).reshape(out_shape)


def repeat(x: Array, repeats: int | Array, *, axis: int | None = None) -> Array:
    source = reshape(x, (-1,)) if axis is None else x
    axis = 0 if axis is None else _normalize_axis(axis, source.ndim)
    axis_len = source.shape[axis]
    if isinstance(repeats, Array):
        repeat_values = [int(value) for value in repeats.tolist()]
        if len(repeat_values) == 1:
            repeat_values *= axis_len
        if len(repeat_values) != axis_len:
            raise ValueError("repeat counts must have length one or match the selected axis")
    else:
        repeat_values = [int(repeats)] * axis_len
    out_shape = list(source.shape)
    out_shape[axis] = _builtins.sum(repeat_values)
    cumulative = []
    current = 0
    for count in repeat_values:
        current += count
        cumulative.append(current)
    values = []
    for out_idx in _ndindex(out_shape):
        out_axis_coord = out_idx[axis]
        source_axis = 0
        while source_axis < len(cumulative) and out_axis_coord >= cumulative[source_axis]:
            source_axis += 1
        source_idx = list(out_idx)
        source_idx[axis] = source_axis
        values.append(_scalar_at(source, tuple(source_idx)))
    return asarray(values, dtype=DType(x.dtype)).reshape(out_shape)


def roll(x: Array, shift: int | Sequence[int], *, axis: int | Sequence[int] | None = None) -> Array:
    if axis is None:
        flat = reshape(x, (-1,))
        size = flat.shape[0]
        if size == 0:
            return flat.reshape(x.shape)
        amount = int(shift) % size
        values = [None] * size
        for source_index, value in enumerate(flat.tolist()):
            values[(source_index + amount) % size] = value
        return asarray(values, dtype=DType(x.dtype)).reshape(x.shape)
    axes = (_normalize_axis(axis, x.ndim),) if isinstance(axis, int) else tuple(_normalize_axis(item, x.ndim) for item in axis)
    shifts = (int(shift),) if isinstance(shift, int) else tuple(int(item) for item in shift)
    if len(axes) != len(shifts):
        raise ValueError("roll shift and axis must have the same length")
    values = [None] * x.size
    for source_idx in _ndindex(x.shape):
        out_idx = list(source_idx)
        for amount, axis_value in zip(shifts, axes):
            dim = x.shape[axis_value]
            if dim:
                out_idx[axis_value] = (out_idx[axis_value] + amount) % dim
        values[_row_major_offset(x.shape, tuple(out_idx))] = _scalar_at(x, source_idx)
    return asarray(values, dtype=DType(x.dtype)).reshape(x.shape)


def tile(x: Array, repetitions: Sequence[int]) -> Array:
    repetitions = tuple(int(value) for value in repetitions)
    if len(x.shape) > len(repetitions):
        source_shape = x.shape
        reps = (1,) * (len(x.shape) - len(repetitions)) + repetitions
    else:
        source_shape = (1,) * (len(repetitions) - len(x.shape)) + x.shape
        reps = repetitions
    reshaped = reshape(x, source_shape)
    out_shape = tuple(rep * side for rep, side in zip(reps, source_shape))
    values = []
    for out_idx in _ndindex(out_shape):
        source_idx = tuple(coord % side if side else 0 for coord, side in zip(out_idx, source_shape))
        values.append(_scalar_at(reshaped, source_idx))
    return asarray(values, dtype=DType(x.dtype)).reshape(out_shape)


def unstack(x: Array, *, axis: int = 0) -> tuple[Array, ...]:
    axis = _normalize_axis(axis, x.ndim)
    out = []
    for index in range(x.shape[axis]):
        key = [slice(None)] * x.ndim
        key[axis] = index
        out.append(x[tuple(key)])
    return tuple(out)


def take(x: Array, indices: Array, *, axis: int = 0) -> Array:
    return x.take_axis(indices, int(axis))


def matrix_transpose(x: Array) -> Array:
    return x.matrix_transpose()


def diagonal(x: Array, *, offset: int = 0) -> Array:
    return x.diagonal(int(offset))


def tril(x: Array, *, k: int = 0) -> Array:
    return _triangular(x, int(k), upper=False)


def triu(x: Array, *, k: int = 0) -> Array:
    return _triangular(x, int(k), upper=True)


def cross(x1: Array, x2: Array, *, axis: int = -1) -> Array:
    dtype = _promote_dtype_array_api(DType(x1.dtype), DType(x2.dtype))
    left, right = broadcast_arrays(asarray(x1, dtype=dtype), asarray(x2, dtype=dtype))
    axis = _normalize_axis(axis, left.ndim)
    if left.shape[axis] != 3 or right.shape[axis] != 3:
        raise ValueError("cross requires both inputs to have length 3 along axis")

    def component(array: Array, component_index: int) -> Array:
        key = [slice(None)] * array.ndim
        key[axis] = component_index
        return array[tuple(key)]

    a0, a1, a2 = (component(left, i) for i in range(3))
    b0, b1, b2 = (component(right, i) for i in range(3))
    return stack(
        [
            a1 * b2 - a2 * b1,
            a2 * b0 - a0 * b2,
            a0 * b1 - a1 * b0,
        ],
        axis=axis,
    )


def _linalg_cholesky(x: Array, *, upper: bool = False) -> Array:
    is_complex = isdtype(x.dtype, "complex floating")
    if not (isdtype(x.dtype, "real floating") or is_complex):
        raise TypeError("cholesky is currently implemented for floating and complex arrays")
    if x.ndim < 2 or x.shape[-1] != x.shape[-2]:
        raise ValueError("cholesky requires square matrices")
    rows = x.shape[-1]
    stack_shape = x.shape[:-2]
    values = []
    for stack_idx in _ndindex(stack_shape):
        matrix = [[complex(_scalar_at(x, stack_idx + (row, col))) for col in range(rows)] for row in range(rows)]
        factor = [[0j for _ in range(rows)] for _ in range(rows)]
        for row in range(rows):
            for col in range(row + 1):
                correction = _builtins.sum(factor[row][inner] * factor[col][inner].conjugate() for inner in range(col))
                if row == col:
                    value = (matrix[row][row] - correction).real
                    if value < 0 and abs(value) < 1e-10:
                        value = 0.0
                    if value < 0:
                        raise ValueError("matrix is not positive definite")
                    factor[row][col] = _math.sqrt(value)
                else:
                    factor[row][col] = (matrix[row][col] - correction) / factor[col][col]
        if upper:
            raw = (factor[col][row].conjugate() if col <= row else 0j for row in range(rows) for col in range(rows))
        else:
            raw = (factor[row][col] if col <= row else 0j for row in range(rows) for col in range(rows))
        if is_complex:
            values.extend(raw)
        else:
            values.extend(value.real for value in raw)
    return asarray(values, dtype=DType(x.dtype)).reshape(x.shape)


def _linalg_det(x: Array) -> Array:
    return x.det()


def _linalg_outer(x1: Array, x2: Array) -> Array:
    dtype = _promote_dtype_array_api(DType(x1.dtype), DType(x2.dtype))
    left = asarray(x1, dtype=dtype)
    right = asarray(x2, dtype=dtype)
    return reshape(left, (left.shape[0], 1)) * reshape(right, (1, right.shape[0]))


def _linalg_tensordot(x1: Array, x2: Array, axes: int | Sequence[Sequence[int]] = 2) -> Array:
    dtype = _promote_dtype_array_api(DType(x1.dtype), DType(x2.dtype))
    left = asarray(x1, dtype=dtype)
    right = asarray(x2, dtype=dtype)
    if isinstance(axes, int):
        left_axes = tuple(range(left.ndim - int(axes), left.ndim))
        right_axes = tuple(range(int(axes)))
    else:
        left_axes = tuple(_normalize_axis(axis, left.ndim) for axis in axes[0])
        right_axes = tuple(_normalize_axis(axis, right.ndim) for axis in axes[1])
    if len(left_axes) != len(right_axes):
        raise ValueError("tensordot axis lists must have the same length")
    for left_axis, right_axis in zip(left_axes, right_axes):
        if left.shape[left_axis] != right.shape[right_axis]:
            raise ValueError("tensordot contracted dimensions must match")
    left_keep = tuple(axis for axis in range(left.ndim) if axis not in left_axes)
    right_keep = tuple(axis for axis in range(right.ndim) if axis not in right_axes)
    out_shape = tuple(left.shape[axis] for axis in left_keep) + tuple(right.shape[axis] for axis in right_keep)
    contracted_shape = tuple(left.shape[axis] for axis in left_axes)
    values = []
    for out_idx in _ndindex(out_shape):
        left_keep_idx = out_idx[: len(left_keep)]
        right_keep_idx = out_idx[len(left_keep) :]
        total = 0j if dtype in {complex64, complex128} else 0
        for contract_idx in _ndindex(contracted_shape):
            left_idx = [0] * left.ndim
            right_idx = [0] * right.ndim
            for axis, coord in zip(left_keep, left_keep_idx):
                left_idx[axis] = coord
            for axis, coord in zip(right_keep, right_keep_idx):
                right_idx[axis] = coord
            for axis, coord in zip(left_axes, contract_idx):
                left_idx[axis] = coord
            for axis, coord in zip(right_axes, contract_idx):
                right_idx[axis] = coord
            total += _scalar_at(left, tuple(left_idx)) * _scalar_at(right, tuple(right_idx))
        values.append(total)
    return asarray(values, dtype=dtype).reshape(out_shape)


def _linalg_trace(x: Array, *, offset: int = 0, dtype: str | DType | None = None) -> Array:
    if x.ndim < 2:
        raise ValueError("trace requires at least two dimensions")
    rows, cols = x.shape[-2:]
    offset = int(offset)
    stack_shape = x.shape[:-2]
    result_dtype = _sum_dtype(DType(x.dtype)) if dtype is None else DType(dtype)
    values = []
    for stack_idx in _ndindex(stack_shape):
        total = 0j if DType(result_dtype) in {complex64, complex128} else 0
        if offset >= 0:
            diag_len = _builtins.min(rows, _builtins.max(cols - offset, 0))
            for diag in range(diag_len):
                total += _scalar_at(x, stack_idx + (diag, diag + offset))
        else:
            diag_len = _builtins.min(_builtins.max(rows + offset, 0), cols)
            for diag in range(diag_len):
                total += _scalar_at(x, stack_idx + (diag - offset, diag))
        values.append(total)
    return asarray(values, dtype=result_dtype).reshape(stack_shape)


def _linalg_vecdot(x1: Array, x2: Array, *, axis: int = -1) -> Array:
    dtype = _promote_dtype_array_api(DType(x1.dtype), DType(x2.dtype))
    out_broadcast_shape = _broadcast_shape(x1.shape, x2.shape)
    ndim = len(out_broadcast_shape)
    axis = _normalize_axis(axis, ndim)
    left_aligned = (1,) * (ndim - x1.ndim) + tuple(x1.shape)
    right_aligned = (1,) * (ndim - x2.ndim) + tuple(x2.shape)
    if left_aligned[axis] != right_aligned[axis]:
        raise ValueError("vecdot dimensions along axis must match")
    left, right = broadcast_arrays(asarray(x1, dtype=dtype), asarray(x2, dtype=dtype))
    out_shape = left.shape[:axis] + left.shape[axis + 1 :]
    values = []
    for out_idx in _ndindex(out_shape):
        total = 0j if dtype in {complex64, complex128} else 0
        for coord in range(left.shape[axis]):
            source_idx = out_idx[:axis] + (coord,) + out_idx[axis:]
            left_value = _scalar_at(left, source_idx)
            if dtype in {complex64, complex128}:
                left_value = complex(left_value).conjugate()
            total += left_value * _scalar_at(right, source_idx)
        values.append(total)
    return asarray(values, dtype=dtype).reshape(out_shape)


def _linalg_vector_norm(x: Array, *, axis: int | Sequence[int] | None = None, keepdims: bool = False, ord: Any = 2) -> Array:
    axes = set(_axes_or_all(axis, x.ndim))
    out_shape = tuple(1 if idx in axes else side for idx, side in enumerate(x.shape)) if keepdims else tuple(
        side for idx, side in enumerate(x.shape) if idx not in axes
    )
    reduced_shape = tuple(x.shape[idx] for idx in range(x.ndim) if idx in axes)
    values = []
    for out_idx in _ndindex(out_shape):
        magnitudes = []
        for reduced_idx in _ndindex(reduced_shape):
            source_idx = []
            out_pos = 0
            reduced_pos = 0
            for idx in range(x.ndim):
                if idx in axes:
                    source_idx.append(reduced_idx[reduced_pos])
                    reduced_pos += 1
                    if keepdims:
                        out_pos += 1
                else:
                    source_idx.append(out_idx[out_pos])
                    out_pos += 1
            magnitudes.append(_builtins.abs(_scalar_at(x, tuple(source_idx))))
        values.append(_vector_norm_value(magnitudes, ord))
    return asarray(values, dtype=_real_result_dtype(DType(x.dtype))).reshape(out_shape)


def tensordot(x1: Array, x2: Array, *, axes: int | Sequence[Sequence[int]] = 2) -> Array:
    return _linalg_tensordot(x1, x2, axes=axes)


def vecdot(x1: Array, x2: Array, *, axis: int = -1) -> Array:
    return _linalg_vecdot(x1, x2, axis=axis)


def _linalg_solve(x1: Array, x2: Array) -> Array:
    dtype = _promote_dtype_array_api(DType(x1.dtype), DType(x2.dtype))
    left = asarray(x1, dtype=dtype)
    right = asarray(x2, dtype=dtype)
    if left.ndim < 2 or left.shape[-1] != left.shape[-2]:
        raise ValueError("solve requires square coefficient matrices")
    size = left.shape[-1]
    if right.ndim == 1:
        if right.shape[0] != size:
            raise ValueError("solve vector RHS length must match matrix size")
        out_shape = left.shape[:-2] + (size,)
        values = []
        for stack_idx in _ndindex(left.shape[:-2]):
            matrix = [[complex(_scalar_at(left, stack_idx + (row, col))) for col in range(size)] for row in range(size)]
            rhs = [complex(_scalar_at(right, (row,))) for row in range(size)]
            solution = _matrix_vector_multiply(_matrix_inverse(matrix), rhs)
            values.extend(solution if dtype in {complex64, complex128} else (value.real for value in solution))
        return asarray(values, dtype=dtype).reshape(out_shape)
    if right.ndim < 2 or right.shape[-2] != size:
        raise ValueError("solve matrix RHS must have shape (..., M, K)")
    columns = right.shape[-1]
    stack_shape = _broadcast_shape(left.shape[:-2], right.shape[:-2])
    out_shape = stack_shape + (size, columns)
    values = []
    for stack_idx in _ndindex(stack_shape):
        left_idx = _broadcast_index(left.shape[:-2], stack_idx)
        right_idx = _broadcast_index(right.shape[:-2], stack_idx)
        matrix = [[complex(_scalar_at(left, left_idx + (row, col))) for col in range(size)] for row in range(size)]
        rhs = [[complex(_scalar_at(right, right_idx + (row, col))) for col in range(columns)] for row in range(size)]
        solution = _matrix_multiply_rect(_matrix_inverse(matrix), rhs)
        values.extend(_flatten_matrix_values(solution, complex_output=dtype in {complex64, complex128}))
    return asarray(values, dtype=dtype).reshape(out_shape)


def _linalg_inv(x: Array) -> Array:
    if x.ndim < 2 or x.shape[-1] != x.shape[-2]:
        raise ValueError("inv requires square matrices")
    if not (isdtype(x.dtype, "real floating") or isdtype(x.dtype, "complex floating")):
        raise TypeError("inv is currently implemented for floating and complex arrays")
    rows = x.shape[-1]
    is_complex = isdtype(x.dtype, "complex floating")
    values = []
    for stack_idx in _ndindex(x.shape[:-2]):
        matrix = [[complex(_scalar_at(x, stack_idx + (row, col))) for col in range(rows)] for row in range(rows)]
        inverse = _matrix_inverse(matrix)
        raw = (inverse[row][col] for row in range(rows) for col in range(rows))
        if is_complex:
            values.extend(raw)
        else:
            values.extend(value.real for value in raw)
    return asarray(values, dtype=DType(x.dtype)).reshape(x.shape)


def _linalg_matrix_power(x: Array, n: int) -> Array:
    if x.ndim < 2 or x.shape[-1] != x.shape[-2]:
        raise ValueError("matrix_power requires square matrices")
    if not (isdtype(x.dtype, "real floating") or isdtype(x.dtype, "complex floating")):
        raise TypeError("matrix_power is currently implemented for floating and complex arrays")
    n = int(n)
    rows = x.shape[-1]
    is_complex = isdtype(x.dtype, "complex floating")
    values = []
    for stack_idx in _ndindex(x.shape[:-2]):
        matrix = [[complex(_scalar_at(x, stack_idx + (row, col))) for col in range(rows)] for row in range(rows)]
        if n < 0:
            base = _matrix_inverse(matrix)
            exponent = -n
        else:
            base = matrix
            exponent = n
        result = _identity_matrix(rows)
        while exponent:
            if exponent & 1:
                result = _matrix_multiply(result, base)
            base = _matrix_multiply(base, base)
            exponent >>= 1
        raw = (result[row][col] for row in range(rows) for col in range(rows))
        if is_complex:
            values.extend(raw)
        else:
            values.extend(value.real for value in raw)
    return asarray(values, dtype=DType(x.dtype)).reshape(x.shape)


def _linalg_matrix_norm(x: Array, *, keepdims: bool = False, ord: Any = "fro") -> Array:
    if x.ndim < 2:
        raise ValueError("matrix_norm requires at least two dimensions")
    rows, cols = x.shape[-2:]
    out_shape = x.shape[:-2] + ((1, 1) if keepdims else ())
    values = []
    for stack_idx in _ndindex(x.shape[:-2]):
        matrix = [[complex(_scalar_at(x, stack_idx + (row, col))) for col in range(cols)] for row in range(rows)]
        values.append(_matrix_norm_value(matrix, ord))
    return asarray(values, dtype=_real_result_dtype(DType(x.dtype))).reshape(out_shape)


def _linalg_matrix_rank(x: Array, *, rtol: Any = None) -> Array:
    if x.ndim < 2:
        raise ValueError("matrix_rank requires at least two dimensions")
    rows, cols = x.shape[-2:]
    tolerance = 1e-7 if rtol is None else float(_first_scalar(rtol))
    values = []
    for stack_idx in _ndindex(x.shape[:-2]):
        matrix = [[complex(_scalar_at(x, stack_idx + (row, col))) for col in range(cols)] for row in range(rows)]
        values.append(_matrix_rank_value(matrix, tolerance))
    return asarray(values, dtype=int64).reshape(x.shape[:-2])


_SlogDet = _namedtuple("slogdet", ["sign", "logabsdet"])
_Eigh = _namedtuple("eigh", ["eigenvalues", "eigenvectors"])
_QR = _namedtuple("qr", ["Q", "R"])
_SVD = _namedtuple("svd", ["U", "S", "Vh"])


def _linalg_eigh(x: Array, /) -> _Eigh:
    array = asarray(x)
    if array.ndim < 2 or array.shape[-1] != array.shape[-2]:
        raise ValueError("eigh requires square matrices")
    size = array.shape[-1]
    dtype = DType(array.dtype)
    complex_output = dtype in {complex64, complex128}
    eigenvalue_values = []
    eigenvector_values = []
    for stack_idx in _ndindex(array.shape[:-2]):
        matrix = [[complex(_scalar_at(array, stack_idx + (row, col))) for col in range(size)] for row in range(size)]
        eigenvalues, eigenvectors = _symmetric_eigh(matrix)
        eigenvalue_values.extend(eigenvalues if not complex_output else (complex(value) for value in eigenvalues))
        eigenvector_values.extend(_flatten_matrix_values(eigenvectors, complex_output=complex_output))
    return _Eigh(
        asarray(eigenvalue_values, dtype=dtype).reshape(array.shape[:-1]),
        asarray(eigenvector_values, dtype=dtype).reshape(array.shape),
    )


def _linalg_eigvalsh(x: Array, /) -> Array:
    return _linalg_eigh(x).eigenvalues


def _linalg_slogdet(x: Array) -> _SlogDet:
    if x.ndim < 2 or x.shape[-1] != x.shape[-2]:
        raise ValueError("slogdet requires square matrices")
    if not (isdtype(x.dtype, "real floating") or isdtype(x.dtype, "complex floating")):
        raise TypeError("slogdet is currently implemented for floating and complex arrays")
    rows = x.shape[-1]
    is_complex = isdtype(x.dtype, "complex floating")
    signs = []
    logabs = []
    for stack_idx in _ndindex(x.shape[:-2]):
        matrix = [[complex(_scalar_at(x, stack_idx + (row, col))) for col in range(rows)] for row in range(rows)]
        det = _matrix_determinant(matrix)
        magnitude = _builtins.abs(det)
        if magnitude == 0:
            sign = 0j if is_complex else 0.0
            log_value = -_math.inf
        else:
            sign_value = det / magnitude
            sign = sign_value if is_complex else float(sign_value.real)
            log_value = _math.log(magnitude)
        signs.append(sign)
        logabs.append(log_value)
    shape = x.shape[:-2]
    return _SlogDet(
        asarray(signs, dtype=DType(x.dtype)).reshape(shape),
        asarray(logabs, dtype=_real_result_dtype(DType(x.dtype))).reshape(shape),
    )


def _linalg_pinv(x: Array, /, *, rtol: Any = None) -> Array:
    array = asarray(x)
    if array.ndim < 2:
        raise ValueError("pinv requires at least two dimensions")
    rows, cols = array.shape[-2:]
    dtype = DType(array.dtype)
    complex_output = dtype in {complex64, complex128}
    eps = 1.1920929e-7 if dtype in {float32, complex64} else 2.220446049250313e-16
    tolerance_scale = float(_first_scalar(rtol)) if rtol is not None else _builtins.max(rows, cols) * eps
    values = []
    for stack_idx in _ndindex(array.shape[:-2]):
        matrix = [[complex(_scalar_at(array, stack_idx + (row, col))) for col in range(cols)] for row in range(rows)]
        if rows == 0 or cols == 0:
            pseudo = [[0j for _ in range(rows)] for _ in range(cols)]
        else:
            u, singular_values, vh = _svd_decomposition(matrix, full_matrices=False)
            v = _matrix_conj_transpose(vh)
            uh = _matrix_conj_transpose(u)
            max_singular = _builtins.max(singular_values, default=0.0)
            diagonal = [
                [((1.0 / singular_values[row]) if row == col and singular_values[row] > tolerance_scale * max_singular else 0.0) + 0j for col in range(len(singular_values))]
                for row in range(len(singular_values))
            ]
            pseudo = _matrix_multiply_rect(_matrix_multiply_rect(v, diagonal), uh)
        values.extend(_flatten_matrix_values(pseudo, complex_output=complex_output))
    return asarray(values, dtype=dtype).reshape(array.shape[:-2] + (cols, rows))


def _linalg_qr(x: Array, /, *, mode: str = "reduced") -> tuple[Array, Array]:
    if mode not in {"reduced", "complete"}:
        raise ValueError("qr mode must be 'reduced' or 'complete'")
    array = asarray(x)
    if array.ndim < 2:
        raise ValueError("qr requires at least two dimensions")
    rows, cols = array.shape[-2:]
    k = _builtins.min(rows, cols)
    complete = mode == "complete"
    q_shape = array.shape[:-2] + (rows, rows if complete else k)
    r_shape = array.shape[:-2] + (rows if complete else k, cols)
    dtype = DType(array.dtype)
    complex_output = dtype in {complex64, complex128}
    q_values = []
    r_values = []
    for stack_idx in _ndindex(array.shape[:-2]):
        matrix = [[complex(_scalar_at(array, stack_idx + (row, col))) for col in range(cols)] for row in range(rows)]
        if rows == 0:
            q, r = [], []
        else:
            q, r = _qr_decomposition(matrix, complete=complete)
        q_values.extend(_flatten_matrix_values(q, complex_output=complex_output))
        r_values.extend(_flatten_matrix_values(r, complex_output=complex_output))
    return _QR(asarray(q_values, dtype=dtype).reshape(q_shape), asarray(r_values, dtype=dtype).reshape(r_shape))


def _linalg_svd(x: Array, /, *, full_matrices: bool = True) -> tuple[Array, Array, Array]:
    array = asarray(x)
    if array.ndim < 2:
        raise ValueError("svd requires at least two dimensions")
    rows, cols = array.shape[-2:]
    k = _builtins.min(rows, cols)
    dtype = DType(array.dtype)
    real_dtype = _real_result_dtype(dtype)
    complex_output = dtype in {complex64, complex128}
    u_shape = array.shape[:-2] + (rows, rows if full_matrices else k)
    s_shape = array.shape[:-2] + (k,)
    vh_shape = array.shape[:-2] + (cols if full_matrices else k, cols)
    u_values = []
    s_values = []
    vh_values = []
    for stack_idx in _ndindex(array.shape[:-2]):
        matrix = [[complex(_scalar_at(array, stack_idx + (row, col))) for col in range(cols)] for row in range(rows)]
        if rows == 0 or cols == 0:
            u = [[0j for _ in range(u_shape[-1])] for _ in range(rows)]
            singular_values: list[float] = []
            vh = [[0j for _ in range(cols)] for _ in range(vh_shape[-2])]
        else:
            u, singular_values, vh = _svd_decomposition(matrix, full_matrices=full_matrices)
        u_values.extend(_flatten_matrix_values(u, complex_output=complex_output))
        s_values.extend(singular_values)
        vh_values.extend(_flatten_matrix_values(vh, complex_output=complex_output))
    return _SVD(
        asarray(u_values, dtype=dtype).reshape(u_shape),
        asarray(s_values, dtype=real_dtype).reshape(s_shape),
        asarray(vh_values, dtype=dtype).reshape(vh_shape),
    )


def _linalg_svdvals(x: Array, /) -> Array:
    return _linalg_svd(x, full_matrices=False).S


def _fft_fft(x: Array, n: int | None = None, axis: int = -1, norm: str = "backward") -> Array:
    return x.fft_axis(None if n is None else int(n), int(axis), False, norm)


def _fft_ifft(x: Array, n: int | None = None, axis: int = -1, norm: str = "backward") -> Array:
    return x.fft_axis(None if n is None else int(n), int(axis), True, norm)


def _fft_fftn(
    x: Array,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: str = "backward",
) -> Array:
    sizes, normalized_axes, _ = _fft_axes_and_sizes(x, s, axes)
    out = x
    for size, axis in zip(sizes, normalized_axes):
        out = out.fft_axis(size, axis, False, norm)
    return out


def _fft_ifftn(
    x: Array,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: str = "backward",
) -> Array:
    sizes, normalized_axes, _ = _fft_axes_and_sizes(x, s, axes)
    out = x
    for size, axis in zip(sizes, normalized_axes):
        out = out.fft_axis(size, axis, True, norm)
    return out


def _fft_rfft(x: Array, n: int | None = None, axis: int = -1, norm: str = "backward") -> Array:
    return x.rfft_axis(None if n is None else int(n), int(axis), norm)


def _fft_irfft(x: Array, n: int | None = None, axis: int = -1, norm: str = "backward") -> Array:
    return x.irfft_axis(None if n is None else int(n), int(axis), norm)


def _fft_rfftn(
    x: Array,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: str = "backward",
) -> Array:
    sizes, normalized_axes, _ = _fft_axes_and_sizes(x, s, axes)
    if not normalized_axes:
        return x
    out = x.rfft_axis(sizes[-1], normalized_axes[-1], norm)
    for size, axis in zip(sizes[:-1], normalized_axes[:-1]):
        out = out.fft_axis(size, axis, False, norm)
    return out


def _fft_irfftn(
    x: Array,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: str = "backward",
) -> Array:
    sizes, normalized_axes, explicit_sizes = _fft_axes_and_sizes(x, s, axes)
    if not normalized_axes:
        return x
    out = x
    for size, axis in zip(sizes[:-1], normalized_axes[:-1]):
        out = out.fft_axis(size, axis, True, norm)
    last_size = sizes[-1] if explicit_sizes else None
    return out.irfft_axis(last_size, normalized_axes[-1], norm)


def _fft_hfft(x: Array, n: int | None = None, axis: int = -1, norm: str = "backward") -> Array:
    return x.irfft_axis(None if n is None else int(n), int(axis), norm)


def _fft_ihfft(x: Array, n: int | None = None, axis: int = -1, norm: str = "backward") -> Array:
    return x.rfft_axis(None if n is None else int(n), int(axis), norm)


def _fft_fftfreq(n: int, d: float = 1.0, dtype: str | DType | None = None, *, device: str | None = None) -> Array:
    _check_device(device)
    return _numrust.fftfreq(int(n), float(d), _dtype_name(dtype, default=float64))


def _fft_rfftfreq(n: int, d: float = 1.0, dtype: str | DType | None = None, *, device: str | None = None) -> Array:
    _check_device(device)
    return _numrust.rfftfreq(int(n), float(d), _dtype_name(dtype, default=float64))


def _fft_fftshift(x: Array, axes: int | Sequence[int] | None = None) -> Array:
    return x.fft_shift(_shift_axes(axes), False)


def _fft_ifftshift(x: Array, axes: int | Sequence[int] | None = None) -> Array:
    return x.fft_shift(_shift_axes(axes), True)


def _fft_axes_and_sizes(
    x: Array,
    sizes: Sequence[int] | None,
    axes: Sequence[int] | None,
) -> tuple[tuple[int, ...], tuple[int, ...], bool]:
    if axes is None:
        normalized_axes = tuple(range(x.ndim))
    else:
        normalized_axes = tuple(_normalize_axis(axis, x.ndim) for axis in axes)
    if sizes is None:
        normalized_sizes = tuple(int(x.shape[axis]) for axis in normalized_axes)
        return normalized_sizes, normalized_axes, False
    normalized_sizes = tuple(int(size) for size in sizes)
    if len(normalized_sizes) != len(normalized_axes):
        raise ValueError("FFT size and axis lists must have the same length")
    return normalized_sizes, normalized_axes, True


def _shift_axes(axes: int | Sequence[int] | None) -> list[int] | None:
    if axes is None:
        return None
    if isinstance(axes, int):
        return [axes]
    return [int(axis) for axis in axes]


class _Info:
    def __init__(self, **values: float | int) -> None:
        self.__dict__.update(values)

    def __repr__(self) -> str:
        values = ", ".join(f"{key}={value!r}" for key, value in self.__dict__.items())
        return f"numrust.info({values})"


class _DTypeValue:
    def __init__(self, dtype: DType) -> None:
        self.dtype = dtype

    def __eq__(self, other: object) -> bool:
        return self.dtype == other

    def __hash__(self) -> int:
        return hash(self.dtype)

    def __repr__(self) -> str:
        return repr(self.dtype)


def iinfo(dtype: str | DType | Array) -> _Info:
    if isinstance(dtype, Array):
        dtype = DType(dtype.dtype)
    dtype_name = _dtype_name(dtype)
    ranges = {
        int8: (-128, 127, 8),
        int16: (-32768, 32767, 16),
        int32: (-2147483648, 2147483647, 32),
        int64: (-9223372036854775808, 9223372036854775807, 64),
        uint8: (0, 255, 8),
        uint16: (0, 65535, 16),
        uint32: (0, 4294967295, 32),
        uint64: (0, 18446744073709551615, 64),
    }
    if dtype_name not in ranges:
        raise TypeError(f"iinfo is only defined for integer dtypes, got {dtype!r}")
    min_value, max_value, bits = ranges[DType(dtype_name)]
    return _Info(min=min_value, max=max_value, bits=bits, dtype=DType(dtype_name))


def finfo(dtype: str | DType | Array) -> _Info:
    if isinstance(dtype, Array):
        dtype = DType(dtype.dtype)
    dtype_name = _dtype_name(dtype)
    if dtype_name in {float32, complex64}:
        return _Info(
            bits=32,
            dtype=_DTypeValue(float32),
            eps=1.1920928955078125e-07,
            max=3.4028234663852886e38,
            min=-3.4028234663852886e38,
            smallest_normal=1.1754943508222875e-38,
        )
    if dtype_name in {float64, complex128}:
        return _Info(
            bits=64,
            dtype=_DTypeValue(float64),
            eps=2.220446049250313e-16,
            max=1.7976931348623157e308,
            min=-1.7976931348623157e308,
            smallest_normal=2.2250738585072014e-308,
        )
    raise TypeError(f"finfo is only defined for floating dtypes, got {dtype!r}")


class _NamespaceInfo:
    def capabilities(self) -> dict[str, bool]:
        return {"boolean indexing": False, "data-dependent shapes": False}

    def devices(self) -> list[str]:
        return ["cpu"]

    def default_device(self) -> str:
        return "cpu"

    def default_dtypes(self, *, device: str | None = None) -> dict[str, DType]:
        _check_device(device)
        return {
            "real floating": float64,
            "complex floating": complex128,
            "integral": int64,
            "indexing": int64,
        }

    def dtypes(self, kind: str | tuple[str, ...] | None = None, *, device: str | None = None) -> dict[str, DType]:
        _check_device(device)
        if kind is None:
            names = _PRIMITIVE_DTYPES | {complex64, complex128}
        elif isinstance(kind, tuple):
            names = set().union(*(_KIND_TO_DTYPES[item] for item in kind))
        else:
            names = _KIND_TO_DTYPES[kind]
        return {str(dtype): dtype for dtype in names}


def __array_namespace_info__() -> _NamespaceInfo:
    return _NamespaceInfo()


def isdtype(dtype: str | DType, kind: str | DType | tuple[str | DType, ...]) -> bool:
    dtype_name = _dtype_name(dtype)
    if isinstance(kind, tuple):
        return _builtins.any(isdtype(dtype_name, candidate) for candidate in kind)
    if isinstance(kind, str) and kind in _KIND_TO_DTYPES:
        return dtype_name in _KIND_TO_DTYPES[kind]
    return dtype_name == _dtype_name(kind)


def _normalize_shape(shape: int | Sequence[int]) -> list[int]:
    if isinstance(shape, int):
        return [shape]
    return [int(dim) for dim in shape]


def _size_of_shape(shape: Sequence[int]) -> int:
    size = 1
    for dim in shape:
        size *= int(dim)
    return size


def _check_device(device: str | None) -> None:
    if device not in (None, "cpu"):
        raise ValueError(f"unsupported device {device!r}; only 'cpu' is available")


def _normalize_axis(axis: int, ndim: int) -> int:
    axis = int(axis)
    if axis < 0:
        axis += ndim
    if axis < 0 or axis >= ndim:
        raise ValueError(f"axis {axis} is out of bounds for array of dimension {ndim}")
    return axis


def _axes_or_all(axis: int | Sequence[int] | None, ndim: int) -> tuple[int, ...]:
    if axis is None:
        return tuple(range(ndim))
    if isinstance(axis, int):
        return (_normalize_axis(axis, ndim),)
    return tuple(_normalize_axis(item, ndim) for item in axis)


def _scalar_at(array: Array, index: Sequence[int]) -> Any:
    return array[tuple(index)].tolist()[0]


def _first_scalar(value: Any) -> Any:
    if isinstance(value, Array):
        values = value.tolist()
        if not values:
            return 0
        return values[0]
    return value


def _triangular(x: Array, k: int, *, upper: bool) -> Array:
    if x.ndim < 2:
        raise ValueError("triangular extraction requires at least two dimensions")
    values = []
    for idx in _ndindex(x.shape):
        row, col = idx[-2], idx[-1]
        keep = col - row >= k if upper else col - row <= k
        values.append(_scalar_at(x, idx) if keep else 0)
    return asarray(values, dtype=DType(x.dtype)).reshape(x.shape)


def _floating_unary(x: Array, real_op, complex_op) -> Array:
    dtype = DType(x.dtype)
    if dtype in {complex64, complex128}:
        values = []
        for value in x.tolist():
            try:
                values.append(complex_op(complex(value)))
            except ValueError:
                values.append(complex(_math.nan, _math.nan))
            except OverflowError:
                values.append(complex(_math.inf, _math.inf))
        return asarray(values, dtype=dtype).reshape(x.shape)
    values = []
    for value in x.tolist():
        try:
            values.append(real_op(float(value)))
        except ValueError:
            values.append(_math.nan)
        except OverflowError:
            values.append(_math.inf)
    return asarray(values, dtype=dtype).reshape(x.shape)


def _real_float_binary(x1: Array, x2: Array, op) -> Array:
    left, right = broadcast_arrays(asarray(x1), asarray(x2))
    dtype = _promote_dtype_array_api(DType(left.dtype), DType(right.dtype))
    left = asarray(left, dtype=dtype)
    right = asarray(right, dtype=dtype)
    values = []
    for left_value, right_value in zip(left.tolist(), right.tolist()):
        try:
            values.append(op(float(left_value), float(right_value)))
        except ValueError:
            values.append(_math.nan)
        except OverflowError:
            values.append(_math.inf)
    return asarray(values, dtype=dtype).reshape(left.shape)


def _round_real_scalar(value: Any) -> Any:
    value = float(value) if isinstance(value, float) else value
    if isinstance(value, float) and value == 0.0:
        return value
    if isinstance(value, float) and not _math.isfinite(value):
        return value
    return _builtins.round(value)


def _logical_binary(x1: Array, x2: Array, op) -> Array:
    left, right = broadcast_arrays(asarray(x1, dtype=bool), asarray(x2, dtype=bool))
    values = [
        op(_builtins.bool(left_value), _builtins.bool(right_value))
        for left_value, right_value in zip(left.tolist(), right.tolist())
    ]
    return asarray(values, dtype=bool).reshape(left.shape)


def _identity_matrix(size: int) -> list[list[complex]]:
    return [[1 + 0j if row == col else 0j for col in range(size)] for row in range(size)]


def _matrix_multiply(left: list[list[complex]], right: list[list[complex]]) -> list[list[complex]]:
    size = len(left)
    return [
        [_builtins.sum(left[row][inner] * right[inner][col] for inner in range(size)) for col in range(size)]
        for row in range(size)
    ]


def _matrix_inverse(matrix: list[list[complex]]) -> list[list[complex]]:
    size = len(matrix)
    work = [row[:] + ident_row[:] for row, ident_row in zip(matrix, _identity_matrix(size))]
    for col in range(size):
        pivot = _builtins.max(range(col, size), key=lambda row: _builtins.abs(work[row][col]))
        if work[pivot][col] == 0:
            raise ValueError("matrix is singular")
        if pivot != col:
            work[col], work[pivot] = work[pivot], work[col]
        pivot_value = work[col][col]
        work[col] = [value / pivot_value for value in work[col]]
        for row in range(size):
            if row == col:
                continue
            factor = work[row][col]
            work[row] = [value - factor * basis_value for value, basis_value in zip(work[row], work[col])]
    return [row[size:] for row in work]


def _matrix_determinant(matrix: list[list[complex]]) -> complex:
    size = len(matrix)
    if size == 0:
        return 1 + 0j
    work = [row[:] for row in matrix]
    det = 1 + 0j
    for col in range(size):
        pivot = _builtins.max(range(col, size), key=lambda row: _builtins.abs(work[row][col]))
        if work[pivot][col] == 0:
            return 0j
        if pivot != col:
            work[col], work[pivot] = work[pivot], work[col]
            det = -det
        pivot_value = work[col][col]
        det *= pivot_value
        for row in range(col + 1, size):
            factor = work[row][col] / pivot_value
            for inner in range(col + 1, size):
                work[row][inner] -= factor * work[col][inner]
    return det


def _matrix_norm_value(matrix: list[list[complex]], ord: Any) -> float:
    if not matrix or not matrix[0]:
        return 0.0
    rows = len(matrix)
    cols = len(matrix[0])
    if ord in (None, "fro", 2):
        return _frobenius_norm(matrix)
    if ord == 1:
        return _builtins.max(_builtins.sum(_builtins.abs(matrix[row][col]) for row in range(rows)) for col in range(cols))
    if ord == -1:
        return _builtins.min(_builtins.sum(_builtins.abs(matrix[row][col]) for row in range(rows)) for col in range(cols))
    if ord == _math.inf:
        return _builtins.max(_builtins.sum(_builtins.abs(value) for value in row) for row in matrix)
    if ord == -_math.inf:
        return _builtins.min(_builtins.sum(_builtins.abs(value) for value in row) for row in matrix)
    return _frobenius_norm(matrix)


def _vector_norm_value(values: Sequence[float], ord: Any) -> float:
    if not values:
        return 0.0
    if ord in (None, 2):
        return _math.sqrt(_builtins.sum(value * value for value in values))
    if ord == 0:
        return float(_builtins.sum(1 for value in values if value != 0))
    if ord == 1:
        return _builtins.sum(values)
    if ord == _math.inf:
        return _builtins.max(values)
    if ord == -_math.inf:
        return _builtins.min(values)
    power = float(ord)
    if power < 0 and _builtins.any(value == 0 for value in values):
        return 0.0
    try:
        total = _builtins.sum(value**power for value in values)
        if total == 0:
            return 0.0
        return total ** (1.0 / power)
    except OverflowError:
        return _math.inf


def _frobenius_norm(matrix: list[list[complex]]) -> float:
    norm = 0.0
    for row in matrix:
        for value in row:
            norm = _math.hypot(norm, _builtins.abs(value))
    return norm


def _matrix_rank_value(matrix: list[list[complex]], tolerance: float) -> int:
    if not matrix:
        return 0
    work = [row[:] for row in matrix]
    rows = len(work)
    cols = len(work[0])
    if cols == 0:
        return 0
    rank = 0
    for col in range(cols):
        pivot = _builtins.max(range(rank, rows), key=lambda row: _builtins.abs(work[row][col]), default=rank)
        if rank >= rows or _builtins.abs(work[pivot][col]) <= tolerance:
            continue
        work[rank], work[pivot] = work[pivot], work[rank]
        pivot_value = work[rank][col]
        for inner in range(col, cols):
            work[rank][inner] /= pivot_value
        for row in range(rows):
            if row == rank:
                continue
            factor = work[row][col]
            for inner in range(col, cols):
                work[row][inner] -= factor * work[rank][inner]
        rank += 1
        if rank == rows:
            break
    return rank


def _matrix_conj_transpose(matrix: list[list[complex]]) -> list[list[complex]]:
    if not matrix:
        return []
    return [[matrix[row][col].conjugate() for row in range(len(matrix))] for col in range(len(matrix[0]))]


def _matrix_multiply_rect(left: list[list[complex]], right: list[list[complex]]) -> list[list[complex]]:
    rows = len(left)
    inner = len(left[0]) if rows else 0
    cols = len(right[0]) if right else 0
    return [
        [_builtins.sum(left[row][k] * right[k][col] for k in range(inner)) for col in range(cols)]
        for row in range(rows)
    ]


def _matrix_vector_multiply(matrix: list[list[complex]], vector: Sequence[complex]) -> list[complex]:
    return [_builtins.sum(value * vector[col] for col, value in enumerate(row)) for row in matrix]


def _dot_conj(left: Sequence[complex], right: Sequence[complex]) -> complex:
    return _builtins.sum(a.conjugate() * b for a, b in zip(left, right))


def _vector_norm_complex(values: Sequence[complex]) -> float:
    return _math.sqrt(_builtins.sum(_builtins.abs(value) ** 2 for value in values))


def _orthogonal_vector(existing: Sequence[Sequence[complex]], size: int) -> list[complex]:
    for basis in range(size):
        candidate = [0j] * size
        candidate[basis] = 1 + 0j
        for q in existing:
            coeff = _dot_conj(q, candidate)
            candidate = [value - coeff * q[row] for row, value in enumerate(candidate)]
        norm = _vector_norm_complex(candidate)
        if norm > 1e-12:
            return [value / norm for value in candidate]
    return [0j] * size


def _qr_decomposition(matrix: list[list[complex]], *, complete: bool) -> tuple[list[list[complex]], list[list[complex]]]:
    rows = len(matrix)
    cols = len(matrix[0]) if rows else 0
    reduced_cols = _builtins.min(rows, cols)
    q_cols: list[list[complex]] = []
    r_rows = rows if complete else reduced_cols
    r = [[0j for _ in range(cols)] for _ in range(r_rows)]
    for col in range(cols):
        column = [matrix[row][col] for row in range(rows)]
        work = column[:]
        for q_idx, q_col in enumerate(q_cols):
            coeff = _dot_conj(q_col, column)
            if q_idx < r_rows:
                r[q_idx][col] = coeff
            work = [value - coeff * q_col[row] for row, value in enumerate(work)]
        if len(q_cols) < reduced_cols:
            norm = _vector_norm_complex(work)
            if norm <= 1e-12:
                q_col = _orthogonal_vector(q_cols, rows)
                norm = 0.0
            else:
                q_col = [value / norm for value in work]
            q_cols.append(q_col)
            if len(q_cols) - 1 < r_rows:
                r[len(q_cols) - 1][col] = norm
    while complete and len(q_cols) < rows:
        q_cols.append(_orthogonal_vector(q_cols, rows))
    q = [[q_cols[col][row] for col in range(len(q_cols))] for row in range(rows)]
    return q, r


def _symmetric_eigh(matrix: list[list[complex]]) -> tuple[list[float], list[list[complex]]]:
    size = len(matrix)
    if size == 0:
        return [], []
    work = [[float(matrix[row][col].real) for col in range(size)] for row in range(size)]
    vectors = [[1.0 if row == col else 0.0 for col in range(size)] for row in range(size)]
    for _ in range(80 * size * size):
        pivot_row, pivot_col = 0, 1 if size > 1 else 0
        pivot_value = 0.0
        for row in range(size):
            for col in range(row + 1, size):
                value = _builtins.abs(work[row][col])
                if value > pivot_value:
                    pivot_row, pivot_col, pivot_value = row, col, value
        if pivot_value <= 1e-10:
            break
        app = work[pivot_row][pivot_row]
        aqq = work[pivot_col][pivot_col]
        apq = work[pivot_row][pivot_col]
        angle = 0.5 * _math.atan2(2.0 * apq, aqq - app)
        cosine = _math.cos(angle)
        sine = _math.sin(angle)
        for idx in range(size):
            aip = work[idx][pivot_row]
            aiq = work[idx][pivot_col]
            work[idx][pivot_row] = cosine * aip - sine * aiq
            work[idx][pivot_col] = sine * aip + cosine * aiq
        for idx in range(size):
            api = work[pivot_row][idx]
            aqi = work[pivot_col][idx]
            work[pivot_row][idx] = cosine * api - sine * aqi
            work[pivot_col][idx] = sine * api + cosine * aqi
            vip = vectors[idx][pivot_row]
            viq = vectors[idx][pivot_col]
            vectors[idx][pivot_row] = cosine * vip - sine * viq
            vectors[idx][pivot_col] = sine * vip + cosine * viq
    pairs = sorted((work[idx][idx], idx) for idx in range(size))
    eigenvalues = [value for value, _ in pairs]
    eigenvectors = [[complex(vectors[row][idx]) for _, idx in pairs] for row in range(size)]
    return eigenvalues, eigenvectors


def _svd_decomposition(
    matrix: list[list[complex]],
    *,
    full_matrices: bool,
) -> tuple[list[list[complex]], list[float], list[list[complex]]]:
    rows = len(matrix)
    cols = len(matrix[0]) if rows else 0
    k = _builtins.min(rows, cols)
    gram = _matrix_multiply_rect(_matrix_conj_transpose(matrix), matrix) if cols else []
    eigenvalues, eigenvectors = _symmetric_eigh(gram)
    order = sorted(range(len(eigenvalues)), key=lambda idx: eigenvalues[idx], reverse=True)
    singular_values = []
    for idx in order[:k]:
        value = eigenvalues[idx]
        if _math.isnan(value):
            singular_values.append(_math.inf)
        elif value == _math.inf:
            singular_values.append(_math.inf)
        else:
            singular_values.append(_math.sqrt(_builtins.max(value, 0.0)))
    singular_values.sort(reverse=True)
    v_cols = [[eigenvectors[row][idx] for row in range(cols)] for idx in order[:k]]
    u_cols: list[list[complex]] = []
    for sigma, v_col in zip(singular_values, v_cols):
        av = _matrix_vector_multiply(matrix, v_col)
        if sigma > 1e-12:
            u_cols.append([value / sigma for value in av])
        else:
            u_cols.append(_orthogonal_vector(u_cols, rows))
    u_target = rows if full_matrices else k
    while len(u_cols) < u_target:
        u_cols.append(_orthogonal_vector(u_cols, rows))
    v_target = cols if full_matrices else k
    while len(v_cols) < v_target:
        v_cols.append(_orthogonal_vector(v_cols, cols))
    u = [[u_cols[col][row] for col in range(u_target)] for row in range(rows)]
    vh = [[v_cols[row][col].conjugate() for col in range(cols)] for row in range(v_target)]
    return u, singular_values, vh


def _flatten_matrix_values(matrix: list[list[complex]], *, complex_output: bool) -> list[Any]:
    if complex_output:
        return [value for row in matrix for value in row]
    return [value.real for row in matrix for value in row]


def _broadcast_index(source_shape: Sequence[int], out_index: Sequence[int]) -> tuple[int, ...]:
    if not source_shape:
        return ()
    aligned = (1,) * (len(out_index) - len(source_shape)) + tuple(source_shape)
    source = [0 if dim == 1 else coord for coord, dim in zip(out_index, aligned)]
    return tuple(source[len(source) - len(source_shape) :])


def _real_result_dtype(dtype: DType) -> DType:
    return float32 if dtype in {float32, complex64} else float64


def _promote_dtype_array_api(left: DType, right: DType) -> DType:
    if left == right:
        return left
    if complex128 in (left, right):
        return complex128
    if complex64 in (left, right):
        other = right if left == complex64 else left
        return complex128 if other in {float64, int64, uint64} else complex64
    if float64 in (left, right):
        return float64
    if float32 in (left, right):
        other = right if left == float32 else left
        return float64 if other in {int64, uint64} else float32
    if left == bool:
        return right
    if right == bool:
        return left
    left_signed = left in {int8, int16, int32, int64}
    right_signed = right in {int8, int16, int32, int64}
    left_bits = _dtype_bits(left)
    right_bits = _dtype_bits(right)
    if left_signed and right_signed:
        return _signed_dtype_for_bits(_builtins.max(left_bits, right_bits))
    if not left_signed and not right_signed:
        return _unsigned_dtype_for_bits(_builtins.max(left_bits, right_bits))
    signed_bits = left_bits if left_signed else right_bits
    unsigned_bits = right_bits if left_signed else left_bits
    return _signed_dtype_for_bits(signed_bits if signed_bits > unsigned_bits else unsigned_bits * 2)


def _dtype_bits(dtype: DType) -> int:
    return {
        bool: 1,
        int8: 8,
        uint8: 8,
        int16: 16,
        uint16: 16,
        int32: 32,
        uint32: 32,
        int64: 64,
        uint64: 64,
    }[dtype]


def _signed_dtype_for_bits(bits: int) -> DType:
    if bits <= 8:
        return int8
    if bits <= 16:
        return int16
    if bits <= 32:
        return int32
    return int64


def _unsigned_dtype_for_bits(bits: int) -> DType:
    if bits <= 8:
        return uint8
    if bits <= 16:
        return uint16
    if bits <= 32:
        return uint32
    return uint64


def _flatten(obj: Any) -> tuple[list[int], list[Any]]:
    if isinstance(obj, Array):
        if obj.shape == ():
            return [], [obj.tolist()[0]]
        return list(obj.shape), obj.tolist()
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        items = list(obj)
        if not items:
            return [0], []

        child_shapes = []
        flat: list[Any] = []
        for item in items:
            child_shape, child_flat = _flatten(item)
            child_shapes.append(child_shape)
            flat.extend(child_flat)

        first = child_shapes[0]
        if _builtins.any(shape != first for shape in child_shapes):
            raise ValueError("ragged arrays are not supported")
        return [len(items), *first], flat
    return [], [obj]


def _infer_dtype(flat: list[Any]) -> DType:
    if _builtins.any(isinstance(value, complex) for value in flat):
        return complex128
    if _builtins.any(isinstance(value, float) for value in flat):
        return float64
    if flat and _builtins.all(isinstance(value, _builtins.bool) for value in flat):
        return bool
    return int64


def _infer_scalar_dtype(value: Any) -> DType:
    if isinstance(value, _builtins.bool):
        return bool
    if isinstance(value, complex):
        return complex128
    if isinstance(value, float):
        return float64
    return int64


def _dtype_name(dtype: str | DType | None, default: DType | None = None) -> str:
    if dtype is None and default is not None:
        return str(default)
    if dtype in _DTYPES:
        return str(dtype)
    raise TypeError(f"unsupported dtype {dtype!r}")


def _complex_pair(value: Any) -> tuple[float, float]:
    z = complex(value)
    return float(z.real), float(z.imag)


def _wrap_unsigned_int(value: Any, bits: int) -> int:
    return int(value) % (1 << bits)


def _wrap_signed_int(value: Any, bits: int) -> int:
    modulus = 1 << bits
    wrapped = int(value) % modulus
    sign_bit = 1 << (bits - 1)
    return wrapped - modulus if wrapped >= sign_bit else wrapped


def _isinf_scalar(value: Any) -> bool:
    if isinstance(value, complex):
        return _math.isinf(value.real) or _math.isinf(value.imag)
    return _math.isinf(float(value))


def _isnan_scalar(value: Any) -> bool:
    if isinstance(value, complex):
        return _math.isnan(value.real) or _math.isnan(value.imag)
    return _math.isnan(float(value))


def _isfinite_scalar(value: Any) -> bool:
    if isinstance(value, complex):
        return _math.isfinite(value.real) and _math.isfinite(value.imag)
    return _math.isfinite(float(value))


def _result_dtype_for_where(x1: Any, x2: Any) -> DType:
    left = DType(x1.dtype) if isinstance(x1, Array) else _infer_dtype([x1])
    right = DType(x2.dtype) if isinstance(x2, Array) else _infer_dtype([x2])
    return _promote_dtype_array_api(left, right)


def _sum_dtype(dtype: DType) -> DType:
    if dtype == bool or dtype in {int8, int16, int32, int64}:
        return int64
    if dtype in {uint8, uint16, uint32, uint64}:
        return uint64
    return dtype


def _arg_reduce(x: Array, *, axis: int | None, keepdims: bool, choose_max: bool) -> Array:
    array = asarray(x)
    if axis is None:
        values = array.tolist()
        best = _best_index(values, choose_max)
        shape = (1,) * array.ndim if keepdims else ()
        return asarray([best], dtype=int64).reshape(shape)

    axis = _normalize_axis(axis, array.ndim)
    reduced = array.shape[axis]
    out_shape = array.shape[:axis] + array.shape[axis + 1 :]
    values = []
    for out_idx in _ndindex(out_shape):
        candidates = []
        for coord in range(reduced):
            source_idx = out_idx[:axis] + (coord,) + out_idx[axis:]
            candidates.append(_scalar_at(array, source_idx))
        values.append(_best_index(candidates, choose_max))
    if keepdims:
        out_shape = out_shape[:axis] + (1,) + out_shape[axis:]
    return asarray(values, dtype=int64).reshape(out_shape)


def _stat_reduce(
    x: Array,
    *,
    axis: int | Sequence[int] | None,
    keepdims: bool,
    op,
    dtype: DType,
) -> Array:
    array = asarray(x)
    axes = set(_axes_or_all(axis, array.ndim))
    out_shape = tuple(1 if idx in axes else side for idx, side in enumerate(array.shape)) if keepdims else tuple(
        side for idx, side in enumerate(array.shape) if idx not in axes
    )
    reduced_shape = tuple(array.shape[idx] for idx in range(array.ndim) if idx in axes)
    values = []
    for out_idx in _ndindex(out_shape):
        items = []
        for reduced_idx in _ndindex(reduced_shape):
            source_idx = []
            out_pos = 0
            reduced_pos = 0
            for idx in range(array.ndim):
                if idx in axes:
                    source_idx.append(reduced_idx[reduced_pos])
                    reduced_pos += 1
                    if keepdims:
                        out_pos += 1
                else:
                    source_idx.append(out_idx[out_pos])
                    out_pos += 1
            items.append(_scalar_at(array, tuple(source_idx)))
        try:
            values.append(op(items))
        except ValueError:
            values.append(_math.nan)
    return asarray(values, dtype=dtype).reshape(out_shape)


def _unique_parts(array: Array) -> tuple[Array, Array, Array, Array]:
    dtype = DType(array.dtype)
    flat = array.tolist()
    keys: dict[Any, int] = {}
    values: list[Any] = []
    indices: list[int] = []
    inverse: list[int] = []
    counts: list[int] = []
    for flat_idx, value in enumerate(flat):
        key = _unique_key(value, flat_idx)
        if key not in keys:
            keys[key] = len(values)
            values.append(value)
            indices.append(flat_idx)
            counts.append(0)
        unique_idx = keys[key]
        inverse.append(unique_idx)
        counts[unique_idx] += 1
    return (
        asarray(values, dtype=dtype),
        asarray(indices, dtype=int64),
        asarray(inverse, dtype=int64).reshape(array.shape),
        asarray(counts, dtype=int64),
    )


def _unique_key(value: Any, position: int) -> Any:
    try:
        if _isnan_scalar(value):
            return ("nan", position)
    except (TypeError, ValueError):
        pass
    return value


def _best_index(values: Sequence[Any], choose_max: bool) -> int:
    best = 0
    for idx, value in enumerate(values[1:], start=1):
        is_better = value > values[best] if choose_max else value < values[best]
        if is_better:
            best = idx
    return best


def _broadcast_shape_for_where(*arrays: Array) -> tuple[int, ...]:
    return _broadcast_shape(*(array.shape for array in arrays))


def _flat_for_shape(array: Array, shape: Sequence[int]) -> list[Any]:
    if array.shape == tuple(shape):
        return array.tolist()
    return _broadcast_flat_values(array, shape)


def _broadcast_shape(*shapes: Sequence[int]) -> tuple[int, ...]:
    out: list[int] = []
    reversed_shapes = [tuple(reversed(shape)) for shape in shapes]
    max_ndim = _builtins.max((len(shape) for shape in shapes), default=0)
    for axis in range(max_ndim):
        dims = [shape[axis] if axis < len(shape) else 1 for shape in reversed_shapes]
        dim = 0 if 0 in dims else _builtins.max(dims)
        if dim == 0:
            valid = _builtins.all(candidate in (0, 1) for candidate in dims)
        else:
            valid = not _builtins.any(candidate not in (1, dim) for candidate in dims)
        if not valid:
            raise ValueError(f"cannot broadcast shapes {shapes!r}")
        out.append(dim)
    return tuple(reversed(out))


def _broadcast_flat_values(array: Array, shape: Sequence[int]) -> list[Any]:
    source_shape = array.shape
    if source_shape == tuple(shape):
        return array.tolist()
    if len(source_shape) > len(shape):
        raise ValueError(f"cannot broadcast shape {source_shape!r} to {tuple(shape)!r}")
    aligned = (1,) * (len(shape) - len(source_shape)) + source_shape
    values = []
    for idx in _ndindex(shape):
        source_idx = tuple(0 if dim == 1 else coord for coord, dim in zip(idx, aligned))
        source_idx = source_idx[len(source_idx) - len(source_shape) :]
        values.append(array[source_idx].tolist()[0])
    return values


def _ndindex(shape: Sequence[int]):
    if not shape:
        yield ()
        return
    total = _size_of_shape(shape)
    for linear in range(total):
        remainder = linear
        idx = [0] * len(shape)
        for axis in range(len(shape) - 1, -1, -1):
            dim = int(shape[axis])
            idx[axis] = remainder % dim if dim else 0
            remainder = remainder // dim if dim else 0
        yield tuple(idx)


def _row_major_offset(shape: Sequence[int], index: Sequence[int]) -> int:
    offset = 0
    for dim, coord in zip(shape, index):
        offset = offset * int(dim) + int(coord)
    return offset


def _promoted_dtype_for_arrays(arrays: Sequence[Array]) -> DType:
    dtype = DType(arrays[0].dtype)
    for array in arrays[1:]:
        dtype = _promote_dtype_array_api(dtype, DType(array.dtype))
    return dtype


_KIND_TO_DTYPES = {
    "bool": {bool},
    "signed integer": {int8, int16, int32, int64},
    "unsigned integer": {uint8, uint16, uint32, uint64},
    "integral": {int8, int16, int32, int64, uint8, uint16, uint32, uint64},
    "real floating": {float32, float64},
    "complex floating": {complex64, complex128},
    "numeric": {
        int8,
        int16,
        int32,
        int64,
        uint8,
        uint16,
        uint32,
        uint64,
        float32,
        float64,
        complex64,
        complex128,
    },
}


for _stub_name in [
    "abs",
    "acos",
    "acosh",
    "argmax",
    "argmin",
    "argsort",
    "asin",
    "asinh",
    "atan",
    "atan2",
    "atanh",
    "bitwise_and",
    "bitwise_left_shift",
    "bitwise_invert",
    "bitwise_or",
    "bitwise_right_shift",
    "bitwise_xor",
    "ceil",
    "clip",
    "concat",
    "conj",
    "copysign",
    "cos",
    "cosh",
    "cumulative_sum",
    "exp",
    "expand_dims",
    "expm1",
    "flip",
    "floor",
    "floor_divide",
    "hypot",
    "log",
    "log1p",
    "log2",
    "log10",
    "logaddexp",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "matrix_transpose",
    "max",
    "maximum",
    "min",
    "minimum",
    "moveaxis",
    "negative",
    "nonzero",
    "positive",
    "pow",
    "prod",
    "remainder",
    "repeat",
    "result_type",
    "roll",
    "round",
    "searchsorted",
    "sign",
    "signbit",
    "sin",
    "sinh",
    "sort",
    "sqrt",
    "square",
    "squeeze",
    "stack",
    "std",
    "tan",
    "tanh",
    "take",
    "tensordot",
    "tile",
    "tril",
    "triu",
    "trunc",
    "unique_all",
    "unique_counts",
    "unique_inverse",
    "unique_values",
    "unstack",
    "var",
    "vecdot",
]:
    globals().setdefault(_stub_name, _not_implemented)


linalg = _SimpleNamespace(
    cholesky=_linalg_cholesky,
    cross=cross,
    det=_linalg_det,
    diagonal=diagonal,
    eigh=_linalg_eigh,
    eigvalsh=_linalg_eigvalsh,
    inv=_linalg_inv,
    matmul=matmul,
    matrix_norm=_linalg_matrix_norm,
    matrix_power=_linalg_matrix_power,
    matrix_rank=_linalg_matrix_rank,
    matrix_transpose=matrix_transpose,
    outer=_linalg_outer,
    pinv=_linalg_pinv,
    qr=_linalg_qr,
    slogdet=_linalg_slogdet,
    solve=_linalg_solve,
    svd=_linalg_svd,
    svdvals=_linalg_svdvals,
    tensordot=_linalg_tensordot,
    trace=_linalg_trace,
    vecdot=_linalg_vecdot,
    vector_norm=_linalg_vector_norm,
)

fft = _SimpleNamespace(
    fft=_fft_fft,
    ifft=_fft_ifft,
    fftn=_fft_fftn,
    ifftn=_fft_ifftn,
    rfft=_fft_rfft,
    irfft=_fft_irfft,
    rfftn=_fft_rfftn,
    irfftn=_fft_irfftn,
    hfft=_fft_hfft,
    ihfft=_fft_ihfft,
    fftfreq=_fft_fftfreq,
    rfftfreq=_fft_rfftfreq,
    fftshift=_fft_fftshift,
    ifftshift=_fft_ifftshift,
)


__all__ = [
    "Array",
    "DType",
    "__array_api_version__",
    "__array_namespace_info__",
    "abs",
    "acos",
    "acosh",
    "add",
    "all",
    "any",
    "argmax",
    "argmin",
    "argsort",
    "arange",
    "asin",
    "asinh",
    "asarray",
    "astype",
    "atan",
    "atan2",
    "atanh",
    "bitwise_and",
    "bitwise_left_shift",
    "bitwise_invert",
    "bitwise_or",
    "bitwise_right_shift",
    "bitwise_xor",
    "bool",
    "broadcast_arrays",
    "broadcast_to",
    "can_cast",
    "ceil",
    "clip",
    "complex64",
    "complex128",
    "concat",
    "conj",
    "copysign",
    "cos",
    "cosh",
    "cross",
    "diagonal",
    "divide",
    "e",
    "equal",
    "empty",
    "empty_like",
    "exp",
    "expm1",
    "expand_dims",
    "eye",
    "finfo",
    "fft",
    "flip",
    "floor",
    "floor_divide",
    "float32",
    "float64",
    "from_dlpack",
    "full",
    "full_like",
    "greater",
    "greater_equal",
    "hypot",
    "imag",
    "int8",
    "int16",
    "int32",
    "int64",
    "inf",
    "iinfo",
    "isfinite",
    "isinf",
    "isnan",
    "isdtype",
    "less",
    "less_equal",
    "linspace",
    "linalg",
    "log",
    "log1p",
    "log2",
    "log10",
    "logaddexp",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "matmul",
    "matrix_transpose",
    "maximum",
    "mean",
    "meshgrid",
    "minimum",
    "moveaxis",
    "multiply",
    "nan",
    "negative",
    "newaxis",
    "not_equal",
    "ones",
    "ones_like",
    "pi",
    "positive",
    "pow",
    "permute_dims",
    "real",
    "repeat",
    "reshape",
    "roll",
    "round",
    "searchsorted",
    "sign",
    "signbit",
    "sin",
    "sinh",
    "sort",
    "sqrt",
    "square",
    "squeeze",
    "stack",
    "subtract",
    "sum",
    "tan",
    "tanh",
    "take",
    "tensordot",
    "tile",
    "tril",
    "triu",
    "trunc",
    "unique_all",
    "unique_counts",
    "unique_inverse",
    "unique_values",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "unstack",
    "var",
    "vecdot",
    "where",
    "zeros",
    "zeros_like",
]
