use num_complex::{Complex32, Complex64};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DTypeKind {
    F64,
    F32,
    Complex128,
    Complex64,
    I64,
    I32,
    I16,
    I8,
    U64,
    U32,
    U16,
    U8,
    Bool,
}

impl DTypeKind {
    pub fn is_float(self) -> bool {
        matches!(
            self,
            Self::F32 | Self::F64 | Self::Complex64 | Self::Complex128
        )
    }

    pub fn is_signed_integer(self) -> bool {
        matches!(self, Self::I8 | Self::I16 | Self::I32 | Self::I64)
    }

    pub fn is_unsigned_integer(self) -> bool {
        matches!(self, Self::U8 | Self::U16 | Self::U32 | Self::U64)
    }

    pub fn itemsize(self) -> usize {
        match self {
            Self::Complex128 => 16,
            Self::F64 | Self::I64 | Self::U64 | Self::Complex64 => 8,
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::I16 | Self::U16 => 2,
            Self::I8 | Self::U8 | Self::Bool => 1,
        }
    }

    fn integer_bits(self) -> Option<usize> {
        match self {
            Self::Bool => Some(1),
            Self::I8 | Self::U8 => Some(8),
            Self::I16 | Self::U16 => Some(16),
            Self::I32 | Self::U32 => Some(32),
            Self::I64 | Self::U64 => Some(64),
            Self::F32 | Self::F64 | Self::Complex64 | Self::Complex128 => None,
        }
    }
}

pub trait DType: Clone + 'static {
    const KIND: DTypeKind;
}

pub trait CastElement<U>: Copy {
    fn cast(self) -> U;
}

impl DType for f64 {
    const KIND: DTypeKind = DTypeKind::F64;
}

impl DType for f32 {
    const KIND: DTypeKind = DTypeKind::F32;
}

impl DType for Complex64 {
    const KIND: DTypeKind = DTypeKind::Complex128;
}

impl DType for Complex32 {
    const KIND: DTypeKind = DTypeKind::Complex64;
}

impl DType for i64 {
    const KIND: DTypeKind = DTypeKind::I64;
}

impl DType for i32 {
    const KIND: DTypeKind = DTypeKind::I32;
}

impl DType for i16 {
    const KIND: DTypeKind = DTypeKind::I16;
}

impl DType for i8 {
    const KIND: DTypeKind = DTypeKind::I8;
}

impl DType for u64 {
    const KIND: DTypeKind = DTypeKind::U64;
}

impl DType for u32 {
    const KIND: DTypeKind = DTypeKind::U32;
}

impl DType for u16 {
    const KIND: DTypeKind = DTypeKind::U16;
}

impl DType for u8 {
    const KIND: DTypeKind = DTypeKind::U8;
}

impl DType for bool {
    const KIND: DTypeKind = DTypeKind::Bool;
}

pub fn promote_dtype(left: DTypeKind, right: DTypeKind) -> DTypeKind {
    use DTypeKind::*;

    if left == right {
        return left;
    }

    if left == Complex128 || right == Complex128 {
        return Complex128;
    }

    if left == Complex64 || right == Complex64 {
        let other = if left == Complex64 { right } else { left };
        return if matches!(other, F64 | I64 | U64) {
            Complex128
        } else {
            Complex64
        };
    }

    if left == F64 || right == F64 {
        return F64;
    }

    if left == F32 || right == F32 {
        let other = if left == F32 { right } else { left };
        return if matches!(other, I64 | U64) { F64 } else { F32 };
    }

    if left == Bool {
        return right;
    }
    if right == Bool {
        return left;
    }

    match (left.is_signed_integer(), right.is_signed_integer()) {
        (true, true) => signed_for_bits(
            left.integer_bits()
                .unwrap()
                .max(right.integer_bits().unwrap()),
        ),
        (false, false) => unsigned_for_bits(
            left.integer_bits()
                .unwrap()
                .max(right.integer_bits().unwrap()),
        ),
        _ => {
            let signed_bits = if left.is_signed_integer() {
                left.integer_bits().unwrap()
            } else {
                right.integer_bits().unwrap()
            };
            let unsigned_bits = if left.is_unsigned_integer() {
                left.integer_bits().unwrap()
            } else {
                right.integer_bits().unwrap()
            };

            if signed_bits > unsigned_bits {
                signed_for_bits(signed_bits)
            } else {
                signed_for_bits(unsigned_bits * 2)
            }
        }
    }
}

fn signed_for_bits(bits: usize) -> DTypeKind {
    match bits {
        0..=8 => DTypeKind::I8,
        9..=16 => DTypeKind::I16,
        17..=32 => DTypeKind::I32,
        _ => DTypeKind::I64,
    }
}

fn unsigned_for_bits(bits: usize) -> DTypeKind {
    match bits {
        0..=8 => DTypeKind::U8,
        9..=16 => DTypeKind::U16,
        17..=32 => DTypeKind::U32,
        _ => DTypeKind::U64,
    }
}

macro_rules! impl_numeric_casts {
    ($src:ty => $($dst:ty),+ $(,)?) => {
        $(
            impl CastElement<$dst> for $src {
                fn cast(self) -> $dst {
                    self as $dst
                }
            }
        )+
    };
}

macro_rules! impl_numeric_to_bool {
    ($($src:ty),+ $(,)?) => {
        $(
            impl CastElement<bool> for $src {
                fn cast(self) -> bool {
                    self != 0 as $src
                }
            }
        )+
    };
}

macro_rules! impl_bool_to_numeric {
    ($($dst:ty),+ $(,)?) => {
        $(
            impl CastElement<$dst> for bool {
                fn cast(self) -> $dst {
                    if self { 1 as $dst } else { 0 as $dst }
                }
            }
        )+
    };
}

impl_numeric_casts!(f64 => f64, f32, i64, i32, i16, i8, u64, u32, u16, u8);
impl_numeric_casts!(f32 => f64, f32, i64, i32, i16, i8, u64, u32, u16, u8);
impl_numeric_casts!(i64 => f64, f32, i64, i32, i16, i8, u64, u32, u16, u8);
impl_numeric_casts!(i32 => f64, f32, i64, i32, i16, i8, u64, u32, u16, u8);
impl_numeric_casts!(i16 => f64, f32, i64, i32, i16, i8, u64, u32, u16, u8);
impl_numeric_casts!(i8 => f64, f32, i64, i32, i16, i8, u64, u32, u16, u8);
impl_numeric_casts!(u64 => f64, f32, i64, i32, i16, i8, u64, u32, u16, u8);
impl_numeric_casts!(u32 => f64, f32, i64, i32, i16, i8, u64, u32, u16, u8);
impl_numeric_casts!(u16 => f64, f32, i64, i32, i16, i8, u64, u32, u16, u8);
impl_numeric_casts!(u8 => f64, f32, i64, i32, i16, i8, u64, u32, u16, u8);
impl_numeric_to_bool!(f64, f32, i64, i32, i16, i8, u64, u32, u16, u8);
impl_bool_to_numeric!(f64, f32, i64, i32, i16, i8, u64, u32, u16, u8);

impl CastElement<bool> for bool {
    fn cast(self) -> bool {
        self
    }
}
