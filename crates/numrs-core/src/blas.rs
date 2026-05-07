use std::mem::{self, MaybeUninit};

fn uninit_vec<T>(len: usize) -> Vec<MaybeUninit<T>> {
    let mut out = Vec::with_capacity(len);
    unsafe {
        out.set_len(len);
    }
    out
}

unsafe fn assume_init_vec<T>(mut out: Vec<MaybeUninit<T>>) -> Vec<T> {
    let ptr = out.as_mut_ptr() as *mut T;
    let len = out.len();
    let cap = out.capacity();
    mem::forget(out);
    Vec::from_raw_parts(ptr, len, cap)
}

#[cfg(target_os = "macos")]
mod imp {
    use std::os::raw::{c_double, c_float, c_int, c_long};

    const CBLAS_COL_MAJOR: c_int = 102;
    const CBLAS_ROW_MAJOR: c_int = 101;
    const CBLAS_NO_TRANS: c_int = 111;
    const CBLAS_TRANS: c_int = 112;

    #[link(name = "Accelerate", kind = "framework")]
    extern "C" {
        fn cblas_dgemm(
            order: c_int,
            trans_a: c_int,
            trans_b: c_int,
            m: c_int,
            n: c_int,
            k: c_int,
            alpha: c_double,
            a: *const c_double,
            lda: c_int,
            b: *const c_double,
            ldb: c_int,
            beta: c_double,
            c: *mut c_double,
            ldc: c_int,
        );
        fn cblas_sgemm(
            order: c_int,
            trans_a: c_int,
            trans_b: c_int,
            m: c_int,
            n: c_int,
            k: c_int,
            alpha: c_float,
            a: *const c_float,
            lda: c_int,
            b: *const c_float,
            ldb: c_int,
            beta: c_float,
            c: *mut c_float,
            ldc: c_int,
        );
        fn cblas_dger(
            order: c_int,
            m: c_int,
            n: c_int,
            alpha: c_double,
            x: *const c_double,
            inc_x: c_int,
            y: *const c_double,
            inc_y: c_int,
            a: *mut c_double,
            lda: c_int,
        );
        fn cblas_dgemv(
            order: c_int,
            trans_a: c_int,
            m: c_int,
            n: c_int,
            alpha: c_double,
            a: *const c_double,
            lda: c_int,
            x: *const c_double,
            inc_x: c_int,
            beta: c_double,
            y: *mut c_double,
            inc_y: c_int,
        );
        fn cblas_sgemv(
            order: c_int,
            trans_a: c_int,
            m: c_int,
            n: c_int,
            alpha: c_float,
            a: *const c_float,
            lda: c_int,
            x: *const c_float,
            inc_x: c_int,
            beta: c_float,
            y: *mut c_float,
            inc_y: c_int,
        );
        fn vDSP_dotprD(
            a: *const c_double,
            stride_a: c_long,
            b: *const c_double,
            stride_b: c_long,
            c: *mut c_double,
            n: c_ulong,
        );
        fn vDSP_dotpr(
            a: *const c_float,
            stride_a: c_long,
            b: *const c_float,
            stride_b: c_long,
            c: *mut c_float,
            n: c_ulong,
        );
        fn vDSP_sveD(a: *const c_double, stride: c_long, c: *mut c_double, n: c_ulong);
        fn vDSP_sve(a: *const c_float, stride: c_long, c: *mut c_float, n: c_ulong);
    }

    #[allow(non_camel_case_types)]
    type c_ulong = u64;

    pub(crate) fn dgemm_row_major(
        rows: usize,
        inner: usize,
        cols: usize,
        a: &[f64],
        b: &[f64],
    ) -> Vec<f64> {
        if rows == 0 || cols == 0 {
            return Vec::new();
        }
        if inner == 0 {
            return vec![0.0; rows * cols];
        }
        let mut out = super::uninit_vec(rows * cols);
        unsafe {
            cblas_dgemm(
                CBLAS_COL_MAJOR,
                CBLAS_NO_TRANS,
                CBLAS_NO_TRANS,
                cols as c_int,
                rows as c_int,
                inner as c_int,
                1.0,
                b.as_ptr(),
                cols as c_int,
                a.as_ptr(),
                inner as c_int,
                0.0,
                out.as_mut_ptr().cast::<f64>(),
                cols as c_int,
            );
            super::assume_init_vec(out)
        }
    }

    pub(crate) fn dgemm_row_major_trans_a_f64(
        rows: usize,
        inner: usize,
        cols: usize,
        a_transposed: &[f64],
        b: &[f64],
    ) -> Vec<f64> {
        if rows == 0 || cols == 0 {
            return Vec::new();
        }
        if inner == 0 {
            return vec![0.0; rows * cols];
        }
        let mut out = super::uninit_vec(rows * cols);
        unsafe {
            cblas_dgemm(
                CBLAS_COL_MAJOR,
                CBLAS_NO_TRANS,
                CBLAS_TRANS,
                cols as c_int,
                rows as c_int,
                inner as c_int,
                1.0,
                b.as_ptr(),
                cols as c_int,
                a_transposed.as_ptr(),
                rows as c_int,
                0.0,
                out.as_mut_ptr().cast::<f64>(),
                cols as c_int,
            );
            super::assume_init_vec(out)
        }
    }

    pub(crate) fn dgemm_row_major_trans_b_f64(
        rows: usize,
        inner: usize,
        cols: usize,
        a: &[f64],
        b_transposed: &[f64],
    ) -> Vec<f64> {
        if rows == 0 || cols == 0 {
            return Vec::new();
        }
        if inner == 0 {
            return vec![0.0; rows * cols];
        }
        let mut out = super::uninit_vec(rows * cols);
        unsafe {
            cblas_dgemm(
                CBLAS_COL_MAJOR,
                CBLAS_TRANS,
                CBLAS_NO_TRANS,
                cols as c_int,
                rows as c_int,
                inner as c_int,
                1.0,
                b_transposed.as_ptr(),
                inner as c_int,
                a.as_ptr(),
                inner as c_int,
                0.0,
                out.as_mut_ptr().cast::<f64>(),
                cols as c_int,
            );
            super::assume_init_vec(out)
        }
    }

    pub(crate) fn sgemm_row_major(
        rows: usize,
        inner: usize,
        cols: usize,
        a: &[f32],
        b: &[f32],
    ) -> Vec<f32> {
        if rows == 0 || cols == 0 {
            return Vec::new();
        }
        if inner == 0 {
            return vec![0.0; rows * cols];
        }
        let mut out = super::uninit_vec(rows * cols);
        unsafe {
            cblas_sgemm(
                CBLAS_ROW_MAJOR,
                CBLAS_NO_TRANS,
                CBLAS_NO_TRANS,
                rows as c_int,
                cols as c_int,
                inner as c_int,
                1.0,
                a.as_ptr(),
                inner as c_int,
                b.as_ptr(),
                cols as c_int,
                0.0,
                out.as_mut_ptr().cast::<f32>(),
                cols as c_int,
            );
            super::assume_init_vec(out)
        }
    }

    pub(crate) fn sgemm_row_major_trans_a(
        rows: usize,
        inner: usize,
        cols: usize,
        a_transposed: &[f32],
        b: &[f32],
    ) -> Vec<f32> {
        if rows == 0 || cols == 0 {
            return Vec::new();
        }
        if inner == 0 {
            return vec![0.0; rows * cols];
        }
        let mut out = super::uninit_vec(rows * cols);
        unsafe {
            cblas_sgemm(
                CBLAS_COL_MAJOR,
                CBLAS_NO_TRANS,
                CBLAS_TRANS,
                cols as c_int,
                rows as c_int,
                inner as c_int,
                1.0,
                b.as_ptr(),
                cols as c_int,
                a_transposed.as_ptr(),
                rows as c_int,
                0.0,
                out.as_mut_ptr().cast::<f32>(),
                cols as c_int,
            );
            super::assume_init_vec(out)
        }
    }

    pub(crate) fn sgemm_row_major_trans_b(
        rows: usize,
        inner: usize,
        cols: usize,
        a: &[f32],
        b_transposed: &[f32],
    ) -> Vec<f32> {
        if rows == 0 || cols == 0 {
            return Vec::new();
        }
        if inner == 0 {
            return vec![0.0; rows * cols];
        }
        let mut out = super::uninit_vec(rows * cols);
        unsafe {
            cblas_sgemm(
                CBLAS_COL_MAJOR,
                CBLAS_TRANS,
                CBLAS_NO_TRANS,
                cols as c_int,
                rows as c_int,
                inner as c_int,
                1.0,
                b_transposed.as_ptr(),
                inner as c_int,
                a.as_ptr(),
                inner as c_int,
                0.0,
                out.as_mut_ptr().cast::<f32>(),
                cols as c_int,
            );
            super::assume_init_vec(out)
        }
    }

    pub(crate) fn dgemv_row_major(
        rows: usize,
        cols: usize,
        matrix: &[f64],
        vector: &[f64],
    ) -> Vec<f64> {
        if rows == 0 {
            return Vec::new();
        }
        if cols == 0 {
            return vec![0.0; rows];
        }
        let mut out = super::uninit_vec(rows);
        unsafe {
            cblas_dgemv(
                CBLAS_ROW_MAJOR,
                CBLAS_NO_TRANS,
                rows as c_int,
                cols as c_int,
                1.0,
                matrix.as_ptr(),
                cols as c_int,
                vector.as_ptr(),
                1,
                0.0,
                out.as_mut_ptr().cast::<f64>(),
                1,
            );
            super::assume_init_vec(out)
        }
    }

    pub(crate) fn dgemv_row_major_trans(
        rows: usize,
        cols: usize,
        matrix: &[f64],
        vector: &[f64],
    ) -> Vec<f64> {
        if cols == 0 {
            return Vec::new();
        }
        if rows == 0 {
            return vec![0.0; cols];
        }
        let mut out = super::uninit_vec(cols);
        unsafe {
            cblas_dgemv(
                CBLAS_ROW_MAJOR,
                CBLAS_TRANS,
                rows as c_int,
                cols as c_int,
                1.0,
                matrix.as_ptr(),
                cols as c_int,
                vector.as_ptr(),
                1,
                0.0,
                out.as_mut_ptr().cast::<f64>(),
                1,
            );
            super::assume_init_vec(out)
        }
    }

    pub(crate) fn sgemv_row_major(
        rows: usize,
        cols: usize,
        matrix: &[f32],
        vector: &[f32],
    ) -> Vec<f32> {
        if rows == 0 {
            return Vec::new();
        }
        if cols == 0 {
            return vec![0.0; rows];
        }
        let mut out = super::uninit_vec(rows);
        unsafe {
            cblas_sgemv(
                CBLAS_ROW_MAJOR,
                CBLAS_NO_TRANS,
                rows as c_int,
                cols as c_int,
                1.0,
                matrix.as_ptr(),
                cols as c_int,
                vector.as_ptr(),
                1,
                0.0,
                out.as_mut_ptr().cast::<f32>(),
                1,
            );
            super::assume_init_vec(out)
        }
    }

    pub(crate) fn sgemv_row_major_trans(
        rows: usize,
        cols: usize,
        matrix: &[f32],
        vector: &[f32],
    ) -> Vec<f32> {
        if cols == 0 {
            return Vec::new();
        }
        if rows == 0 {
            return vec![0.0; cols];
        }
        let mut out = super::uninit_vec(cols);
        unsafe {
            cblas_sgemv(
                CBLAS_ROW_MAJOR,
                CBLAS_TRANS,
                rows as c_int,
                cols as c_int,
                1.0,
                matrix.as_ptr(),
                cols as c_int,
                vector.as_ptr(),
                1,
                0.0,
                out.as_mut_ptr().cast::<f32>(),
                1,
            );
            super::assume_init_vec(out)
        }
    }

    pub(crate) fn dot_f64(left: &[f64], right: &[f64]) -> f64 {
        if left.is_empty() {
            return 0.0;
        }
        let mut out = 0.0;
        unsafe {
            vDSP_dotprD(
                left.as_ptr(),
                1,
                right.as_ptr(),
                1,
                &mut out,
                left.len() as c_ulong,
            );
        }
        out
    }

    pub(crate) fn dot_f32(left: &[f32], right: &[f32]) -> f32 {
        if left.is_empty() {
            return 0.0;
        }
        let mut out = 0.0;
        unsafe {
            vDSP_dotpr(
                left.as_ptr(),
                1,
                right.as_ptr(),
                1,
                &mut out,
                left.len() as c_ulong,
            );
        }
        out
    }

    pub(crate) fn outer_product_f64(rows: usize, cols: usize, x: &[f64], y: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; rows * cols];
        if out.is_empty() {
            return out;
        }
        unsafe {
            cblas_dger(
                CBLAS_ROW_MAJOR,
                rows as c_int,
                cols as c_int,
                1.0,
                x.as_ptr(),
                1,
                y.as_ptr(),
                1,
                out.as_mut_ptr(),
                cols as c_int,
            );
        }
        out
    }

    #[inline(always)]
    pub(crate) fn mul_scalar_f64(values: &[f64], scalar: f64) -> Vec<f64> {
        #[cfg(target_arch = "aarch64")]
        {
            unsafe { mul_scalar_f64_neon(values, scalar) }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            mul_scalar_f64_scalar(values, scalar)
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    unsafe fn mul_scalar_f64_neon(values: &[f64], scalar: f64) -> Vec<f64> {
        use std::arch::aarch64::{vdupq_n_f64, vld1q_f64, vmulq_f64, vst1q_f64};

        let len = values.len();
        let mut out: Vec<f64> = Vec::with_capacity(len);
        let src = values.as_ptr();
        let dst = out.as_mut_ptr();
        let scalar_vec = vdupq_n_f64(scalar);
        let mut idx = 0usize;
        while idx + 8 <= len {
            let v0 = vld1q_f64(src.add(idx));
            let v1 = vld1q_f64(src.add(idx + 2));
            let v2 = vld1q_f64(src.add(idx + 4));
            let v3 = vld1q_f64(src.add(idx + 6));
            vst1q_f64(dst.add(idx), vmulq_f64(v0, scalar_vec));
            vst1q_f64(dst.add(idx + 2), vmulq_f64(v1, scalar_vec));
            vst1q_f64(dst.add(idx + 4), vmulq_f64(v2, scalar_vec));
            vst1q_f64(dst.add(idx + 6), vmulq_f64(v3, scalar_vec));
            idx += 8;
        }
        while idx + 2 <= len {
            let v = vld1q_f64(src.add(idx));
            vst1q_f64(dst.add(idx), vmulq_f64(v, scalar_vec));
            idx += 2;
        }
        while idx < len {
            dst.add(idx).write(*src.add(idx) * scalar);
            idx += 1;
        }
        out.set_len(len);
        out
    }

    #[cfg(not(target_arch = "aarch64"))]
    #[inline(always)]
    fn mul_scalar_f64_scalar(values: &[f64], scalar: f64) -> Vec<f64> {
        let len = values.len();
        let mut out: Vec<f64> = Vec::with_capacity(len);
        unsafe {
            let src = values.as_ptr();
            let dst = out.as_mut_ptr();
            let mut idx = 0usize;
            while idx + 4 <= len {
                dst.add(idx).write(*src.add(idx) * scalar);
                dst.add(idx + 1).write(*src.add(idx + 1) * scalar);
                dst.add(idx + 2).write(*src.add(idx + 2) * scalar);
                dst.add(idx + 3).write(*src.add(idx + 3) * scalar);
                idx += 4;
            }
            while idx < len {
                dst.add(idx).write(*src.add(idx) * scalar);
                idx += 1;
            }
            out.set_len(len);
        }
        out
    }

    #[inline(always)]
    pub(crate) fn mul_repeated_tile_f64(tile: &[f64], values: &[f64]) -> Vec<f64> {
        #[cfg(target_arch = "aarch64")]
        {
            unsafe { mul_repeated_tile_f64_neon(tile, values) }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            mul_repeated_tile_f64_scalar(tile, values)
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    unsafe fn mul_repeated_tile_f64_neon(tile: &[f64], values: &[f64]) -> Vec<f64> {
        use std::arch::aarch64::{vld1q_f64, vmulq_f64, vst1q_f64};

        let len = values.len();
        let tile_len = tile.len();
        debug_assert!(tile_len > 0);
        debug_assert_eq!(len % tile_len, 0);

        let mut out: Vec<f64> = Vec::with_capacity(len);
        let tile_ptr = tile.as_ptr();
        let value_ptr = values.as_ptr();
        let dst = out.as_mut_ptr();
        let mut base = 0usize;
        while base < len {
            let mut idx = 0usize;
            while idx + 8 <= tile_len {
                let t0 = vld1q_f64(tile_ptr.add(idx));
                let t1 = vld1q_f64(tile_ptr.add(idx + 2));
                let t2 = vld1q_f64(tile_ptr.add(idx + 4));
                let t3 = vld1q_f64(tile_ptr.add(idx + 6));
                let v0 = vld1q_f64(value_ptr.add(base + idx));
                let v1 = vld1q_f64(value_ptr.add(base + idx + 2));
                let v2 = vld1q_f64(value_ptr.add(base + idx + 4));
                let v3 = vld1q_f64(value_ptr.add(base + idx + 6));
                vst1q_f64(dst.add(base + idx), vmulq_f64(t0, v0));
                vst1q_f64(dst.add(base + idx + 2), vmulq_f64(t1, v1));
                vst1q_f64(dst.add(base + idx + 4), vmulq_f64(t2, v2));
                vst1q_f64(dst.add(base + idx + 6), vmulq_f64(t3, v3));
                idx += 8;
            }
            while idx + 2 <= tile_len {
                let tile_values = vld1q_f64(tile_ptr.add(idx));
                let values = vld1q_f64(value_ptr.add(base + idx));
                vst1q_f64(dst.add(base + idx), vmulq_f64(tile_values, values));
                idx += 2;
            }
            while idx < tile_len {
                dst.add(base + idx)
                    .write(*tile_ptr.add(idx) * *value_ptr.add(base + idx));
                idx += 1;
            }
            base += tile_len;
        }
        out.set_len(len);
        out
    }

    #[cfg(not(target_arch = "aarch64"))]
    #[inline(always)]
    fn mul_repeated_tile_f64_scalar(tile: &[f64], values: &[f64]) -> Vec<f64> {
        let len = values.len();
        let tile_len = tile.len();
        debug_assert!(tile_len > 0);
        debug_assert_eq!(len % tile_len, 0);

        let mut out: Vec<f64> = Vec::with_capacity(len);
        unsafe {
            let tile_ptr = tile.as_ptr();
            let value_ptr = values.as_ptr();
            let dst = out.as_mut_ptr();
            let mut base = 0usize;
            while base < len {
                let mut idx = 0usize;
                while idx + 4 <= tile_len {
                    dst.add(base + idx)
                        .write(*tile_ptr.add(idx) * *value_ptr.add(base + idx));
                    dst.add(base + idx + 1)
                        .write(*tile_ptr.add(idx + 1) * *value_ptr.add(base + idx + 1));
                    dst.add(base + idx + 2)
                        .write(*tile_ptr.add(idx + 2) * *value_ptr.add(base + idx + 2));
                    dst.add(base + idx + 3)
                        .write(*tile_ptr.add(idx + 3) * *value_ptr.add(base + idx + 3));
                    idx += 4;
                }
                while idx < tile_len {
                    dst.add(base + idx)
                        .write(*tile_ptr.add(idx) * *value_ptr.add(base + idx));
                    idx += 1;
                }
                base += tile_len;
            }
            out.set_len(len);
        }
        out
    }

    pub(crate) fn weighted_axis1_sum_f64(
        rows: usize,
        cols: usize,
        matrix: &[f64],
        weights: &[f64],
    ) -> f64 {
        if rows == 0 || cols == 0 {
            return 0.0;
        }
        let mut row_sums = super::uninit_vec(rows);
        unsafe {
            cblas_dgemv(
                CBLAS_ROW_MAJOR,
                CBLAS_NO_TRANS,
                rows as c_int,
                cols as c_int,
                1.0,
                matrix.as_ptr(),
                cols as c_int,
                weights.as_ptr(),
                1,
                0.0,
                row_sums.as_mut_ptr().cast::<f64>(),
                1,
            );
            let row_sums = super::assume_init_vec(row_sums);
            sum_f64(&row_sums)
        }
    }

    pub(crate) fn sum_f64(values: &[f64]) -> f64 {
        let mut out = 0.0;
        unsafe {
            vDSP_sveD(values.as_ptr(), 1, &mut out, values.len() as c_ulong);
        }
        out
    }

    pub(crate) fn sum_f32(values: &[f32]) -> f32 {
        let mut out = 0.0;
        unsafe {
            vDSP_sve(values.as_ptr(), 1, &mut out, values.len() as c_ulong);
        }
        out
    }
}

#[cfg(not(target_os = "macos"))]
mod imp {
    pub(crate) fn dgemm_row_major(
        rows: usize,
        inner: usize,
        cols: usize,
        a: &[f64],
        b: &[f64],
    ) -> Vec<f64> {
        if rows == 0 || cols == 0 {
            return Vec::new();
        }
        if inner == 0 {
            return vec![0.0; rows * cols];
        }
        let mut out = super::uninit_vec(rows * cols);
        unsafe {
            matrixmultiply::dgemm(
                rows,
                inner,
                cols,
                1.0,
                a.as_ptr(),
                inner as isize,
                1,
                b.as_ptr(),
                cols as isize,
                1,
                0.0,
                out.as_mut_ptr().cast::<f64>(),
                cols as isize,
                1,
            );
            super::assume_init_vec(out)
        }
    }

    pub(crate) fn sgemm_row_major(
        rows: usize,
        inner: usize,
        cols: usize,
        a: &[f32],
        b: &[f32],
    ) -> Vec<f32> {
        if rows == 0 || cols == 0 {
            return Vec::new();
        }
        if inner == 0 {
            return vec![0.0; rows * cols];
        }
        let mut out = super::uninit_vec(rows * cols);
        unsafe {
            matrixmultiply::sgemm(
                rows,
                inner,
                cols,
                1.0,
                a.as_ptr(),
                inner as isize,
                1,
                b.as_ptr(),
                cols as isize,
                1,
                0.0,
                out.as_mut_ptr().cast::<f32>(),
                cols as isize,
                1,
            );
            super::assume_init_vec(out)
        }
    }

    pub(crate) fn dgemm_row_major_trans_a_f64(
        rows: usize,
        inner: usize,
        cols: usize,
        a_transposed: &[f64],
        b: &[f64],
    ) -> Vec<f64> {
        if rows == 0 || cols == 0 {
            return Vec::new();
        }
        if inner == 0 {
            return vec![0.0; rows * cols];
        }
        let mut out = super::uninit_vec(rows * cols);
        unsafe {
            matrixmultiply::dgemm(
                rows,
                inner,
                cols,
                1.0,
                a_transposed.as_ptr(),
                1,
                rows as isize,
                b.as_ptr(),
                cols as isize,
                1,
                0.0,
                out.as_mut_ptr().cast::<f64>(),
                cols as isize,
                1,
            );
            super::assume_init_vec(out)
        }
    }

    pub(crate) fn dgemm_row_major_trans_b_f64(
        rows: usize,
        inner: usize,
        cols: usize,
        a: &[f64],
        b_transposed: &[f64],
    ) -> Vec<f64> {
        if rows == 0 || cols == 0 {
            return Vec::new();
        }
        if inner == 0 {
            return vec![0.0; rows * cols];
        }
        let mut out = super::uninit_vec(rows * cols);
        unsafe {
            matrixmultiply::dgemm(
                rows,
                inner,
                cols,
                1.0,
                a.as_ptr(),
                inner as isize,
                1,
                b_transposed.as_ptr(),
                1,
                inner as isize,
                0.0,
                out.as_mut_ptr().cast::<f64>(),
                cols as isize,
                1,
            );
            super::assume_init_vec(out)
        }
    }

    pub(crate) fn sgemm_row_major_trans_a(
        rows: usize,
        inner: usize,
        cols: usize,
        a_transposed: &[f32],
        b: &[f32],
    ) -> Vec<f32> {
        if rows == 0 || cols == 0 {
            return Vec::new();
        }
        if inner == 0 {
            return vec![0.0; rows * cols];
        }
        let mut out = super::uninit_vec(rows * cols);
        unsafe {
            matrixmultiply::sgemm(
                rows,
                inner,
                cols,
                1.0,
                a_transposed.as_ptr(),
                1,
                rows as isize,
                b.as_ptr(),
                cols as isize,
                1,
                0.0,
                out.as_mut_ptr().cast::<f32>(),
                cols as isize,
                1,
            );
            super::assume_init_vec(out)
        }
    }

    pub(crate) fn sgemm_row_major_trans_b(
        rows: usize,
        inner: usize,
        cols: usize,
        a: &[f32],
        b_transposed: &[f32],
    ) -> Vec<f32> {
        if rows == 0 || cols == 0 {
            return Vec::new();
        }
        if inner == 0 {
            return vec![0.0; rows * cols];
        }
        let mut out = super::uninit_vec(rows * cols);
        unsafe {
            matrixmultiply::sgemm(
                rows,
                inner,
                cols,
                1.0,
                a.as_ptr(),
                inner as isize,
                1,
                b_transposed.as_ptr(),
                1,
                inner as isize,
                0.0,
                out.as_mut_ptr().cast::<f32>(),
                cols as isize,
                1,
            );
            super::assume_init_vec(out)
        }
    }

    pub(crate) fn outer_product_f64(rows: usize, cols: usize, x: &[f64], y: &[f64]) -> Vec<f64> {
        let mut out = Vec::with_capacity(rows * cols);
        for x_value in x.iter().take(rows).copied() {
            for y_value in y.iter().take(cols).copied() {
                out.push(x_value * y_value);
            }
        }
        out
    }

    pub(crate) fn dgemv_row_major(
        rows: usize,
        cols: usize,
        matrix: &[f64],
        vector: &[f64],
    ) -> Vec<f64> {
        let mut out = vec![0.0; rows];
        for (row_out, row) in out.iter_mut().zip(matrix.chunks_exact(cols)) {
            for (value, weight) in row.iter().zip(vector.iter()) {
                *row_out += value * weight;
            }
        }
        out
    }

    pub(crate) fn dgemv_row_major_trans(
        rows: usize,
        cols: usize,
        matrix: &[f64],
        vector: &[f64],
    ) -> Vec<f64> {
        let mut out = vec![0.0; cols];
        for (row, weight) in matrix.chunks_exact(cols).take(rows).zip(vector.iter()) {
            for (col, value) in row.iter().enumerate() {
                out[col] += value * weight;
            }
        }
        out
    }

    pub(crate) fn sgemv_row_major(
        rows: usize,
        cols: usize,
        matrix: &[f32],
        vector: &[f32],
    ) -> Vec<f32> {
        let mut out = vec![0.0; rows];
        for (row_out, row) in out.iter_mut().zip(matrix.chunks_exact(cols)) {
            for (value, weight) in row.iter().zip(vector.iter()) {
                *row_out += value * weight;
            }
        }
        out
    }

    pub(crate) fn sgemv_row_major_trans(
        rows: usize,
        cols: usize,
        matrix: &[f32],
        vector: &[f32],
    ) -> Vec<f32> {
        let mut out = vec![0.0; cols];
        for (row, weight) in matrix.chunks_exact(cols).take(rows).zip(vector.iter()) {
            for (col, value) in row.iter().enumerate() {
                out[col] += value * weight;
            }
        }
        out
    }

    pub(crate) fn dot_f64(left: &[f64], right: &[f64]) -> f64 {
        left.iter()
            .zip(right.iter())
            .map(|(left, right)| left * right)
            .sum()
    }

    pub(crate) fn dot_f32(left: &[f32], right: &[f32]) -> f32 {
        left.iter()
            .zip(right.iter())
            .map(|(left, right)| left * right)
            .sum()
    }

    pub(crate) fn mul_scalar_f64(values: &[f64], scalar: f64) -> Vec<f64> {
        values.iter().map(|value| value * scalar).collect()
    }

    pub(crate) fn mul_repeated_tile_f64(tile: &[f64], values: &[f64]) -> Vec<f64> {
        let tile_len = tile.len();
        debug_assert!(tile_len > 0);
        debug_assert_eq!(values.len() % tile_len, 0);

        let mut out = Vec::with_capacity(values.len());
        for chunk in values.chunks_exact(tile_len) {
            for (tile_value, value) in tile.iter().zip(chunk.iter()) {
                out.push(*tile_value * *value);
            }
        }
        out
    }

    pub(crate) fn weighted_axis1_sum_f64(
        rows: usize,
        cols: usize,
        matrix: &[f64],
        weights: &[f64],
    ) -> f64 {
        let mut acc = 0.0;
        for row in matrix.chunks_exact(cols).take(rows) {
            for (value, weight) in row.iter().zip(weights.iter()) {
                acc += value * weight;
            }
        }
        acc
    }

    pub(crate) fn sum_f64(values: &[f64]) -> f64 {
        values.iter().copied().sum()
    }

    pub(crate) fn sum_f32(values: &[f32]) -> f32 {
        values.iter().copied().sum()
    }
}

pub(crate) use imp::*;
