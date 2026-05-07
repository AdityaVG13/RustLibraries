use std::fs;
use std::hint::black_box;
use std::time::{Duration, Instant};

use numrs_core::Array;

struct Case {
    name: &'static str,
    millis: f64,
    checksum: f64,
}

fn median_ms(mut samples: Vec<Duration>) -> f64 {
    samples.sort_unstable();
    samples[samples.len() / 2].as_secs_f64() * 1000.0
}

fn bench<F>(mut f: F, rounds: usize) -> (f64, f64)
where
    F: FnMut() -> f64,
{
    let mut samples = Vec::with_capacity(rounds);
    let mut checksum = black_box(f());
    for _ in 0..rounds {
        let start = Instant::now();
        checksum = black_box(f());
        samples.push(start.elapsed());
    }
    (median_ms(samples), checksum)
}

fn edge_sum_f64(array: &Array<f64>) -> f64 {
    let values = array.as_slice();
    values[0] + values[values.len() - 1]
}

fn edge_sum_i64(array: &Array<i64>) -> f64 {
    let values = array.as_slice();
    (values[0] + values[values.len() - 1]) as f64
}

fn shape_checksum(shape: &[usize]) -> f64 {
    shape.iter().map(|dim| *dim as f64).sum::<f64>() + shape.len() as f64
}

fn read_lstsq_fixture() -> numrs_core::Result<(Array<f64>, Array<f64>)> {
    let bytes = fs::read("benchmark-results/numpy-asv-lstsq-f64.bin").map_err(|err| {
        numrs_core::NumRsError::InvalidShape(format!(
            "missing lstsq fixture; run benchmarks/external_numpy_cases.py: {err}"
        ))
    })?;
    let expected = (100 * 100 + 100) * std::mem::size_of::<f64>();
    if bytes.len() != expected {
        return Err(numrs_core::NumRsError::InvalidShape(format!(
            "lstsq fixture expected {expected} bytes, got {}",
            bytes.len()
        )));
    }

    let mut values = Vec::with_capacity(100 * 100 + 100);
    for chunk in bytes.chunks_exact(8) {
        let mut word = [0u8; 8];
        word.copy_from_slice(chunk);
        values.push(f64::from_le_bytes(word));
    }
    let b = values.split_off(100 * 100);
    Ok((
        Array::from_vec(vec![100, 100], values)?,
        Array::from_vec(vec![100], b)?,
    ))
}

fn main() -> numrs_core::Result<()> {
    let mut cases = Vec::new();

    let d = Array::from_shape_fn(vec![50_000, 100], |_| 1.0_f64)?;
    let e = Array::from_shape_fn(vec![100], |_| 1.0_f64)?;
    let (millis, checksum) = bench(
        || {
            let out = d.sub(&e).unwrap();
            edge_sum_f64(&out)
        },
        7,
    );
    cases.push(Case {
        name: "asv_ufunc_broadcast_sub_f64",
        millis,
        checksum,
    });

    let astype_source = Array::from_shape_fn(vec![100, 100], |idx| (idx[0] * 100 + idx[1]) as i32)?;
    let (millis, checksum) = bench(
        || {
            let mut checksum = 0.0;
            for _ in 0..5_000 {
                let out = astype_source.astype::<f64>().unwrap();
                checksum += edge_sum_f64(&out);
            }
            checksum
        },
        7,
    );
    cases.push(Case {
        name: "asv_ufunc_astype_i32_to_f64_100x100",
        millis,
        checksum,
    });

    let at_values = Array::from_shape_fn(vec![10_000_000], |idx| {
        (((idx[0] * 17) % 1000) as f64) / 1000.0
    })?;
    let at_indices: Vec<isize> = (0..10_000_000)
        .map(|idx| ((idx * 37) % 1000) as isize)
        .collect();
    let (millis, checksum) = bench(
        || {
            let mut result = Array::<f64>::zeros(vec![1000]).unwrap();
            result.add_at(&at_indices, at_values.as_slice()).unwrap();
            edge_sum_f64(&result)
        },
        7,
    );
    cases.push(Case {
        name: "asv_ufunc_add_at_f64_10000000",
        millis,
        checksum,
    });

    let (millis, checksum) = bench(
        || {
            let mut result = Array::<f64>::zeros(vec![1000]).unwrap();
            result
                .maximum_at(&at_indices, at_values.as_slice())
                .unwrap();
            edge_sum_f64(&result)
        },
        7,
    );
    cases.push(Case {
        name: "asv_ufunc_maximum_at_f64_10000000",
        millis,
        checksum,
    });

    let small = Array::full(vec![100], 1.0_f32)?;
    let (millis, checksum) = bench(
        || {
            let mut checksum = 0.0;
            for _ in 0..200_000 {
                checksum += small.sum_all().unwrap() as f64;
            }
            checksum
        },
        7,
    );
    cases.push(Case {
        name: "asv_reduce_small_sum_f32_100",
        millis,
        checksum,
    });

    let stats_data = Array::full(vec![200], 1.0_f64)?;
    let (millis, checksum) = bench(
        || {
            let mut checksum = 0.0;
            for _ in 0..100_000 {
                checksum += stats_data.min_all().unwrap();
            }
            checksum
        },
        7,
    );
    cases.push(Case {
        name: "asv_reduce_stats_min_f64_200",
        millis,
        checksum,
    });

    let (millis, checksum) = bench(
        || {
            let mut checksum = 0.0;
            for _ in 0..100_000 {
                checksum += stats_data.max_all().unwrap();
            }
            checksum
        },
        7,
    );
    cases.push(Case {
        name: "asv_reduce_stats_max_f64_200",
        millis,
        checksum,
    });

    let (millis, checksum) = bench(
        || {
            let mut checksum = 0.0;
            for _ in 0..100_000 {
                checksum += stats_data.mean_all().unwrap();
            }
            checksum
        },
        7,
    );
    cases.push(Case {
        name: "asv_reduce_stats_mean_f64_200",
        millis,
        checksum,
    });

    let (millis, checksum) = bench(
        || {
            let mut checksum = 0.0;
            for _ in 0..100_000 {
                checksum += stats_data.std_all().unwrap();
            }
            checksum
        },
        7,
    );
    cases.push(Case {
        name: "asv_reduce_stats_std_f64_200",
        millis,
        checksum,
    });

    let (millis, checksum) = bench(
        || {
            let mut checksum = 0.0;
            for _ in 0..100_000 {
                checksum += stats_data.prod_all().unwrap();
            }
            checksum
        },
        7,
    );
    cases.push(Case {
        name: "asv_reduce_stats_prod_f64_200",
        millis,
        checksum,
    });

    let (millis, checksum) = bench(
        || {
            let mut checksum = 0.0;
            for _ in 0..100_000 {
                checksum += stats_data.var_all().unwrap();
            }
            checksum
        },
        7,
    );
    cases.push(Case {
        name: "asv_reduce_stats_var_f64_200",
        millis,
        checksum,
    });

    let argmax_data = Array::<i64>::zeros(vec![200_000])?;
    let (millis, checksum) = bench(
        || {
            let mut checksum = 0.0;
            for _ in 0..20_000 {
                checksum += argmax_data.argmax().unwrap() as f64;
            }
            checksum
        },
        7,
    );
    cases.push(Case {
        name: "asv_reduce_argmax_i64_200000",
        millis,
        checksum,
    });

    let argmin_data = Array::full(vec![200_000], 1_i64)?;
    let (millis, checksum) = bench(
        || {
            let mut checksum = 0.0;
            for _ in 0..20_000 {
                checksum += argmin_data.argmin().unwrap() as f64;
            }
            checksum
        },
        7,
    );
    cases.push(Case {
        name: "asv_reduce_argmin_i64_200000",
        millis,
        checksum,
    });

    let take_arr = Array::full(vec![1000, 1], 1_i64)?;
    let take_indices: Vec<isize> = (0..1000).map(|idx| idx as isize).collect();
    let (millis, checksum) = bench(
        || {
            let mut checksum = 0.0;
            for _ in 0..2_000 {
                let out = take_arr.take_axis(&take_indices, -2).unwrap();
                checksum += edge_sum_i64(&out);
            }
            checksum
        },
        7,
    );
    cases.push(Case {
        name: "asv_itemselection_take_i64_1000x1",
        millis,
        checksum,
    });

    let dense_mask = Array::full(vec![1000], true)?;
    let sparse_mask = Array::full(vec![1000], false)?;
    let (millis, checksum) = bench(
        || {
            let mut arr = Array::full(vec![1000], 1.0_f64).unwrap();
            let mut checksum = 0.0;
            for _ in 0..10_000 {
                arr.putmask(&dense_mask, &[1.0]).unwrap();
                checksum += edge_sum_f64(&arr);
            }
            checksum
        },
        7,
    );
    cases.push(Case {
        name: "asv_itemselection_putmask_dense_scalar_f64_1000",
        millis,
        checksum,
    });

    let (millis, checksum) = bench(
        || {
            let mut arr = Array::full(vec![1000], 1.0_f64).unwrap();
            let mut checksum = 0.0;
            for _ in 0..10_000 {
                arr.putmask(&sparse_mask, &[1.0]).unwrap();
                checksum += edge_sum_f64(&arr);
            }
            checksum
        },
        7,
    );
    cases.push(Case {
        name: "asv_itemselection_putmask_sparse_scalar_f64_1000",
        millis,
        checksum,
    });

    let put_indices: Vec<isize> = (0..1000).map(|idx| idx as isize).collect();
    let put_values = vec![1.0_f64; 1000];
    let (millis, checksum) = bench(
        || {
            let mut arr = Array::full(vec![1000], 1.0_f64).unwrap();
            let mut checksum = 0.0;
            for _ in 0..10_000 {
                arr.put(&put_indices, &put_values).unwrap();
                checksum += edge_sum_f64(&arr);
            }
            checksum
        },
        7,
    );
    cases.push(Case {
        name: "asv_itemselection_put_ordered_f64_1000",
        millis,
        checksum,
    });

    let broadcast_source = Array::from_shape_fn(vec![512], |idx| idx[0] as f64)?;
    let (millis, checksum) = bench(
        || {
            let mut checksum = 0.0;
            for _ in 0..200_000 {
                let out = broadcast_source.broadcast_to(&[512, 512]).unwrap();
                checksum += shape_checksum(out.shape());
            }
            checksum
        },
        7,
    );
    cases.push(Case {
        name: "asv_manipulate_broadcast_to_f64_512",
        millis,
        checksum,
    });

    let concat_arrays = (0..5)
        .map(|array_idx| {
            Array::from_shape_fn(vec![32, 64], |idx| {
                (array_idx * 32 * 64 + idx[0] * 64 + idx[1]) as f64
            })
        })
        .collect::<numrs_core::Result<Vec<_>>>()?;
    let (millis, checksum) = bench(
        || {
            let mut checksum = 0.0;
            for _ in 0..2_000 {
                let out = Array::concatenate(&concat_arrays, 0).unwrap();
                checksum += edge_sum_f64(&out);
            }
            checksum
        },
        7,
    );
    cases.push(Case {
        name: "asv_manipulate_concatenate_ax0_f64_32x64_n5",
        millis,
        checksum,
    });

    let (millis, checksum) = bench(
        || {
            let mut checksum = 0.0;
            for _ in 0..2_000 {
                let out = Array::concatenate(&concat_arrays, 1).unwrap();
                checksum += edge_sum_f64(&out);
            }
            checksum
        },
        7,
    );
    cases.push(Case {
        name: "asv_manipulate_concatenate_ax1_f64_32x64_n5",
        millis,
        checksum,
    });

    let (millis, checksum) = bench(
        || {
            let mut checksum = 0.0;
            for _ in 0..2_000 {
                let out = Array::stack(&concat_arrays, 0).unwrap();
                checksum += edge_sum_f64(&out);
            }
            checksum
        },
        7,
    );
    cases.push(Case {
        name: "asv_manipulate_stack_ax0_f64_32x64_n5",
        millis,
        checksum,
    });

    let (millis, checksum) = bench(
        || {
            let mut checksum = 0.0;
            for _ in 0..2_000 {
                let out = Array::stack(&concat_arrays, 1).unwrap();
                checksum += edge_sum_f64(&out);
            }
            checksum
        },
        7,
    );
    cases.push(Case {
        name: "asv_manipulate_stack_ax1_f64_32x64_n5",
        millis,
        checksum,
    });

    let dims_source = Array::full(vec![5, 2, 3, 1], 1.0_f64)?;
    let (millis, checksum) = bench(
        || {
            let mut checksum = 0.0;
            for _ in 0..200_000 {
                let out = dims_source.expand_dims(1).unwrap();
                checksum += shape_checksum(out.shape());
            }
            checksum
        },
        7,
    );
    cases.push(Case {
        name: "asv_manipulate_expand_dims_f64_5x2x3x1_axis1",
        millis,
        checksum,
    });

    let (millis, checksum) = bench(
        || {
            let mut checksum = 0.0;
            for _ in 0..200_000 {
                let out = dims_source.expand_dims(-1).unwrap();
                checksum += shape_checksum(out.shape());
            }
            checksum
        },
        7,
    );
    cases.push(Case {
        name: "asv_manipulate_expand_dims_neg_f64_5x2x3x1",
        millis,
        checksum,
    });

    let (millis, checksum) = bench(
        || {
            let mut checksum = 0.0;
            for _ in 0..200_000 {
                let out = dims_source.squeeze(None).unwrap();
                checksum += shape_checksum(out.shape());
            }
            checksum
        },
        7,
    );
    cases.push(Case {
        name: "asv_manipulate_squeeze_dims_f64_5x2x3x1",
        millis,
        checksum,
    });

    let (millis, checksum) = bench(
        || {
            let mut checksum = 0.0;
            for _ in 0..200_000 {
                let out = dims_source.reshape(&[1, 5, 2, 3]).unwrap();
                checksum += shape_checksum(out.shape());
            }
            checksum
        },
        7,
    );
    cases.push(Case {
        name: "asv_manipulate_reshape_f64_5x2x3x1_to_1x5x2x3",
        millis,
        checksum,
    });

    let a = Array::from_shape_fn(vec![150, 400], |idx| (idx[0] * 400 + idx[1]) as f64)?;
    let b = Array::from_shape_fn(vec![400, 600], |idx| (idx[0] * 600 + idx[1]) as f64)?;
    let (millis, checksum) = bench(
        || {
            let mut checksum = 0.0;
            for _ in 0..1000 {
                let out = a.dot2d(&b).unwrap();
                checksum += edge_sum_f64(&out);
            }
            checksum
        },
        7,
    );
    cases.push(Case {
        name: "asv_linalg_dot_a_b_f64_150x400_400x600",
        millis,
        checksum,
    });

    let (millis, checksum) = bench(
        || {
            let mut checksum = 0.0;
            for _ in 0..1000 {
                let out = a.matmul(&b).unwrap();
                checksum += edge_sum_f64(&out);
            }
            checksum
        },
        7,
    );
    cases.push(Case {
        name: "asv_linalg_matmul_a_b_f64_150x400_400x600",
        millis,
        checksum,
    });

    let c = Array::from_shape_fn(vec![600], |idx| idx[0] as f64)?;
    let d_vec = Array::from_shape_fn(vec![400], |idx| idx[0] as f64)?;
    let (millis, checksum) = bench(
        || {
            let mut checksum = 0.0;
            for _ in 0..1000 {
                let tmp = b.matmul(&c).unwrap();
                let out = d_vec.matmul(&tmp).unwrap();
                checksum += edge_sum_f64(&out);
            }
            checksum
        },
        7,
    );
    cases.push(Case {
        name: "asv_linalg_matmul_d_matmul_b_c_f64",
        millis,
        checksum,
    });

    let (millis, checksum) = bench(
        || {
            let mut checksum = 0.0;
            for _ in 0..1000 {
                let tmp = b.matmul(&c).unwrap();
                let out = d_vec.matmul(&tmp).unwrap();
                checksum += edge_sum_f64(&out);
            }
            checksum
        },
        7,
    );
    cases.push(Case {
        name: "asv_linalg_dot_d_dot_b_c_f64",
        millis,
        checksum,
    });

    let a_view = a.view();
    let at = a_view.transpose();
    let atc = Array::from_shape_fn(vec![400, 150], |idx| (idx[1] * 400 + idx[0]) as f64)?;
    let (millis, checksum) = bench(
        || {
            let mut checksum = 0.0;
            for _ in 0..1000 {
                let out = a_view.dot2d(&at).unwrap();
                checksum += edge_sum_f64(&out);
            }
            checksum
        },
        7,
    );
    cases.push(Case {
        name: "asv_linalg_dot_trans_a_at_f64_150x400_400x150",
        millis,
        checksum,
    });

    let (millis, checksum) = bench(
        || {
            let mut checksum = 0.0;
            for _ in 0..1000 {
                let out = a_view.dot2d(&atc.view()).unwrap();
                checksum += edge_sum_f64(&out);
            }
            checksum
        },
        7,
    );
    cases.push(Case {
        name: "asv_linalg_dot_trans_a_atc_f64_150x400_400x150",
        millis,
        checksum,
    });

    let (millis, checksum) = bench(
        || {
            let mut checksum = 0.0;
            for _ in 0..1000 {
                let out = at.dot2d(&a_view).unwrap();
                checksum += edge_sum_f64(&out);
            }
            checksum
        },
        7,
    );
    cases.push(Case {
        name: "asv_linalg_dot_trans_at_a_f64_400x150_150x400",
        millis,
        checksum,
    });

    let (millis, checksum) = bench(
        || {
            let mut checksum = 0.0;
            for _ in 0..1000 {
                let out = atc.view().dot2d(&a_view).unwrap();
                checksum += edge_sum_f64(&out);
            }
            checksum
        },
        7,
    );
    cases.push(Case {
        name: "asv_linalg_dot_trans_atc_a_f64_400x150_150x400",
        millis,
        checksum,
    });

    let ac = a.clone();
    let (millis, checksum) = bench(
        || {
            let mut checksum = 0.0;
            for _ in 0..1000 {
                let out = a.inner2d(&a).unwrap();
                checksum += edge_sum_f64(&out);
            }
            checksum
        },
        7,
    );
    cases.push(Case {
        name: "asv_linalg_inner_a_a_f64_150x400_150x400",
        millis,
        checksum,
    });

    let (millis, checksum) = bench(
        || {
            let mut checksum = 0.0;
            for _ in 0..1000 {
                let out = a.inner2d(&ac).unwrap();
                checksum += edge_sum_f64(&out);
            }
            checksum
        },
        7,
    );
    cases.push(Case {
        name: "asv_linalg_inner_a_ac_f64_150x400_150x400",
        millis,
        checksum,
    });

    let (millis, checksum) = bench(
        || {
            let mut checksum = 0.0;
            for _ in 0..1000 {
                let out = a_view.matmul(&at).unwrap();
                checksum += edge_sum_f64(&out);
            }
            checksum
        },
        7,
    );
    cases.push(Case {
        name: "asv_linalg_matmul_trans_a_at_f64_150x400_400x150",
        millis,
        checksum,
    });

    let (millis, checksum) = bench(
        || {
            let mut checksum = 0.0;
            for _ in 0..1000 {
                let out = a_view.matmul(&atc.view()).unwrap();
                checksum += edge_sum_f64(&out);
            }
            checksum
        },
        7,
    );
    cases.push(Case {
        name: "asv_linalg_matmul_trans_a_atc_f64_150x400_400x150",
        millis,
        checksum,
    });

    let (millis, checksum) = bench(
        || {
            let mut checksum = 0.0;
            for _ in 0..1000 {
                let out = at.matmul(&a_view).unwrap();
                checksum += edge_sum_f64(&out);
            }
            checksum
        },
        7,
    );
    cases.push(Case {
        name: "asv_linalg_matmul_trans_at_a_f64_400x150_150x400",
        millis,
        checksum,
    });

    let (millis, checksum) = bench(
        || {
            let mut checksum = 0.0;
            for _ in 0..1000 {
                let out = atc.view().matmul(&a_view).unwrap();
                checksum += edge_sum_f64(&out);
            }
            checksum
        },
        7,
    );
    cases.push(Case {
        name: "asv_linalg_matmul_trans_atc_a_f64_400x150_150x400",
        millis,
        checksum,
    });

    let a3 = Array::from_shape_fn(vec![60, 80, 100], |idx| {
        (idx[0] * 80 * 100 + idx[1] * 100 + idx[2]) as f64
    })?;
    let b3 = Array::from_shape_fn(vec![80, 60, 40], |idx| {
        (idx[0] * 60 * 40 + idx[1] * 40 + idx[2]) as f64
    })?;
    let (millis, checksum) = bench(
        || {
            let mut checksum = 0.0;
            for _ in 0..10 {
                let out = a3.tensordot_axes(&b3, &[1, 0], &[0, 1]).unwrap();
                checksum += edge_sum_f64(&out);
            }
            checksum
        },
        7,
    );
    cases.push(Case {
        name: "asv_linalg_tensordot_a3_b3_axes_10_01",
        millis,
        checksum,
    });

    let (millis, checksum) = bench(
        || {
            let mut checksum = 0.0;
            for _ in 0..1000 {
                checksum += b.bilinear_form(&d_vec, &c).unwrap();
            }
            checksum
        },
        7,
    );
    cases.push(Case {
        name: "asv_linalg_einsum_i_ij_j_f64_400_400x600_600",
        millis,
        checksum,
    });

    let (millis, checksum) = bench(
        || {
            let mut checksum = 0.0;
            for _ in 0..1000 {
                let out = a.dot2d(&b).unwrap();
                checksum += edge_sum_f64(&out);
            }
            checksum
        },
        7,
    );
    cases.push(Case {
        name: "asv_linalg_einsum_ij_jk_f64_150x400_400x600",
        millis,
        checksum,
    });

    let array_5 = Array::from_shape_fn(vec![5], |idx| idx[0] as f64)?;
    let (millis, checksum) = bench(
        || {
            let mut checksum = 0.0;
            for _ in 0..100_000 {
                checksum += array_5.norm_l2().unwrap();
            }
            checksum
        },
        7,
    );
    cases.push(Case {
        name: "asv_linalg_norm_small_array_f64_5",
        millis,
        checksum,
    });

    let array_5_5 = Array::from_shape_fn(vec![5, 5], |idx| (idx[0] * 5 + idx[1]) as f64)?;
    let (millis, checksum) = bench(
        || {
            let mut checksum = 0.0;
            for _ in 0..100_000 {
                checksum += array_5_5.det().unwrap();
            }
            checksum
        },
        7,
    );
    cases.push(Case {
        name: "asv_linalg_det_small_array_f64_5x5",
        millis,
        checksum,
    });

    let array_3_3 = Array::from_shape_fn(vec![3, 3], |idx| {
        let eye = if idx[0] == idx[1] { 1.0 } else { 0.0 };
        eye + (idx[0] * 3 + idx[1]) as f64
    })?;
    let (millis, checksum) = bench(
        || {
            let mut checksum = 0.0;
            for _ in 0..100_000 {
                checksum += array_3_3.det().unwrap();
            }
            checksum
        },
        7,
    );
    cases.push(Case {
        name: "asv_linalg_det_3x3_f64",
        millis,
        checksum,
    });

    let array_3 = Array::from_shape_fn(vec![3], |idx| idx[0] as f64)?;
    let (millis, checksum) = bench(
        || {
            let mut checksum = 0.0;
            for _ in 0..100_000 {
                let out = array_3_3.solve(&array_3).unwrap();
                checksum += edge_sum_f64(&out);
            }
            checksum
        },
        7,
    );
    cases.push(Case {
        name: "asv_linalg_solve_3x3_f64",
        millis,
        checksum,
    });

    let (lstsq_a, lstsq_b) = read_lstsq_fixture()?;
    let (millis, checksum) = bench(
        || {
            let mut checksum = 0.0;
            for _ in 0..100 {
                let out = lstsq_a.solve(&lstsq_b).unwrap();
                checksum += edge_sum_f64(&out);
            }
            checksum
        },
        7,
    );
    cases.push(Case {
        name: "asv_linalg_lstsq_square_f64_100x100",
        millis,
        checksum,
    });

    let one_dim = Array::from_shape_fn(vec![3000], |idx| idx[0] as f64)?;
    let (millis, checksum) = bench(
        || {
            let out = one_dim.outer_product(&one_dim).unwrap();
            edge_sum_f64(&out)
        },
        7,
    );
    cases.push(Case {
        name: "asv_linalg_einsum_outer_f64_3000",
        millis,
        checksum,
    });

    let two_dim_small = Array::from_shape_fn(vec![30, 40], |idx| (idx[0] * 40 + idx[1]) as f64)?;
    let three_dim = Array::from_shape_fn(vec![20, 30, 40], |idx| {
        (idx[0] * 30 * 40 + idx[1] * 40 + idx[2]) as f64
    })?;
    let (millis, checksum) = bench(
        || {
            let out = two_dim_small.mul(&three_dim).unwrap();
            edge_sum_f64(&out)
        },
        7,
    );
    cases.push(Case {
        name: "asv_linalg_einsum_multiply_f64_30x40_20x30x40",
        millis,
        checksum,
    });

    let three_dim_small = Array::from_shape_fn(vec![10, 100, 10], |idx| {
        (idx[0] * 100 * 10 + idx[1] * 10 + idx[2]) as f64
    })?;
    let (millis, checksum) = bench(
        || {
            let mut checksum = 0.0;
            for _ in 0..100 {
                checksum += three_dim_small.sum_all().unwrap() * 300.0;
            }
            checksum
        },
        7,
    );
    cases.push(Case {
        name: "asv_linalg_einsum_sum_mul_f64_scalar_10x100x10",
        millis,
        checksum,
    });

    let (millis, checksum) = bench(
        || {
            let mut checksum = 0.0;
            for _ in 0..100 {
                checksum += three_dim_small.sum_all().unwrap() * 300.0;
            }
            checksum
        },
        7,
    );
    cases.push(Case {
        name: "asv_linalg_einsum_sum_mul2_f64_10x100x10_scalar",
        millis,
        checksum,
    });

    let one_dim_big = Array::from_shape_fn(vec![480_000], |idx| idx[0] as f64)?;
    let (millis, checksum) = bench(
        || {
            let mut checksum = 0.0;
            for _ in 0..100 {
                let out = one_dim_big.mul_scalar(300.0).unwrap();
                checksum += edge_sum_f64(&out);
            }
            checksum
        },
        7,
    );
    cases.push(Case {
        name: "asv_linalg_einsum_scalar_mul_f64_480000",
        millis,
        checksum,
    });

    let (millis, checksum) = bench(
        || {
            let mut checksum = 0.0;
            for _ in 0..100 {
                checksum += one_dim_big.sum_all().unwrap();
            }
            checksum
        },
        7,
    );
    cases.push(Case {
        name: "asv_linalg_einsum_sum_f64_480000",
        millis,
        checksum,
    });

    let two_dim = Array::from_shape_fn(vec![400, 600], |idx| (idx[0] * 600 + idx[1]) as f64)?;
    let one_dim_small = Array::from_shape_fn(vec![600], |idx| idx[0] as f64)?;
    let (millis, checksum) = bench(
        || {
            let mut checksum = 0.0;
            for _ in 0..100 {
                checksum += two_dim.weighted_axis1_sum(&one_dim_small).unwrap();
            }
            checksum
        },
        7,
    );
    cases.push(Case {
        name: "asv_linalg_einsum_weighted_sum_f64_400x600",
        millis,
        checksum,
    });

    let noncon_dim1 = Array::from_shape_fn(vec![2000], |idx| (1 + idx[0] * 2) as f64)?;
    let (millis, checksum) = bench(
        || {
            let out = noncon_dim1.outer_product(&noncon_dim1).unwrap();
            edge_sum_f64(&out)
        },
        7,
    );
    cases.push(Case {
        name: "asv_linalg_einsum_noncon_outer_f64_2000",
        millis,
        checksum,
    });

    let noncon_dim2 =
        Array::from_shape_fn(vec![30, 40], |idx| (1 + (idx[0] * 40 + idx[1]) * 2) as f64)?;
    let noncon_dim3 = Array::from_shape_fn(vec![20, 30, 40], |idx| {
        (1 + (idx[0] * 30 * 40 + idx[1] * 40 + idx[2]) * 2) as f64
    })?;
    let (millis, checksum) = bench(
        || {
            let out = noncon_dim2.mul(&noncon_dim3).unwrap();
            edge_sum_f64(&out)
        },
        7,
    );
    cases.push(Case {
        name: "asv_linalg_einsum_noncon_multiply_f64_30x40_20x30x40",
        millis,
        checksum,
    });

    let (millis, checksum) = bench(
        || {
            let mut checksum = 0.0;
            for _ in 0..100 {
                checksum += noncon_dim3.sum_all().unwrap() * 300.0;
            }
            checksum
        },
        7,
    );
    cases.push(Case {
        name: "asv_linalg_einsum_noncon_sum_mul_f64_scalar_20x30x40",
        millis,
        checksum,
    });

    let (millis, checksum) = bench(
        || {
            let mut checksum = 0.0;
            for _ in 0..100 {
                checksum += noncon_dim3.sum_all().unwrap() * 300.0;
            }
            checksum
        },
        7,
    );
    cases.push(Case {
        name: "asv_linalg_einsum_noncon_sum_mul2_f64_20x30x40_scalar",
        millis,
        checksum,
    });

    let (millis, checksum) = bench(
        || {
            let mut checksum = 0.0;
            for _ in 0..100 {
                let out = noncon_dim1.mul_scalar(300.0).unwrap();
                checksum += edge_sum_f64(&out);
            }
            checksum
        },
        7,
    );
    cases.push(Case {
        name: "asv_linalg_einsum_noncon_scalar_mul_f64_2000",
        millis,
        checksum,
    });

    let noncon_dim1_small = Array::from_shape_fn(vec![40], |idx| (1 + idx[0] * 2) as f64)?;
    let (millis, checksum) = bench(
        || {
            let mut checksum = 0.0;
            for _ in 0..100 {
                checksum += noncon_dim2.weighted_axis1_sum(&noncon_dim1_small).unwrap();
            }
            checksum
        },
        7,
    );
    cases.push(Case {
        name: "asv_linalg_einsum_noncon_weighted_sum_f64_30x40",
        millis,
        checksum,
    });

    let (millis, checksum) = bench(
        || {
            let mut checksum = 0.0;
            for _ in 0..100 {
                checksum += noncon_dim1.sum_all().unwrap();
            }
            checksum
        },
        7,
    );
    cases.push(Case {
        name: "asv_linalg_einsum_noncon_sum_f64_2000",
        millis,
        checksum,
    });

    print!("{{\"engine\":\"numrust\",\"cases\":[");
    for (idx, case) in cases.iter().enumerate() {
        if idx > 0 {
            print!(",");
        }
        print!(
            "{{\"name\":\"{}\",\"millis\":{:.6},\"checksum\":{:.12}}}",
            case.name, case.millis, case.checksum
        );
    }
    println!("]}}");

    Ok(())
}
