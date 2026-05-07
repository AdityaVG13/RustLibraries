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
    let mut checksum = std::hint::black_box(f());
    for _ in 0..rounds {
        let start = Instant::now();
        checksum = std::hint::black_box(f());
        samples.push(start.elapsed());
    }
    (median_ms(samples), checksum)
}

fn emit_json(cases: &[Case]) {
    print!("{{\"engine\":\"numrust\",\"cases\":[");
    for (index, case) in cases.iter().enumerate() {
        if index > 0 {
            print!(",");
        }
        print!(
            "{{\"name\":\"{}\",\"millis\":{:.6},\"checksum\":{:.6}}}",
            case.name, case.millis, case.checksum
        );
    }
    println!("]}}");
}

fn main() -> numrs_core::Result<()> {
    let mut cases = Vec::new();

    let small_a = Array::from_shape_fn(vec![32], |idx| idx[0] as f64)?;
    let small_b = Array::from_shape_fn(vec![32], |idx| (idx[0] as f64) * 0.5)?;
    let (millis, checksum) = bench(
        || {
            let mut checksum = 0.0;
            for _ in 0..20_000 {
                checksum += small_a.add(&small_b).unwrap().sum_all().unwrap();
            }
            checksum
        },
        7,
    );
    cases.push(Case {
        name: "small_add_f64_loop",
        millis,
        checksum,
    });

    let add_a = Array::from_shape_fn(vec![250_000], |idx| idx[0] as f64)?;
    let add_b = Array::from_shape_fn(vec![250_000], |idx| (idx[0] as f64) * 0.5)?;
    let (millis, checksum) = bench(
        || {
            let mut checksum = 0.0;
            for _ in 0..50 {
                checksum += add_a.add(&add_b).unwrap().sum_all().unwrap();
            }
            checksum
        },
        7,
    );
    cases.push(Case {
        name: "large_add_f64_loop",
        millis,
        checksum,
    });

    let (millis, checksum) = bench(
        || {
            let mut checksum = 0.0;
            for _ in 0..50 {
                checksum += add_a.add_sum_f64(&add_b).unwrap();
            }
            checksum
        },
        7,
    );
    cases.push(Case {
        name: "fused_add_sum_f64_loop",
        millis,
        checksum,
    });

    let col = Array::from_shape_fn(vec![1024, 1], |idx| idx[0] as f64)?;
    let row = Array::from_shape_fn(vec![1, 1024], |idx| idx[1] as f64)?;
    let (millis, checksum) = bench(
        || {
            let out = col.add_outer2d_f64(&row).unwrap();
            out.sum_all().unwrap()
        },
        7,
    );
    cases.push(Case {
        name: "broadcast_add_f64",
        millis,
        checksum,
    });

    let sum_data = Array::from_shape_fn(vec![1_000_000], |idx| (idx[0] % 1024) as f64)?;
    let (millis, checksum) = bench(
        || {
            let mut checksum = 0.0;
            for _ in 0..100 {
                checksum += sum_data.sum_all().unwrap();
            }
            checksum
        },
        7,
    );
    cases.push(Case {
        name: "sum_f64_loop",
        millis,
        checksum,
    });

    let metadata = Array::from_shape_fn(vec![4, 8], |idx| (idx[0] * 8 + idx[1]) as f64)?;
    let (millis, checksum) = bench(
        || {
            let mut checksum = 0.0;
            for _ in 0..200_000 {
                let view = metadata
                    .transpose()
                    .expand_dims(0)
                    .unwrap()
                    .squeeze(Some(0))
                    .unwrap();
                checksum += view.shape()[0] as f64;
            }
            checksum
        },
        7,
    );
    cases.push(Case {
        name: "metadata_view_loop",
        millis,
        checksum,
    });

    let take_source = Array::from_shape_fn(vec![256, 16], |idx| (idx[0] * 16 + idx[1]) as i64)?;
    let take_indices = [15, 3, 1, 7, 0, 14, 2, 10];
    let (millis, checksum) = bench(
        || {
            let mut checksum = 0_i64;
            for _ in 0..5_000 {
                checksum += take_source
                    .take_axis(&take_indices, 1)
                    .unwrap()
                    .sum_all()
                    .unwrap();
            }
            checksum as f64
        },
        7,
    );
    cases.push(Case {
        name: "take_axis_i64_loop",
        millis,
        checksum,
    });

    let where_mask =
        Array::from_shape_fn(vec![512, 512], |idx| (idx[0] * 17 + idx[1] * 31) % 5 == 0)?;
    let where_true =
        Array::from_shape_fn(vec![512, 512], |idx| ((idx[0] * 13 + idx[1]) % 101) as f64)?;
    let where_false = Array::from_shape_fn(vec![1, 512], |idx| -((idx[1] % 97) as f64))?;
    let (millis, checksum) = bench(
        || {
            let mut checksum = 0.0;
            for _ in 0..400 {
                checksum += where_mask
                    .where_select_f64(&where_true, &where_false)
                    .unwrap()
                    .sum_all()
                    .unwrap();
            }
            checksum
        },
        7,
    );
    cases.push(Case {
        name: "where_select_f64_loop",
        millis,
        checksum,
    });

    let (millis, checksum) = bench(
        || {
            let mut checksum = 0.0;
            for _ in 0..2_000 {
                let coordinates = where_mask.nonzero().unwrap();
                for axis in &coordinates {
                    if let (Some(first), Some(last)) =
                        (axis.as_slice().first(), axis.as_slice().last())
                    {
                        checksum += (*first + *last) as f64;
                    }
                }
            }
            checksum
        },
        7,
    );
    cases.push(Case {
        name: "nonzero_bool_loop",
        millis,
        checksum,
    });

    let left = Array::from_shape_fn(vec![192, 192], |idx| ((idx[0] * 17 + idx[1]) % 97) as f64)?;
    let right = Array::from_shape_fn(vec![192, 192], |idx| ((idx[0] + idx[1] * 31) % 89) as f64)?;
    let (millis, checksum) = bench(
        || {
            let mut checksum = 0.0;
            for _ in 0..500 {
                let out = left.dot2d(&right).unwrap();
                checksum += out.as_slice()[0];
            }
            checksum
        },
        7,
    );
    cases.push(Case {
        name: "dot_f64_192",
        millis,
        checksum,
    });

    emit_json(&cases);
    Ok(())
}
