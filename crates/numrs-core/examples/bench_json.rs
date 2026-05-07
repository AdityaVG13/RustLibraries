use std::hint::black_box;
use std::time::{Duration, Instant};

use numrs_core::Array;

fn median_ms(mut samples: Vec<Duration>) -> f64 {
    samples.sort_unstable();
    samples[samples.len() / 2].as_secs_f64() * 1000.0
}

fn bench<F>(mut f: F, rounds: usize) -> f64
where
    F: FnMut(),
{
    let mut samples = Vec::with_capacity(rounds);
    f();
    for _ in 0..rounds {
        let start = Instant::now();
        f();
        samples.push(start.elapsed());
    }
    median_ms(samples)
}

fn main() -> numrs_core::Result<()> {
    let small_a = Array::from_shape_fn(vec![32], |idx| idx[0] as f64)?;
    let small_b = Array::from_shape_fn(vec![32], |idx| (idx[0] as f64) * 0.5)?;
    let add_small = bench(
        || {
            let mut checksum = 0.0;
            for _ in 0..20_000 {
                checksum += small_a.add(&small_b).unwrap().sum_all().unwrap();
            }
            black_box(checksum);
        },
        7,
    );

    let add_a = Array::from_shape_fn(vec![250_000], |idx| idx[0] as f64)?;
    let add_b = Array::from_shape_fn(vec![250_000], |idx| (idx[0] as f64) * 0.5)?;
    let add_large = bench(
        || {
            let mut checksum = 0.0;
            for _ in 0..50 {
                checksum += add_a.add(&add_b).unwrap().sum_all().unwrap();
            }
            black_box(checksum);
        },
        7,
    );

    let fused_add_sum = bench(
        || {
            let mut checksum = 0.0;
            for _ in 0..50 {
                checksum += add_a.add_sum_f64(&add_b).unwrap();
            }
            black_box(checksum);
        },
        7,
    );

    let col = Array::from_shape_fn(vec![1024, 1], |idx| idx[0] as f64)?;
    let row = Array::from_shape_fn(vec![1, 1024], |idx| idx[1] as f64)?;
    let broadcast = bench(
        || {
            let out = col.add_outer2d_f64(&row).unwrap();
            black_box(out.sum_all().unwrap());
        },
        7,
    );

    let sum_data = Array::from_shape_fn(vec![1_000_000], |idx| (idx[0] % 1024) as f64)?;
    let sum = bench(
        || {
            let mut checksum = 0.0;
            for _ in 0..100 {
                checksum += sum_data.sum_all().unwrap();
            }
            black_box(checksum);
        },
        7,
    );

    let metadata = Array::from_shape_fn(vec![4, 8], |idx| (idx[0] * 8 + idx[1]) as f64)?;
    let metadata_views = bench(
        || {
            let mut checksum = 0usize;
            for _ in 0..200_000 {
                let view = metadata
                    .transpose()
                    .expand_dims(0)
                    .unwrap()
                    .squeeze(Some(0))
                    .unwrap();
                checksum += view.shape()[0];
            }
            black_box(checksum);
        },
        7,
    );

    let take_source = Array::from_shape_fn(vec![256, 16], |idx| (idx[0] * 16 + idx[1]) as i64)?;
    let take_indices = [15, 3, 1, 7, 0, 14, 2, 10];
    let take_axis = bench(
        || {
            let mut checksum = 0_i64;
            for _ in 0..5_000 {
                checksum += take_source
                    .take_axis(&take_indices, 1)
                    .unwrap()
                    .sum_all()
                    .unwrap();
            }
            black_box(checksum);
        },
        7,
    );

    let left = Array::from_shape_fn(vec![192, 192], |idx| ((idx[0] * 17 + idx[1]) % 97) as f64)?;
    let right = Array::from_shape_fn(vec![192, 192], |idx| ((idx[0] + idx[1] * 31) % 89) as f64)?;
    let dot = bench(
        || {
            let mut checksum = 0.0;
            for _ in 0..500 {
                let out = left.dot2d(&right).unwrap();
                checksum += out.as_slice()[0];
            }
            black_box(checksum);
        },
        7,
    );

    println!(
        "{{\"engine\":\"numrust\",\"cases\":[\
         {{\"name\":\"small_add_f64_loop\",\"millis\":{add_small:.6}}},\
         {{\"name\":\"large_add_f64_loop\",\"millis\":{add_large:.6}}},\
         {{\"name\":\"fused_add_sum_f64_loop\",\"millis\":{fused_add_sum:.6}}},\
         {{\"name\":\"broadcast_add_f64\",\"millis\":{broadcast:.6}}},\
         {{\"name\":\"sum_f64_loop\",\"millis\":{sum:.6}}},\
         {{\"name\":\"metadata_view_loop\",\"millis\":{metadata_views:.6}}},\
         {{\"name\":\"take_axis_i64_loop\",\"millis\":{take_axis:.6}}},\
         {{\"name\":\"dot_f64_192\",\"millis\":{dot:.6}}}\
         ]}}"
    );

    Ok(())
}
