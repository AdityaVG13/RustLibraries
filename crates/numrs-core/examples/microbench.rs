use std::hint::black_box;
use std::time::Instant;

use numrs_core::Array;

fn main() -> numrs_core::Result<()> {
    let n = 250_000usize;
    let a = Array::from_shape_fn(vec![n], |idx| idx[0] as f64)?;
    let b = Array::from_shape_fn(vec![n], |idx| (idx[0] as f64) * 0.5)?;

    let start = Instant::now();
    let mut checksum = 0.0;
    for _ in 0..50 {
        let out = a.add(&b)?;
        checksum += out.sum_all()?;
    }
    let elapsed = start.elapsed();
    println!(
        "contiguous add: elems={} iters=50 elapsed_ms={} checksum={}",
        n,
        elapsed.as_millis(),
        black_box(checksum)
    );

    let column = Array::from_shape_fn(vec![1024, 1], |idx| idx[0] as f64)?;
    let row = Array::from_shape_fn(vec![1, 1024], |idx| idx[1] as f64)?;

    let start = Instant::now();
    let out = column.add(&row)?;
    let elapsed = start.elapsed();
    println!(
        "broadcast add: shape={:?} elapsed_ms={} checksum={}",
        out.shape(),
        elapsed.as_millis(),
        black_box(out.sum_all()?)
    );

    Ok(())
}
