use numrs_core::{Array, Slice};

fn main() -> numrs_core::Result<()> {
    let a = Array::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
    let row_bias = Array::from_vec(vec![1, 3], vec![10.0, 20.0, 30.0])?;

    let shifted = a.add(&row_bias)?;
    println!("{:?}", shifted.as_slice());

    let second_col = shifted.slice(&[Slice::All, Slice::Index(1)])?;
    println!("{:?}", second_col.to_vec()?);

    Ok(())
}
