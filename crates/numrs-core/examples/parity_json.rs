use numrs_core::Array;

fn fmt_f64(values: &[f64]) -> String {
    values
        .iter()
        .map(|value| format!("{value:.12}"))
        .collect::<Vec<_>>()
        .join(",")
}

fn fmt_i64(values: &[i64]) -> String {
    values
        .iter()
        .map(|value| value.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

fn main() -> numrs_core::Result<()> {
    let col = Array::from_vec(vec![3, 1], vec![1.0_f64, 2.0, 3.0])?;
    let row = Array::from_vec(vec![1, 4], vec![10.0_f64, 20.0, 30.0, 40.0])?;
    let broadcast = col.add_outer2d_f64(&row)?;

    let a = Array::from_vec(vec![3, 4], (0_i64..12).collect())?;
    let take_axis = a.take_axis(&[-1, 1], 1)?;
    let mask = Array::from_vec(
        vec![3, 4],
        vec![
            true, false, false, true, false, true, false, false, true, false, true, false,
        ],
    )?;
    let masked = a.boolean_mask(&mask)?;

    let reductions = Array::from_vec(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0])?;
    let sum_axis0 = reductions.sum_axis(0)?;
    let mean_axis1 = reductions.mean_axis(1)?;

    let left = Array::from_vec(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0])?;
    let right = Array::from_vec(vec![3, 2], vec![7.0_f64, 8.0, 9.0, 10.0, 11.0, 12.0])?;
    let dot = left.dot2d(&right)?;
    let batched_left = Array::from_vec(
        vec![2, 2, 3],
        vec![
            1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 2.0, 0.0, 1.0, 3.0, 1.0, 4.0,
        ],
    )?;
    let batched_right = Array::from_vec(vec![1, 3, 2], vec![7.0_f64, 8.0, 9.0, 10.0, 11.0, 12.0])?;
    let matmul = batched_left.matmul(&batched_right)?;
    let tensor_left = Array::from_vec(
        vec![2, 3, 4],
        (0..24).map(|value| value as f64).collect::<Vec<_>>(),
    )?;
    let tensor_right = Array::from_vec(
        vec![3, 2, 5],
        (0..30).map(|value| value as f64).collect::<Vec<_>>(),
    )?;
    let tensordot = tensor_left.tensordot_axes(&tensor_right, &[1, 0], &[0, 1])?;

    println!(
        "{{\
         \"broadcast_outer_add\":[{}],\
         \"take_axis_i64\":[{}],\
         \"boolean_mask_i64\":[{}],\
         \"sum_axis0\":[{}],\
         \"mean_axis1\":[{}],\
         \"dot_f64\":[{}],\
         \"batched_matmul_f64\":[{}],\
         \"tensordot_f64\":[{}]\
         }}",
        fmt_f64(broadcast.as_slice()),
        fmt_i64(take_axis.as_slice()),
        fmt_i64(masked.as_slice()),
        fmt_f64(sum_axis0.as_slice()),
        fmt_f64(mean_axis1.as_slice()),
        fmt_f64(dot.as_slice()),
        fmt_f64(matmul.as_slice()),
        fmt_f64(tensordot.as_slice()),
    );

    Ok(())
}
