use framerust::{bench_median_ms, Aggregation, Column, Frame};

fn checksum(frame: &Frame) -> f64 {
    let sales_sum = frame.column("sales_sum").unwrap().as_f64().unwrap();
    let weight_mean = frame.column("weight_mean").unwrap().as_f64().unwrap();
    let rows = frame.column("rows").unwrap().as_i64().unwrap();
    sales_sum.iter().sum::<f64>()
        + weight_mean.iter().sum::<f64>()
        + rows.iter().map(|value| *value as f64).sum::<f64>()
}

fn main() {
    let rows = 250_000;
    let groups = 1_000;
    let mut keys = Vec::with_capacity(rows);
    let mut sales = Vec::with_capacity(rows);
    let mut weights = Vec::with_capacity(rows);
    for row in 0..rows {
        keys.push((row % groups) as i64);
        sales.push(((row % 97) as f64) * 0.5 + 1.0);
        weights.push((row % 31) as f64);
    }

    let frame = Frame::from_columns([
        ("store", Column::I64(keys)),
        ("sales", Column::F64(sales)),
        ("weight", Column::F64(weights)),
    ])
    .unwrap();
    let aggs = [
        Aggregation::count("rows"),
        Aggregation::sum("sales", "sales_sum"),
        Aggregation::mean("weight", "weight_mean"),
    ];
    let warmup = frame.groupby("store").agg(&aggs).unwrap();
    let checksum_value = checksum(&warmup);
    let millis = bench_median_ms(9, || {
        let out = frame.groupby("store").agg(&aggs).unwrap();
        std::hint::black_box(out);
    });

    println!(
        "{{\"engine\":\"framerust\",\"cases\":[{{\"name\":\"groupby_i64_sum_mean_count_250k_1000\",\"millis\":{millis:.6},\"checksum\":{checksum_value:.12}}}]}}"
    );
}
