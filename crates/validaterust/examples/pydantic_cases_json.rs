use validaterust::{bench_median_ms, Field, Schema, Value, ValueType};

fn schema() -> Schema {
    Schema::new([
        Field::required("id", ValueType::Int).number_range(0, 1_000_000),
        Field::required("email", ValueType::Text).min_len(5),
        Field::required("score", ValueType::Float),
        Field::required("active", ValueType::Bool),
        Field::required("tags", ValueType::List(Box::new(ValueType::Text))),
    ])
}

fn records(count: usize) -> Vec<Value> {
    (0..count)
        .map(|idx| {
            Value::object([
                ("id", Value::Int(idx as i64)),
                ("email", Value::Text(format!("user{idx}@example.test"))),
                ("score", Value::Float((idx % 100) as f64 + 0.25)),
                ("active", Value::Bool(idx % 2 == 0)),
                (
                    "tags",
                    Value::list([
                        Value::Text(format!("segment{}", idx % 17)),
                        Value::Text(format!("cohort{}", idx % 11)),
                    ]),
                ),
            ])
        })
        .collect()
}

fn checksum(records: &[validaterust::ValidatedObject<'_>]) -> u64 {
    let mut out = 0u64;
    for record in records {
        if let Some(Value::Int(id)) = record.get("id") {
            out = out.wrapping_add(*id as u64);
        }
        if let Some(Value::Text(email)) = record.get("email") {
            out = out.wrapping_add(email.len() as u64);
        }
        if let Some(Value::List(tags)) = record.get("tags") {
            out = out.wrapping_add(tags.len() as u64);
        }
    }
    out
}

fn main() {
    let schema = schema();
    let records = records(100_000);
    let warmup = schema.validate_many(&records).unwrap();
    let checksum_value = checksum(&warmup);
    let millis = bench_median_ms(9, || {
        let validated = schema.validate_many(&records).unwrap();
        std::hint::black_box(validated);
    });
    println!(
        "{{\"engine\":\"validaterust\",\"cases\":[{{\"name\":\"user_schema_100000\",\"millis\":{millis:.6},\"checksum\":{checksum_value}}}]}}"
    );
}
