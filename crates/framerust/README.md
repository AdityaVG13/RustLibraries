# FrameRust

FrameRust is the Rust data aggregation crate in this workspace.

Current scope:

- Typed in-memory columns: `f64`, `i64`, `bool`, and `String`.
- Data-frame construction with length validation.
- First-seen-order `groupby`.
- Aggregations: `count`, `sum`, `mean`, `min`, and `max`.
- A same-data Pandas comparison harness.

It is not a Pandas replacement yet. Joins, indexes, time series, Arrow/Parquet interop, rolling windows, missing-value semantics, and expression planning are still roadmap items.
