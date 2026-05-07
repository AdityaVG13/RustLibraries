# FrameRust vs Pandas

Same-data local benchmark for the implemented FrameRust aggregation slice.
This is not a full Pandas replacement claim.

- Pandas version: `3.0.2`
- FrameRust wins: 1
- Pandas wins: 0
- Checksum failures: 0
- Global Pandas replacement claim: false

| Case | FrameRust ms | Pandas ms | Speedup | Winner | Checksum |
| --- | ---: | ---: | ---: | --- | --- |
| `groupby_i64_sum_mean_count_250k_1000` | 1.144 | 2.450 | 2.14x | framerust | ok |
