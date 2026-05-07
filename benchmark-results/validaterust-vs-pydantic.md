# ValidateRust vs Pydantic

Same-data local benchmark for the implemented schema validation slice.
This is not a full Pydantic replacement claim.

- Pydantic version: `2.13.4`
- ValidateRust wins: 1
- Pydantic wins: 0
- Checksum failures: 0
- Global Pydantic replacement claim: false

| Case | ValidateRust ms | Pydantic ms | Speedup | Winner | Checksum |
| --- | ---: | ---: | ---: | --- | --- |
| `user_schema_100000` | 5.003 | 178.539 | 35.69x | validaterust | ok |
