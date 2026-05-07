# Array API Tests Probe

This report runs the pinned upstream `array-api-tests` pytest suite without patching the suite.

## Source

- Repo: `https://github.com/data-apis/array-api-tests.git`
- Commit: `55fcc60179efa2680ddd6cd926ddf17b83530e2b`
- API version: `2023.12`
- Targets: `array_api_tests/test_statistical_functions.py::test_cumulative_sum, array_api_tests/test_special_cases.py::test_empty_arrays, array_api_tests/test_special_cases.py::test_nan_propagation`
- Full suite: False

## Result

- Status: passed
- Return code: 0
- Duration seconds: 2.17
- Summary: `{"collected": 13, "passed": 13, "total": 13}`

## Command

```sh
python -m pytest <repo>/target/external/array-api-tests/array_api_tests/test_statistical_functions.py::test_cumulative_sum <repo>/target/external/array-api-tests/array_api_tests/test_special_cases.py::test_empty_arrays <repo>/target/external/array-api-tests/array_api_tests/test_special_cases.py::test_nan_propagation -q --tb=short --json-report --json-report-file=<repo>/target/array-api-tests-report.json
```
