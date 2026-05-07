# Array API Tests Probe

This report runs the pinned upstream `array-api-tests` pytest suite without patching the suite.

## Source

- Repo: `https://github.com/data-apis/array-api-tests.git`
- Commit: `55fcc60179efa2680ddd6cd926ddf17b83530e2b`
- API version: `2023.12`
- Targets: `array_api_tests/test_linalg.py::test_inv, array_api_tests/test_linalg.py::test_matrix_norm, array_api_tests/test_linalg.py::test_matrix_rank, array_api_tests/test_linalg.py::test_slogdet`
- Full suite: False

## Result

- Status: passed
- Return code: 0
- Duration seconds: 0.73
- Summary: `{"collected": 4, "passed": 4, "total": 4}`

## Command

```sh
python -m pytest <repo>/target/external/array-api-tests/array_api_tests/test_linalg.py::test_inv <repo>/target/external/array-api-tests/array_api_tests/test_linalg.py::test_matrix_norm <repo>/target/external/array-api-tests/array_api_tests/test_linalg.py::test_matrix_rank <repo>/target/external/array-api-tests/array_api_tests/test_linalg.py::test_slogdet -q --tb=short --json-report --json-report-file=<repo>/target/array-api-tests-report.json --maxfail=10
```
