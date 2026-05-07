# Array API Tests Probe

This report runs the pinned upstream `array-api-tests` pytest suite without patching the suite.

## Source

- Repo: `https://github.com/data-apis/array-api-tests.git`
- Commit: `55fcc60179efa2680ddd6cd926ddf17b83530e2b`
- API version: `2023.12`
- Targets: `array_api_tests/test_operators_and_elementwise_functions.py::test_floor, array_api_tests/test_operators_and_elementwise_functions.py::test_floor_divide[floor_divide(x1, x2)], array_api_tests/test_operators_and_elementwise_functions.py::test_floor_divide[__floordiv__(x1, x2)], array_api_tests/test_operators_and_elementwise_functions.py::test_floor_divide[__floordiv__(x, s)], array_api_tests/test_operators_and_elementwise_functions.py::test_floor_divide[__ifloordiv__(x1, x2)], array_api_tests/test_operators_and_elementwise_functions.py::test_floor_divide[__ifloordiv__(x, s)], array_api_tests/test_operators_and_elementwise_functions.py::test_hypot, array_api_tests/test_operators_and_elementwise_functions.py::test_log, array_api_tests/test_operators_and_elementwise_functions.py::test_log1p, array_api_tests/test_operators_and_elementwise_functions.py::test_log2, array_api_tests/test_operators_and_elementwise_functions.py::test_log10, array_api_tests/test_operators_and_elementwise_functions.py::test_logaddexp, array_api_tests/test_operators_and_elementwise_functions.py::test_maximum`
- Full suite: False

## Result

- Status: passed
- Return code: 0
- Duration seconds: 0.91
- Summary: `{"collected": 13, "passed": 13, "total": 13}`

## Command

```sh
python -m pytest <repo>/target/external/array-api-tests/array_api_tests/test_operators_and_elementwise_functions.py::test_floor <repo>/target/external/array-api-tests/array_api_tests/test_operators_and_elementwise_functions.py::test_floor_divide[floor_divide(x1, x2)] <repo>/target/external/array-api-tests/array_api_tests/test_operators_and_elementwise_functions.py::test_floor_divide[__floordiv__(x1, x2)] <repo>/target/external/array-api-tests/array_api_tests/test_operators_and_elementwise_functions.py::test_floor_divide[__floordiv__(x, s)] <repo>/target/external/array-api-tests/array_api_tests/test_operators_and_elementwise_functions.py::test_floor_divide[__ifloordiv__(x1, x2)] <repo>/target/external/array-api-tests/array_api_tests/test_operators_and_elementwise_functions.py::test_floor_divide[__ifloordiv__(x, s)] <repo>/target/external/array-api-tests/array_api_tests/test_operators_and_elementwise_functions.py::test_hypot <repo>/target/external/array-api-tests/array_api_tests/test_operators_and_elementwise_functions.py::test_log <repo>/target/external/array-api-tests/array_api_tests/test_operators_and_elementwise_functions.py::test_log1p <repo>/target/external/array-api-tests/array_api_tests/test_operators_and_elementwise_functions.py::test_log2 <repo>/target/external/array-api-tests/array_api_tests/test_operators_and_elementwise_functions.py::test_log10 <repo>/target/external/array-api-tests/array_api_tests/test_operators_and_elementwise_functions.py::test_logaddexp <repo>/target/external/array-api-tests/array_api_tests/test_operators_and_elementwise_functions.py::test_maximum -q --tb=short --json-report --json-report-file=<repo>/target/array-api-tests-report.json --maxfail=20
```
