# Array API Tests Probe

This report runs the pinned upstream `array-api-tests` pytest suite without patching the suite.

## Source

- Repo: `https://github.com/data-apis/array-api-tests.git`
- Commit: `55fcc60179efa2680ddd6cd926ddf17b83530e2b`
- API version: `2023.12`
- Targets: `array_api_tests/test_operators_and_elementwise_functions.py::test_ceil, array_api_tests/test_operators_and_elementwise_functions.py::test_clip, array_api_tests/test_operators_and_elementwise_functions.py::test_copysign, array_api_tests/test_operators_and_elementwise_functions.py::test_cos, array_api_tests/test_operators_and_elementwise_functions.py::test_cosh, array_api_tests/test_operators_and_elementwise_functions.py::test_divide[divide(x1, x2)], array_api_tests/test_operators_and_elementwise_functions.py::test_divide[__truediv__(x1, x2)], array_api_tests/test_operators_and_elementwise_functions.py::test_divide[__truediv__(x, s)], array_api_tests/test_operators_and_elementwise_functions.py::test_divide[__itruediv__(x1, x2)], array_api_tests/test_operators_and_elementwise_functions.py::test_divide[__itruediv__(x, s)], array_api_tests/test_operators_and_elementwise_functions.py::test_exp, array_api_tests/test_operators_and_elementwise_functions.py::test_expm1`
- Full suite: False

## Result

- Status: passed
- Return code: 0
- Duration seconds: 0.79
- Summary: `{"collected": 12, "passed": 12, "total": 12}`

## Command

```sh
python -m pytest <repo>/target/external/array-api-tests/array_api_tests/test_operators_and_elementwise_functions.py::test_ceil <repo>/target/external/array-api-tests/array_api_tests/test_operators_and_elementwise_functions.py::test_clip <repo>/target/external/array-api-tests/array_api_tests/test_operators_and_elementwise_functions.py::test_copysign <repo>/target/external/array-api-tests/array_api_tests/test_operators_and_elementwise_functions.py::test_cos <repo>/target/external/array-api-tests/array_api_tests/test_operators_and_elementwise_functions.py::test_cosh <repo>/target/external/array-api-tests/array_api_tests/test_operators_and_elementwise_functions.py::test_divide[divide(x1, x2)] <repo>/target/external/array-api-tests/array_api_tests/test_operators_and_elementwise_functions.py::test_divide[__truediv__(x1, x2)] <repo>/target/external/array-api-tests/array_api_tests/test_operators_and_elementwise_functions.py::test_divide[__truediv__(x, s)] <repo>/target/external/array-api-tests/array_api_tests/test_operators_and_elementwise_functions.py::test_divide[__itruediv__(x1, x2)] <repo>/target/external/array-api-tests/array_api_tests/test_operators_and_elementwise_functions.py::test_divide[__itruediv__(x, s)] <repo>/target/external/array-api-tests/array_api_tests/test_operators_and_elementwise_functions.py::test_exp <repo>/target/external/array-api-tests/array_api_tests/test_operators_and_elementwise_functions.py::test_expm1 -q --tb=short --json-report --json-report-file=<repo>/target/array-api-tests-report.json --maxfail=20
```
