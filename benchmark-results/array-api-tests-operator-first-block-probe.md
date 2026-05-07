# Array API Tests Probe

This report runs the pinned upstream `array-api-tests` pytest suite without patching the suite.

## Source

- Repo: `https://github.com/data-apis/array-api-tests.git`
- Commit: `55fcc60179efa2680ddd6cd926ddf17b83530e2b`
- API version: `2023.12`
- Targets: `array_api_tests/test_operators_and_elementwise_functions.py::test_abs[__abs__], array_api_tests/test_operators_and_elementwise_functions.py::test_acos, array_api_tests/test_operators_and_elementwise_functions.py::test_acosh, array_api_tests/test_operators_and_elementwise_functions.py::test_add[add(x1, x2)], array_api_tests/test_operators_and_elementwise_functions.py::test_add[__add__(x1, x2)], array_api_tests/test_operators_and_elementwise_functions.py::test_add[__add__(x, s)], array_api_tests/test_operators_and_elementwise_functions.py::test_add[__iadd__(x1, x2)], array_api_tests/test_operators_and_elementwise_functions.py::test_add[__iadd__(x, s)], array_api_tests/test_operators_and_elementwise_functions.py::test_asin, array_api_tests/test_operators_and_elementwise_functions.py::test_asinh, array_api_tests/test_operators_and_elementwise_functions.py::test_atan, array_api_tests/test_operators_and_elementwise_functions.py::test_atan2, array_api_tests/test_operators_and_elementwise_functions.py::test_atanh, array_api_tests/test_operators_and_elementwise_functions.py::test_bitwise_and[bitwise_and(x1, x2)], array_api_tests/test_operators_and_elementwise_functions.py::test_bitwise_and[__and__(x1, x2)], array_api_tests/test_operators_and_elementwise_functions.py::test_bitwise_and[__and__(x, s)], array_api_tests/test_operators_and_elementwise_functions.py::test_bitwise_and[__iand__(x1, x2)], array_api_tests/test_operators_and_elementwise_functions.py::test_bitwise_and[__iand__(x, s)]`
- Full suite: False

## Result

- Status: passed
- Return code: 0
- Duration seconds: 1.02
- Summary: `{"collected": 18, "passed": 18, "total": 18}`

## Command

```sh
python -m pytest <repo>/target/external/array-api-tests/array_api_tests/test_operators_and_elementwise_functions.py::test_abs[__abs__] <repo>/target/external/array-api-tests/array_api_tests/test_operators_and_elementwise_functions.py::test_acos <repo>/target/external/array-api-tests/array_api_tests/test_operators_and_elementwise_functions.py::test_acosh <repo>/target/external/array-api-tests/array_api_tests/test_operators_and_elementwise_functions.py::test_add[add(x1, x2)] <repo>/target/external/array-api-tests/array_api_tests/test_operators_and_elementwise_functions.py::test_add[__add__(x1, x2)] <repo>/target/external/array-api-tests/array_api_tests/test_operators_and_elementwise_functions.py::test_add[__add__(x, s)] <repo>/target/external/array-api-tests/array_api_tests/test_operators_and_elementwise_functions.py::test_add[__iadd__(x1, x2)] <repo>/target/external/array-api-tests/array_api_tests/test_operators_and_elementwise_functions.py::test_add[__iadd__(x, s)] <repo>/target/external/array-api-tests/array_api_tests/test_operators_and_elementwise_functions.py::test_asin <repo>/target/external/array-api-tests/array_api_tests/test_operators_and_elementwise_functions.py::test_asinh <repo>/target/external/array-api-tests/array_api_tests/test_operators_and_elementwise_functions.py::test_atan <repo>/target/external/array-api-tests/array_api_tests/test_operators_and_elementwise_functions.py::test_atan2 <repo>/target/external/array-api-tests/array_api_tests/test_operators_and_elementwise_functions.py::test_atanh <repo>/target/external/array-api-tests/array_api_tests/test_operators_and_elementwise_functions.py::test_bitwise_and[bitwise_and(x1, x2)] <repo>/target/external/array-api-tests/array_api_tests/test_operators_and_elementwise_functions.py::test_bitwise_and[__and__(x1, x2)] <repo>/target/external/array-api-tests/array_api_tests/test_operators_and_elementwise_functions.py::test_bitwise_and[__and__(x, s)] <repo>/target/external/array-api-tests/array_api_tests/test_operators_and_elementwise_functions.py::test_bitwise_and[__iand__(x1, x2)] <repo>/target/external/array-api-tests/array_api_tests/test_operators_and_elementwise_functions.py::test_bitwise_and[__iand__(x, s)] -q --tb=short --json-report --json-report-file=<repo>/target/array-api-tests-report.json --maxfail=20
```
