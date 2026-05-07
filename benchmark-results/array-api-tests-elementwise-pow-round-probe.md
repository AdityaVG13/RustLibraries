# Array API Tests Probe

This report runs the pinned upstream `array-api-tests` pytest suite without patching the suite.

## Source

- Repo: `https://github.com/data-apis/array-api-tests.git`
- Commit: `55fcc60179efa2680ddd6cd926ddf17b83530e2b`
- API version: `2023.12`
- Targets: `array_api_tests/test_operators_and_elementwise_functions.py::test_minimum, array_api_tests/test_operators_and_elementwise_functions.py::test_multiply[__mul__(x, s)], array_api_tests/test_operators_and_elementwise_functions.py::test_multiply[__imul__(x, s)], array_api_tests/test_operators_and_elementwise_functions.py::test_negative[negative], array_api_tests/test_operators_and_elementwise_functions.py::test_negative[__neg__], array_api_tests/test_operators_and_elementwise_functions.py::test_positive[positive], array_api_tests/test_operators_and_elementwise_functions.py::test_pow[pow(x1, x2)], array_api_tests/test_operators_and_elementwise_functions.py::test_pow[__pow__(x1, x2)], array_api_tests/test_operators_and_elementwise_functions.py::test_pow[__pow__(x, s)], array_api_tests/test_operators_and_elementwise_functions.py::test_pow[__ipow__(x1, x2)], array_api_tests/test_operators_and_elementwise_functions.py::test_pow[__ipow__(x, s)], array_api_tests/test_operators_and_elementwise_functions.py::test_round, array_api_tests/test_operators_and_elementwise_functions.py::test_signbit`
- Full suite: False

## Result

- Status: passed
- Return code: 0
- Duration seconds: 0.86
- Summary: `{"collected": 13, "passed": 13, "total": 13}`

## Command

```sh
python -m pytest <repo>/target/external/array-api-tests/array_api_tests/test_operators_and_elementwise_functions.py::test_minimum <repo>/target/external/array-api-tests/array_api_tests/test_operators_and_elementwise_functions.py::test_multiply[__mul__(x, s)] <repo>/target/external/array-api-tests/array_api_tests/test_operators_and_elementwise_functions.py::test_multiply[__imul__(x, s)] <repo>/target/external/array-api-tests/array_api_tests/test_operators_and_elementwise_functions.py::test_negative[negative] <repo>/target/external/array-api-tests/array_api_tests/test_operators_and_elementwise_functions.py::test_negative[__neg__] <repo>/target/external/array-api-tests/array_api_tests/test_operators_and_elementwise_functions.py::test_positive[positive] <repo>/target/external/array-api-tests/array_api_tests/test_operators_and_elementwise_functions.py::test_pow[pow(x1, x2)] <repo>/target/external/array-api-tests/array_api_tests/test_operators_and_elementwise_functions.py::test_pow[__pow__(x1, x2)] <repo>/target/external/array-api-tests/array_api_tests/test_operators_and_elementwise_functions.py::test_pow[__pow__(x, s)] <repo>/target/external/array-api-tests/array_api_tests/test_operators_and_elementwise_functions.py::test_pow[__ipow__(x1, x2)] <repo>/target/external/array-api-tests/array_api_tests/test_operators_and_elementwise_functions.py::test_pow[__ipow__(x, s)] <repo>/target/external/array-api-tests/array_api_tests/test_operators_and_elementwise_functions.py::test_round <repo>/target/external/array-api-tests/array_api_tests/test_operators_and_elementwise_functions.py::test_signbit -q --tb=short --json-report --json-report-file=<repo>/target/array-api-tests-report.json --maxfail=20
```
