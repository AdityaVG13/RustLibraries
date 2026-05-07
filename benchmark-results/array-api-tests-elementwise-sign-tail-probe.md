# Array API Tests Probe

This report runs the pinned upstream `array-api-tests` pytest suite without patching the suite.

## Source

- Repo: `https://github.com/data-apis/array-api-tests.git`
- Commit: `55fcc60179efa2680ddd6cd926ddf17b83530e2b`
- API version: `2023.12`
- Targets: `array_api_tests/test_operators_and_elementwise_functions.py::test_sign, array_api_tests/test_operators_and_elementwise_functions.py::test_sin, array_api_tests/test_operators_and_elementwise_functions.py::test_sinh, array_api_tests/test_operators_and_elementwise_functions.py::test_square, array_api_tests/test_operators_and_elementwise_functions.py::test_sqrt, array_api_tests/test_operators_and_elementwise_functions.py::test_subtract[__sub__(x, s)], array_api_tests/test_operators_and_elementwise_functions.py::test_subtract[__isub__(x, s)], array_api_tests/test_operators_and_elementwise_functions.py::test_tan, array_api_tests/test_operators_and_elementwise_functions.py::test_tanh, array_api_tests/test_operators_and_elementwise_functions.py::test_trunc`
- Full suite: False

## Result

- Status: passed
- Return code: 0
- Duration seconds: 1.38
- Summary: `{"collected": 10, "passed": 10, "total": 10}`

## Command

```sh
/Users/aditya/.cache/uv/builds-v0/.tmpf9wg01/bin/python -m pytest /Users/aditya/AI/GitHub Research/FromPythonToRust/target/external/array-api-tests/array_api_tests/test_operators_and_elementwise_functions.py::test_sign /Users/aditya/AI/GitHub Research/FromPythonToRust/target/external/array-api-tests/array_api_tests/test_operators_and_elementwise_functions.py::test_sin /Users/aditya/AI/GitHub Research/FromPythonToRust/target/external/array-api-tests/array_api_tests/test_operators_and_elementwise_functions.py::test_sinh /Users/aditya/AI/GitHub Research/FromPythonToRust/target/external/array-api-tests/array_api_tests/test_operators_and_elementwise_functions.py::test_square /Users/aditya/AI/GitHub Research/FromPythonToRust/target/external/array-api-tests/array_api_tests/test_operators_and_elementwise_functions.py::test_sqrt /Users/aditya/AI/GitHub Research/FromPythonToRust/target/external/array-api-tests/array_api_tests/test_operators_and_elementwise_functions.py::test_subtract[__sub__(x, s)] /Users/aditya/AI/GitHub Research/FromPythonToRust/target/external/array-api-tests/array_api_tests/test_operators_and_elementwise_functions.py::test_subtract[__isub__(x, s)] /Users/aditya/AI/GitHub Research/FromPythonToRust/target/external/array-api-tests/array_api_tests/test_operators_and_elementwise_functions.py::test_tan /Users/aditya/AI/GitHub Research/FromPythonToRust/target/external/array-api-tests/array_api_tests/test_operators_and_elementwise_functions.py::test_tanh /Users/aditya/AI/GitHub Research/FromPythonToRust/target/external/array-api-tests/array_api_tests/test_operators_and_elementwise_functions.py::test_trunc -q --tb=short --json-report --json-report-file=/Users/aditya/AI/GitHub Research/FromPythonToRust/target/array-api-tests-report.json --maxfail=20
```
