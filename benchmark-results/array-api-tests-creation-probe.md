# Array API Tests Probe

This report runs the pinned upstream `array-api-tests` pytest suite without patching the suite.

## Source

- Repo: `https://github.com/data-apis/array-api-tests.git`
- Commit: `55fcc60179efa2680ddd6cd926ddf17b83530e2b`
- API version: `2023.12`
- Targets: `array_api_tests/test_creation_functions.py::test_zeros, array_api_tests/test_creation_functions.py::test_ones, array_api_tests/test_creation_functions.py::test_full`
- Full suite: False

## Result

- Status: passed
- Return code: 0
- Duration seconds: 0.75
- Summary: `{"collected": 3, "passed": 3, "total": 3}`

## Command

```sh
/Users/aditya/.cache/uv/builds-v0/.tmppRmjDi/bin/python -m pytest /Users/aditya/AI/GitHub Research/FromPythonToRust/target/external/array-api-tests/array_api_tests/test_creation_functions.py::test_zeros /Users/aditya/AI/GitHub Research/FromPythonToRust/target/external/array-api-tests/array_api_tests/test_creation_functions.py::test_ones /Users/aditya/AI/GitHub Research/FromPythonToRust/target/external/array-api-tests/array_api_tests/test_creation_functions.py::test_full -q --tb=short --json-report --json-report-file=/Users/aditya/AI/GitHub Research/FromPythonToRust/target/array-api-tests-report.json --maxfail=10
```
