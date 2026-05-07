# Array API Tests Probe

This report runs the pinned upstream `array-api-tests` pytest suite without patching the suite.

## Source

- Repo: `https://github.com/data-apis/array-api-tests.git`
- Commit: `55fcc60179efa2680ddd6cd926ddf17b83530e2b`
- API version: `2023.12`
- Targets: `array_api_tests/test_searching_functions.py::test_nonzero, array_api_tests/test_searching_functions.py::test_where, array_api_tests/test_searching_functions.py::test_searchsorted, array_api_tests/test_set_functions.py::test_unique_all, array_api_tests/test_set_functions.py::test_unique_counts, array_api_tests/test_set_functions.py::test_unique_inverse, array_api_tests/test_set_functions.py::test_unique_values`
- Full suite: False

## Result

- Status: passed
- Return code: 0
- Duration seconds: 1.07
- Summary: `{"collected": 7, "passed": 7, "total": 7}`

## Command

```sh
/Users/aditya/.cache/uv/builds-v0/.tmpIBGuO6/bin/python -m pytest /Users/aditya/AI/GitHub Research/FromPythonToRust/target/external/array-api-tests/array_api_tests/test_searching_functions.py::test_nonzero /Users/aditya/AI/GitHub Research/FromPythonToRust/target/external/array-api-tests/array_api_tests/test_searching_functions.py::test_where /Users/aditya/AI/GitHub Research/FromPythonToRust/target/external/array-api-tests/array_api_tests/test_searching_functions.py::test_searchsorted /Users/aditya/AI/GitHub Research/FromPythonToRust/target/external/array-api-tests/array_api_tests/test_set_functions.py::test_unique_all /Users/aditya/AI/GitHub Research/FromPythonToRust/target/external/array-api-tests/array_api_tests/test_set_functions.py::test_unique_counts /Users/aditya/AI/GitHub Research/FromPythonToRust/target/external/array-api-tests/array_api_tests/test_set_functions.py::test_unique_inverse /Users/aditya/AI/GitHub Research/FromPythonToRust/target/external/array-api-tests/array_api_tests/test_set_functions.py::test_unique_values -q --tb=short --json-report --json-report-file=/Users/aditya/AI/GitHub Research/FromPythonToRust/target/array-api-tests-report.json --maxfail=20
```
