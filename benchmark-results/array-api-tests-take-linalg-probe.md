# Array API Tests Probe

This report runs the pinned upstream `array-api-tests` pytest suite without patching the suite.

## Source

- Repo: `https://github.com/data-apis/array-api-tests.git`
- Commit: `55fcc60179efa2680ddd6cd926ddf17b83530e2b`
- API version: `2023.12`
- Targets: `array_api_tests/test_indexing_functions.py::test_take, array_api_tests/test_linalg.py::test_cholesky, array_api_tests/test_linalg.py::test_cross, array_api_tests/test_linalg.py::test_det, array_api_tests/test_linalg.py::test_diagonal, array_api_tests/test_linalg.py::test_linalg_matrix_transpose, array_api_tests/test_linalg.py::test_matrix_transpose, array_api_tests/test_linalg.py::test_outer`
- Full suite: False

## Result

- Status: passed
- Return code: 0
- Duration seconds: 1.05
- Summary: `{"collected": 8, "passed": 8, "total": 8}`

## Command

```sh
/Users/aditya/.cache/uv/builds-v0/.tmp0PHgvC/bin/python -m pytest /Users/aditya/AI/GitHub Research/FromPythonToRust/target/external/array-api-tests/array_api_tests/test_indexing_functions.py::test_take /Users/aditya/AI/GitHub Research/FromPythonToRust/target/external/array-api-tests/array_api_tests/test_linalg.py::test_cholesky /Users/aditya/AI/GitHub Research/FromPythonToRust/target/external/array-api-tests/array_api_tests/test_linalg.py::test_cross /Users/aditya/AI/GitHub Research/FromPythonToRust/target/external/array-api-tests/array_api_tests/test_linalg.py::test_det /Users/aditya/AI/GitHub Research/FromPythonToRust/target/external/array-api-tests/array_api_tests/test_linalg.py::test_diagonal /Users/aditya/AI/GitHub Research/FromPythonToRust/target/external/array-api-tests/array_api_tests/test_linalg.py::test_linalg_matrix_transpose /Users/aditya/AI/GitHub Research/FromPythonToRust/target/external/array-api-tests/array_api_tests/test_linalg.py::test_matrix_transpose /Users/aditya/AI/GitHub Research/FromPythonToRust/target/external/array-api-tests/array_api_tests/test_linalg.py::test_outer -q --tb=short --json-report --json-report-file=/Users/aditya/AI/GitHub Research/FromPythonToRust/target/array-api-tests-report.json --maxfail=20
```
