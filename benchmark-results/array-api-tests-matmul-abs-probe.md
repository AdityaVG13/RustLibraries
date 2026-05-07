# Array API Tests Probe

This report runs the pinned upstream `array-api-tests` pytest suite without patching the suite.

## Source

- Repo: `https://github.com/data-apis/array-api-tests.git`
- Commit: `55fcc60179efa2680ddd6cd926ddf17b83530e2b`
- API version: `2023.12`
- Targets: `array_api_tests/test_linalg.py::test_linalg_matmul, array_api_tests/test_linalg.py::test_matmul, array_api_tests/test_linalg.py::test_matrix_power`
- Full suite: False

## Result

- Status: passed
- Return code: 0
- Duration seconds: 1.54
- Summary: `{"collected": 3, "passed": 3, "total": 3}`

## Command

```sh
/Users/aditya/.cache/uv/builds-v0/.tmpFkvrPH/bin/python -m pytest /Users/aditya/AI/GitHub Research/FromPythonToRust/target/external/array-api-tests/array_api_tests/test_linalg.py::test_linalg_matmul /Users/aditya/AI/GitHub Research/FromPythonToRust/target/external/array-api-tests/array_api_tests/test_linalg.py::test_matmul /Users/aditya/AI/GitHub Research/FromPythonToRust/target/external/array-api-tests/array_api_tests/test_linalg.py::test_matrix_power -q --tb=short --json-report --json-report-file=/Users/aditya/AI/GitHub Research/FromPythonToRust/target/array-api-tests-report.json --maxfail=10
```
