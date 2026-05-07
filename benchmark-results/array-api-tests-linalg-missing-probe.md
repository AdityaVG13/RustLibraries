# Array API Tests Probe

This report runs the pinned upstream `array-api-tests` pytest suite without patching the suite.

## Source

- Repo: `https://github.com/data-apis/array-api-tests.git`
- Commit: `55fcc60179efa2680ddd6cd926ddf17b83530e2b`
- API version: `2023.12`
- Targets: `array_api_tests/test_linalg.py::test_eigh, array_api_tests/test_linalg.py::test_eigvalsh, array_api_tests/test_linalg.py::test_pinv, array_api_tests/test_linalg.py::test_qr, array_api_tests/test_linalg.py::test_solve, array_api_tests/test_linalg.py::test_svd, array_api_tests/test_linalg.py::test_svdvals`
- Full suite: False

## Result

- Status: failed
- Return code: 1
- Duration seconds: 18.91
- Summary: `{"collected": 7, "failed": 1, "passed": 6, "total": 7}`

## First Failures

- `array_api_tests/test_linalg.py::test_svd`: failed

## Command

```sh
/Users/aditya/.cache/uv/builds-v0/.tmp7wDP18/bin/python -m pytest /Users/aditya/AI/GitHub Research/FromPythonToRust/target/external/array-api-tests/array_api_tests/test_linalg.py::test_eigh /Users/aditya/AI/GitHub Research/FromPythonToRust/target/external/array-api-tests/array_api_tests/test_linalg.py::test_eigvalsh /Users/aditya/AI/GitHub Research/FromPythonToRust/target/external/array-api-tests/array_api_tests/test_linalg.py::test_pinv /Users/aditya/AI/GitHub Research/FromPythonToRust/target/external/array-api-tests/array_api_tests/test_linalg.py::test_qr /Users/aditya/AI/GitHub Research/FromPythonToRust/target/external/array-api-tests/array_api_tests/test_linalg.py::test_solve /Users/aditya/AI/GitHub Research/FromPythonToRust/target/external/array-api-tests/array_api_tests/test_linalg.py::test_svd /Users/aditya/AI/GitHub Research/FromPythonToRust/target/external/array-api-tests/array_api_tests/test_linalg.py::test_svdvals -q --tb=short --json-report --json-report-file=/Users/aditya/AI/GitHub Research/FromPythonToRust/target/array-api-tests-report.json
```
