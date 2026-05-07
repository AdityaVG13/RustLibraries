# Array API Tests Probe

This report runs the pinned upstream `array-api-tests` pytest suite without patching the suite.

## Source

- Repo: `https://github.com/data-apis/array-api-tests.git`
- Commit: `55fcc60179efa2680ddd6cd926ddf17b83530e2b`
- API version: `2023.12`
- Targets: `array_api_tests/test_manipulation_functions.py::test_repeat, array_api_tests/test_searching_functions.py::test_argmax, array_api_tests/test_searching_functions.py::test_argmin, array_api_tests/test_linalg.py::test_linalg_vecdot`
- Full suite: False

## Result

- Status: passed
- Return code: 0
- Duration seconds: 0.79
- Summary: `{"collected": 4, "passed": 4, "total": 4}`

## Command

```sh
/Users/aditya/.cache/uv/builds-v0/.tmpXHWAIN/bin/python -m pytest /Users/aditya/AI/GitHub Research/FromPythonToRust/target/external/array-api-tests/array_api_tests/test_manipulation_functions.py::test_repeat /Users/aditya/AI/GitHub Research/FromPythonToRust/target/external/array-api-tests/array_api_tests/test_searching_functions.py::test_argmax /Users/aditya/AI/GitHub Research/FromPythonToRust/target/external/array-api-tests/array_api_tests/test_searching_functions.py::test_argmin /Users/aditya/AI/GitHub Research/FromPythonToRust/target/external/array-api-tests/array_api_tests/test_linalg.py::test_linalg_vecdot -q --tb=short --json-report --json-report-file=/Users/aditya/AI/GitHub Research/FromPythonToRust/target/array-api-tests-report.json --maxfail=10
```
