# Array API Tests Probe

This report runs the pinned upstream `array-api-tests` pytest suite without patching the suite.

## Source

- Repo: `https://github.com/data-apis/array-api-tests.git`
- Commit: `55fcc60179efa2680ddd6cd926ddf17b83530e2b`
- API version: `2023.12`
- Targets: `array_api_tests/test_constants.py, array_api_tests/test_array_object.py::test_scalar_casting, array_api_tests/test_array_object.py::test_getitem, array_api_tests/test_array_object.py::test_getitem_masking`
- Full suite: False

## Result

- Status: passed
- Return code: 0
- Duration seconds: 1.88
- Summary: `{"collected": 26, "passed": 26, "total": 26}`

## Command

```sh
/Users/aditya/.cache/uv/builds-v0/.tmpC4PNi7/bin/python -m pytest /Users/aditya/AI/GitHub Research/FromPythonToRust/target/external/array-api-tests/array_api_tests/test_constants.py /Users/aditya/AI/GitHub Research/FromPythonToRust/target/external/array-api-tests/array_api_tests/test_array_object.py::test_scalar_casting /Users/aditya/AI/GitHub Research/FromPythonToRust/target/external/array-api-tests/array_api_tests/test_array_object.py::test_getitem /Users/aditya/AI/GitHub Research/FromPythonToRust/target/external/array-api-tests/array_api_tests/test_array_object.py::test_getitem_masking -q --tb=short --json-report --json-report-file=/Users/aditya/AI/GitHub Research/FromPythonToRust/target/array-api-tests-report.json --maxfail=20
```
