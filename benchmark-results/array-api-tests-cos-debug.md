# Array API Tests Probe

This report runs the pinned upstream `array-api-tests` pytest suite without patching the suite.

## Source

- Repo: `https://github.com/data-apis/array-api-tests.git`
- Commit: `55fcc60179efa2680ddd6cd926ddf17b83530e2b`
- API version: `2023.12`
- Targets: `array_api_tests/test_operators_and_elementwise_functions.py::test_cos`
- Full suite: False

## Result

- Status: failed
- Return code: 1
- Duration seconds: 0.67
- Summary: `{"collected": 1, "failed": 1, "total": 1}`

## First Failures

- `array_api_tests/test_operators_and_elementwise_functions.py::test_cos`: failed

## Command

```sh
/Users/aditya/.cache/uv/builds-v0/.tmpUCHH7H/bin/python -m pytest /Users/aditya/AI/GitHub Research/FromPythonToRust/target/external/array-api-tests/array_api_tests/test_operators_and_elementwise_functions.py::test_cos -q --tb=short --json-report --json-report-file=/Users/aditya/AI/GitHub Research/FromPythonToRust/target/array-api-tests-report.json --maxfail=1
```
