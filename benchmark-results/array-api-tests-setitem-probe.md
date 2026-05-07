# Array API Tests Probe

This report runs the pinned upstream `array-api-tests` pytest suite without patching the suite.

## Source

- Repo: `https://github.com/data-apis/array-api-tests.git`
- Commit: `55fcc60179efa2680ddd6cd926ddf17b83530e2b`
- API version: `2023.12`
- Targets: `array_api_tests/test_array_object.py::test_setitem, array_api_tests/test_array_object.py::test_setitem_masking`
- Full suite: False

## Result

- Status: passed
- Return code: 0
- Duration seconds: 0.58
- Summary: `{"collected": 2, "passed": 2, "total": 2}`

## Command

```sh
python -m pytest <repo>/target/external/array-api-tests/array_api_tests/test_array_object.py::test_setitem <repo>/target/external/array-api-tests/array_api_tests/test_array_object.py::test_setitem_masking -q --tb=short --json-report --json-report-file=<repo>/target/array-api-tests-report.json --maxfail=20
```
