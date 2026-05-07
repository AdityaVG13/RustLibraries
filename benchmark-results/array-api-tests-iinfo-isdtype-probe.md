# Array API Tests Probe

This report runs the pinned upstream `array-api-tests` pytest suite without patching the suite.

## Source

- Repo: `https://github.com/data-apis/array-api-tests.git`
- Commit: `55fcc60179efa2680ddd6cd926ddf17b83530e2b`
- API version: `2023.12`
- Targets: `array_api_tests/test_data_type_functions.py::test_iinfo, array_api_tests/test_data_type_functions.py::test_iinfo_dtype, array_api_tests/test_data_type_functions.py::test_isdtype`
- Full suite: False

## Result

- Status: passed
- Return code: 0
- Duration seconds: 0.58
- Summary: `{"collected": 17, "passed": 17, "total": 17}`

## Command

```sh
python -m pytest <repo>/target/external/array-api-tests/array_api_tests/test_data_type_functions.py::test_iinfo <repo>/target/external/array-api-tests/array_api_tests/test_data_type_functions.py::test_iinfo_dtype <repo>/target/external/array-api-tests/array_api_tests/test_data_type_functions.py::test_isdtype -q --tb=short --json-report --json-report-file=<repo>/target/array-api-tests-report.json --maxfail=20
```
