# Array API Tests Probe

This report runs the pinned upstream `array-api-tests` pytest suite without patching the suite.

## Source

- Repo: `https://github.com/data-apis/array-api-tests.git`
- Commit: `55fcc60179efa2680ddd6cd926ddf17b83530e2b`
- API version: `2023.12`
- Targets: `array_api_tests/test_linalg.py::test_linalg_tensordot, array_api_tests/test_linalg.py::test_tensordot, array_api_tests/test_linalg.py::test_trace, array_api_tests/test_linalg.py::test_linalg_vecdot, array_api_tests/test_linalg.py::test_vecdot, array_api_tests/test_linalg.py::test_vecdot_conj, array_api_tests/test_linalg.py::test_vector_norm`
- Full suite: False

## Result

- Status: passed
- Return code: 0
- Duration seconds: 1.17
- Summary: `{"collected": 7, "passed": 7, "total": 7}`

## Command

```sh
python -m pytest <repo>/target/external/array-api-tests/array_api_tests/test_linalg.py::test_linalg_tensordot <repo>/target/external/array-api-tests/array_api_tests/test_linalg.py::test_tensordot <repo>/target/external/array-api-tests/array_api_tests/test_linalg.py::test_trace <repo>/target/external/array-api-tests/array_api_tests/test_linalg.py::test_linalg_vecdot <repo>/target/external/array-api-tests/array_api_tests/test_linalg.py::test_vecdot <repo>/target/external/array-api-tests/array_api_tests/test_linalg.py::test_vecdot_conj <repo>/target/external/array-api-tests/array_api_tests/test_linalg.py::test_vector_norm -q --tb=short --json-report --json-report-file=<repo>/target/array-api-tests-report.json --maxfail=20
```
