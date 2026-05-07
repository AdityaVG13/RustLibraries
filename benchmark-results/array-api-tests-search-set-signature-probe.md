# Array API Tests Probe

This report runs the pinned upstream `array-api-tests` pytest suite without patching the suite.

## Source

- Repo: `https://github.com/data-apis/array-api-tests.git`
- Commit: `55fcc60179efa2680ddd6cd926ddf17b83530e2b`
- API version: `2023.12`
- Targets: `array_api_tests/test_searching_functions.py::test_nonzero, array_api_tests/test_searching_functions.py::test_where, array_api_tests/test_searching_functions.py::test_searchsorted, array_api_tests/test_set_functions.py::test_unique_all, array_api_tests/test_set_functions.py::test_unique_counts, array_api_tests/test_set_functions.py::test_unique_inverse, array_api_tests/test_set_functions.py::test_unique_values, array_api_tests/test_signatures.py::test_func_signature[cumulative_sum], array_api_tests/test_signatures.py::test_func_signature[max], array_api_tests/test_signatures.py::test_func_signature[mean], array_api_tests/test_signatures.py::test_func_signature[min], array_api_tests/test_signatures.py::test_func_signature[prod], array_api_tests/test_signatures.py::test_func_signature[std], array_api_tests/test_signatures.py::test_func_signature[sum]`
- Full suite: False

## Result

- Status: passed
- Return code: 0
- Duration seconds: 1.42
- Summary: `{"collected": 14, "passed": 14, "total": 14}`

## Command

```sh
python -m pytest <repo>/target/external/array-api-tests/array_api_tests/test_searching_functions.py::test_nonzero <repo>/target/external/array-api-tests/array_api_tests/test_searching_functions.py::test_where <repo>/target/external/array-api-tests/array_api_tests/test_searching_functions.py::test_searchsorted <repo>/target/external/array-api-tests/array_api_tests/test_set_functions.py::test_unique_all <repo>/target/external/array-api-tests/array_api_tests/test_set_functions.py::test_unique_counts <repo>/target/external/array-api-tests/array_api_tests/test_set_functions.py::test_unique_inverse <repo>/target/external/array-api-tests/array_api_tests/test_set_functions.py::test_unique_values <repo>/target/external/array-api-tests/array_api_tests/test_signatures.py::test_func_signature[cumulative_sum] <repo>/target/external/array-api-tests/array_api_tests/test_signatures.py::test_func_signature[max] <repo>/target/external/array-api-tests/array_api_tests/test_signatures.py::test_func_signature[mean] <repo>/target/external/array-api-tests/array_api_tests/test_signatures.py::test_func_signature[min] <repo>/target/external/array-api-tests/array_api_tests/test_signatures.py::test_func_signature[prod] <repo>/target/external/array-api-tests/array_api_tests/test_signatures.py::test_func_signature[std] <repo>/target/external/array-api-tests/array_api_tests/test_signatures.py::test_func_signature[sum] -q --tb=short --json-report --json-report-file=<repo>/target/array-api-tests-report.json --maxfail=20
```
