# Array API Tests Probe

This report runs the pinned upstream `array-api-tests` pytest suite without patching the suite.

## Source

- Repo: `https://github.com/data-apis/array-api-tests.git`
- Commit: `55fcc60179efa2680ddd6cd926ddf17b83530e2b`
- API version: `2023.12`
- Targets: `array_api_tests/test_signatures.py::test_func_signature, array_api_tests/test_signatures.py::test_extension_func_signature, array_api_tests/test_signatures.py::test_array_method_signature, array_api_tests/test_sorting_functions.py::test_argsort, array_api_tests/test_sorting_functions.py::test_sort, array_api_tests/test_special_cases.py::test_unary`
- Full suite: False

## Result

- Status: passed
- Return code: 0
- Duration seconds: 1.06
- Summary: `{"collected": 457, "passed": 457, "total": 457}`

## Command

```sh
python -m pytest <repo>/target/external/array-api-tests/array_api_tests/test_signatures.py::test_func_signature <repo>/target/external/array-api-tests/array_api_tests/test_signatures.py::test_extension_func_signature <repo>/target/external/array-api-tests/array_api_tests/test_signatures.py::test_array_method_signature <repo>/target/external/array-api-tests/array_api_tests/test_sorting_functions.py::test_argsort <repo>/target/external/array-api-tests/array_api_tests/test_sorting_functions.py::test_sort <repo>/target/external/array-api-tests/array_api_tests/test_special_cases.py::test_unary -q --tb=short --json-report --json-report-file=<repo>/target/array-api-tests-report.json
```
