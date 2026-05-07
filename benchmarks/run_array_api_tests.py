from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from verify_array_api_namespace import build_extension, prepare_import_tree

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "benchmark-results" / "external-source-lock.json"
DEFAULT_TARGET = "array_api_tests/test_creation_functions.py::test_zeros"
FOCUSED_TARGETS = [
    "array_api_tests/test_constants.py",
    "array_api_tests/test_array_object.py::test_scalar_casting",
    "array_api_tests/test_array_object.py::test_getitem",
    "array_api_tests/test_array_object.py::test_getitem_masking",
    "array_api_tests/test_array_object.py::test_setitem",
    "array_api_tests/test_array_object.py::test_setitem_masking",
    "array_api_tests/test_creation_functions.py::test_arange",
    "array_api_tests/test_creation_functions.py::test_asarray_scalars",
    "array_api_tests/test_creation_functions.py::test_asarray_arrays",
    "array_api_tests/test_creation_functions.py::test_empty_like",
    "array_api_tests/test_creation_functions.py::test_eye",
    "array_api_tests/test_creation_functions.py::test_full_like",
    "array_api_tests/test_creation_functions.py::test_linspace",
    "array_api_tests/test_creation_functions.py::test_meshgrid",
    "array_api_tests/test_creation_functions.py::test_ones_like",
    "array_api_tests/test_creation_functions.py::test_zeros_like",
    "array_api_tests/test_data_type_functions.py::test_astype",
    "array_api_tests/test_data_type_functions.py::test_broadcast_arrays",
    "array_api_tests/test_data_type_functions.py::test_broadcast_to",
    "array_api_tests/test_data_type_functions.py::test_can_cast",
    "array_api_tests/test_data_type_functions.py::test_finfo",
    "array_api_tests/test_data_type_functions.py::test_finfo_dtype",
    "array_api_tests/test_data_type_functions.py::test_iinfo",
    "array_api_tests/test_data_type_functions.py::test_iinfo_dtype",
    "array_api_tests/test_data_type_functions.py::test_isdtype",
    "array_api_tests/test_dlpack.py",
    "array_api_tests/test_fft.py",
    "array_api_tests/test_indexing_functions.py::test_take",
    "array_api_tests/test_linalg.py",
    "array_api_tests/test_manipulation_functions.py",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_abs[__abs__]",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_acos",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_acosh",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_add[add(x1, x2)]",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_add[__add__(x1, x2)]",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_add[__add__(x, s)]",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_add[__iadd__(x1, x2)]",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_add[__iadd__(x, s)]",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_asin",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_asinh",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_atan",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_atan2",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_atanh",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_bitwise_and[bitwise_and(x1, x2)]",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_bitwise_and[__and__(x1, x2)]",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_bitwise_and[__and__(x, s)]",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_bitwise_and[__iand__(x1, x2)]",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_bitwise_and[__iand__(x, s)]",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_bitwise_left_shift[bitwise_left_shift(x1, x2)]",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_bitwise_left_shift[__lshift__(x1, x2)]",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_bitwise_left_shift[__lshift__(x, s)]",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_bitwise_left_shift[__ilshift__(x1, x2)]",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_bitwise_left_shift[__ilshift__(x, s)]",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_bitwise_invert[bitwise_invert]",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_bitwise_invert[__invert__]",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_bitwise_or[bitwise_or(x1, x2)]",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_bitwise_or[__or__(x1, x2)]",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_bitwise_or[__or__(x, s)]",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_bitwise_or[__ior__(x1, x2)]",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_bitwise_or[__ior__(x, s)]",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_bitwise_right_shift[bitwise_right_shift(x1, x2)]",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_bitwise_right_shift[__rshift__(x1, x2)]",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_bitwise_right_shift[__rshift__(x, s)]",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_bitwise_right_shift[__irshift__(x1, x2)]",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_bitwise_right_shift[__irshift__(x, s)]",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_bitwise_xor[bitwise_xor(x1, x2)]",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_bitwise_xor[__xor__(x1, x2)]",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_bitwise_xor[__xor__(x, s)]",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_bitwise_xor[__ixor__(x1, x2)]",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_bitwise_xor[__ixor__(x, s)]",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_ceil",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_clip",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_copysign",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_cos",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_cosh",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_divide[divide(x1, x2)]",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_divide[__truediv__(x1, x2)]",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_divide[__truediv__(x, s)]",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_divide[__itruediv__(x1, x2)]",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_divide[__itruediv__(x, s)]",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_exp",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_expm1",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_floor",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_floor_divide[floor_divide(x1, x2)]",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_floor_divide[__floordiv__(x1, x2)]",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_floor_divide[__floordiv__(x, s)]",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_floor_divide[__ifloordiv__(x1, x2)]",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_floor_divide[__ifloordiv__(x, s)]",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_hypot",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_log",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_log1p",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_log2",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_log10",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_logaddexp",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_maximum",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_minimum",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_multiply[__mul__(x, s)]",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_multiply[__imul__(x, s)]",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_negative[negative]",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_negative[__neg__]",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_positive[positive]",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_pow[pow(x1, x2)]",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_pow[__pow__(x1, x2)]",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_pow[__pow__(x, s)]",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_pow[__ipow__(x1, x2)]",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_pow[__ipow__(x, s)]",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_round",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_signbit",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_sign",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_sin",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_sinh",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_square",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_sqrt",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_subtract[__sub__(x, s)]",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_subtract[__isub__(x, s)]",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_tan",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_tanh",
    "array_api_tests/test_operators_and_elementwise_functions.py::test_trunc",
    "array_api_tests/test_searching_functions.py::test_argmax",
    "array_api_tests/test_searching_functions.py::test_argmin",
    "array_api_tests/test_searching_functions.py::test_nonzero",
    "array_api_tests/test_searching_functions.py::test_where",
    "array_api_tests/test_searching_functions.py::test_searchsorted",
    "array_api_tests/test_sorting_functions.py::test_argsort",
    "array_api_tests/test_sorting_functions.py::test_sort",
    "array_api_tests/test_set_functions.py::test_unique_all",
    "array_api_tests/test_set_functions.py::test_unique_counts",
    "array_api_tests/test_set_functions.py::test_unique_inverse",
    "array_api_tests/test_set_functions.py::test_unique_values",
    "array_api_tests/test_signatures.py::test_func_signature",
    "array_api_tests/test_signatures.py::test_extension_func_signature",
    "array_api_tests/test_signatures.py::test_array_method_signature",
    "array_api_tests/test_special_cases.py",
    "array_api_tests/test_statistical_functions.py",
    "array_api_tests/test_has_names.py",
]


def run(cmd: list[str], cwd: Path, *, env: dict[str, str] | None = None) -> None:
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def array_api_source() -> dict[str, Any]:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    for source in lock["sources"]:
        if source["id"] == "array-api-tests":
            return source
    raise RuntimeError("array-api-tests source is missing from external-source-lock.json")


def ensure_suite() -> tuple[Path, dict[str, Any]]:
    source = array_api_source()
    suite = ROOT / "target" / "external" / "array-api-tests"
    suite.parent.mkdir(parents=True, exist_ok=True)
    if not (suite / ".git").exists():
        if suite.exists():
            shutil.rmtree(suite)
        run(["git", "clone", source["repo"], str(suite)], cwd=ROOT)
    run(["git", "fetch", "--quiet", "origin", source["commit"]], cwd=suite)
    run(["git", "checkout", "--quiet", "--detach", source["commit"]], cwd=suite)
    run(["git", "submodule", "update", "--init", "--quiet"], cwd=suite)
    return suite, source


def pytest_targets(suite: Path, targets: list[str], full: bool) -> list[str]:
    if full:
        return [str(suite / "array_api_tests")]
    return [str(suite / target) for target in targets]


def tail(text: str, limit: int = 4000) -> str:
    return text[-limit:]


def result_paths(stem: str) -> tuple[Path, Path]:
    result_dir = ROOT / "benchmark-results"
    return result_dir / f"{stem}.json", result_dir / f"{stem}.md"


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# Array API Tests Probe",
        "",
        "This report runs the pinned upstream `array-api-tests` pytest suite without patching the suite.",
        "",
        "## Source",
        "",
        f"- Repo: `{result['source']['repo']}`",
        f"- Commit: `{result['source']['commit']}`",
        f"- API version: `{result['api_version']}`",
        f"- Targets: `{', '.join(result['targets'])}`",
        f"- Full suite: {result['full_suite']}",
        "",
        "## Result",
        "",
        f"- Status: {result['status']}",
        f"- Return code: {result['returncode']}",
        f"- Duration seconds: {result['duration_seconds']:.2f}",
        f"- Summary: `{json.dumps(result['summary'], sort_keys=True)}`",
        "",
    ]
    failures = result.get("failures", [])
    if failures:
        lines.extend(["## First Failures", ""])
        for failure in failures:
            lines.append(f"- `{failure['nodeid']}`: {failure['outcome']}")
        lines.append("")
    lines.extend(
        [
            "## Command",
            "",
            "```sh",
            result["command"],
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-version", default="2023.12")
    parser.add_argument("--target", action="append", default=[])
    parser.add_argument("--focused", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--maxfail", type=int, default=0)
    parser.add_argument("--output-stem", default="array-api-tests-probe")
    args = parser.parse_args()
    if args.full and (args.focused or args.target):
        parser.error("--full cannot be combined with --focused or --target")
    if args.focused and args.target:
        parser.error("--focused cannot be combined with explicit --target values")

    suite, source = ensure_suite()
    build_extension()
    targets = FOCUSED_TARGETS if args.focused else args.target or [DEFAULT_TARGET]
    report_file = ROOT / "target" / "array-api-tests-report.json"
    pytest_args = [
        sys.executable,
        "-m",
        "pytest",
        *pytest_targets(suite, targets, args.full),
        "-q",
        "--tb=short",
        "--json-report",
        f"--json-report-file={report_file}",
    ]
    if args.maxfail:
        pytest_args.append(f"--maxfail={args.maxfail}")

    start = time.perf_counter()
    with tempfile.TemporaryDirectory() as raw_tmp:
        package_root = prepare_import_tree(Path(raw_tmp))
        env = os.environ.copy()
        env["PYTHONPATH"] = str(package_root) + os.pathsep + str(suite)
        env["ARRAY_API_TESTS_MODULE"] = "numrust"
        env["ARRAY_API_TESTS_VERSION"] = args.api_version
        proc = subprocess.run(
            pytest_args,
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
        )
    duration = time.perf_counter() - start

    report = json.loads(report_file.read_text(encoding="utf-8")) if report_file.exists() else {}
    failures = [
        {
            "nodeid": test.get("nodeid", ""),
            "outcome": test.get("outcome", ""),
        }
        for test in report.get("tests", [])
        if test.get("outcome") in {"failed", "error"}
    ][:20]
    result = {
        "status": "passed" if proc.returncode == 0 else "failed",
        "returncode": proc.returncode,
        "duration_seconds": duration,
        "source": {
            "repo": source["repo"],
            "commit": source["commit"],
            "commit_url": source["commit_url"],
        },
        "api_version": args.api_version,
        "targets": ["array_api_tests/"] if args.full else targets,
        "full_suite": args.full,
        "summary": report.get("summary", {}),
        "failures": failures,
        "command": " ".join(pytest_args),
        "stdout_tail": tail(proc.stdout),
        "stderr_tail": tail(proc.stderr),
    }
    result_json, result_md = result_paths(args.output_stem)
    result_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(result, result_md)
    print(json.dumps({k: result[k] for k in ("status", "returncode", "summary")}, indent=2))
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
