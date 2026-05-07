from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "benchmark-results"
LOCK_OUT = RESULT_DIR / "external-source-lock.json"


SOURCES: list[dict[str, Any]] = [
    {
        "id": "numpy-asv",
        "name": "NumPy ASV benchmark suite",
        "repo": "https://github.com/numpy/numpy.git",
        "github": "numpy/numpy",
        "ref": "refs/heads/main",
        "web_url": "https://github.com/numpy/numpy/tree/main/benchmarks/benchmarks",
        "files": [
            {
                "path": "benchmarks/benchmarks/bench_ufunc.py",
                "symbols": [
                    "Broadcast.time_broadcast",
                    "At.time_sum_at",
                    "At.time_maximum_at",
                    "NDArrayAsType.time_astype",
                ],
            },
            {
                "path": "benchmarks/benchmarks/bench_reduce.py",
                "symbols": [
                    "SmallReduction.time_small",
                    "StatsReductions.time_min",
                    "StatsReductions.time_max",
                    "StatsReductions.time_mean",
                    "StatsReductions.time_std",
                    "StatsReductions.time_prod",
                    "StatsReductions.time_var",
                    "ArgMax.time_argmax",
                    "ArgMin.time_argmin",
                ],
            },
            {
                "path": "benchmarks/benchmarks/bench_itemselection.py",
                "symbols": [
                    "Take.time_contiguous",
                    "PutMask.time_dense",
                    "PutMask.time_sparse",
                    "Put.time_ordered",
                ],
            },
            {
                "path": "benchmarks/benchmarks/bench_linalg.py",
                "symbols": [
                    "Eindot.time_dot_a_b",
                    "Eindot.time_dot_d_dot_b_c",
                    "Eindot.time_matmul_a_b",
                    "Eindot.time_matmul_d_matmul_b_c",
                    "Eindot.time_dot_trans_a_at",
                    "Eindot.time_dot_trans_a_atc",
                    "Eindot.time_dot_trans_at_a",
                    "Eindot.time_dot_trans_atc_a",
                    "Eindot.time_inner_trans_a_a",
                    "Eindot.time_inner_trans_a_ac",
                    "Eindot.time_einsum_i_ij_j",
                    "Eindot.time_einsum_ij_jk_a_b",
                    "Eindot.time_matmul_trans_a_at",
                    "Eindot.time_matmul_trans_a_atc",
                    "Eindot.time_matmul_trans_at_a",
                    "Eindot.time_matmul_trans_atc_a",
                    "Eindot.time_tensordot_a_b_axes_1_0_0_1",
                    "LinalgSmallArrays.time_norm_small_array",
                    "LinalgSmallArrays.time_det_small_array",
                    "LinalgSmallArrays.time_det_3x3",
                    "LinalgSmallArrays.time_solve_3x3",
                    "Lstsq.time_numpy_linalg_lstsq_a__b_float64",
                    "Einsum.time_einsum_outer",
                    "Einsum.time_einsum_multiply",
                    "Einsum.time_einsum_sum_mul",
                    "Einsum.time_einsum_sum_mul2",
                    "Einsum.time_einsum_mul",
                    "Einsum.time_einsum_contig_outstride0",
                    "Einsum.time_einsum_contig_contig",
                    "Einsum.time_einsum_noncon_outer",
                    "Einsum.time_einsum_noncon_multiply",
                    "Einsum.time_einsum_noncon_sum_mul",
                    "Einsum.time_einsum_noncon_sum_mul2",
                    "Einsum.time_einsum_noncon_mul",
                    "Einsum.time_einsum_noncon_contig_contig",
                    "Einsum.time_einsum_noncon_contig_outstride0",
                ],
            },
            {
                "path": "benchmarks/benchmarks/bench_manipulate.py",
                "symbols": [
                    "BroadcastArraysTo.time_broadcast_to",
                    "DimsManipulations.time_expand_dims",
                    "DimsManipulations.time_expand_dims_neg",
                    "DimsManipulations.time_squeeze_dims",
                    "DimsManipulations.time_reshape",
                ],
            },
            {
                "path": "benchmarks/benchmarks/common.py",
                "symbols": ["Benchmark"],
            },
        ],
    },
    {
        "id": "array-api-tests",
        "name": "Python Array API conformance tests",
        "repo": "https://github.com/data-apis/array-api-tests.git",
        "github": "data-apis/array-api-tests",
        "ref": "HEAD",
        "web_url": "https://data-apis.org/array-api-tests/",
        "files": [
            {
                "path": "README.md",
                "symbols": ["ARRAY_API_TESTS_MODULE", "pytest array_api_tests/"],
            }
        ],
    },
    {
        "id": "scipy-asv",
        "name": "SciPy ASV benchmark suite",
        "repo": "https://github.com/scipy/scipy.git",
        "github": "scipy/scipy",
        "ref": "refs/heads/main",
        "web_url": "https://github.com/scipy/scipy/tree/main/benchmarks/benchmarks",
        "files": [
            {
                "path": "benchmarks/benchmarks/optimize_zeros.py",
                "symbols": ["Zeros.time_zeros"],
            },
            {
                "path": "scipy/optimize/_tstutils.py",
                "symbols": ["f2", "methods", "mstrings", "functions", "fstrings"],
            },
            {
                "path": "benchmarks/benchmarks/integrate.py",
                "symbols": ["CumulativeSimpson.time_1d", "CumulativeSimpson.time_multid"],
            },
        ],
    },
    {
        "id": "statsmodels-source",
        "name": "StatsModels source APIs",
        "repo": "https://github.com/statsmodels/statsmodels.git",
        "github": "statsmodels/statsmodels",
        "ref": "refs/heads/main",
        "web_url": "https://github.com/statsmodels/statsmodels",
        "files": [
            {
                "path": "statsmodels/regression/linear_model.py",
                "symbols": ["OLS", "WLS"],
            },
            {
                "path": "statsmodels/discrete/discrete_model.py",
                "symbols": ["Logit"],
            },
            {
                "path": "README.rst",
                "symbols": ["Linear regression models", "Discrete models"],
            },
        ],
    },
]


def git_rev(repo: str, ref: str) -> str:
    proc = subprocess.run(
        ["git", "ls-remote", repo, ref],
        check=True,
        capture_output=True,
        text=True,
    )
    first = proc.stdout.strip().splitlines()[0]
    return first.split()[0]


def raw_url(github: str, commit: str, path: str) -> str:
    return f"https://raw.githubusercontent.com/{github}/{commit}/{path}"


def fetch_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "numrust-external-evidence"})
    with urllib.request.urlopen(request, timeout=45) as response:
        return response.read()


def build_lock() -> dict[str, Any]:
    generated_at = dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()
    lock: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": generated_at,
        "purpose": "Pinned online sources for benchmark, conformance, and upstream API evidence.",
        "sources": [],
    }

    for source in SOURCES:
        commit = git_rev(source["repo"], source["ref"])
        locked_files = []
        for file_spec in source["files"]:
            url = raw_url(source["github"], commit, file_spec["path"])
            body = fetch_bytes(url)
            locked_files.append(
                {
                    "path": file_spec["path"],
                    "raw_url": url,
                    "sha256": hashlib.sha256(body).hexdigest(),
                    "bytes": len(body),
                    "symbols": file_spec["symbols"],
                }
            )
        lock["sources"].append(
            {
                "id": source["id"],
                "name": source["name"],
                "repo": source["repo"],
                "ref": source["ref"],
                "commit": commit,
                "commit_url": f"https://github.com/{source['github']}/commit/{commit}",
                "web_url": source["web_url"],
                "files": locked_files,
            }
        )

    return lock


def verify_pinned_files(lock: dict[str, Any]) -> list[str]:
    failures = []
    for source in lock.get("sources", []):
        for file_spec in source.get("files", []):
            body = fetch_bytes(file_spec["raw_url"])
            actual = hashlib.sha256(body).hexdigest()
            if actual != file_spec["sha256"]:
                failures.append(f"{source['id']}:{file_spec['path']}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--update-lock", action="store_true")
    parser.add_argument("--verify-pinned", action="store_true")
    args = parser.parse_args()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    if args.update_lock:
        lock = build_lock()
        LOCK_OUT.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"status": "updated", "lock": str(LOCK_OUT)}, indent=2))
        return 0

    if not LOCK_OUT.exists():
        print(json.dumps({"status": "missing-lock", "lock": str(LOCK_OUT)}, indent=2))
        return 1

    lock = json.loads(LOCK_OUT.read_text(encoding="utf-8"))
    if args.verify_pinned:
        failures = verify_pinned_files(lock)
        if failures:
            print(json.dumps({"status": "failed", "hash_mismatches": failures}, indent=2))
            return 1

    summary = {
        "status": "ok",
        "sources": [
            {
                "id": source["id"],
                "commit": source["commit"],
                "files": len(source["files"]),
            }
            for source in lock["sources"]
        ],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
