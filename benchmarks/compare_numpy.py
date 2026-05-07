#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=2.0"]
# ///

from __future__ import annotations

import json
import math
import os
import statistics
import subprocess
import time
from pathlib import Path
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "benchmark-results"
JSON_OUT = RESULT_DIR / "numrust-vs-numpy.json"
MD_OUT = RESULT_DIR / "numrust-vs-numpy.md"


def median_ms(fn: Callable[[], float], rounds: int = 7) -> tuple[float, float]:
    samples = []
    checksum = fn()
    for _ in range(rounds):
        start = time.perf_counter()
        checksum = fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples), checksum


def run_numrust() -> dict:
    env = os.environ.copy()
    env.setdefault("RUSTFLAGS", "-C target-cpu=native")
    proc = subprocess.run(
        ["cargo", "run", "--release", "-p", "numrs-core", "--example", "bench_json"],
        cwd=ROOT,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
    for line in reversed(proc.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    raise RuntimeError("numrust benchmark did not emit JSON")


def edge_nonzero_checksum(mask: np.ndarray) -> float:
    checksum = 0.0
    for axis in np.nonzero(mask):
        if axis.size:
            checksum += float(axis[0] + axis[-1])
    return checksum


def bench_numpy() -> dict:
    cases = []

    small_a = np.arange(32, dtype=np.float64)
    small_b = small_a * 0.5
    millis, checksum = median_ms(
        lambda: sum(float((small_a + small_b).sum()) for _ in range(20_000))
    )
    cases.append(
        {"name": "small_add_f64_loop", "millis": millis, "checksum": checksum}
    )

    add_a = np.arange(250_000, dtype=np.float64)
    add_b = add_a * 0.5
    millis, checksum = median_ms(
        lambda: sum(float((add_a + add_b).sum()) for _ in range(50))
    )
    cases.append(
        {"name": "large_add_f64_loop", "millis": millis, "checksum": checksum}
    )

    millis, checksum = median_ms(
        lambda: sum(float((add_a + add_b).sum()) for _ in range(50))
    )
    cases.append(
        {"name": "fused_add_sum_f64_loop", "millis": millis, "checksum": checksum}
    )

    col = np.arange(1024, dtype=np.float64).reshape(1024, 1)
    row = np.arange(1024, dtype=np.float64).reshape(1, 1024)
    millis, checksum = median_ms(lambda: float((col + row).sum()))
    cases.append({"name": "broadcast_add_f64", "millis": millis, "checksum": checksum})

    sum_data = np.arange(1_000_000, dtype=np.float64) % 1024.0
    millis, checksum = median_ms(lambda: sum(float(sum_data.sum()) for _ in range(100)))
    cases.append({"name": "sum_f64_loop", "millis": millis, "checksum": checksum})

    metadata = np.arange(32, dtype=np.float64).reshape(4, 8)
    millis, checksum = median_ms(
        lambda: sum(
            float(metadata.T[None, :, :].squeeze(0).shape[0])
            for _ in range(200_000)
        )
    )
    cases.append({"name": "metadata_view_loop", "millis": millis, "checksum": checksum})

    take_source = np.fromfunction(lambda i, j: i * 16 + j, (256, 16), dtype=np.int64)
    take_indices = np.array([15, 3, 1, 7, 0, 14, 2, 10], dtype=np.int64)
    millis, checksum = median_ms(
        lambda: sum(
            float(np.take(take_source, take_indices, axis=1).sum())
            for _ in range(5_000)
        )
    )
    cases.append({"name": "take_axis_i64_loop", "millis": millis, "checksum": checksum})

    where_mask = np.fromfunction(
        lambda i, j: ((i * 17 + j * 31) % 5) == 0,
        (512, 512),
        dtype=np.int64,
    )
    where_true = np.fromfunction(
        lambda i, j: (i * 13 + j) % 101,
        (512, 512),
        dtype=np.float64,
    )
    where_false = np.fromfunction(
        lambda _i, j: -(j % 97),
        (1, 512),
        dtype=np.float64,
    )
    millis, checksum = median_ms(
        lambda: sum(
            float(np.where(where_mask, where_true, where_false).sum())
            for _ in range(400)
        )
    )
    cases.append(
        {"name": "where_select_f64_loop", "millis": millis, "checksum": checksum}
    )

    millis, checksum = median_ms(
        lambda: sum(edge_nonzero_checksum(where_mask) for _ in range(2_000))
    )
    cases.append({"name": "nonzero_bool_loop", "millis": millis, "checksum": checksum})

    left = np.fromfunction(
        lambda i, j: (i * 17 + j) % 97,
        (192, 192),
        dtype=np.float64,
    )
    right = np.fromfunction(
        lambda i, j: (i + j * 31) % 89,
        (192, 192),
        dtype=np.float64,
    )
    millis, checksum = median_ms(lambda: sum(float((left @ right)[0, 0]) for _ in range(500)))
    cases.append({"name": "dot_f64_192", "millis": millis, "checksum": checksum})

    return {
        "engine": "numpy",
        "numpy_version": np.__version__,
        "cases": cases,
    }


def checksum_ok(numrust_checksum: float, numpy_checksum: float) -> tuple[bool, float]:
    difference = abs(numrust_checksum - numpy_checksum)
    tolerance = max(1e-6, abs(numpy_checksum) * 1e-6)
    return difference <= tolerance, difference


def compare(numrust: dict, numpy_result: dict) -> dict:
    rust_cases = {case["name"]: case for case in numrust["cases"]}
    numpy_cases = {case["name"]: case for case in numpy_result["cases"]}
    missing = sorted(set(rust_cases) ^ set(numpy_cases))
    if missing:
        raise RuntimeError(f"benchmark case mismatch: {missing}")

    rows = []
    for name in rust_cases:
        rust_case = rust_cases[name]
        numpy_case = numpy_cases[name]
        rust_ms = rust_case["millis"]
        numpy_ms = numpy_case["millis"]
        speedup = numpy_ms / rust_ms
        ok, checksum_diff = checksum_ok(rust_case["checksum"], numpy_case["checksum"])
        if not ok:
            winner = "checksum_failed"
        else:
            winner = "numrust" if speedup > 1.0 else "numpy"
        rows.append(
            {
                "name": name,
                "numrust_ms": rust_ms,
                "numpy_ms": numpy_ms,
                "speedup_vs_numpy": speedup,
                "numrust_checksum": rust_case["checksum"],
                "numpy_checksum": numpy_case["checksum"],
                "checksum_abs_diff": checksum_diff,
                "checksum_ok": ok,
                "winner": winner,
            }
        )

    valid_rows = [row for row in rows if row["checksum_ok"]]
    wins = sum(1 for row in valid_rows if row["winner"] == "numrust")
    numpy_wins = sum(1 for row in valid_rows if row["winner"] == "numpy")
    checksum_failures = len(rows) - len(valid_rows)
    geomean = math.prod(row["speedup_vs_numpy"] for row in valid_rows) ** (
        1 / len(valid_rows)
    )
    return {
        "numrust": numrust,
        "numpy": numpy_result,
        "comparison": rows,
        "score": {
            "numrust_wins": wins,
            "numpy_wins": numpy_wins,
            "checksum_failures": checksum_failures,
            "geomean_speedup_vs_numpy": geomean,
            "numrust_ranked_higher_on_this_suite": wins > numpy_wins
            and checksum_failures == 0,
            "global_numpy_replacement_claim": False,
        },
    }


def write_markdown(result: dict) -> None:
    lines = [
        "# NumRust vs NumPy Benchmark",
        "",
        "Targeted same-data benchmark for implemented NumRust kernels. "
        "This is not a full NumPy replacement claim.",
        "",
        f"NumPy version: `{result['numpy']['numpy_version']}`",
        "",
        "| Case | NumRust ms | NumPy ms | Speedup vs NumPy | Checksum diff | Checksum ok | Winner |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in result["comparison"]:
        lines.append(
            f"| `{row['name']}` | {row['numrust_ms']:.3f} | {row['numpy_ms']:.3f} | "
            f"{row['speedup_vs_numpy']:.2f}x | {row['checksum_abs_diff']:.6g} | "
            f"{row['checksum_ok']} | {row['winner']} |"
        )
    score = result["score"]
    lines += [
        "",
        "## Score",
        "",
        f"- NumRust wins: {score['numrust_wins']}",
        f"- NumPy wins: {score['numpy_wins']}",
        f"- Checksum failures: {score['checksum_failures']}",
        f"- Geomean speedup vs NumPy: {score['geomean_speedup_vs_numpy']:.2f}x",
        f"- Ranked higher on this suite: {score['numrust_ranked_higher_on_this_suite']}",
        "- Global NumPy replacement claim: false",
        "",
        "A row with a checksum failure is not counted as a Rust win, regardless of timing.",
        "",
    ]
    MD_OUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    numrust = run_numrust()
    numpy_result = bench_numpy()
    result = compare(numrust, numpy_result)
    JSON_OUT.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(result)
    print(json.dumps(result["score"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
