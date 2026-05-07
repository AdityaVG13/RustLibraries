#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=2.0", "scipy>=1.13"]
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
import scipy
from scipy import integrate, optimize


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "benchmark-results"
JSON_OUT = RESULT_DIR / "scirust-vs-scipy.json"
MD_OUT = RESULT_DIR / "scirust-vs-scipy.md"
ASV_CUMULATIVE_1D_REPETITIONS = 1_000


def median_ms(fn: Callable[[], float], rounds: int = 5) -> tuple[float, float]:
    samples = []
    checksum = fn()
    for _ in range(rounds):
        start = time.perf_counter()
        checksum = fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples), checksum


def run_scirust() -> dict:
    env = os.environ.copy()
    env.setdefault("RUSTFLAGS", "-C target-cpu=native")
    proc = subprocess.run(
        ["cargo", "run", "--release", "-p", "scirust", "--example", "scipy_cases_json"],
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
    raise RuntimeError("scirust benchmark did not emit JSON")


def integration_data(samples: int) -> tuple[np.ndarray, np.ndarray, float]:
    lower = 0.0
    upper = math.pi
    x = np.linspace(lower, upper, samples, dtype=np.float64)
    y = np.sin(x) + 0.05 * np.cos(3.0 * x)
    dx = (upper - lower) / (samples - 1)
    return x, y, dx


def cumulative_simpson_asv_data() -> tuple[np.ndarray, np.ndarray, float]:
    x, dx = np.linspace(0, 5, 1000, retstep=True)
    y = np.sin(2 * np.pi * x)
    y2 = np.tile(y, (100, 100, 1))
    return y, y2, dx


def objective(x: float) -> float:
    centered = x - 1.2345
    return centered * centered + 2.0


def root_function(x: float) -> float:
    return x * x * x - x - 2.0


def scipy_asv_f2(x: float) -> float:
    return x * x - 1.0


def bench_scipy() -> dict:
    cases = []
    x, y, dx = integration_data(10_001)

    def trapezoid_case() -> float:
        checksum = 0.0
        for _ in range(2_000):
            checksum += float(integrate.trapezoid(y, x=x))
        return checksum

    millis, checksum = median_ms(trapezoid_case)
    cases.append({"name": "scipy_integrate_trapezoid_10001", "millis": millis, "checksum": checksum})

    def simpson_case() -> float:
        checksum = 0.0
        for _ in range(2_000):
            checksum += float(integrate.simpson(y, dx=dx))
        return checksum

    millis, checksum = median_ms(simpson_case)
    cases.append({"name": "scipy_integrate_simpson_10001", "millis": millis, "checksum": checksum})

    asv_y, asv_y2, asv_dx = cumulative_simpson_asv_data()

    def asv_cumulative_simpson_1d() -> float:
        checksum = 0.0
        for _ in range(ASV_CUMULATIVE_1D_REPETITIONS):
            checksum += float(np.sum(integrate.cumulative_simpson(asv_y, dx=asv_dx)))
        return checksum

    millis, checksum = median_ms(asv_cumulative_simpson_1d)
    cases.append(
        {
            "name": "scipy_asv_cumulative_simpson_1d_1000",
            "millis": millis,
            "checksum": checksum,
        }
    )

    def asv_cumulative_simpson_multid() -> float:
        return float(np.sum(integrate.cumulative_simpson(asv_y2, dx=asv_dx)))

    millis, checksum = median_ms(asv_cumulative_simpson_multid)
    cases.append(
        {
            "name": "scipy_asv_cumulative_simpson_multid_100x100x1000",
            "millis": millis,
            "checksum": checksum,
        }
    )

    def minimize_case() -> float:
        checksum = 0.0
        for _ in range(10_000):
            result = optimize.minimize_scalar(
                objective,
                bounds=(-5.0, 5.0),
                method="bounded",
                options={"xatol": 1e-10, "maxiter": 100},
            )
            checksum += float(result.x + result.fun)
        return checksum

    millis, checksum = median_ms(minimize_case)
    cases.append({"name": "scipy_optimize_minimize_scalar_bounded", "millis": millis, "checksum": checksum})

    def bisect_case() -> float:
        checksum = 0.0
        for _ in range(10_000):
            root = optimize.bisect(root_function, 1.0, 2.0, xtol=1e-12, maxiter=100)
            checksum += float(root + abs(root_function(root)))
        return checksum

    millis, checksum = median_ms(bisect_case)
    cases.append({"name": "scipy_optimize_bisect", "millis": millis, "checksum": checksum})

    def asv_zeros_f2_bisect() -> float:
        checksum = 0.0
        for _ in range(10_000):
            root = optimize.bisect(scipy_asv_f2, 0.5, math.sqrt(3.0), xtol=1e-12, maxiter=100)
            checksum += float(root + abs(scipy_asv_f2(root)))
        return checksum

    millis, checksum = median_ms(asv_zeros_f2_bisect)
    cases.append({"name": "scipy_asv_zeros_f2_bisect", "millis": millis, "checksum": checksum})

    def brentq_case() -> float:
        checksum = 0.0
        for _ in range(10_000):
            root = optimize.brentq(root_function, 1.0, 2.0, xtol=1e-12, rtol=1e-12, maxiter=100)
            checksum += float(root + abs(root_function(root)))
        return checksum

    millis, checksum = median_ms(brentq_case)
    cases.append({"name": "scipy_optimize_brentq", "millis": millis, "checksum": checksum})

    def asv_zeros_f2_brentq() -> float:
        checksum = 0.0
        for _ in range(10_000):
            root = optimize.brentq(
                scipy_asv_f2, 0.5, math.sqrt(3.0), xtol=1e-12, rtol=1e-12, maxiter=100
            )
            checksum += float(root + abs(scipy_asv_f2(root)))
        return checksum

    millis, checksum = median_ms(asv_zeros_f2_brentq)
    cases.append({"name": "scipy_asv_zeros_f2_brentq", "millis": millis, "checksum": checksum})

    return {
        "engine": "scipy",
        "scipy_version": scipy.__version__,
        "cases": cases,
    }


def compare(scirust: dict, scipy_result: dict) -> dict:
    rust_cases = {case["name"]: case for case in scirust["cases"]}
    python_cases = {case["name"]: case for case in scipy_result["cases"]}
    rows = []
    for name in rust_cases:
        rust_case = rust_cases[name]
        python_case = python_cases[name]
        rust_ms = rust_case["millis"]
        python_ms = python_case["millis"]
        checksum_abs_diff = abs(float(rust_case["checksum"]) - float(python_case["checksum"]))
        checksum_ok = checksum_abs_diff <= max(1e-5, abs(float(python_case["checksum"])) * 1e-7)
        speedup = python_ms / rust_ms
        rows.append(
            {
                "name": name,
                "scirust_ms": rust_ms,
                "scipy_ms": python_ms,
                "speedup_vs_scipy": speedup,
                "winner": "scirust" if speedup > 1.0 else "scipy",
                "checksum_abs_diff": checksum_abs_diff,
                "checksum_ok": checksum_ok,
            }
        )

    wins = sum(1 for row in rows if row["winner"] == "scirust")
    checksum_failures = [row["name"] for row in rows if not row["checksum_ok"]]
    geomean = math.prod(row["speedup_vs_scipy"] for row in rows) ** (1 / len(rows))
    return {
        "scirust": scirust,
        "scipy": scipy_result,
        "comparison": rows,
        "score": {
            "scirust_wins": wins,
            "scipy_wins": len(rows) - wins,
            "geomean_speedup_vs_scipy": geomean,
            "checksum_failures": checksum_failures,
            "scirust_ranked_higher_on_this_suite": wins > len(rows) / 2 and not checksum_failures,
            "global_scipy_replacement_claim": False,
        },
    }


def write_markdown(result: dict) -> None:
    lines = [
        "# SciRust vs SciPy Benchmark",
        "",
        f"SciPy version: `{result['scipy']['scipy_version']}`",
        "",
        "| Case | SciRust ms | SciPy ms | Speedup vs SciPy | Winner | Checksum |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ]
    for row in result["comparison"]:
        checksum = "ok" if row["checksum_ok"] else f"fail ({row['checksum_abs_diff']:.3e})"
        lines.append(
            f"| `{row['name']}` | {row['scirust_ms']:.3f} | {row['scipy_ms']:.3f} | "
            f"{row['speedup_vs_scipy']:.2f}x | {row['winner']} | {checksum} |"
        )
    score = result["score"]
    lines += [
        "",
        "## Score",
        "",
        f"- SciRust wins: {score['scirust_wins']}",
        f"- SciPy wins: {score['scipy_wins']}",
        f"- Geomean speedup vs SciPy: {score['geomean_speedup_vs_scipy']:.2f}x",
        f"- Checksum failures: {len(score['checksum_failures'])}",
        f"- Ranked higher on this suite: {score['scirust_ranked_higher_on_this_suite']}",
        "- Global SciPy replacement claim: false",
        "",
        "This suite includes translated SciPy ASV root-finding and cumulative Simpson cases, plus same-data integration/optimization cases for the implemented slice. It does not claim full SciPy API parity.",
        "",
    ]
    MD_OUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    scirust = run_scirust()
    scipy_result = bench_scipy()
    result = compare(scirust, scipy_result)
    JSON_OUT.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(result)
    print(json.dumps(result["score"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
