#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=2.0", "statsmodels>=0.14"]
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
import statsmodels
import statsmodels.api as sm


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "benchmark-results"
JSON_OUT = RESULT_DIR / "statsrust-vs-statsmodels.json"
MD_OUT = RESULT_DIR / "statsrust-vs-statsmodels.md"


def median_ms(fn: Callable[[], float], rounds: int = 5) -> tuple[float, float]:
    samples = []
    checksum = fn()
    for _ in range(rounds):
        start = time.perf_counter()
        checksum = fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples), checksum


def run_statsrust() -> dict:
    env = os.environ.copy()
    env.setdefault("RUSTFLAGS", "-C target-cpu=native")
    proc = subprocess.run(
        ["cargo", "run", "--release", "-p", "statsrust", "--example", "statsmodels_cases_json"],
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
    raise RuntimeError("statsrust benchmark did not emit JSON")


def regression_data(rows: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.empty((rows, 3), dtype=np.float64)
    y = np.empty(rows, dtype=np.float64)
    weights = np.empty(rows, dtype=np.float64)
    for row in range(rows):
        x0 = (row % 97) / 48.0 - 1.0
        x1 = ((row * 7) % 113) / 56.0 - 1.0
        x2 = ((row * 13) % 89) / 44.0 - 1.0
        noise = (((row * 17) % 23) - 11.0) * 0.001
        x[row] = (x0, x1, x2)
        y[row] = 1.0 + 2.0 * x0 - 3.0 * x1 + 0.5 * x2 + noise
        weights[row] = 0.5 + ((row * 5) % 17) / 16.0
    return x, y, weights


def logit_data(rows: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.empty((rows, 2), dtype=np.float64)
    y = np.empty(rows, dtype=np.float64)
    for row in range(rows):
        x0 = (row % 101) / 50.0 - 1.0
        x1 = ((row * 11) % 97) / 48.0 - 1.0
        eta = -0.35 + 1.2 * x0 - 0.8 * x1
        probability = 1.0 / (1.0 + math.exp(-eta))
        draw = (((row * 37) % 101) + 0.5) / 101.0
        x[row] = (x0, x1)
        y[row] = float(draw < probability)
    return x, y


def coefficient_checksum(values: np.ndarray) -> float:
    return float(sum((idx + 1) * value for idx, value in enumerate(values)))


def edge_checksum(values: np.ndarray) -> float:
    return float(values[0] + values[len(values) // 2] + values[-1])


def bench_statsmodels() -> dict:
    cases = []
    x, y, weights = regression_data(2000)
    design = sm.add_constant(x, has_constant="add")

    def ols_fit() -> float:
        checksum = 0.0
        for _ in range(50):
            model = sm.OLS(y, design).fit()
            checksum += coefficient_checksum(np.asarray(model.params))
        return checksum

    millis, checksum = median_ms(ols_fit)
    cases.append({"name": "statsmodels_ols_fit_2000x3", "millis": millis, "checksum": checksum})

    ols_model = sm.OLS(y, design).fit()

    def ols_predict() -> float:
        checksum = 0.0
        for _ in range(5_000):
            checksum += edge_checksum(np.asarray(ols_model.predict(design)))
        return checksum

    millis, checksum = median_ms(ols_predict)
    cases.append({"name": "statsmodels_ols_predict_2000x3", "millis": millis, "checksum": checksum})

    def wls_fit() -> float:
        checksum = 0.0
        for _ in range(50):
            model = sm.WLS(y, design, weights=weights).fit()
            checksum += coefficient_checksum(np.asarray(model.params))
        return checksum

    millis, checksum = median_ms(wls_fit)
    cases.append({"name": "statsmodels_wls_fit_2000x3", "millis": millis, "checksum": checksum})

    logit_x, logit_y = logit_data(800)
    logit_design = sm.add_constant(logit_x, has_constant="add")

    def logit_fit() -> float:
        checksum = 0.0
        for _ in range(10):
            model = sm.Logit(logit_y, logit_design).fit(disp=0, maxiter=100, tol=1e-10)
            checksum += coefficient_checksum(np.asarray(model.params))
        return checksum

    millis, checksum = median_ms(logit_fit)
    cases.append({"name": "statsmodels_logit_fit_800x2", "millis": millis, "checksum": checksum})

    return {
        "engine": "statsmodels",
        "statsmodels_version": statsmodels.__version__,
        "cases": cases,
    }


def compare(statsrust: dict, statsmodels_result: dict) -> dict:
    rust_cases = {case["name"]: case for case in statsrust["cases"]}
    python_cases = {case["name"]: case for case in statsmodels_result["cases"]}
    rows = []
    for name in rust_cases:
        rust_case = rust_cases[name]
        python_case = python_cases[name]
        rust_ms = rust_case["millis"]
        python_ms = python_case["millis"]
        checksum_abs_diff = abs(float(rust_case["checksum"]) - float(python_case["checksum"]))
        checksum_ok = checksum_abs_diff <= max(1e-6, abs(float(python_case["checksum"])) * 1e-7)
        speedup = python_ms / rust_ms
        rows.append(
            {
                "name": name,
                "statsrust_ms": rust_ms,
                "statsmodels_ms": python_ms,
                "speedup_vs_statsmodels": speedup,
                "winner": "statsrust" if speedup > 1.0 else "statsmodels",
                "checksum_abs_diff": checksum_abs_diff,
                "checksum_ok": checksum_ok,
            }
        )

    wins = sum(1 for row in rows if row["winner"] == "statsrust")
    checksum_failures = [row["name"] for row in rows if not row["checksum_ok"]]
    geomean = math.prod(row["speedup_vs_statsmodels"] for row in rows) ** (1 / len(rows))
    return {
        "statsrust": statsrust,
        "statsmodels": statsmodels_result,
        "comparison": rows,
        "score": {
            "statsrust_wins": wins,
            "statsmodels_wins": len(rows) - wins,
            "geomean_speedup_vs_statsmodels": geomean,
            "checksum_failures": checksum_failures,
            "statsrust_ranked_higher_on_this_suite": wins > len(rows) / 2 and not checksum_failures,
            "global_statsmodels_replacement_claim": False,
        },
    }


def write_markdown(result: dict) -> None:
    lines = [
        "# StatsRust vs StatsModels Benchmark",
        "",
        f"StatsModels version: `{result['statsmodels']['statsmodels_version']}`",
        "",
        "| Case | StatsRust ms | StatsModels ms | Speedup vs StatsModels | Winner | Checksum |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ]
    for row in result["comparison"]:
        checksum = "ok" if row["checksum_ok"] else f"fail ({row['checksum_abs_diff']:.3e})"
        lines.append(
            f"| `{row['name']}` | {row['statsrust_ms']:.3f} | {row['statsmodels_ms']:.3f} | "
            f"{row['speedup_vs_statsmodels']:.2f}x | {row['winner']} | {checksum} |"
        )
    score = result["score"]
    lines += [
        "",
        "## Score",
        "",
        f"- StatsRust wins: {score['statsrust_wins']}",
        f"- StatsModels wins: {score['statsmodels_wins']}",
        f"- Geomean speedup vs StatsModels: {score['geomean_speedup_vs_statsmodels']:.2f}x",
        f"- Checksum failures: {len(score['checksum_failures'])}",
        f"- Ranked higher on this suite: {score['statsrust_ranked_higher_on_this_suite']}",
        "- Global StatsModels replacement claim: false",
        "",
        "This is a same-data benchmark for the implemented OLS, WLS, and Logit slice. It does not claim full StatsModels API parity.",
        "",
    ]
    MD_OUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    statsrust = run_statsrust()
    statsmodels_result = bench_statsmodels()
    result = compare(statsrust, statsmodels_result)
    JSON_OUT.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(result)
    print(json.dumps(result["score"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
