#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=2.0", "pandas>=2.0"]
# ///

from __future__ import annotations

import json
import os
import statistics
import subprocess
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "benchmark-results"
JSON_OUT = RESULT_DIR / "framerust-vs-pandas.json"
MD_OUT = RESULT_DIR / "framerust-vs-pandas.md"
CASE_NAME = "groupby_i64_sum_mean_count_250k_1000"


def median_ms(fn: Callable[[], float], rounds: int = 9) -> tuple[float, float]:
    samples = []
    checksum = fn()
    for _ in range(rounds):
        start = time.perf_counter()
        checksum = fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples), checksum


def run_framerust() -> dict:
    env = os.environ.copy()
    env.setdefault("RUSTFLAGS", "-C target-cpu=native")
    proc = subprocess.run(
        ["cargo", "run", "--release", "-p", "framerust", "--example", "pandas_cases_json"],
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
    raise RuntimeError("framerust benchmark did not emit JSON")


def build_frame(rows: int = 250_000, groups: int = 1_000) -> pd.DataFrame:
    idx = np.arange(rows, dtype=np.int64)
    return pd.DataFrame(
        {
            "store": idx % groups,
            "sales": (idx % 97).astype(np.float64) * 0.5 + 1.0,
            "weight": (idx % 31).astype(np.float64),
        }
    )


def checksum(result: pd.DataFrame) -> float:
    return float(
        result["sales_sum"].sum() + result["weight_mean"].sum() + result["rows"].sum()
    )


def bench_pandas() -> dict:
    frame = build_frame()

    def groupby_case() -> float:
        result = (
            frame.groupby("store", sort=False)
            .agg(
                rows=("store", "size"),
                sales_sum=("sales", "sum"),
                weight_mean=("weight", "mean"),
            )
            .reset_index()
        )
        return checksum(result)

    millis, case_checksum = median_ms(groupby_case)
    return {
        "engine": "pandas",
        "pandas_version": pd.__version__,
        "cases": [
            {
                "name": CASE_NAME,
                "millis": millis,
                "checksum": case_checksum,
            }
        ],
    }


def compare(framerust: dict, pandas_result: dict) -> dict:
    rust_case = framerust["cases"][0]
    pandas_case = pandas_result["cases"][0]
    speedup = pandas_case["millis"] / rust_case["millis"]
    checksum_abs_diff = abs(float(rust_case["checksum"]) - float(pandas_case["checksum"]))
    checksum_match = checksum_abs_diff <= 1e-6
    return {
        "framerust": framerust,
        "pandas": pandas_result,
        "comparison": [
            {
                "name": CASE_NAME,
                "framerust_ms": rust_case["millis"],
                "pandas_ms": pandas_case["millis"],
                "speedup_vs_pandas": speedup,
                "winner": "framerust" if speedup > 1.0 else "pandas",
                "checksum_abs_diff": checksum_abs_diff,
                "checksum_match": checksum_match,
            }
        ],
        "score": {
            "cases": 1,
            "framerust_wins": int(speedup > 1.0),
            "pandas_wins": int(speedup <= 1.0),
            "global_pandas_replacement_claim": False,
            "checksum_failures": [] if checksum_match else [CASE_NAME],
        },
    }


def write_markdown(result: dict) -> None:
    row = result["comparison"][0]
    lines = [
        "# FrameRust vs Pandas",
        "",
        "Same-data local benchmark for the implemented FrameRust aggregation slice.",
        "This is not a full Pandas replacement claim.",
        "",
        f"- Pandas version: `{result['pandas']['pandas_version']}`",
        f"- FrameRust wins: {result['score']['framerust_wins']}",
        f"- Pandas wins: {result['score']['pandas_wins']}",
        f"- Checksum failures: {len(result['score']['checksum_failures'])}",
        "- Global Pandas replacement claim: false",
        "",
        "| Case | FrameRust ms | Pandas ms | Speedup | Winner | Checksum |",
        "| --- | ---: | ---: | ---: | --- | --- |",
        (
            f"| `{row['name']}` | {row['framerust_ms']:.3f} | "
            f"{row['pandas_ms']:.3f} | {row['speedup_vs_pandas']:.2f}x | "
            f"{row['winner']} | {'ok' if row['checksum_match'] else 'failed'} |"
        ),
        "",
    ]
    MD_OUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    RESULT_DIR.mkdir(exist_ok=True)
    result = compare(run_framerust(), bench_pandas())
    JSON_OUT.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(result)
    print(
        json.dumps(
            {
                "framerust_wins": result["score"]["framerust_wins"],
                "pandas_wins": result["score"]["pandas_wins"],
                "checksum_failures": result["score"]["checksum_failures"],
                "status": "written",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
