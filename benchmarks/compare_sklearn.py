#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scikit-learn"]
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
import sklearn
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "benchmark-results"
JSON_OUT = RESULT_DIR / "learnrust-vs-sklearn.json"
MD_OUT = RESULT_DIR / "learnrust-vs-sklearn.md"


def median_ms(fn: Callable[[], float], rounds: int = 9) -> tuple[float, float]:
    samples = []
    checksum = fn()
    for _ in range(rounds):
        start = time.perf_counter()
        checksum = fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples), checksum


def scaler_matrix(rows: int = 200_000, cols: int = 12) -> np.ndarray:
    row = np.arange(rows, dtype=np.int64)[:, None]
    col = np.arange(cols, dtype=np.int64)[None, :]
    values = ((row * 31 + col * 17) % 1009).astype(np.float64) / 37.0
    values += col.astype(np.float64) * 0.25
    values -= (row % 11).astype(np.float64) * 0.03
    return np.ascontiguousarray(values)


def classification_matrix(
    rows: int, cols: int = 8, classes: int = 6
) -> tuple[np.ndarray, np.ndarray]:
    row = np.arange(rows, dtype=np.int64)[:, None]
    col = np.arange(cols, dtype=np.int64)[None, :]
    labels = (np.arange(rows, dtype=np.int64) % classes).astype(np.int64)
    values = labels[:, None].astype(np.float64) * 4.0 + col.astype(np.float64) * 0.2
    values = values + ((row * 19 + col * 23) % 97).astype(np.float64) / 500.0
    return np.ascontiguousarray(values), labels


def matrix_checksum(matrix: np.ndarray) -> float:
    values = matrix.ravel()
    if values.size == 0:
        return 0.0
    mid = values.size // 2
    return float(values[0] + values[mid] + values[-1])


def labels_checksum(values: np.ndarray) -> float:
    checksum = 0.0
    for idx, value in enumerate(values.tolist()):
        checksum += float((idx % 31) + 1) * float(value)
    return checksum


def counts_checksum(values: np.ndarray) -> float:
    checksum = 0.0
    for idx, value in enumerate(values.ravel().tolist()):
        checksum += float((idx % 17) + 1) * float(value)
    return checksum


def run_learnrust() -> dict:
    env = os.environ.copy()
    env.setdefault("RUSTFLAGS", "-C target-cpu=native")
    proc = subprocess.run(
        ["cargo", "run", "--release", "-p", "learnrust", "--example", "learnrust_cases_json"],
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
    raise RuntimeError("learnrust benchmark did not emit JSON")


def bench_sklearn() -> dict:
    scaler_input = scaler_matrix()

    def scaler_case() -> float:
        transformed = StandardScaler().fit_transform(scaler_input)
        return matrix_checksum(transformed)

    scaler_ms, scaler_checksum = median_ms(scaler_case)

    train, train_labels = classification_matrix(120_000)
    test, test_labels = classification_matrix(40_000)

    def nearest_centroid_case() -> float:
        predicted = NearestCentroid().fit(train, train_labels).predict(test)
        return labels_checksum(predicted) + float(accuracy_score(test_labels, predicted))

    centroid_ms, centroid_checksum = median_ms(nearest_centroid_case)

    metric_len = 1_000_000
    y_true = (np.arange(metric_len, dtype=np.int64) % 8).astype(np.int64)
    y_pred = ((np.arange(metric_len, dtype=np.int64) + (np.arange(metric_len) % 19 == 0)) % 8).astype(
        np.int64
    )
    labels = np.union1d(y_true, y_pred)

    def metrics_case() -> float:
        accuracy = accuracy_score(y_true, y_pred)
        matrix = confusion_matrix(y_true, y_pred, labels=labels)
        return float(accuracy) + counts_checksum(matrix)

    metrics_ms, metrics_checksum = median_ms(metrics_case)

    return {
        "engine": "sklearn",
        "sklearn_version": sklearn.__version__,
        "cases": [
            {
                "name": "standard_scaler_fit_transform_200000x12",
                "millis": scaler_ms,
                "checksum": scaler_checksum,
            },
            {
                "name": "nearest_centroid_fit_predict_120000x8_40000x8_6",
                "millis": centroid_ms,
                "checksum": centroid_checksum,
            },
            {
                "name": "accuracy_confusion_1000000_8",
                "millis": metrics_ms,
                "checksum": metrics_checksum,
            },
        ],
    }


def checksum_match(left: float, right: float) -> bool:
    return abs(left - right) <= max(1e-9, abs(right) * 1e-9)


def compare(learnrust: dict, sklearn_result: dict) -> dict:
    rust_cases = {case["name"]: case for case in learnrust["cases"]}
    sklearn_cases = {case["name"]: case for case in sklearn_result["cases"]}
    rows = []
    for name in sklearn_cases:
        rust_case = rust_cases[name]
        sklearn_case = sklearn_cases[name]
        speedup = sklearn_case["millis"] / rust_case["millis"]
        checksums_match = checksum_match(rust_case["checksum"], sklearn_case["checksum"])
        rows.append(
            {
                "name": name,
                "learnrust_ms": rust_case["millis"],
                "sklearn_ms": sklearn_case["millis"],
                "speedup_vs_sklearn": speedup,
                "winner": "learnrust" if speedup > 1.0 else "sklearn",
                "checksum_match": checksums_match,
                "checksum_abs_diff": abs(rust_case["checksum"] - sklearn_case["checksum"]),
            }
        )
    learnrust_wins = sum(1 for row in rows if row["winner"] == "learnrust")
    sklearn_wins = len(rows) - learnrust_wins
    checksum_failures = [row["name"] for row in rows if not row["checksum_match"]]
    geomean = math.exp(
        statistics.mean(math.log(row["speedup_vs_sklearn"]) for row in rows)
    )
    return {
        "learnrust": learnrust,
        "sklearn": sklearn_result,
        "comparison": rows,
        "score": {
            "cases": len(rows),
            "learnrust_wins": learnrust_wins,
            "sklearn_wins": sklearn_wins,
            "geomean_speedup_vs_sklearn": geomean,
            "checksum_failures": checksum_failures,
            "global_sklearn_replacement_claim": False,
        },
    }


def write_markdown(result: dict) -> None:
    score = result["score"]
    lines = [
        "# LearnRust vs scikit-learn",
        "",
        "Same-data local benchmark for the implemented preprocessing, nearest-centroid, and metrics slice.",
        "This is not a full scikit-learn replacement claim.",
        "",
        f"- scikit-learn version: `{result['sklearn']['sklearn_version']}`",
        f"- LearnRust wins: {score['learnrust_wins']}",
        f"- scikit-learn wins: {score['sklearn_wins']}",
        f"- Geomean speedup vs scikit-learn: {score['geomean_speedup_vs_sklearn']:.2f}x",
        f"- Checksum failures: {len(score['checksum_failures'])}",
        "- Global scikit-learn replacement claim: false",
        "",
        "| Case | LearnRust ms | scikit-learn ms | Speedup | Winner | Checksum |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ]
    for row in result["comparison"]:
        checksum = "ok" if row["checksum_match"] else f"diff {row['checksum_abs_diff']:.3g}"
        lines.append(
            f"| `{row['name']}` | {row['learnrust_ms']:.3f} | "
            f"{row['sklearn_ms']:.3f} | {row['speedup_vs_sklearn']:.2f}x | "
            f"{row['winner']} | {checksum} |"
        )
    lines.append("")
    MD_OUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    RESULT_DIR.mkdir(exist_ok=True)
    result = compare(run_learnrust(), bench_sklearn())
    JSON_OUT.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(result)
    print(
        json.dumps(
            {
                "checksum_failures": result["score"]["checksum_failures"],
                "learnrust_wins": result["score"]["learnrust_wins"],
                "sklearn_wins": result["score"]["sklearn_wins"],
                "status": "written",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
