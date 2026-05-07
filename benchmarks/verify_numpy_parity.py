#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=2.0"]
# ///

# /// script
# dependencies = ["numpy"]
# ///

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def run_numrust() -> dict[str, list[float]]:
    proc = subprocess.run(
        ["cargo", "run", "--release", "-p", "numrs-core", "--example", "parity_json"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    for line in reversed(proc.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    raise RuntimeError("parity_json did not emit JSON")


def expected() -> dict[str, np.ndarray]:
    col = np.array([[1.0], [2.0], [3.0]])
    row = np.array([[10.0, 20.0, 30.0, 40.0]])
    a = np.arange(12, dtype=np.int64).reshape(3, 4)
    mask = np.array(
        [
            [True, False, False, True],
            [False, True, False, False],
            [True, False, True, False],
        ]
    )
    reductions = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    left = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    right = np.array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
    batched_left = np.array(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[2.0, 0.0, 1.0], [3.0, 1.0, 4.0]],
        ]
    )
    batched_right = np.array([[[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]])
    tensor_left = np.arange(24.0).reshape(2, 3, 4)
    tensor_right = np.arange(30.0).reshape(3, 2, 5)
    return {
        "broadcast_outer_add": (col + row).ravel(),
        "take_axis_i64": np.take(a, [-1, 1], axis=1).ravel(),
        "boolean_mask_i64": a[mask],
        "sum_axis0": reductions.sum(axis=0),
        "mean_axis1": reductions.mean(axis=1),
        "dot_f64": (left @ right).ravel(),
        "batched_matmul_f64": (batched_left @ batched_right).ravel(),
        "tensordot_f64": np.tensordot(
            tensor_left, tensor_right, axes=([1, 0], [0, 1])
        ).ravel(),
    }


def main() -> int:
    actual = run_numrust()
    expected_values = expected()
    failures = []
    for name, expected_array in expected_values.items():
        actual_array = np.array(actual[name])
        if not np.allclose(actual_array, expected_array):
            failures.append(
                {
                    "case": name,
                    "actual": actual_array.tolist(),
                    "expected": expected_array.tolist(),
                }
            )
    if failures:
        print(json.dumps({"status": "failed", "failures": failures}, indent=2))
        return 1
    print(json.dumps({"status": "passed", "cases": sorted(expected_values)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
