#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = ["pydantic>=2.0"]
# ///

from __future__ import annotations

import json
import os
import statistics
import subprocess
import time
from pathlib import Path
from typing import Callable

import pydantic
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "benchmark-results"
JSON_OUT = RESULT_DIR / "validaterust-vs-pydantic.json"
MD_OUT = RESULT_DIR / "validaterust-vs-pydantic.md"
CASE_NAME = "user_schema_100000"


class UserRecord(BaseModel):
    id: int = Field(ge=0, le=1_000_000)
    email: str = Field(min_length=5)
    score: float
    active: bool
    tags: list[str]


def records(count: int = 100_000) -> list[dict]:
    return [
        {
            "id": idx,
            "email": f"user{idx}@example.test",
            "score": float(idx % 100) + 0.25,
            "active": idx % 2 == 0,
            "tags": [f"segment{idx % 17}", f"cohort{idx % 11}"],
        }
        for idx in range(count)
    ]


def checksum(values: list[UserRecord]) -> int:
    out = 0
    for value in values:
        out += value.id
        out += len(value.email)
        out += len(value.tags)
    return out


def median_ms(fn: Callable[[], int], rounds: int = 9) -> tuple[float, int]:
    samples = []
    checksum_value = fn()
    for _ in range(rounds):
        start = time.perf_counter()
        checksum_value = fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples), checksum_value


def run_validaterust() -> dict:
    env = os.environ.copy()
    env.setdefault("RUSTFLAGS", "-C target-cpu=native")
    proc = subprocess.run(
        ["cargo", "run", "--release", "-p", "validaterust", "--example", "pydantic_cases_json"],
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
    raise RuntimeError("validaterust benchmark did not emit JSON")


def bench_pydantic() -> dict:
    data = records()

    def validate_case() -> int:
        return checksum([UserRecord.model_validate(record) for record in data])

    millis, checksum_value = median_ms(validate_case)
    return {
        "engine": "pydantic",
        "pydantic_version": pydantic.__version__,
        "cases": [{"name": CASE_NAME, "millis": millis, "checksum": checksum_value}],
    }


def compare(validaterust: dict, pydantic_result: dict) -> dict:
    rust_case = validaterust["cases"][0]
    pydantic_case = pydantic_result["cases"][0]
    speedup = pydantic_case["millis"] / rust_case["millis"]
    checksum_match = int(rust_case["checksum"]) == int(pydantic_case["checksum"])
    return {
        "validaterust": validaterust,
        "pydantic": pydantic_result,
        "comparison": [
            {
                "name": CASE_NAME,
                "validaterust_ms": rust_case["millis"],
                "pydantic_ms": pydantic_case["millis"],
                "speedup_vs_pydantic": speedup,
                "winner": "validaterust" if speedup > 1.0 else "pydantic",
                "checksum_match": checksum_match,
            }
        ],
        "score": {
            "cases": 1,
            "validaterust_wins": int(speedup > 1.0),
            "pydantic_wins": int(speedup <= 1.0),
            "checksum_failures": [] if checksum_match else [CASE_NAME],
            "global_pydantic_replacement_claim": False,
        },
    }


def write_markdown(result: dict) -> None:
    score = result["score"]
    row = result["comparison"][0]
    lines = [
        "# ValidateRust vs Pydantic",
        "",
        "Same-data local benchmark for the implemented schema validation slice.",
        "This is not a full Pydantic replacement claim.",
        "",
        f"- Pydantic version: `{result['pydantic']['pydantic_version']}`",
        f"- ValidateRust wins: {score['validaterust_wins']}",
        f"- Pydantic wins: {score['pydantic_wins']}",
        f"- Checksum failures: {len(score['checksum_failures'])}",
        "- Global Pydantic replacement claim: false",
        "",
        "| Case | ValidateRust ms | Pydantic ms | Speedup | Winner | Checksum |",
        "| --- | ---: | ---: | ---: | --- | --- |",
        (
            f"| `{row['name']}` | {row['validaterust_ms']:.3f} | "
            f"{row['pydantic_ms']:.3f} | {row['speedup_vs_pydantic']:.2f}x | "
            f"{row['winner']} | {'ok' if row['checksum_match'] else 'failed'} |"
        ),
        "",
    ]
    MD_OUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    RESULT_DIR.mkdir(exist_ok=True)
    result = compare(run_validaterust(), bench_pydantic())
    JSON_OUT.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(result)
    print(
        json.dumps(
            {
                "checksum_failures": result["score"]["checksum_failures"],
                "pydantic_wins": result["score"]["pydantic_wins"],
                "status": "written",
                "validaterust_wins": result["score"]["validaterust_wins"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
