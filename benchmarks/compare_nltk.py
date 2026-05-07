#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = ["nltk>=3.9"]
# ///

from __future__ import annotations

import json
import os
import statistics
import subprocess
import time
from pathlib import Path
from typing import Callable

import nltk
from nltk.tokenize import wordpunct_tokenize


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "benchmark-results"
JSON_OUT = RESULT_DIR / "textrust-vs-nltk.json"
MD_OUT = RESULT_DIR / "textrust-vs-nltk.md"
CASE_NAME = "wordpunct_tokenize_200000_sentences"


def corpus(sentences: int = 200_000) -> str:
    parts = []
    for idx in range(sentences):
        parts.append(
            "Revenue, cost, and margin changed in segment_"
            f"{idx % 19}! Model {idx % 7} scored {idx % 101} points; retry? yes.\n"
        )
    return "".join(parts)


def token_checksum(tokens: list[str]) -> int:
    out = 0
    for idx, token in enumerate(tokens):
        kind_add = 17 if (token.replace("_", "").isalnum()) else 31
        out += (idx + 1) * len(token)
        out += kind_add
    return out


def median_ms(fn: Callable[[], int], rounds: int = 9) -> tuple[float, int]:
    samples = []
    checksum_value = fn()
    for _ in range(rounds):
        start = time.perf_counter()
        checksum_value = fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples), checksum_value


def run_textrust() -> dict:
    env = os.environ.copy()
    env.setdefault("RUSTFLAGS", "-C target-cpu=native")
    proc = subprocess.run(
        ["cargo", "run", "--release", "-p", "textrust", "--example", "nltk_cases_json"],
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
    raise RuntimeError("textrust benchmark did not emit JSON")


def bench_nltk() -> dict:
    text = corpus()

    def tokenize_case() -> int:
        return token_checksum(wordpunct_tokenize(text))

    millis, checksum_value = median_ms(tokenize_case)
    return {
        "engine": "nltk",
        "nltk_version": nltk.__version__,
        "cases": [{"name": CASE_NAME, "millis": millis, "checksum": checksum_value}],
    }


def compare(textrust: dict, nltk_result: dict) -> dict:
    rust_case = textrust["cases"][0]
    nltk_case = nltk_result["cases"][0]
    speedup = nltk_case["millis"] / rust_case["millis"]
    checksum_match = int(rust_case["checksum"]) == int(nltk_case["checksum"])
    return {
        "textrust": textrust,
        "nltk": nltk_result,
        "comparison": [
            {
                "name": CASE_NAME,
                "textrust_ms": rust_case["millis"],
                "nltk_ms": nltk_case["millis"],
                "speedup_vs_nltk": speedup,
                "winner": "textrust" if speedup > 1.0 else "nltk",
                "checksum_match": checksum_match,
            }
        ],
        "score": {
            "cases": 1,
            "textrust_wins": int(speedup > 1.0),
            "nltk_wins": int(speedup <= 1.0),
            "checksum_failures": [] if checksum_match else [CASE_NAME],
            "global_nltk_replacement_claim": False,
        },
    }


def write_markdown(result: dict) -> None:
    score = result["score"]
    row = result["comparison"][0]
    lines = [
        "# TextRust vs NLTK",
        "",
        "Same-data local benchmark for the implemented word/punctuation tokenization slice.",
        "This is not a full NLTK or spaCy replacement claim.",
        "",
        f"- NLTK version: `{result['nltk']['nltk_version']}`",
        f"- TextRust wins: {score['textrust_wins']}",
        f"- NLTK wins: {score['nltk_wins']}",
        f"- Checksum failures: {len(score['checksum_failures'])}",
        "- Global NLTK replacement claim: false",
        "",
        "| Case | TextRust ms | NLTK ms | Speedup | Winner | Checksum |",
        "| --- | ---: | ---: | ---: | --- | --- |",
        (
            f"| `{row['name']}` | {row['textrust_ms']:.3f} | "
            f"{row['nltk_ms']:.3f} | {row['speedup_vs_nltk']:.2f}x | "
            f"{row['winner']} | {'ok' if row['checksum_match'] else 'failed'} |"
        ),
        "",
    ]
    MD_OUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    RESULT_DIR.mkdir(exist_ok=True)
    result = compare(run_textrust(), bench_nltk())
    JSON_OUT.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(result)
    print(
        json.dumps(
            {
                "checksum_failures": result["score"]["checksum_failures"],
                "nltk_wins": result["score"]["nltk_wins"],
                "status": "written",
                "textrust_wins": result["score"]["textrust_wins"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
