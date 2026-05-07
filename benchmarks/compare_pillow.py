#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = ["pillow>=10.0"]
# ///

from __future__ import annotations

from io import BytesIO
import json
import os
import statistics
import subprocess
import time
from pathlib import Path
from typing import Callable

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "benchmark-results"
JSON_OUT = RESULT_DIR / "imagerust-vs-pillow.json"
MD_OUT = RESULT_DIR / "imagerust-vs-pillow.md"
CASE_NAME = "ppm_grayscale_resize_threshold_2048x1024"


def ppm_image(width: int = 2048, height: int = 1024) -> bytes:
    out = bytearray(f"P6\n{width} {height}\n255\n".encode())
    for y in range(height):
        for x in range(width):
            out.append((x * 13 + y * 3) & 255)
            out.append((x * 7 + y * 11) & 255)
            out.append((x * 5 + y * 17) & 255)
    return bytes(out)


def checksum(data: bytes) -> int:
    total = 0
    for idx, value in enumerate(data):
        total = (total + (idx + 1) * value) & 0xFFFF_FFFF_FFFF_FFFF
    return total


def median_ms(fn: Callable[[], int], rounds: int = 9) -> tuple[float, int]:
    samples = []
    checksum_value = fn()
    for _ in range(rounds):
        start = time.perf_counter()
        checksum_value = fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples), checksum_value


def run_imagerust() -> dict:
    env = os.environ.copy()
    env.setdefault("RUSTFLAGS", "-C target-cpu=native")
    proc = subprocess.run(
        ["cargo", "run", "--release", "-p", "imagerust", "--example", "pillow_cases_json"],
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
    raise RuntimeError("imagerust benchmark did not emit JSON")


def bench_pillow() -> dict:
    image_bytes = ppm_image()

    def image_case() -> int:
        with Image.open(BytesIO(image_bytes)) as image:
            gray = image.convert("L")
            resized = gray.resize((1024, 512), Image.Resampling.NEAREST)
            thresholded = resized.point(lambda value: 255 if value >= 128 else 0)
            return checksum(thresholded.tobytes())

    millis, checksum_value = median_ms(image_case)
    return {
        "engine": "pillow",
        "pillow_version": Image.__version__,
        "cases": [{"name": CASE_NAME, "millis": millis, "checksum": checksum_value}],
    }


def compare(imagerust: dict, pillow: dict) -> dict:
    rust_case = imagerust["cases"][0]
    pillow_case = pillow["cases"][0]
    speedup = pillow_case["millis"] / rust_case["millis"]
    checksum_match = int(rust_case["checksum"]) == int(pillow_case["checksum"])
    return {
        "imagerust": imagerust,
        "pillow": pillow,
        "comparison": [
            {
                "name": CASE_NAME,
                "imagerust_ms": rust_case["millis"],
                "pillow_ms": pillow_case["millis"],
                "speedup_vs_pillow": speedup,
                "winner": "imagerust" if speedup > 1.0 else "pillow",
                "checksum_match": checksum_match,
            }
        ],
        "score": {
            "cases": 1,
            "imagerust_wins": int(speedup > 1.0),
            "pillow_wins": int(speedup <= 1.0),
            "checksum_failures": [] if checksum_match else [CASE_NAME],
            "global_pillow_replacement_claim": False,
        },
    }


def write_markdown(result: dict) -> None:
    score = result["score"]
    row = result["comparison"][0]
    lines = [
        "# ImageRust vs Pillow",
        "",
        "Same-data local benchmark for the implemented PPM grayscale, resize, and threshold slice.",
        "This is not a full Pillow replacement claim.",
        "",
        f"- Pillow version: `{result['pillow']['pillow_version']}`",
        f"- ImageRust wins: {score['imagerust_wins']}",
        f"- Pillow wins: {score['pillow_wins']}",
        f"- Checksum failures: {len(score['checksum_failures'])}",
        "- Global Pillow replacement claim: false",
        "",
        "| Case | ImageRust ms | Pillow ms | Speedup | Winner | Checksum |",
        "| --- | ---: | ---: | ---: | --- | --- |",
        (
            f"| `{row['name']}` | {row['imagerust_ms']:.3f} | "
            f"{row['pillow_ms']:.3f} | {row['speedup_vs_pillow']:.2f}x | "
            f"{row['winner']} | {'ok' if row['checksum_match'] else 'failed'} |"
        ),
        "",
    ]
    MD_OUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    RESULT_DIR.mkdir(exist_ok=True)
    result = compare(run_imagerust(), bench_pillow())
    JSON_OUT.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(result)
    print(
        json.dumps(
            {
                "checksum_failures": result["score"]["checksum_failures"],
                "imagerust_wins": result["score"]["imagerust_wins"],
                "pillow_wins": result["score"]["pillow_wins"],
                "status": "written",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
