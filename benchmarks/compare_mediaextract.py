#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = ["beautifulsoup4>=4.12", "pypdf>=5.0"]
# ///

from __future__ import annotations

from io import BytesIO
import html
import json
import os
import statistics
import subprocess
import time
from pathlib import Path
from typing import Callable

from bs4 import BeautifulSoup
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "benchmark-results"
JSON_OUT = RESULT_DIR / "mediaextractrust-vs-python.json"
MD_OUT = RESULT_DIR / "mediaextractrust-vs-python.md"


def html_doc(repetitions: int = 20_000) -> bytes:
    parts = ["<html><body>"]
    for idx in range(repetitions):
        parts.append(
            f"<section><h2>Invoice {idx}</h2>"
            f"<p>Total &amp; tax for North&nbsp;America segment {idx % 17}</p></section>"
        )
    parts.append("</body></html>")
    return "".join(parts).encode()


def pdf_doc(repetitions: int = 20_000) -> bytes:
    stream_parts = ["BT\n/F1 12 Tf\n72 720 Td\n"]
    for idx in range(repetitions):
        stream_parts.append(f"(Invoice {idx} total tax segment {idx % 17}) Tj\n0 -14 Td\n")
    stream_parts.append("ET\n")
    stream = "".join(stream_parts).encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"endstream",
    ]
    output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for obj_id, obj in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{obj_id} 0 obj\n".encode())
        output.extend(obj)
        output.extend(b"\nendobj\n")
    xref_offset = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode())
    output.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode()
    )
    return bytes(output)


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def checksum(data: bytes) -> int:
    value = 14_695_981_039_346_656_037
    for byte in data:
        value ^= byte
        value = (value * 1_099_511_628_211) & 0xFFFF_FFFF_FFFF_FFFF
    return value


def text_checksum(text: str) -> int:
    return checksum(normalize_whitespace(text).encode())


def median_ms(fn: Callable[[], int], rounds: int = 9) -> tuple[float, int]:
    samples = []
    checksum_value = fn()
    for _ in range(rounds):
        start = time.perf_counter()
        checksum_value = fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples), checksum_value


def run_mediaextractrust() -> dict:
    env = os.environ.copy()
    env.setdefault("RUSTFLAGS", "-C target-cpu=native")
    proc = subprocess.run(
        ["cargo", "run", "--release", "-p", "mediaextractrust", "--example", "python_cases_json"],
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
    raise RuntimeError("mediaextractrust benchmark did not emit JSON")


def bench_python() -> dict:
    html_bytes = html_doc()
    pdf_bytes = pdf_doc()

    def html_case() -> int:
        soup = BeautifulSoup(html_bytes, "html.parser")
        return text_checksum(html.unescape(soup.get_text(" ")))

    def pdf_case() -> int:
        reader = PdfReader(BytesIO(pdf_bytes))
        return text_checksum(" ".join(page.extract_text() or "" for page in reader.pages))

    html_ms, html_sum = median_ms(html_case)
    pdf_ms, pdf_sum = median_ms(pdf_case)
    return {
        "engine": "python",
        "libraries": {"beautifulsoup4": "bs4", "pypdf": "pypdf"},
        "cases": [
            {"name": "html_text_20000_sections", "millis": html_ms, "checksum": html_sum},
            {"name": "pdf_literal_text_20000_lines", "millis": pdf_ms, "checksum": pdf_sum},
        ],
    }


def compare(rust: dict, python: dict) -> dict:
    rust_cases = {case["name"]: case for case in rust["cases"]}
    python_cases = {case["name"]: case for case in python["cases"]}
    rows = []
    failures = []
    for name in python_cases:
        rust_case = rust_cases[name]
        python_case = python_cases[name]
        speedup = python_case["millis"] / rust_case["millis"]
        checksum_match = int(rust_case["checksum"]) == int(python_case["checksum"])
        if not checksum_match:
            failures.append(name)
        rows.append(
            {
                "name": name,
                "mediaextractrust_ms": rust_case["millis"],
                "python_ms": python_case["millis"],
                "speedup_vs_python": speedup,
                "winner": "mediaextractrust" if speedup > 1.0 else "python",
                "checksum_match": checksum_match,
            }
        )
    wins = sum(row["winner"] == "mediaextractrust" for row in rows)
    return {
        "mediaextractrust": rust,
        "python": python,
        "comparison": rows,
        "score": {
            "cases": len(rows),
            "mediaextractrust_wins": wins,
            "python_wins": len(rows) - wins,
            "checksum_failures": failures,
            "global_python_document_pipeline_replacement_claim": False,
        },
    }


def write_markdown(result: dict) -> None:
    score = result["score"]
    lines = [
        "# MediaExtractRust vs Python Extraction Libraries",
        "",
        "Same-data local benchmark for the implemented HTML and uncompressed PDF text extraction slice.",
        "This is not a full OCR, Office document, or general document pipeline replacement claim.",
        "",
        f"- MediaExtractRust wins: {score['mediaextractrust_wins']}",
        f"- Python wins: {score['python_wins']}",
        f"- Checksum failures: {len(score['checksum_failures'])}",
        "- Global Python document pipeline replacement claim: false",
        "",
        "| Case | MediaExtractRust ms | Python ms | Speedup | Winner | Checksum |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ]
    for row in result["comparison"]:
        lines.append(
            f"| `{row['name']}` | {row['mediaextractrust_ms']:.3f} | "
            f"{row['python_ms']:.3f} | {row['speedup_vs_python']:.2f}x | "
            f"{row['winner']} | {'ok' if row['checksum_match'] else 'failed'} |"
        )
    lines.append("")
    MD_OUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    RESULT_DIR.mkdir(exist_ok=True)
    result = compare(run_mediaextractrust(), bench_python())
    JSON_OUT.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(result)
    print(
        json.dumps(
            {
                "checksum_failures": result["score"]["checksum_failures"],
                "mediaextractrust_wins": result["score"]["mediaextractrust_wins"],
                "python_wins": result["score"]["python_wins"],
                "status": "written",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
