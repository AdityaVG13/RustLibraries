#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = ["networkx>=3.0"]
# ///

from __future__ import annotations

import json
import os
import statistics
import subprocess
import time
from pathlib import Path
from typing import Callable

import networkx as nx


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "benchmark-results"
JSON_OUT = RESULT_DIR / "graphrust-vs-networkx.json"
MD_OUT = RESULT_DIR / "graphrust-vs-networkx.md"
CASE_NAME = "bfs_undirected_5000_30000"


def graph_edges(node_count: int = 5_000) -> list[tuple[int, int]]:
    edges = []
    for node in range(node_count):
        edges.append((node, (node + 1) % node_count))
        for step in range(1, 6):
            edges.append((node, (node * 37 + step * 97) % node_count))
    return edges


def checksum(distances: dict[int, int], node_count: int = 5_000) -> float:
    return float(sum((idx + 1) * distances.get(idx, 0) for idx in range(node_count)))


def median_ms(fn: Callable[[], float], rounds: int = 11) -> tuple[float, float]:
    samples = []
    checksum_value = fn()
    for _ in range(rounds):
        start = time.perf_counter()
        checksum_value = fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples), checksum_value


def run_graphrust() -> dict:
    env = os.environ.copy()
    env.setdefault("RUSTFLAGS", "-C target-cpu=native")
    proc = subprocess.run(
        ["cargo", "run", "--release", "-p", "graphrust", "--example", "networkx_cases_json"],
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
    raise RuntimeError("graphrust benchmark did not emit JSON")


def bench_networkx() -> dict:
    graph = nx.Graph()
    graph.add_nodes_from(range(5_000))
    graph.add_edges_from(graph_edges())

    def bfs_case() -> float:
        return checksum(nx.single_source_shortest_path_length(graph, 0))

    millis, checksum_value = median_ms(bfs_case)
    return {
        "engine": "networkx",
        "networkx_version": nx.__version__,
        "cases": [{"name": CASE_NAME, "millis": millis, "checksum": checksum_value}],
    }


def compare(graphrust: dict, networkx_result: dict) -> dict:
    rust_case = graphrust["cases"][0]
    networkx_case = networkx_result["cases"][0]
    speedup = networkx_case["millis"] / rust_case["millis"]
    checksum_abs_diff = abs(float(rust_case["checksum"]) - float(networkx_case["checksum"]))
    checksum_match = checksum_abs_diff <= 1e-9
    return {
        "graphrust": graphrust,
        "networkx": networkx_result,
        "comparison": [
            {
                "name": CASE_NAME,
                "graphrust_ms": rust_case["millis"],
                "networkx_ms": networkx_case["millis"],
                "speedup_vs_networkx": speedup,
                "winner": "graphrust" if speedup > 1.0 else "networkx",
                "checksum_abs_diff": checksum_abs_diff,
                "checksum_match": checksum_match,
            }
        ],
        "score": {
            "cases": 1,
            "graphrust_wins": int(speedup > 1.0),
            "networkx_wins": int(speedup <= 1.0),
            "global_networkx_replacement_claim": False,
            "checksum_failures": [] if checksum_match else [CASE_NAME],
        },
    }


def write_markdown(result: dict) -> None:
    row = result["comparison"][0]
    lines = [
        "# GraphRust vs NetworkX",
        "",
        "Same-data local benchmark for the implemented GraphRust BFS slice.",
        "This is not a full NetworkX replacement claim.",
        "",
        f"- NetworkX version: `{result['networkx']['networkx_version']}`",
        f"- GraphRust wins: {result['score']['graphrust_wins']}",
        f"- NetworkX wins: {result['score']['networkx_wins']}",
        f"- Checksum failures: {len(result['score']['checksum_failures'])}",
        "- Global NetworkX replacement claim: false",
        "",
        "| Case | GraphRust ms | NetworkX ms | Speedup | Winner | Checksum |",
        "| --- | ---: | ---: | ---: | --- | --- |",
        (
            f"| `{row['name']}` | {row['graphrust_ms']:.3f} | "
            f"{row['networkx_ms']:.3f} | {row['speedup_vs_networkx']:.2f}x | "
            f"{row['winner']} | {'ok' if row['checksum_match'] else 'failed'} |"
        ),
        "",
    ]
    MD_OUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    RESULT_DIR.mkdir(exist_ok=True)
    result = compare(run_graphrust(), bench_networkx())
    JSON_OUT.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(result)
    print(
        json.dumps(
            {
                "checksum_failures": result["score"]["checksum_failures"],
                "graphrust_wins": result["score"]["graphrust_wins"],
                "networkx_wins": result["score"]["networkx_wins"],
                "status": "written",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
