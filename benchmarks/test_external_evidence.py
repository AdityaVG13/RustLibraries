from __future__ import annotations

import json
import unittest
from pathlib import Path

import external_numpy_cases


ROOT = Path(__file__).resolve().parents[1]


class ExternalEvidenceSchemaTests(unittest.TestCase):
    def load_report(self) -> dict:
        report_path = ROOT / "benchmark-results" / "external-numpy-asv-inspired.json"
        if not report_path.exists():
            self.skipTest("external NumPy report has not been generated")
        return json.loads(report_path.read_text(encoding="utf-8"))

    def test_runnable_cases_have_source_metadata(self) -> None:
        for case in external_numpy_cases.RUNNABLE_CASES:
            self.assertTrue(case.name.startswith("asv_"))
            self.assertEqual(case.source_id, "numpy-asv")
            self.assertTrue(case.source_path.startswith("benchmarks/benchmarks/"))
            self.assertGreater(case.repetitions, 0)

    def test_unsupported_cases_are_explicit(self) -> None:
        self.assertGreaterEqual(len(external_numpy_cases.UNSUPPORTED_EXTERNAL_CASES), 1)
        unsupported = {case["case"] for case in external_numpy_cases.UNSUPPORTED_EXTERNAL_CASES}
        self.assertIn(
            "eig/remaining LAPACK, remaining strided/batched matmul, and full linalg/einsum grammar",
            unsupported,
        )
        for case in external_numpy_cases.UNSUPPORTED_EXTERNAL_CASES:
            self.assertIn("source", case)
            self.assertIn("case", case)
            self.assertIn("reason", case)

    def test_source_lock_schema_when_present(self) -> None:
        lock_path = ROOT / "benchmark-results" / "external-source-lock.json"
        if not lock_path.exists():
            self.skipTest("external source lock has not been generated")
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        self.assertEqual(lock["schema_version"], 1)
        source_ids = {source["id"] for source in lock["sources"]}
        self.assertIn("numpy-asv", source_ids)
        self.assertIn("array-api-tests", source_ids)
        self.assertIn("scipy-asv", source_ids)
        self.assertIn("statsmodels-source", source_ids)

    def test_runnable_case_symbols_are_locked_when_present(self) -> None:
        lock_path = ROOT / "benchmark-results" / "external-source-lock.json"
        if not lock_path.exists():
            self.skipTest("external source lock has not been generated")
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        locked_symbols = {
            (source["id"], file_spec["path"], symbol)
            for source in lock["sources"]
            for file_spec in source["files"]
            for symbol in file_spec["symbols"]
        }
        for case in external_numpy_cases.RUNNABLE_CASES:
            root_symbol = case.source_symbol.split("(")[0]
            self.assertIn(
                (case.source_id, case.source_path, root_symbol),
                locked_symbols,
                msg=f"{case.name} source symbol is not present in external-source-lock.json",
            )

    def test_report_covers_every_runnable_case_when_present(self) -> None:
        report = self.load_report()
        specs = {case.name: case for case in external_numpy_cases.RUNNABLE_CASES}
        rows = {row["name"]: row for row in report["comparison"]}
        self.assertEqual(set(rows), set(specs))
        self.assertEqual(report["score"]["supported_external_cases"], len(specs))
        for name, spec in specs.items():
            row = rows[name]
            self.assertEqual(row["source_id"], spec.source_id)
            self.assertEqual(row["source_path"], spec.source_path)
            self.assertEqual(row["source_symbol"], spec.source_symbol)
            self.assertEqual(row["translation"], spec.translation)
            self.assertEqual(row["repetitions"], spec.repetitions)
            self.assertEqual(len(row["numrust_pass_millis"]), report["benchmark_passes_per_engine"])
            self.assertEqual(len(row["numpy_pass_millis"]), report["benchmark_passes_per_engine"])

    def test_report_score_matches_raw_rows_when_present(self) -> None:
        report = self.load_report()
        rows = report["comparison"]
        score = report["score"]
        numrust_wins = sum(row["winner"] == "numrust" for row in rows)
        numpy_wins = sum(row["winner"] == "numpy" for row in rows)
        checksum_failures = [row["name"] for row in rows if not row["checksum_match"]]
        near_ties = [row["name"] for row in rows if row["near_tie"]]
        self.assertEqual(score["numrust_wins"], numrust_wins)
        self.assertEqual(score["numpy_wins"], numpy_wins)
        self.assertEqual(score["checksum_failures"], checksum_failures)
        self.assertEqual(score["near_tie_cases"], near_ties)
        self.assertEqual(score["unsupported_external_cases"], len(report["unsupported_external_cases"]))
        self.assertFalse(score["global_numpy_replacement_claim"])

    def test_loss_triage_matches_numpy_winners_when_present(self) -> None:
        report = self.load_report()
        expected = [
            row["name"]
            for row in sorted(
                (row for row in report["comparison"] if row["winner"] == "numpy"),
                key=lambda row: row["speedup_vs_numpy"],
            )
        ]
        triage = external_numpy_cases.loss_triage_payload(report)
        self.assertEqual([row["name"] for row in triage["rows"]], expected)
        self.assertEqual(triage["numpy_wins"], len(expected))
        self.assertFalse(triage["global_numpy_replacement_claim"])
        for priority, row in enumerate(triage["rows"], start=1):
            self.assertEqual(row["priority"], priority)
            self.assertLess(row["speedup_vs_numpy"], 1.0)
            self.assertGreater(row["numrust_slowdown_vs_numpy"], 0.0)
            self.assertEqual(len(row["numrust_pass_millis"]), report["benchmark_passes_per_engine"])
            self.assertEqual(len(row["numpy_pass_millis"]), report["benchmark_passes_per_engine"])

    def test_focused_loss_payload_uses_current_numpy_winners(self) -> None:
        report = self.load_report()
        loss_names = [
            row["name"]
            for row in sorted(
                (row for row in report["comparison"] if row["winner"] == "numpy"),
                key=lambda row: row["speedup_vs_numpy"],
            )
        ]
        numrust = {
            "engine": "numrust",
            "cases": [
                {
                    "name": name,
                    "millis": float(idx + 1),
                    "checksum": float(idx),
                }
                for idx, name in enumerate(loss_names)
            ],
        }
        numpy = {
            "engine": "numpy",
            "cases": [
                {
                    "name": name,
                    "millis": float((idx + 1) * 2),
                    "checksum": float(idx),
                }
                for idx, name in enumerate(loss_names)
            ],
        }
        payload = external_numpy_cases.focused_loss_payload(report, numrust, numpy)
        self.assertEqual([row["name"] for row in payload["rows"]], loss_names)
        self.assertEqual(payload["focused_cases"], len(loss_names))
        self.assertEqual(payload["benchmark_passes_per_engine"], 1)
        self.assertEqual(payload["numrust_wins"], len(loss_names))
        self.assertEqual(payload["numpy_wins"], 0)
        self.assertEqual(payload["checksum_failures"], [])
        self.assertFalse(payload["global_numpy_replacement_claim"])
        self.assertEqual(payload["stability"]["source_numpy_winners"], len(loss_names))
        self.assertEqual(payload["stability"]["focused_numrust_winner_flips"], len(loss_names))
        self.assertEqual(payload["stability"]["focused_still_numpy_winners"], 0)
        self.assertEqual(payload["stability"]["numrust_flip_cases"], loss_names)
        self.assertEqual(payload["stability"]["stable_numpy_win_cases"], [])
        self.assertEqual(
            payload["stability"]["authoritative_score_source"],
            "benchmark-results/external-numpy-asv-inspired.json",
        )
        for row in payload["rows"]:
            self.assertEqual(len(row["numrust_pass_millis"]), 1)
            self.assertEqual(len(row["numpy_pass_millis"]), 1)

    def test_focused_loss_artifact_stability_matches_rows_when_present(self) -> None:
        artifact_path = ROOT / "benchmark-results" / "external-numpy-loss-focused.json"
        if not artifact_path.exists():
            self.skipTest("focused loss report has not been generated")
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        rows = payload["rows"]
        numrust_flip_cases = [row["name"] for row in rows if row["winner"] == "numrust"]
        stable_numpy_win_cases = [row["name"] for row in rows if row["winner"] == "numpy"]
        near_tie_cases = [row["name"] for row in rows if row["near_tie"]]
        stability = payload["stability"]
        self.assertEqual(stability["source_numpy_winners"], len(rows))
        self.assertEqual(stability["focused_numrust_winner_flips"], len(numrust_flip_cases))
        self.assertEqual(stability["focused_still_numpy_winners"], len(stable_numpy_win_cases))
        self.assertEqual(stability["focused_near_ties"], len(near_tie_cases))
        self.assertEqual(stability["numrust_flip_cases"], numrust_flip_cases)
        self.assertEqual(stability["stable_numpy_win_cases"], stable_numpy_win_cases)
        self.assertEqual(stability["near_tie_cases"], near_tie_cases)
        self.assertEqual(payload["numrust_wins"], len(numrust_flip_cases))
        self.assertEqual(payload["numpy_wins"], len(stable_numpy_win_cases))

    def test_selected_engine_aggregation_preserves_focused_pass_samples(self) -> None:
        case_names = [case.name for case in external_numpy_cases.RUNNABLE_CASES[:2]]
        runs = [
            {
                "engine": "numrust",
                "cases": [
                    {"name": case_names[0], "millis": 3.0, "checksum": 10.0},
                    {"name": case_names[1], "millis": 8.0, "checksum": 20.0},
                ],
            },
            {
                "engine": "numrust",
                "cases": [
                    {"name": case_names[0], "millis": 1.0, "checksum": 11.0},
                    {"name": case_names[1], "millis": 4.0, "checksum": 21.0},
                ],
            },
            {
                "engine": "numrust",
                "cases": [
                    {"name": case_names[0], "millis": 2.0, "checksum": 12.0},
                    {"name": case_names[1], "millis": 6.0, "checksum": 22.0},
                ],
            },
        ]
        result = external_numpy_cases.aggregate_selected_engine_runs(case_names, runs)
        self.assertEqual(result["benchmark_passes"], 3)
        self.assertEqual(result["cases"][0]["millis"], 2.0)
        self.assertEqual(result["cases"][0]["pass_millis"], [3.0, 1.0, 2.0])
        self.assertEqual(result["cases"][0]["checksum"], 12.0)

    def test_sharded_pass_aggregation_preserves_raw_samples(self) -> None:
        def engine_run(engine: str, multiplier: float) -> dict:
            return {
                "engine": engine,
                "cases": [
                    {
                        "name": case.name,
                        "millis": (idx + 1) * multiplier,
                        "checksum": float(idx),
                    }
                    for idx, case in enumerate(external_numpy_cases.RUNNABLE_CASES)
                ],
            }

        pass_payloads = [
            {
                "pass_idx": 1,
                "numrust": engine_run("numrust", 2.0),
                "numpy": engine_run("numpy", 4.0),
            },
            {
                "pass_idx": 0,
                "numrust": engine_run("numrust", 1.0),
                "numpy": engine_run("numpy", 3.0),
            },
        ]
        result = external_numpy_cases.aggregate_pass_payloads(pass_payloads, {"sources": []})
        first = result["comparison"][0]
        self.assertEqual(result["benchmark_passes_per_engine"], 2)
        self.assertEqual(first["numrust_pass_millis"], [1.0, 2.0])
        self.assertEqual(first["numpy_pass_millis"], [3.0, 4.0])
        self.assertEqual(first["winner"], "numrust")
        self.assertEqual(result["score"]["supported_external_cases"], len(external_numpy_cases.RUNNABLE_CASES))
        self.assertEqual(result["score"]["checksum_failures"], [])


if __name__ == "__main__":
    unittest.main()
