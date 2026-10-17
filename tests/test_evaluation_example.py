import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from autopersona_memory.evaluation import ClarificationEvaluationReport, ClarificationEvaluationResult


class EvaluationExampleTests(unittest.TestCase):
    def test_public_example_runs_and_exercises_both_decisions(self):
        result = subprocess.run(
            [sys.executable, "examples/evaluate_retrieval_clarification.py"],
            capture_output=True,
            text=True,
            check=True,
        )
        report = json.loads(result.stdout)
        self.assertEqual(report["case_count"], 2)
        self.assertEqual(report["accuracy"], 1.0)
        self.assertEqual(report["clarification_precision"], 1.0)
        self.assertEqual(report["clarification_recall"], 1.0)
        self.assertTrue(report["quality_gate"]["passed"])
        self.assertEqual(
            [case["actual_action"] for case in report["cases"]],
            ["final", "clarify"],
        )

    def test_public_example_rejects_invalid_quality_gate_threshold(self):
        result = subprocess.run(
            [sys.executable, "examples/evaluate_retrieval_clarification.py", "--min-accuracy", "1.1"],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("thresholds", result.stderr)

    def test_public_example_can_write_a_ci_artifact(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "reports" / "evaluation.json"
            result = subprocess.run([sys.executable, "examples/evaluate_retrieval_clarification.py", "--output", str(output)], capture_output=True, text=True, check=True)
            self.assertEqual(json.loads(output.read_text())["quality_gate"]["passed"], True)
            self.assertIn('"quality_gate"', result.stdout)
            self.assertEqual(list(output.parent.glob("*.tmp")), [])

    def test_quality_gate_is_ci_friendly(self):
        report = ClarificationEvaluationReport((
            ClarificationEvaluationResult("good", "final", "final", None, {"trajectory": 0, "workspace": 0, "persona": 0}),
            ClarificationEvaluationResult("needed", "clarify", "clarify", None, {"trajectory": 0, "workspace": 0, "persona": 0}),
        ))
        self.assertTrue(report.quality_gate()["passed"])
        failed = ClarificationEvaluationReport((
            ClarificationEvaluationResult("missed", "clarify", "final", None, {"trajectory": 0, "workspace": 0, "persona": 0}),
        ))
        self.assertEqual(failed.quality_gate()["failures"], ["accuracy", "clarification_recall"])


if __name__ == "__main__":
    unittest.main()
