import json
import subprocess
import sys
import unittest


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
        self.assertEqual(
            [case["actual_action"] for case in report["cases"]],
            ["final", "clarify"],
        )


if __name__ == "__main__":
    unittest.main()
