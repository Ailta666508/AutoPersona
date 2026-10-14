"""Run a public, synthetic retrieval/clarification evaluation without API keys."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from autopersona_memory import JsonlMemoryStore, MemoryRetriever, PersonaAgent
from autopersona_memory.evaluation import (
    ClarificationEvaluationCase,
    evaluate_clarification_policy,
)
from autopersona_memory.models import PersonaMemory, PersonaRequest


def embed(text: str) -> list[float]:
    lowered = text.lower()
    return [
        float("concise" in lowered or "brief" in lowered),
        float("citation" in lowered or "source" in lowered),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-accuracy", type=float, default=1.0)
    parser.add_argument("--min-clarification-recall", type=float, default=1.0)
    args = parser.parse_args()
    with TemporaryDirectory(prefix="autopersona-eval-") as directory:
        store = JsonlMemoryStore(Path(directory) / "memory")
        store.add(
            "synthetic-user",
            "persona",
            PersonaMemory("writing", "concise answers", "Prefer brief explanations"),
        )
        retriever = MemoryRetriever(store, embed)

        def search(query: str) -> list[dict[str, str]]:
            return [{"memory": "persona", "query": query}]

        def decide(task, memories):
            if not memories["persona"]:
                return {
                    "action": "clarify",
                    "question": "What response style should I use?",
                }
            return {"action": "final", "answer": "Use the retrieved preference."}

        agent = PersonaAgent(
            retriever,
            search,
            decide,
            top_k=1,
            min_similarity=0.8,
        )
        cases = [
            ClarificationEvaluationCase(
                "relevant preference",
                PersonaRequest("synthetic-user", "Give me a concise summary"),
                "final",
                ("persona",),
            ),
            ClarificationEvaluationCase(
                "missing preference",
                PersonaRequest("synthetic-user", "Which citation format should I use?"),
                "clarify",
                (),
            ),
        ]
        report = evaluate_clarification_policy(agent, cases)
        payload = report.to_dict()
        payload["quality_gate"] = report.quality_gate(
            min_accuracy=args.min_accuracy,
            min_clarification_recall=args.min_clarification_recall,
        )
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0 if payload["quality_gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
