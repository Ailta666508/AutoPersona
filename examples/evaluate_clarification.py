"""Evaluate AutoPersona's decision boundary on synthetic, API-free cases."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from autopersona_memory import (
    ClarificationEvaluationCase,
    JsonlMemoryStore,
    MemoryRetriever,
    PersonaAgent,
    PersonaMemory,
    PersonaRequest,
    evaluate_clarification_policy,
)


def embed(text: str) -> list[float]:
    lowered = text.lower()
    return [float(term in lowered) for term in ("paper", "open-source", "budget")]


def main() -> None:
    with TemporaryDirectory() as directory:
        store = JsonlMemoryStore(Path(directory) / "memory")
        store.add(
            "known-user",
            "persona",
            PersonaMemory(
                topic="paper search",
                preference="Prefer open-source implementations",
                strategy="Check code availability and reproduction cost",
            ),
        )
        retriever = MemoryRetriever(store, embed)
        agent = PersonaAgent(
            retriever,
            lambda query: [{"memory": "persona", "query": query}],
            lambda task, memories: (
                {"action": "final", "answer": "Apply the retrieved preference."}
                if memories["persona"]
                else {"action": "clarify", "question": "Should code availability be required?"}
            ),
        )
        report = evaluate_clarification_policy(
            agent,
            [
                ClarificationEvaluationCase(
                    "retrieved preference",
                    PersonaRequest("known-user", "Recommend an open-source paper"),
                    "final",
                ),
                ClarificationEvaluationCase(
                    "missing preference",
                    PersonaRequest("new-user", "Recommend a paper"),
                    "clarify",
                ),
            ],
        )
        print("Synthetic software-validation cases; not research benchmark results.")
        print(json.dumps(report.to_dict(), indent=2))


if __name__ == "__main__":
    main()
