from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

from .models import MemoryType, PersonaRequest
from .persona_agent import PersonaAgent


AgentAction = Literal["clarify", "final"]


@dataclass(frozen=True)
class ClarificationEvaluationCase:
    """One labeled, non-sensitive case for testing the agent decision boundary."""

    name: str
    request: PersonaRequest
    expected_action: AgentAction


@dataclass(frozen=True)
class ClarificationEvaluationResult:
    name: str
    expected_action: AgentAction
    actual_action: AgentAction
    retrieved_counts: dict[MemoryType, int]

    @property
    def correct(self) -> bool:
        return self.expected_action == self.actual_action

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "expected_action": self.expected_action,
            "actual_action": self.actual_action,
            "correct": self.correct,
            "retrieved_counts": dict(self.retrieved_counts),
        }


@dataclass(frozen=True)
class ClarificationEvaluationReport:
    results: tuple[ClarificationEvaluationResult, ...]

    @property
    def accuracy(self) -> float:
        return sum(result.correct for result in self.results) / len(self.results)

    @property
    def necessary_clarifications(self) -> int:
        return sum(
            result.expected_action == "clarify" and result.actual_action == "clarify"
            for result in self.results
        )

    @property
    def unnecessary_clarifications(self) -> int:
        return sum(
            result.expected_action == "final" and result.actual_action == "clarify"
            for result in self.results
        )

    @property
    def missed_clarifications(self) -> int:
        return sum(
            result.expected_action == "clarify" and result.actual_action == "final"
            for result in self.results
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "case_count": len(self.results),
            "accuracy": self.accuracy,
            "necessary_clarifications": self.necessary_clarifications,
            "unnecessary_clarifications": self.unnecessary_clarifications,
            "missed_clarifications": self.missed_clarifications,
            "cases": [result.to_dict() for result in self.results],
        }


def evaluate_clarification_policy(
    agent: PersonaAgent,
    cases: Sequence[ClarificationEvaluationCase],
) -> ClarificationEvaluationReport:
    """Evaluate clarify/final decisions without requiring a benchmark or judge model."""

    if not cases:
        raise ValueError("at least one clarification evaluation case is required")

    results = []
    for case in cases:
        decision = agent.decide(case.request)
        results.append(
            ClarificationEvaluationResult(
                name=case.name,
                expected_action=case.expected_action,
                actual_action=decision.action,
                retrieved_counts={
                    "trajectory": len(decision.memories.trajectory),
                    "workspace": len(decision.memories.workspace),
                    "persona": len(decision.memories.persona),
                },
            )
        )
    return ClarificationEvaluationReport(tuple(results))
