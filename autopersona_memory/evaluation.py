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
    expected_memory_types: tuple[MemoryType, ...] | None = None


@dataclass(frozen=True)
class ClarificationEvaluationResult:
    name: str
    expected_action: AgentAction
    actual_action: AgentAction
    expected_memory_types: tuple[MemoryType, ...] | None
    retrieved_counts: dict[MemoryType, int]

    @property
    def correct(self) -> bool:
        return self.expected_action == self.actual_action

    @property
    def missing_expected_memory_types(self) -> tuple[MemoryType, ...]:
        if self.expected_memory_types is None:
            return ()
        return tuple(
            memory_type
            for memory_type in self.expected_memory_types
            if self.retrieved_counts[memory_type] == 0
        )

    @property
    def unexpected_retrieved_memory_types(self) -> tuple[MemoryType, ...]:
        if self.expected_memory_types is None:
            return ()
        return tuple(
            memory_type
            for memory_type, count in self.retrieved_counts.items()
            if count > 0 and memory_type not in self.expected_memory_types
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "expected_action": self.expected_action,
            "actual_action": self.actual_action,
            "correct": self.correct,
            "expected_memory_types": (
                list(self.expected_memory_types)
                if self.expected_memory_types is not None
                else None
            ),
            "missing_expected_memory_types": list(self.missing_expected_memory_types),
            "unexpected_retrieved_memory_types": list(
                self.unexpected_retrieved_memory_types
            ),
            "retrieved_counts": dict(self.retrieved_counts),
        }


@dataclass(frozen=True)
class ClarificationEvaluationReport:
    results: tuple[ClarificationEvaluationResult, ...]

    @staticmethod
    def _ratio(numerator: float, denominator: float) -> float:
        return numerator / denominator if denominator else 0.0

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

    @property
    def correct_final_responses(self) -> int:
        return sum(
            result.expected_action == "final" and result.actual_action == "final"
            for result in self.results
        )

    @property
    def clarification_precision(self) -> float:
        """Fraction of clarification requests that were labeled as necessary."""

        return self._ratio(
            self.necessary_clarifications,
            self.necessary_clarifications + self.unnecessary_clarifications,
        )

    @property
    def clarification_recall(self) -> float:
        """Fraction of required clarifications that the policy requested."""

        return self._ratio(
            self.necessary_clarifications,
            self.necessary_clarifications + self.missed_clarifications,
        )

    @property
    def clarification_f1(self) -> float:
        """Balance clarification precision and recall in one decision metric."""

        precision = self.clarification_precision
        recall = self.clarification_recall
        return self._ratio(2 * precision * recall, precision + recall)

    @property
    def retrieval_coverage(self) -> float:
        """Fraction of labeled memory layers that supplied retrieved evidence."""

        labeled = [
            result for result in self.results if result.expected_memory_types is not None
        ]
        expected = sum(len(result.expected_memory_types or ()) for result in labeled)
        missing = sum(len(result.missing_expected_memory_types) for result in self.results)
        return self._ratio(expected - missing, expected)

    @property
    def retrieval_precision(self) -> float:
        """Fraction of retrieved memory layers that matched labeled expectations."""

        labeled = [
            result for result in self.results if result.expected_memory_types is not None
        ]
        retrieved = sum(
            count > 0
            for result in labeled
            for count in result.retrieved_counts.values()
        )
        unexpected = sum(
            len(result.unexpected_retrieved_memory_types) for result in labeled
        )
        return self._ratio(retrieved - unexpected, retrieved)

    @property
    def retrieval_f1(self) -> float:
        precision = self.retrieval_precision
        recall = self.retrieval_coverage
        return self._ratio(2 * precision * recall, precision + recall)

    def to_dict(self) -> dict[str, object]:
        return {
            "case_count": len(self.results),
            "accuracy": self.accuracy,
            "necessary_clarifications": self.necessary_clarifications,
            "unnecessary_clarifications": self.unnecessary_clarifications,
            "missed_clarifications": self.missed_clarifications,
            "correct_final_responses": self.correct_final_responses,
            "clarification_precision": self.clarification_precision,
            "clarification_recall": self.clarification_recall,
            "clarification_f1": self.clarification_f1,
            "retrieval_coverage": self.retrieval_coverage,
            "retrieval_precision": self.retrieval_precision,
            "retrieval_f1": self.retrieval_f1,
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
                expected_memory_types=case.expected_memory_types,
                retrieved_counts={
                    "trajectory": len(decision.memories.trajectory),
                    "workspace": len(decision.memories.workspace),
                    "persona": len(decision.memories.persona),
                },
            )
        )
    return ClarificationEvaluationReport(tuple(results))
