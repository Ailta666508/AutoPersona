from __future__ import annotations

import time
from collections.abc import Callable


class MvpMetrics:
    def __init__(self, clock: Callable[[], float] = time.perf_counter):
        self.clock = clock
        self.correct = 0
        self.incorrect = 0
        self.necessary_clarifications = 0
        self.unnecessary_clarifications = 0
        self.elapsed_ms = 0.0

    def record(
        self,
        correct: bool,
        clarification: str | None = None,
        elapsed_ms: float = 0.0,
    ) -> None:
        self.correct += int(correct)
        self.incorrect += int(not correct)
        self.necessary_clarifications += int(clarification == "necessary")
        self.unnecessary_clarifications += int(clarification == "unnecessary")
        self.elapsed_ms += elapsed_ms

    def snapshot(self) -> dict[str, float]:
        total = self.correct + self.incorrect
        return {
            "accuracy": self.correct / total if total else 0.0,
            "necessary_clarifications": float(self.necessary_clarifications),
            "unnecessary_clarifications": float(self.unnecessary_clarifications),
            "elapsed_ms": self.elapsed_ms,
        }
