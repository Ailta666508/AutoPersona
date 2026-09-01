from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence

from .models import (
    Memory,
    MemoryType,
    PersonaMemory,
    RawTaskRecord,
    TrajectoryMemory,
    UpdateDecision,
    WorkspaceMemory,
)
from .refinement import refine_persona, refine_trajectory, refine_workspace
from .store import JsonlMemoryStore, MEMORY_CLASSES


Embedder = Callable[[str], Sequence[float]]


class MemoryResolver(Protocol):
    def __call__(
        self,
        memory_type: MemoryType,
        new_memory: dict[str, Any],
        candidates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class IngestionResult:
    trajectory: TrajectoryMemory
    workspace: WorkspaceMemory
    persona: list[PersonaMemory] = field(default_factory=list)
    operations: list[UpdateDecision] = field(default_factory=list)


class MemoryUpdater:
    def __init__(
        self,
        store: JsonlMemoryStore,
        embedder: Embedder,
        resolver: MemoryResolver,
        top_k: int = 3,
    ):
        self.store = store
        self.embedder = embedder
        self.resolver = resolver
        self.top_k = top_k

    def update(
        self,
        user_id: str,
        memory_type: MemoryType,
        memory: Memory,
    ) -> UpdateDecision:
        existing = self.store.list(user_id, memory_type)
        candidates = self._nearest(memory, existing)
        if not candidates:
            self.store.add(user_id, memory_type, memory)
            return UpdateDecision(operation="add", memory=memory.to_dict())
        decision = UpdateDecision(**self.resolver(
            memory_type,
            memory.to_dict(),
            [item.to_dict() for _, item in candidates],
        ))
        if decision.operation == "add":
            selected = self._build(memory_type, decision.memory or memory.to_dict())
            self.store.add(user_id, memory_type, selected)
        elif decision.operation == "update":
            if decision.index is None:
                raise ValueError("update requires an index")
            source_index = candidates[decision.index][0]
            selected = self._build(memory_type, decision.memory or memory.to_dict())
            self.store.update(user_id, memory_type, source_index, selected)
        elif decision.operation == "delete":
            if decision.index is None:
                raise ValueError("delete requires an index")
            self.store.delete(user_id, memory_type, candidates[decision.index][0])
        return decision

    def _nearest(
        self,
        memory: Memory,
        existing: list[Memory],
    ) -> list[tuple[int, Memory]]:
        if not existing:
            return []
        query_vector = self.embedder(_text(memory))
        scored = [
            (_cosine(query_vector, self.embedder(_text(item))), index, item)
            for index, item in enumerate(existing)
        ]
        scored.sort(key=lambda row: row[0], reverse=True)
        return [(index, item) for score, index, item in scored[: self.top_k] if score > 0]

    def _build(self, memory_type: MemoryType, data: dict[str, Any]) -> Memory:
        return MEMORY_CLASSES[memory_type].from_dict(data)


def ingest_task_history(
    store: JsonlMemoryStore,
    updater: MemoryUpdater,
    record: RawTaskRecord,
) -> IngestionResult:
    trajectory = refine_trajectory(record)
    workspace = refine_workspace(record)
    persona = refine_persona(record)
    store.add(record.user_id, "trajectory", trajectory)
    operations = [updater.update(record.user_id, "workspace", workspace)]
    operations.extend(
        updater.update(record.user_id, "persona", item)
        for item in persona
    )
    return IngestionResult(
        trajectory=trajectory,
        workspace=workspace,
        persona=persona,
        operations=operations,
    )


def _text(memory: Memory) -> str:
    return json.dumps(memory.to_dict(), ensure_ascii=False, sort_keys=True)


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)
