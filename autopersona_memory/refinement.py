from __future__ import annotations

from .models import PersonaMemory, RawTaskRecord, TrajectoryMemory, WorkspaceMemory


def refine_trajectory(record: RawTaskRecord) -> TrajectoryMemory:
    return TrajectoryMemory(
        query=record.query,
        trace=list(record.trace),
        outcome=record.outcome,
        insight=record.insight,
    )


def refine_workspace(record: RawTaskRecord) -> WorkspaceMemory:
    return WorkspaceMemory(
        task=record.task,
        interaction=list(record.interaction),
        state=record.state,
    )


def refine_persona(record: RawTaskRecord) -> list[PersonaMemory]:
    return [PersonaMemory.from_dict(item) for item in record.persona]
