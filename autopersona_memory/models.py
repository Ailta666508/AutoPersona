from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


MemoryType = Literal["trajectory", "workspace", "persona"]


@dataclass(frozen=True)
class TrajectoryMemory:
    query: str
    trace: list[Any] = field(default_factory=list)
    outcome: Any = None
    insight: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrajectoryMemory":
        return cls(
            query=str(data.get("query", "")),
            trace=list(data.get("trace", []) or []),
            outcome=data.get("outcome"),
            insight=str(data.get("insight", "")),
        )


@dataclass(frozen=True)
class WorkspaceMemory:
    task: Any = None
    interaction: list[Any] = field(default_factory=list)
    state: Any = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkspaceMemory":
        return cls(
            task=data.get("task"),
            interaction=list(data.get("interaction", []) or []),
            state=data.get("state"),
        )


@dataclass(frozen=True)
class PersonaMemory:
    topic: str
    preference: str
    strategy: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PersonaMemory":
        return cls(
            topic=str(data.get("topic", "")),
            preference=str(data.get("preference", "")),
            strategy=str(data.get("strategy", "")),
        )


Memory = TrajectoryMemory | WorkspaceMemory | PersonaMemory


@dataclass(frozen=True)
class MemoryBundle:
    trajectory: list[TrajectoryMemory] = field(default_factory=list)
    workspace: list[WorkspaceMemory] = field(default_factory=list)
    persona: list[PersonaMemory] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trajectory": [item.to_dict() for item in self.trajectory],
            "workspace": [item.to_dict() for item in self.workspace],
            "persona": [item.to_dict() for item in self.persona],
        }


@dataclass(frozen=True)
class RawTaskRecord:
    user_id: str
    query: str
    trace: list[Any] = field(default_factory=list)
    outcome: Any = None
    insight: str = ""
    task: Any = None
    interaction: list[Any] = field(default_factory=list)
    state: Any = None
    persona: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class SearchRequest:
    memory: MemoryType
    query: str


@dataclass(frozen=True)
class ClarificationRequest:
    question: str


@dataclass(frozen=True)
class PersonaRequest:
    user_id: str
    query: str
    dag: dict[str, Any] = field(default_factory=dict)
    additional_tools: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PersonaDecision:
    action: Literal["clarify", "final"]
    memories: MemoryBundle = field(default_factory=MemoryBundle)
    clarification: ClarificationRequest | None = None
    answer: str | None = None


@dataclass(frozen=True)
class UpdateDecision:
    operation: Literal["add", "update", "delete", "none"]
    index: int | None = None
    memory: dict[str, Any] | None = None
